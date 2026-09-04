"""Pluggable inference-engine backends for the TTS (test-time scaffolding)
harness.

Design goal (per the task's "make it scalable -- I can choose to use remote or
local" requirement): every ROLE in the scaffold loop (the solver, the
skill-writer, the meta-optimizer -- see scaffold.py / optimizer.py) picks its
engine independently from config, and switching a role between a local SGLang
server and a remote OpenAI/Anthropic API is a config change, never a code edit.

A "backend" here answers exactly ONE question:

    async def chat(messages, *, max_tokens, temperature, ...) -> ChatResult

i.e. messages-in / text-out. That is deliberately the SMALLEST contract that
both a local model and a remote API can honor -- unlike miles' training rollout
path (miles/rollout/generate_hub/*), which is token-in/token-out and needs
per-token logprobs + loss masks for the gradient. This harness is TEST-TIME
scaffolding: it optimizes a *prompt* (the skill-generation prompt), not model
weights, so it never needs the token-level training signal. That is exactly why
a remote API (which returns text, not input_ids or per-token logprobs) is a
first-class backend here, whereas it could only ever serve the eval path in the
training rollout code.

Three backends, all over raw async HTTP (httpx) so there are no hard SDK deps
(`openai`/`anthropic` need not be installed) and one code path serves both a
locally-launched server and a hosted API:

  - ``OpenAIChatBackend``  -> POST {base_url}/chat/completions  (OpenAI protocol)
        * a LOCAL SGLang server launched with ``python -m sglang.launch_server``
          exposes exactly this endpoint, so the SAME backend drives Qwen2.5-7B
          on localhost AND a hosted model like gpt-5.6-luna -- only base_url /
          model / api_key differ. This mirrors examples/SDPO/sdpo.py's own
          external-LLM path (_external_llm_chat), which already talks OpenAI
          /chat/completions to a configurable base_url.
  - ``AnthropicChatBackend`` -> POST {base_url}/messages          (Anthropic protocol)
        * system prompt is hoisted to a top-level ``system`` field; the
          x-api-key + anthropic-version headers are set. For Claude models.
  - ``SGLangGenerateBackend`` -> POST {base_url}/generate         (SGLang native)
        * the token-in/token-out endpoint miles' own rollout uses. Text-in via
          the chat template applied client-side. Offered for parity / when you
          want the raw SGLang endpoint rather than its OpenAI shim; returns
          logprobs when asked, which the OpenAI/Anthropic paths cannot.

The harness owns its OWN ``httpx.AsyncClient`` (see ``EngineClient``) rather
than reusing ``miles.utils.http_utils._http_client`` -- that global is only
initialized inside a running miles rollout (init_http_client(args)), and a
standalone script that imports a miles tool without initializing it hits a
silent ``None`` client (see the "http client None breaks standalone tools"
project note). Owning the client keeps this harness runnable on its own.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

Message = dict[str, Any]  # OpenAI-standard {"role", "content", ...}


# --------------------------------------------------------------------------- #
# Result + config value objects
# --------------------------------------------------------------------------- #
@dataclass
class ChatResult:
    """One completion. ``text`` is always the assistant's final text (reasoning
    stripped for models that expose a separate reasoning channel). The rest is
    best-effort and backend-dependent (None when a backend cannot supply it)."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning: str | None = None  # separate reasoning channel, if the backend has one
    raw: dict[str, Any] | None = None  # the raw provider response, for debugging
    tool_calls: list[dict[str, Any]] | None = None  # OpenAI-style [{"id","type","function":{"name","arguments"}}]


@dataclass
class EngineConfig:
    """How to reach ONE engine. Constructed from a config dict (see from_dict);
    ``api_key_env`` names the env var holding the key so keys never live in
    config files. All sampling defaults can be overridden per-call."""

    backend: str  # "openai" | "anthropic" | "sglang"
    model: str
    base_url: str
    api_key_env: str = ""  # env var name; "" means no auth header (local server)
    # sampling defaults (per-call overrides win)
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    # transport
    timeout: float = 600.0
    max_retries: int = 4
    # provider-specific extras merged into the request body verbatim
    extra_body: dict[str, Any] = field(default_factory=dict)
    # optional label for logs (e.g. "solver", "skill_writer", "optimizer")
    name: str = ""

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EngineConfig":
        known = {
            "backend",
            "model",
            "base_url",
            "api_key_env",
            "temperature",
            "top_p",
            "max_tokens",
            "timeout",
            "max_retries",
            "extra_body",
            "name",
        }
        base = {k: v for k, v in d.items() if k in known}
        # any unknown key is treated as a provider extra_body field (forward-compat)
        extras = {k: v for k, v in d.items() if k not in known}
        cfg = EngineConfig(**base)
        if extras:
            cfg.extra_body = {**cfg.extra_body, **extras}
        return cfg

    @property
    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        return os.environ.get(self.api_key_env, "") or ""


# --------------------------------------------------------------------------- #
# Shared HTTP client
# --------------------------------------------------------------------------- #
class EngineClient:
    """Owns a single shared ``httpx.AsyncClient`` reused across every backend
    (connection pooling). Create one per process; pass it to every engine."""

    def __init__(self, timeout: float = 600.0, max_connections: int = 256):
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), limits=limits)

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        max_retries: int,
    ) -> dict[str, Any]:
        """POST JSON with exponential backoff on transport / 5xx / 429. Raises
        the last error after ``max_retries`` attempts."""
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = await self._client.post(url, json=payload, headers=headers)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {resp.text[:500]}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, httpx.HTTPStatusError) as e:
                last_exc = e
                # 4xx (other than 429) are client errors -- do not retry.
                if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                    code = e.response.status_code
                    if 400 <= code < 500 and code != 429:
                        raise
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2.0**attempt, 30.0))
        assert last_exc is not None
        raise last_exc

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class Engine:
    """Base class. One engine == one config + the shared client. Subclasses
    implement ``_request`` (build payload/headers/url, parse response)."""

    def __init__(self, config: EngineConfig, client: EngineClient):
        self.config = config
        self.client = client

    @property
    def name(self) -> str:
        return self.config.name or f"{self.config.backend}:{self.config.model}"

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        c = self.config
        params = {
            "temperature": c.temperature if temperature is None else temperature,
            "top_p": c.top_p if top_p is None else top_p,
            "max_tokens": c.max_tokens if max_tokens is None else max_tokens,
            "stop": stop,
            "tools": tools,
        }
        return await self._request(messages, params, **kwargs)

    async def complete(self, system: str, user: str, **kwargs: Any) -> str:
        """Convenience: single (system, user) -> text. Used by the skill-writer
        and meta-optimizer, which are always one system + one user turn."""
        res = await self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        return res.text

    async def _request(self, messages, params, **kwargs) -> ChatResult:  # pragma: no cover
        raise NotImplementedError


class OpenAIChatBackend(Engine):
    """OpenAI ``/chat/completions``. Serves BOTH a local SGLang OpenAI server and
    a hosted OpenAI-protocol model. Newer reasoning models (gpt-5*, o1/o3/o4)
    reject ``temperature`` and want ``max_completion_tokens`` -- handled here,
    same special-casing as examples/SDPO/sdpo.py::_external_llm_chat."""

    def _is_reasoning_model(self) -> bool:
        return self.config.model.startswith(("gpt-5", "o1", "o3", "o4"))

    async def _request(self, messages, params, **kwargs) -> ChatResult:
        c = self.config
        payload: dict[str, Any] = {"model": c.model, "messages": messages, **c.extra_body}
        if self._is_reasoning_model():
            payload["max_completion_tokens"] = params["max_tokens"]
            # reasoning models reject explicit temperature/top_p; omit them.
            if params.get("tools"):
                # gateway constraint (verified live): function tools are rejected
                # on /chat/completions unless reasoning_effort is 'none' (the
                # reasoning-capable alternative is the separate /v1/responses
                # endpoint, not used here to keep one request shape for all
                # roles). Only applies when tools are actually passed.
                payload["reasoning_effort"] = "none"
        else:
            payload["max_tokens"] = params["max_tokens"]
            payload["temperature"] = params["temperature"]
            payload["top_p"] = params["top_p"]
        if params.get("stop"):
            payload["stop"] = params["stop"]
        if params.get("tools"):
            payload["tools"] = params["tools"]

        headers = {"Content-Type": "application/json"}
        if c.api_key:
            headers["Authorization"] = f"Bearer {c.api_key}"

        out = await self.client.post_json(
            f"{c.base_url.rstrip('/')}/chat/completions", payload, headers, c.max_retries
        )
        choice = out["choices"][0]
        msg = choice.get("message", {})
        text = (msg.get("content") or "").strip()
        usage = out.get("usage", {}) or {}
        return ChatResult(
            text=text,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning=(msg.get("reasoning_content") or None),
            raw=out,
            tool_calls=msg.get("tool_calls") or None,
        )


class AnthropicChatBackend(Engine):
    """Anthropic ``/messages``. System prompt is hoisted to the top-level
    ``system`` field (Anthropic has no system role in the messages array);
    consecutive same-role turns are otherwise passed through. Auth via
    ``x-api-key`` + ``anthropic-version`` headers."""

    ANTHROPIC_VERSION = "2023-06-01"

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
        system_parts, rest = [], []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(m.get("content", "") or "")
            else:
                rest.append({"role": m["role"], "content": m.get("content", "")})
        return "\n\n".join(p for p in system_parts if p), rest

    async def _request(self, messages, params, **kwargs) -> ChatResult:
        c = self.config
        system, rest = self._split_system(messages)
        payload: dict[str, Any] = {
            "model": c.model,
            "messages": rest,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            **c.extra_body,
        }
        if system:
            payload["system"] = system
        if params.get("stop"):
            payload["stop_sequences"] = params["stop"]

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.ANTHROPIC_VERSION,
        }
        if c.api_key:
            headers["x-api-key"] = c.api_key

        out = await self.client.post_json(f"{c.base_url.rstrip('/')}/messages", payload, headers, c.max_retries)
        # content is a list of blocks; concatenate text blocks.
        blocks = out.get("content", []) or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        usage = out.get("usage", {}) or {}
        return ChatResult(
            text=text,
            finish_reason=out.get("stop_reason"),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            raw=out,
        )


class SGLangGenerateBackend(Engine):
    """SGLang native ``/generate`` (token/text-in, text+logprob-out) -- the same
    endpoint miles' own rollout uses. Chat templating is applied CLIENT-side (a
    tokenizer is loaded from ``tokenizer_path``), because /generate takes ``text``
    (or input_ids), not messages. Offered for parity with the training path and
    for when you want raw logprobs; most roles should prefer OpenAIChatBackend
    against the same server (simpler, no local tokenizer needed).

    ``extra_body`` may carry chat-template kwargs, e.g.
    ``{"chat_template_kwargs": {"enable_thinking": false}}``.
    """

    def __init__(self, config: EngineConfig, client: EngineClient):
        super().__init__(config, client)
        self._tokenizer = None  # lazily loaded

    def _tok(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            path = self.config.extra_body.get("tokenizer_path") or self.config.model
            self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return self._tokenizer

    async def _request(self, messages, params, **kwargs) -> ChatResult:
        c = self.config
        tok = self._tok()
        template_kwargs = c.extra_body.get("chat_template_kwargs", {})
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        sampling = {
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "max_new_tokens": params["max_tokens"],
            "skip_special_tokens": False,
        }
        if params.get("stop"):
            sampling["stop"] = params["stop"]
        payload = {"text": text, "sampling_params": sampling, "return_logprob": False}
        out = await self.client.post_json(f"{c.base_url.rstrip('/')}/generate", payload, {}, c.max_retries)
        # /generate returns {"text": ..., "meta_info": {...}} (or a list thereof).
        if isinstance(out, list):
            out = out[0]
        meta = out.get("meta_info", {}) or {}
        finish = meta.get("finish_reason")
        if isinstance(finish, dict):
            finish = finish.get("type")
        return ChatResult(
            text=(out.get("text") or "").strip(),
            finish_reason=finish,
            prompt_tokens=meta.get("prompt_tokens"),
            completion_tokens=meta.get("completion_tokens"),
            raw=out,
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_BACKENDS = {
    "openai": OpenAIChatBackend,
    "anthropic": AnthropicChatBackend,
    "sglang": SGLangGenerateBackend,
}


def build_engine(config: EngineConfig | dict[str, Any], client: EngineClient) -> Engine:
    if isinstance(config, dict):
        config = EngineConfig.from_dict(config)
    backend = _BACKENDS.get(config.backend)
    if backend is None:
        raise ValueError(f"unknown engine backend {config.backend!r}; expected one of {sorted(_BACKENDS)}")
    return backend(config, client)


def build_engines(
    role_configs: dict[str, dict[str, Any]], client: EngineClient
) -> dict[str, Engine]:
    """Build one engine per named role from a ``{role: engine_config}`` mapping
    (see configs/*.yaml). Each role points local or remote independently, so the
    solver can be a local Qwen2.5-7B while the skill-writer/optimizer is a remote
    model -- or all three the same engine. Roles sharing an identical config
    reuse ONE Engine instance (dedup by config identity)."""
    import json

    engines: dict[str, Engine] = {}
    by_key: dict[str, Engine] = {}
    for role, cfg_dict in role_configs.items():
        cfg = EngineConfig.from_dict({**cfg_dict, "name": cfg_dict.get("name", role)})
        # stable identity key: same target server+model+auth+extras -> one Engine.
        key = json.dumps(
            [cfg.backend, cfg.base_url, cfg.model, cfg.api_key_env, cfg.extra_body],
            sort_keys=True,
            default=str,
        )
        eng = by_key.get(key)
        if eng is None:
            eng = build_engine(cfg, client)
            by_key[key] = eng
        engines[role] = eng
    return engines
