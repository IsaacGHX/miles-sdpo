"""Base-model pass@k learnability filter for the search (multi-hop QA) domain.

WHY (see memory "search-needs-passk-difficulty-filter"): in the 3-domain
pure-distill run, search eval REGRESSED while math/code improved. Root cause is
NOT prompt/format -- it is learnability: under --sdpo-pure-distill search has no
task reward, half the groups have no correct peer to distill (`has_prefix`
11/22), and unfiltered FlashRAG hotpot/2wiki is mostly outside the useful band
(too easy -> no gradient, or too hard -> no correct peer for the teacher-prefix).

WHAT: run the BASE model (Qwen3-4B, no-think, web_search tool) k times per
question over the search TRAIN and VAL pools, EM-grade each sample, compute
pass@1 / pass@k, then:
  - TRAIN: keep questions whose pass@k lands in the [--train-lo, --train-hi] band
    (default 15%-75%) -- learnable, and a correct peer exists for distillation.
  - VAL:   keep --val-n questions with pass@1 < --val-hi (default 30%) -- so the
    held-out eval is not saturated and can actually show improvement.

HOW: talks to a running SGLang OpenAI-compatible server (base Qwen3-4B) for
generation and to the running torch retriever (:8000) for web_search, reusing
the SAME native tool-calling contract as training (registry.execute_tool). No
training stack, no Megatron -- just generation + grading.

Run INSIDE the enroot container (has torch, the tokenizer, the repo, and the
mounted data), AFTER launching:
  1. the retriever (:8000)  -- launch_retriever
  2. an SGLang server for /root/Qwen3-4B on the free GPUs, e.g.:
       python -m sglang.launch_server --model-path /root/Qwen3-4B \
         --tool-call-parser qwen25 --host 127.0.0.1 --port 30000 --tp 4
Then:
  python -m examples.SDPO_ReAct.data.passk_filter_search \
    --in /root/data/search_data/search_train.jsonl \
    --out /root/data/search_data/search_train_passk.jsonl \
    --k 8 --mode train --train-lo 0.15 --train-hi 0.75
  python -m examples.SDPO_ReAct.data.passk_filter_search \
    --in /root/data/search_data/hotpotqa_val.jsonl ... --mode val --val-hi 0.30 --val-n 100
"""

import argparse
import asyncio
import json
import os
import re

import aiohttp

from examples.SDPO_ReAct.tools.registry import execute_tool

SGLANG_URL = os.environ.get("PASSK_SGLANG_URL", "http://127.0.0.1:30000/v1/chat/completions")
MAX_TURNS = int(os.environ.get("PASSK_MAX_TURNS", "8"))
MAX_TOKENS_PER_TURN = int(os.environ.get("PASSK_MAX_TOKENS", "2048"))
CONCURRENCY = int(os.environ.get("PASSK_CONCURRENCY", "64"))

# EM grader (search-r1 normalize_answer / em_check), imported lazily.
def _load_em():
    try:
        from examples.search_r1.qa_em_format import em_check
        return em_check
    except Exception:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "search-r1", "qa_em_format.py")
        spec = importlib.util.spec_from_file_location("qa_em_format", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.em_check


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def _extract_answer(text: str):
    m = _ANSWER_RE.findall(text or "")
    return m[-1].strip() if m else None


async def _one_rollout(session, messages, tools):
    """Run ONE native multi-turn tool-calling trajectory to completion.
    Returns the final assistant text (all turns concatenated for answer scan)."""
    convo = list(messages)
    full = []
    for _ in range(MAX_TURNS):
        payload = {
            "model": "qwen3",
            "messages": convo,
            "tools": tools,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": MAX_TOKENS_PER_TURN,
            # Qwen3 no-thinking: the server template honors this via chat_template_kwargs
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            async with session.post(SGLANG_URL, json=payload) as resp:
                data = await resp.json()
        except Exception as e:
            full.append(f"[gen-error {e}]")
            break
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        content = msg.get("content") or ""
        full.append(content)
        tool_calls = msg.get("tool_calls") or []
        # Append the assistant turn to the conversation.
        convo.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        if not tool_calls:
            break  # model gave a final answer (or stopped)
        # Execute each tool call and feed observations back.
        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            try:
                params = json.loads(args) if isinstance(args, str) else args
            except Exception:
                params = {"query": args}
            obs = await execute_tool(name, params)
            convo.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": obs,
            })
    return "\n".join(full)


async def _score_question(session, sem, em_check, row, k):
    """k independent rollouts for one question; return #correct out of k."""
    messages = row["prompt"]
    tools = row["tools"]
    golden = (row.get("metadata") or {}).get("golden_answers") or ([row.get("label")] if row.get("label") else [])

    async def _run():
        async with sem:
            text = await _one_rollout(session, messages, tools)
        pred = _extract_answer(text)
        if pred is None or not golden:
            return 0
        return 1 if em_check(pred, golden) else 0

    results = await asyncio.gather(*[_run() for _ in range(k)])
    return sum(results)


def _shuffle(rows, seed):
    """Deterministic interleave-ish shuffle (no Math.random/Date dependency)."""
    import random
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows


def _ensure_http_client():
    """registry.execute_tool -> _handle_web_search -> miles.utils.http_utils.post
    uses a module-global httpx client that is None until the TRAINING bootstrap
    calls init_http_client(args). This standalone sweep has no such args, so the
    client stays None and every web_search fails with "'NoneType' object has no
    attribute 'post'" -- the model then silently answers from memory and the
    pass@k is measuring the WRONG (tool-free) distribution. Initialize the client
    directly here so web_search actually reaches the retriever."""
    import httpx
    import miles.utils.http_utils as hu
    if hu._http_client is None:
        hu._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=CONCURRENCY * 2),
            timeout=httpx.Timeout(None),
        )
        print("[passk] initialized miles http_utils._http_client for web_search", flush=True)


async def main_async(args):
    _ensure_http_client()
    em_check = _load_em()
    rows = [json.loads(l) for l in open(args.infile) if l.strip()]
    if args.shuffle:
        rows = _shuffle(rows, args.seed)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[passk] {len(rows)} candidates, k={args.k}, mode={args.mode}, "
          f"target_kept={args.target_kept or 'all'}", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    scored = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # process in chunks to bound memory + give progress
        CHUNK = 50
        for i in range(0, len(rows), CHUNK):
            chunk = rows[i : i + CHUNK]
            counts = await asyncio.gather(
                *[_score_question(session, sem, em_check, r, args.k) for r in chunk]
            )
            for r, c in zip(chunk, counts):
                passk = c / args.k
                pass1 = c / args.k  # with k samples, empirical pass@1 == mean correctness
                scored.append((r, c, passk, pass1))
            done = min(i + CHUNK, len(rows))
            band = sum(1 for _, _, pk, _ in scored if args.train_lo <= pk <= args.train_hi)
            print(f"[passk] {done}/{len(rows)}  in-band(train)={band}", flush=True)
            # Early stop once we've collected enough in-band TRAIN rows.
            if args.mode == "train" and args.target_kept and band >= args.target_kept:
                print(f"[passk] reached target_kept={args.target_kept} at {done} scanned; stopping.", flush=True)
                break

    # Selection
    if args.mode == "train":
        kept = [r for (r, c, pk, p1) in scored if args.train_lo <= pk <= args.train_hi]
        if args.target_kept:
            kept = kept[: args.target_kept]
        print(f"[passk] TRAIN kept {len(kept)}/{len(scored)} scanned in [{args.train_lo},{args.train_hi}] band", flush=True)
    else:  # val
        hard = [(r, p1) for (r, c, pk, p1) in scored if p1 < args.val_hi]
        hard.sort(key=lambda x: x[1])  # hardest first
        kept = [r for (r, _) in hard[: args.val_n]]
        print(f"[passk] VAL kept {len(kept)} (pass1<{args.val_hi}, cap {args.val_n}); {len(hard)} eligible", flush=True)

    with open(args.outfile, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # also dump the full pass@k table for inspection
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            for (r, c, pk, p1) in scored:
                q = r["prompt"][1]["content"] if len(r["prompt"]) > 1 else ""
                f.write(json.dumps({"q": q[:200], "correct": c, "k": args.k, "passk": pk}, ensure_ascii=False) + "\n")
    print(f"[passk] wrote {len(kept)} rows -> {args.outfile}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--stats-out", default=None, help="optional per-question pass@k jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--mode", choices=["train", "val"], required=True)
    ap.add_argument("--train-lo", type=float, default=0.15)
    ap.add_argument("--train-hi", type=float, default=0.75)
    ap.add_argument("--val-hi", type=float, default=0.30)
    ap.add_argument("--val-n", type=int, default=100)
    ap.add_argument("--target-kept", type=int, default=0,
                    help="TRAIN: stop scanning once this many in-band rows collected (0=scan all)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle candidates before scanning (mix datasets)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0, help="cap #questions (debug)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
