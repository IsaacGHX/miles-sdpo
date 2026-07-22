"""Tool definitions (OpenAI function-call spec shape) for SDPO_ReAct.

Passed as ``--generate-tool-specs-path`` to
``miles.rollout.generate_hub.multi_turn.generate``, which forwards this list
straight into ``tokenizer.apply_chat_template(prompt, tools=tool_specs, ...)``
(see miles/rollout/generate_utils/generate_endpoint_utils.py::
compute_prompt_ids_from_sample) -- i.e. tool definitions are injected using the
model's OWN chat template, never a hand-written XML/tag block, so they render
identically to however Qwen2.5-Instruct (or any other --tito-model) formats
tool declarations for real tool_calls messages.

Base version wires ONE real tool (code_interpreter, backed by the Docker
sandbox sidecar in tools/docker/). cli_exec and web_search are registered here
as schema-only stubs -- declaring them now keeps the tool_specs contract
stable for a later pass, but tool_client.execute_tool raises NotImplementedError
for them so a misconfigured run fails loudly instead of silently no-opping.
"""

CODE_INTERPRETER_SPEC = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": (
            "Execute Python code in an isolated sandbox and return its stdout. "
            "Use this for calculations, symbolic math (sympy), or verifying a "
            "numeric answer before giving your final answer. The sandbox has "
            "sympy, numpy, and scipy preinstalled; it has no network access."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute. Print anything you want to see.",
                }
            },
            "required": ["code"],
        },
    },
}

# Schema-only stubs for the deferred tools (see module docstring). Kept out of
# `tool_specs` (below) so the model is never told it can call something that
# will raise -- register them there once tool_client.execute_tool implements
# a real backend.
CLI_EXEC_SPEC = {
    "type": "function",
    "function": {
        "name": "cli_exec",
        "description": "NOT YET IMPLEMENTED. Execute a shell/CLI command in the sandbox.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to run."}},
            "required": ["command"],
        },
    },
}

WEB_SEARCH_SPEC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "NOT YET IMPLEMENTED. Search the web and return top results.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query."}},
            "required": ["query"],
        },
    },
}

# The active tool set for the base version. Extend with CLI_EXEC_SPEC /
# WEB_SEARCH_SPEC once tool_client.execute_tool implements real backends for
# them -- see the NotImplementedError branches there.
tool_specs = [CODE_INTERPRETER_SPEC]
