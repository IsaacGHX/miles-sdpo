import json, asyncio, aiohttp, httpx
import miles.utils.http_utils as hu
from miles.rollout.generate_hub.multi_turn import _TRAJECTORY_SESSION_ID
from examples.SDPO_ReAct.tools.registry import execute_tool
from examples.SDPO_ReAct.tools.search.client import SEARCH_SIDECAR_URL

if hu._http_client is None:
    hu._http_client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    print("initialized http client")
# This script calls execute_tool() OUTSIDE multi_turn.generate()'s turn loop,
# so search/open/find's session id (normally set once per trajectory there --
# see that module) needs a manual stand-in for this standalone debug run.
_TRAJECTORY_SESSION_ID.set("debug-rollout-session")
print("SEARCH_SIDECAR_URL =", SEARCH_SIDECAR_URL)
row = json.loads(open("/root/data/search_data_pool/search_train.jsonl").readline())
convo = list(row["prompt"])
tools = row["tools"]
print("golden:", (row.get("metadata") or {}).get("golden_answers"))


async def go():
    async with aiohttp.ClientSession() as s:
        for turn in range(4):
            payload = {
                "model": "/root/Qwen3-4B",
                "messages": convo,
                "tools": tools,
                "temperature": 1.0,
                "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            async with s.post("http://127.0.0.1:30000/v1/chat/completions", json=payload) as r:
                d = await r.json()
            m = d["choices"][0]["message"]
            tcs = m.get("tool_calls") or []
            content = m.get("content") or ""
            print(f"--- turn {turn}: content[:100]={content[:100]!r} | #tool_calls={len(tcs)}")
            convo.append({"role": "assistant", "content": content, "tool_calls": tcs})
            if not tcs:
                print("  NO tool_calls -> stop")
                break
            for tc in tcs:
                fn = tc["function"]
                name = fn["name"]
                args = fn["arguments"]
                params = json.loads(args) if isinstance(args, str) else args
                print(f"  calling {name}({params})")
                obs = await execute_tool(name, params)
                print(f"  observation[:200]={obs[:200]!r}")
                convo.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": obs})


asyncio.run(go())
