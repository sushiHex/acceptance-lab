"""Probe 3: LLMRails with no models, a Colang 1 output rail calling a registered action.

Pass: no LLM key, no import error; a permitted proposal PASSES, a forbidden one and junk are
BLOCKED by the named rail; the action really ran each time and saw the context.
"""
import asyncio
import json
import os
import time

for k in ("OPENAI_API_KEY", "NVIDIA_API_KEY", "NGC_API_KEY"):
    os.environ.pop(k, None)

t0 = time.time()
from nemoguardrails import LLMRails, RailsConfig  # noqa: E402
from nemoguardrails.actions import action  # noqa: E402

CONFIG_YML = """
colang_version: "1.0"
models: []
rails:
  output:
    flows:
      - stagewright preflight
"""
RAILS_CO = """
define bot block proposal
  "PREFLIGHT_BLOCKED"

define flow stagewright preflight
  $permitted = execute preflight_rail(proposal=$bot_message)
  if not $permitted
    bot block proposal
    stop
"""

ALLOWED = {"set_default_prim"}
seen = []


@action(name="preflight_rail")
async def preflight_rail(proposal: str = "", context: dict | None = None) -> bool:
    seen.append(((context or {}).get("attempt_id"), proposal))
    try:
        p = json.loads(proposal)
        return all(op.get("op") in ALLOWED for op in p.get("operations", []))
    except Exception:
        return False


cfg = RailsConfig.from_content(yaml_content=CONFIG_YML, colang_content=RAILS_CO)
rails = LLMRails(cfg)
rails.register_action(preflight_rail, name="preflight_rail")
print(f"INFO  LLMRails init {time.time() - t0:.1f}s")

PERMITTED = json.dumps({"decision": "edit", "operations": [{"op": "set_default_prim", "target": "/World"}]})
FORBIDDEN = json.dumps({"decision": "edit", "operations": [{"op": "set_up_axis_metadata", "token": "Y"}]})


async def run(text):
    from nemoguardrails.rails.llm.options import RailType
    msgs = [{"role": "context", "content": {"attempt_id": "probe-1"}},
            {"role": "user", "content": "probe"},
            {"role": "assistant", "content": text}]
    return await rails.check_async(msgs, rail_types=[RailType.OUTPUT])


async def main():
    ok_r, bad_r, junk_r = await run(PERMITTED), await run(FORBIDDEN), await run("not json")
    for name, r in (("permitted", ok_r), ("forbidden", bad_r), ("junk", junk_r)):
        print(f"INFO  {name}: status={getattr(r, 'status', r)!s} rail={getattr(r, 'rail', None)!s}")
    s = lambda r: str(getattr(r, "status", "")).upper()
    good = ("PASS" in s(ok_r) and "BLOCK" in s(bad_r) and "BLOCK" in s(junk_r)
            and len(seen) == 3 and all(a == "probe-1" for a, _ in seen))
    print(f"INFO  action calls: {len(seen)}  attempt ids: {[a for a, _ in seen]}")
    print("INFO  initialized with no model and no OpenAI/NVIDIA/NGC key; network not instrumented")
    print("PROBE 3", "PASS" if good else "FAIL")
    return good


raise SystemExit(0 if asyncio.run(main()) else 1)
