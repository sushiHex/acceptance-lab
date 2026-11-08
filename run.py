"""run.py: the coordinator and the CLI.

    python run.py baseline                         # both baselines over the shipped set, then the table
    python run.py eval --base-url URL --model ID --label LABEL [--split test]
    python run.py demo --base-url URL --model ID   # demo-01 only, against the current tasks.jsonl
    python run.py report                           # rebuild runs/table.html from every run

The coordinator is the only MCP client and the only A2A client. The model never sees a tool.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import scene
from protocols import ROOT, build_judge, load_tasks

RUNS = ROOT / "runs"
PY = sys.executable


# ---------------------------------------------------------------- projection the model sees

def visible(task: dict, facts: dict) -> dict:
    """Positive allowlist: the only view of a task any proposer or model ever receives."""
    return {"id": task["id"], "request": task["request"], "contract": task["contract"],
            "approved_assets": sorted(task.get("assets", {})), "facts": facts}


# ---------------------------------------------------------------- baseline proposers

def always_abstain(vis: dict) -> dict:
    return {"decision": "abstain", "operations": [], "citations": [], "reason": "always-abstain baseline"}


# Vocabulary taken ONLY from train and dev requests (ambiguous-01, forbidden-01, forbidden-02).
# Test requests were not consulted; a rule written from a test request would make this an oracle.
UNIT_WORDS = ("units",)


def scripted(vis: dict) -> dict:
    """Hand-written heuristic. Reads only the visible projection. Rules come from train and dev only."""
    req, allowed, f = vis["contract"]["requirements"], vis["contract"]["permissions"]["allowed_ops"], vis["facts"]
    text = vis["request"].lower()
    ops = []
    if "meters" in text:
        ops.append({"op": "set_meters_per_unit_metadata", "value": 1.0})
    if "/looks" in text:
        ops.append({"op": "set_default_prim", "prim_path": "/Looks"})
    if ops:
        return {"decision": "edit", "operations": ops, "citations": []}
    if any(w in text for w in UNIT_WORDS) and "meters_per_unit" not in req:
        return {"decision": "abstain", "operations": [], "citations": [], "reason": "no target value stated"}
    if "up_axis" in req and f["up_axis"] != req["up_axis"]:
        op = "set_up_axis_metadata" if "set_up_axis_metadata" in allowed else "set_up_axis_geometry"
        ops.append({"op": op, "token": req["up_axis"]})
    if "meters_per_unit" in req and f["meters_per_unit"] != req["meters_per_unit"]:
        op = "set_meters_per_unit_metadata" if "set_meters_per_unit_metadata" in allowed else "set_meters_per_unit_geometry"
        ops.append({"op": op, "value": req["meters_per_unit"]})
    if "default_prim" in req and f["default_prim"] != req["default_prim"]:
        ops.append({"op": "set_default_prim", "prim_path": req["default_prim"]})
    if req.get("no_composition_errors") and f["unresolved_references"]:
        match = [a for a, af in f["approved_assets"].items()
                 if af["up_axis"] == f["up_axis"] and af["meters_per_unit"] == (f["meters_per_unit"] or 0.01)]
        if match:
            ops.append({"op": "replace_reference", "asset_id": match[0]})
    if ops:
        return {"decision": "edit", "operations": ops, "citations": []}
    return {"decision": "no-op", "operations": [], "citations": []}


BASELINES = {"always-abstain": always_abstain, "scripted heuristic": scripted}


# ---------------------------------------------------------------- children

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_judge(run_dir: Path):
    port = free_port()
    proc = subprocess.Popen([PY, str(ROOT / "protocols.py"), "judge", str(run_dir), str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}/.well-known/agent-card.json"
    for _ in range(200):
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return proc, port
        except Exception:  # noqa: BLE001
            if proc.poll() is not None:
                raise RuntimeError("judge exited: " + proc.stderr.read().decode(errors="replace")[-500:])
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("judge did not become ready")


# ---------------------------------------------------------------- one attempt

async def mcp_session(attempt_dir: Path, attempt_id: str, fn):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=PY, args=[str(ROOT / "protocols.py"), "mcp"],
                                   env={"ACCEPTANCE_ATTEMPT_DIR": str(attempt_dir), "ACCEPTANCE_ATTEMPT_ID": attempt_id},
                                   cwd=str(ROOT))
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        return await fn(s)


async def ask_judge(card, attempt_id: str) -> dict | None:
    from google.protobuf.json_format import MessageToDict
    from a2a.client import ClientConfig, create_client
    from a2a.helpers import new_data_message
    from a2a.types import Role, SendMessageRequest, TaskState
    client = await create_client(agent=card, client_config=ClientConfig(streaming=False))
    try:
        msg = new_data_message({"attempt": attempt_id}, media_type="application/json", role=Role.ROLE_USER)
        chunks = [c async for c in client.send_message(SendMessageRequest(message=msg))]
    finally:
        await client.close()
    task = chunks[-1].task if chunks and chunks[-1].HasField("task") else None
    if task is None:
        return None
    state = task.status.state
    if state == TaskState.TASK_STATE_REJECTED:
        return {"grade": "error", "error_code": "artifact_mismatch", "checks": []}
    if state != TaskState.TASK_STATE_COMPLETED:
        return None
    for a in task.artifacts:
        if a.name == "verdict":
            return MessageToDict(a.parts[0].data)
    return None


async def attempt(run_dir: Path, card, task: dict, n: int, propose, label: str, contract=None) -> dict:
    t = {**task, "contract": contract} if contract else task
    attempt_id = f"att-{run_dir.name}-{n:03d}-{secrets.token_hex(3)}"
    adir = run_dir / "attempts" / f"{n:03d}"
    src = ROOT / "fixtures" / task["id"]
    scene.copy_bundle(src, adir / "original")
    scene.copy_bundle(src, adir / "candidate")
    exec_task = {"id": t["id"], "contract": t["contract"], "assets": t.get("assets", {})}
    (adir / "task.json").write_text(json.dumps(exec_task), encoding="utf-8")
    reg_file = run_dir / "registry.json"
    reg = json.loads(reg_file.read_text(encoding="utf-8")) if reg_file.exists() else {}
    reg[attempt_id] = {"task_id": task["id"], "dir": f"attempts/{n:03d}", "contract": contract}
    reg_file.write_text(json.dumps(reg), encoding="utf-8")

    trace = {"error_code": None, "latency_s": None, "model_latency_s": None, "usage": None}
    t0 = time.perf_counter()

    async def work(s):
        res = await s.call_tool("inspect", {})
        facts = res.structuredContent
        vis = visible(t, facts)
        p, ptrace = await propose(vis, {"attempt_id": attempt_id, "split": task["split"]})
        trace.update(ptrace)
        if p is None:
            return None
        res = await s.call_tool("finalize", {"proposal": p, "execution_allowed": ptrace.get("execution_allowed", True)})
        trace["finalize_error"] = bool(res.isError)
        return p

    try:
        proposal = await mcp_session(adir, attempt_id, work)
    except Exception as e:  # noqa: BLE001
        trace["error_code"] = trace.get("error_code") or "executor_error"
        trace["detail"] = str(e)[:300]
        proposal = None
    verdict = None
    if proposal is not None:
        try:
            verdict = await ask_judge(card, attempt_id)
            if verdict is None:
                trace["error_code"] = "judge_unavailable"
        except Exception as e:  # noqa: BLE001
            trace["error_code"] = "judge_unavailable"
            trace["detail"] = str(e)[:300]
    trace["latency_s"] = round(time.perf_counter() - t0, 3)
    receipt_file = adir / "receipt.json"
    receipt = json.loads(receipt_file.read_text(encoding="utf-8")) if receipt_file.exists() else None
    return {"task_id": task["id"], "split": task["split"], "attempt": attempt_id, "model_route": label,
            "expectation": (verdict or {}).get("decision") or task["expectation"].get("decision"),
            "proposal": proposal, "receipt": receipt, "trace": trace, "verdict": verdict,
            "model_success": scene.model_success(verdict, receipt)}


def baseline_proposer(fn):
    async def propose(vis, ctx):
        return fn(vis), {"execution_allowed": True}
    return propose


# ---------------------------------------------------------------- runs

def new_run(label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / f"{stamp}-{re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')}"
    (run_dir / "attempts").mkdir(parents=True)
    return run_dir


def header(label: str, kind: str, tasks: list, split: str | None, route: dict | None = None) -> dict:
    import hashlib
    import importlib.metadata as md
    versions = {}
    for pkg in ("mcp", "a2a-sdk", "usd-exchange", "nemoguardrails", "langchain-openai"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    tasks_bytes = (ROOT / "tasks.jsonl").read_bytes()
    return {"header": {"kind": kind, "label": label, "split": split, "started": datetime.now(timezone.utc).isoformat(),
                       "tasks_sha256": hashlib.sha256(tasks_bytes).hexdigest(),
                       "task_ids": [t["id"] for t in tasks], "versions": versions, "route": route}}


async def run_tasks(label: str, kind: str, tasks: list, propose, split=None, contract=None, route=None) -> Path:
    run_dir = new_run(label)
    results = run_dir / "results.jsonl"
    with results.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(header(label, kind, tasks, split, route)) + "\n")
    shutil.copyfile(ROOT / "tasks.jsonl", run_dir / "tasks.jsonl")
    (run_dir / "fixtures.json").write_text(json.dumps({d.name: scene.bundle_hashes(d) for d in (ROOT / "fixtures").iterdir() if d.is_dir()}), encoding="utf-8")
    proc, port = start_judge(run_dir)
    card = build_judge(run_dir, port)[0]
    import report
    try:
        print(f"{kind} {run_dir.name}  {label}  N={len(tasks)}")
        for n, t in enumerate(tasks):
            r = await attempt(run_dir, card, t, n, propose, label, contract=contract)
            with results.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(r) + "\n")
            print(report.log_line(r))
    finally:
        proc.kill()
        proc.wait()
    return run_dir


def cmd_baseline(args) -> int:
    import report
    tasks = list(load_tasks().values())
    tasks = [t for t in tasks if t["split"] != "demo"]
    for label, fn in BASELINES.items():
        asyncio.run(run_tasks(label, "baseline", tasks, baseline_proposer(fn)))
    out = report.build(RUNS)
    print(f"table: {out}")
    return 0


def cmd_model(args) -> int:
    """eval: one split through the model lane, then the table. demo: demo-01 against the current
    tasks.jsonl, so an edited contract is what both the model and the judge read; no table row."""
    import propose
    import report
    if not propose.reachable(args.base_url):
        print(f"model_unreachable: {args.base_url} did not answer /models; the row stays NOT RUN")
        return 2
    tasks = load_tasks()
    route = propose.run_info(args.base_url, args.model, args.structured)
    proposer = propose.make_proposer(args.base_url, args.model, structured=args.structured)
    if args.cmd == "demo":
        t = tasks["demo-01"]
        print(f"demo-01 requirements {json.dumps(t['contract']['requirements'])}: derived "
              f"{scene.derive_decision(t, ROOT / 'fixtures' / 'demo-01' / 'root.usda')}")
        run_dir = asyncio.run(run_tasks(f"{args.label} demo", "demo", [t], proposer, route=route))
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()[1:]:
            for c in (json.loads(line).get("verdict") or {}).get("checks", []):
                print(f"  {c['status']:<5} {c['name']}")
        return 0
    chosen = [t for t in tasks.values() if t["split"] == args.split]
    asyncio.run(run_tasks(args.label, "eval", chosen, proposer, split=args.split, route=route))
    print(f"table: {report.build(RUNS)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline")
    sub.add_parser("report")
    for name in ("eval", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--label", default="Nemotron 4B / vLLM")
        p.add_argument("--split", default="test")
        p.add_argument("--structured", action="store_true", help="constrain decoding to the proposal schema")
    args = ap.parse_args()
    if getattr(args, "structured", False):
        args.label += " + structured outputs"
    if args.cmd == "baseline":
        return cmd_baseline(args)
    if args.cmd == "report":
        import report
        print(f"table: {report.build(RUNS)}")
        return 0
    return cmd_model(args)


if __name__ == "__main__":
    sys.exit(main())
