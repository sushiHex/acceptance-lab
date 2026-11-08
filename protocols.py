"""protocols.py: the two protocol surfaces, both thin over scene.py.

MCP server (stdio): the executor. One process per attempt; the attempt is bound at spawn through
ACCEPTANCE_ATTEMPT_DIR, so no tool takes an attempt argument. Two tools: inspect, finalize.

A2A judge (HTTP): receives only an attempt id, maps it through the run's registry to a directory,
reads that directory read-only, and returns a verdict Artifact. It never mutates.

    python protocols.py mcp                 # spawned by the coordinator
    python protocols.py judge RUN_DIR PORT  # spawned by the coordinator
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import scene

ROOT = Path(__file__).resolve().parent


def load_tasks(path: Path = ROOT / "tasks.jsonl") -> dict:
    return {t["id"]: t for t in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def contract_sha(contract: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def binding_error(manifest: dict, attempt_id: str, entry: dict, t: dict, attempt_dir: Path) -> str | None:
    """The executor's manifest must bind this attempt id to this task, contract, proposal, receipt
    and candidate. Any disagreement means the directory is not this attempt's."""
    if manifest.get("attempt") != attempt_id:
        return "manifest attempt id differs"
    if manifest.get("task_id") != entry["task_id"] or manifest.get("contract_sha256") != contract_sha(t["contract"]):
        return "manifest task or contract differs"
    if manifest.get("proposal_sha256") != scene.sha256(attempt_dir / "proposal.json"):
        return "proposal changed after finalize"
    if manifest.get("receipt_sha256") != scene.sha256(attempt_dir / "receipt.json"):
        return "receipt changed after finalize"
    if manifest.get("candidate") != scene.bundle_hashes(attempt_dir / "candidate"):
        return "candidate changed after finalize"
    return None


# ---------------------------------------------------------------- MCP executor

def serve_mcp() -> None:
    from mcp.server.fastmcp import FastMCP

    attempt = Path(os.environ["ACCEPTANCE_ATTEMPT_DIR"])
    task = json.loads((attempt / "task.json").read_text(encoding="utf-8"))
    state = {"finalized": (attempt / "receipt.json").exists()}
    server = FastMCP("acceptance-executor", log_level="WARNING")

    @server.tool(structured_output=True)
    def inspect() -> dict[str, Any]:
        """Typed facts about the frozen original scene and the approved assets."""
        return scene.inspect(attempt / "original", task)

    @server.tool(structured_output=True)
    def finalize(proposal: dict[str, Any], execution_allowed: bool) -> dict[str, Any]:
        """Apply the proposal to the scratch candidate, all-or-nothing. Callable once per attempt."""
        if state["finalized"]:
            raise ValueError("attempt already finalized")
        state["finalized"] = True
        receipt = scene.apply(attempt / "candidate", task, proposal, execution_allowed)
        (attempt / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        (attempt / "proposal.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")
        manifest = {"attempt": os.environ["ACCEPTANCE_ATTEMPT_ID"], "task_id": task["id"],
                    "contract_sha256": contract_sha(task["contract"]),
                    "proposal_sha256": scene.sha256(attempt / "proposal.json"),
                    "receipt_sha256": scene.sha256(attempt / "receipt.json"),
                    "candidate": scene.bundle_hashes(attempt / "candidate")}
        (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if any(o["status"] != "permitted" for o in receipt["operations"]):
            raise ValueError(json.dumps({"blocked": receipt["operations"]}))
        return {"files": sorted(p.relative_to(attempt).as_posix() for p in attempt.rglob("*") if p.is_file())}

    server.run("stdio")


# ---------------------------------------------------------------- A2A judge

def build_judge(run_dir: Path, port: int):
    from starlette.applications import Starlette
    from a2a.helpers import get_data_parts, new_data_part, new_task_from_user_message
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

    tasks = load_tasks(run_dir / "tasks.jsonl")                      # frozen at run start
    fixtures = json.loads((run_dir / "fixtures.json").read_text(encoding="utf-8"))
    registry_file = run_dir / "registry.json"

    class Judge(AgentExecutor):
        async def execute(self, ctx, q):
            task = ctx.current_task or new_task_from_user_message(ctx.message)
            if not ctx.current_task:
                await q.enqueue_event(task)
            up = TaskUpdater(event_queue=q, task_id=task.id, context_id=task.context_id)
            try:
                attempt_id = get_data_parts(ctx.message.parts)[0]["attempt"]
                registry = json.loads(registry_file.read_text(encoding="utf-8"))
                entry = registry.get(attempt_id)
                if entry is None:
                    await up.reject()
                    return
                t = tasks.get(entry["task_id"])
                if t is None or (run_dir / entry["dir"]).resolve().parent != (run_dir / "attempts").resolve():
                    await up.reject()
                    return
                if entry.get("contract") is not None:   # demo: the edited contract the coordinator froze
                    t = {**t, "contract": entry["contract"]}
                attempt_dir = run_dir / entry["dir"]
                if scene.bundle_hashes(attempt_dir / "original") != fixtures.get(t["id"]):
                    await up.reject()
                    return
                mf = attempt_dir / "manifest.json"
                manifest = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
                if binding_error(manifest, attempt_id, entry, t, attempt_dir):
                    await up.reject()
                    return
                v = await asyncio.to_thread(scene.evaluate, t, attempt_dir)
                data = {"attempt": attempt_id, "task_id": t["id"], "grade": v["grade"],
                        "decision": v["decision"], "contained_by": v.get("contained_by") or "",
                        "error_code": v.get("error_code", ""), "contract_sha256": contract_sha(t["contract"]),
                        "manifest_sha256": scene.sha256(mf), "hashes": v["hashes"], "checks": v["checks"]}
                await up.add_artifact([new_data_part(data, "application/json")], name="verdict")
                await up.complete()
            except Exception:  # noqa: BLE001
                await up.failed()

        async def cancel(self, ctx, q):
            raise NotImplementedError

    url = f"http://127.0.0.1:{port}"
    card = AgentCard(
        name="acceptance-judge", description="Grades one attempt against its acceptance contract.",
        version="0.1.0", default_input_modes=["application/json"], default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=False),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")],
        skills=[AgentSkill(id="grade_attempt", name="grade_attempt",
                           description="Grade one attempt directory against its frozen task", tags=["judge"])])
    handler = DefaultRequestHandler(agent_executor=Judge(), task_store=InMemoryTaskStore(), agent_card=card)
    return card, Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])


def serve_judge(run_dir: Path, port: int) -> None:
    import uvicorn
    _, app = build_judge(run_dir, port)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    if sys.argv[1:2] == ["mcp"]:
        serve_mcp()
    elif sys.argv[1:2] == ["judge"]:
        serve_judge(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        sys.exit(__doc__)
