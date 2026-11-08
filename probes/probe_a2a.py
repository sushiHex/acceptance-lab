"""Probe 2: hello-judge on a2a-sdk 1.3.0.

Pass: Agent Card served at the well-known path; a data Part round trip ends TASK_STATE_COMPLETED
with an artifact named "verdict" carrying the attempt id; and we learn how an integer comes back.
"""
import asyncio
import json
import socket
import urllib.request

import uvicorn
from starlette.applications import Starlette

from a2a.client import ClientConfig, create_client
from a2a.helpers import get_data_parts, new_data_message, new_data_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role, SendMessageRequest


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


PORT = free_port()
URL = f"http://127.0.0.1:{PORT}"


class Judge(AgentExecutor):
    async def execute(self, ctx, q):
        task = ctx.current_task or new_task_from_user_message(ctx.message)
        if not ctx.current_task:
            await q.enqueue_event(task)
        up = TaskUpdater(event_queue=q, task_id=task.id, context_id=task.context_id)
        req = get_data_parts(ctx.message.parts)[0]
        await up.add_artifact(
            [new_data_part({"attempt": req["attempt"], "grade": "accepted", "count": 3}, "application/json")],
            name="verdict")
        await up.complete()

    async def cancel(self, ctx, q):
        raise NotImplementedError


card = AgentCard(
    name="judge", description="hello judge", version="0.0.1",
    default_input_modes=["application/json"], default_output_modes=["application/json"],
    capabilities=AgentCapabilities(streaming=False),
    supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=URL, protocol_version="1.0")],
    skills=[AgentSkill(id="grade_attempt", name="grade_attempt", description="grade one attempt", tags=["judge"])])
handler = DefaultRequestHandler(agent_executor=Judge(), task_store=InMemoryTaskStore(), agent_card=card)
app = Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])


async def main():
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    t = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.05)
    ok = True
    try:
        body = await asyncio.to_thread(
            lambda: urllib.request.urlopen(f"{URL}/.well-known/agent-card.json", timeout=5).read())
        served = json.loads(body)
        print(f"INFO  agent card served: name={served.get('name')} skills={[s.get('id') for s in served.get('skills', [])]}")
        ok &= served.get("name") == "judge"

        skills = [s.get("id") for s in served.get("skills", [])]
        ifaces = served.get("supportedInterfaces") or served.get("supported_interfaces") or []
        ok &= skills == ["grade_attempt"] and served.get("version") == "0.0.1" and bool(ifaces)
        print(f"INFO  card version={served.get('version')} interfaces={ifaces}")

        client = await create_client(agent=card, client_config=ClientConfig(streaming=False))
        msg = new_data_message({"attempt": "a-1", "contract_sha256": "0" * 64},
                               media_type="application/json", role=Role.ROLE_USER)
        chunks = [c async for c in client.send_message(SendMessageRequest(message=msg))]
        await client.close()
        from google.protobuf.json_format import MessageToDict
        from a2a.types import TaskState
        task = chunks[-1].task if chunks and chunks[-1].HasField("task") else None
        state_ok = task is not None and task.status.state == TaskState.TASK_STATE_COMPLETED
        verdicts = [a for a in (task.artifacts if task else []) if a.name == "verdict"]
        data = MessageToDict(verdicts[0].parts[0].data) if verdicts else {}
        print(f"INFO  state_completed={state_ok} verdict_artifacts={len(verdicts)} data={data}")
        count = data.get("count")
        print(f"INFO  integer 3 decoded as {count!r} of type {type(count).__name__}")
        ok &= (state_ok and len(verdicts) == 1 and data.get("attempt") == "a-1"
               and data.get("grade") == "accepted" and isinstance(count, float))
    finally:
        srv.should_exit = True
        await t
    print("PROBE 2", "PASS" if ok else "FAIL")
    return ok


raise SystemExit(0 if asyncio.run(main()) else 1)
