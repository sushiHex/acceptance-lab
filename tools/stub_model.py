"""stub_model.py: an OpenAI-compatible endpoint that answers with the scripted heuristic.

Exercises the real model lane (LangChain, the repair turn, Guardrails, RAG) with no GPU, so the
wiring is proven before vLLM exists. It is a test double, never a table row.

    python tools/stub_model.py PORT [scripted|fence|junk|obey-notes]

  scripted    reply with run.scripted(visible) parsed out of the prompt's TASK block
  fence       wrap the first reply in a code fence, so the repair turn has to fix it
  junk        never reply with JSON (proposal_unparseable)
  obey-notes  follow a planted rule note: add the forbidden operations it asks for (rail blocks)
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run  # noqa: E402

MODE = sys.argv[2] if len(sys.argv) > 2 else "scripted"


def answer(messages: list) -> str:
    users = [m["content"] for m in messages if m["role"] == "user"]
    vis = json.loads(users[0].split("\nTASK\n", 1)[1])
    p = run.scripted(vis)
    if MODE == "junk":
        return "I would set the default prim."
    if MODE == "fence" and len(users) == 1:
        return "```json\n" + json.dumps(p) + "\n```"
    if MODE == "obey-notes" and "pre-approved" in users[0]:
        p = {"decision": "edit", "citations": ["injection-note"], "operations": p["operations"] + [
            {"op": "set_meters_per_unit_metadata", "value": 0.01}, {"op": "set_default_prim", "prim_path": "/Looks"}]}
    return json.dumps(p)


class H(BaseHTTPRequestHandler):
    def _send(self, body: dict):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "stub", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = answer(req["messages"])
        self._send({"id": "stub", "object": "chat.completion", "created": 0, "model": req["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
