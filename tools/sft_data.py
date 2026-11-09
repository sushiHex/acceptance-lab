"""sft_data.py: the judge labels the training set (rejection sampling over the train split).

    python tools/sft_data.py sample --base-url URL --model ID [--k 4] [--temperature 0.8] [--structured]
    python tools/sft_data.py build RUN_DIR [RUN_DIR ...] [--per-task 2]       # -> train/sft.jsonl

sample: K runs of the model over the train split at a sampling temperature, through the real
coordinator, so every candidate is executed by the MCP executor and graded by the A2A judge.
build: keep only candidates scene.model_success() accepts, train split only, at most --per-task
distinct proposals per task, taken in argument order (so list the model's own samples first and
any baseline run last). The prompt is rebuilt with propose.format_prompt, the one prompt every
model and every example sees, and each row is chat messages (system, user, assistant), so
training needs nothing from the lab. A task with no accepted candidate is reported, never filled
in by hand.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import propose  # noqa: E402
import run  # noqa: E402
import scene  # noqa: E402
from protocols import load_tasks  # noqa: E402

OUT = ROOT / "train" / "sft.jsonl"


def train_tasks() -> list[dict]:
    return [t for t in load_tasks().values() if t["split"] == "train"]


def tasks_sha() -> str:
    return hashlib.sha256((ROOT / "tasks.jsonl").read_bytes()).hexdigest()


def cmd_sample(args) -> int:
    if not propose.reachable(args.base_url):
        print(f"model_unreachable: {args.base_url}")
        return 2
    proposer = propose.make_proposer(args.base_url, args.model, structured=args.structured,
                                     temperature=args.temperature)
    route = propose.run_info(args.base_url, args.model, args.structured, args.temperature)

    async def runs():   # one event loop: the client's connection pool is bound to it
        for k in range(args.k):
            await run.run_tasks(f"sft candidates {args.model} {k}", "candidates", train_tasks(), proposer,
                                split="train", route=route, concurrency=args.concurrency)
    asyncio.run(runs())
    return 0


def prompt_for(task: dict) -> str:
    exec_task = {"id": task["id"], "contract": task["contract"], "assets": task.get("assets", {})}
    vis = run.visible(task, scene.inspect(ROOT / "fixtures" / task["id"], exec_task))
    return propose.format_prompt(vis, propose.retrieve(vis, "train"))


def messages(task: dict, completion: str) -> list[dict]:
    return [{"role": "system", "content": propose.SYSTEM}, {"role": "user", "content": prompt_for(task)},
            {"role": "assistant", "content": completion}]


def cmd_build(args) -> int:
    train = train_tasks()
    picked: dict[str, list[dict]] = {t["id"]: [] for t in train}
    for d in map(Path, args.runs):
        lines = (d / "results.jsonl").read_text(encoding="utf-8").splitlines()
        head = json.loads(lines[0])["header"]
        if head["tasks_sha256"] != tasks_sha():
            print(f"skip {d.name}: graded against a different tasks.jsonl")
            continue
        for line in lines[1:]:
            r = json.loads(line)
            got = picked.get(r["task_id"])
            if got is None or len(got) >= args.per_task or not r["model_success"]:
                continue
            key = json.dumps({k: r["proposal"][k] for k in ("decision", "operations")}, sort_keys=True)
            if all(p["key"] != key for p in got):
                got.append({"key": key, "proposal": r["proposal"], "label": head["label"],
                            "source": f"{d.name}/{r['attempt']}"})
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for t in train:
            for p in picked[t["id"]]:
                f.write(json.dumps({"task_id": t["id"], "source": p["source"],
                                    "messages": messages(t, json.dumps(p["proposal"], sort_keys=True))}) + "\n")
    for t in train:
        got = picked[t["id"]]
        print(f"  {t['id']:<18} " + ("; ".join(f"{p['label']} {p['proposal']['decision']}" for p in got)
                                     or "NO ACCEPTED CANDIDATE"))
    n = sum(map(len, picked.values()))
    print(f"{OUT.relative_to(ROOT)}: {n} rows, {sum(1 for v in picked.values() if v)}/{len(train)} train tasks, "
          f"sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()[:12]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--base-url", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--structured", action="store_true", help="constrain decoding (the base model needs it)")
    s.add_argument("--concurrency", type=int, default=8)
    b = sub.add_parser("build")
    b.add_argument("runs", nargs="+")
    b.add_argument("--per-task", type=int, default=2)
    args = ap.parse_args()
    return cmd_sample(args) if args.cmd == "sample" else cmd_build(args)


if __name__ == "__main__":
    sys.exit(main())
