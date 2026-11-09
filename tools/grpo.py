"""grpo.py: GRPO with the judge as the reward (Windows coordinator; vLLM and training in WSL).

    python tools/grpo.py --start /root/adapters/sft/epoch-2 [--iterations 5] [--group 8]

Each iteration: serve the current adapter, score it on dev (temperature 0, the table's
settings), sample --group single-turn replies per train task at temperature 1, execute and
judge every one through the real coordinator, and take one GRPO update on the replies whose
group the judge did not score all alike. The reward is scene.model_success and nothing else:
accepted, receipt present, every operation permitted. The adapter that ships is the iteration
with the best dev success, never the last one. train/grpo-manifest.json records every iteration.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import propose  # noqa: E402
import run  # noqa: E402
from protocols import load_tasks  # noqa: E402

URL = "http://127.0.0.1:8000/v1"
DISTRO = ["wsl", "-d", "Ubuntu-24.04", "-u", "root", "--exec"]   # --exec: no outer shell re-expands $vars
MANIFEST = ROOT / "train" / "grpo-manifest.json"


def wsl_path(p: Path) -> str:
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}"


def wsl(cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(DISTRO + ["bash", "-c", cmd], capture_output=True, text=True, **kw)


def serve(adapter: str) -> subprocess.Popen:
    proc = subprocess.Popen(DISTRO + ["bash", "-c", f"LORAS='policy={adapter}' bash {wsl_path(ROOT / 'tools' / 'serve.sh')}"
                                      " > /root/vllm.log 2>&1"])
    for _ in range(600):
        if propose.reachable(URL):
            return proc
        if proc.poll() is not None:
            raise RuntimeError("vLLM exited; see /root/vllm.log")
        time.sleep(1)
    raise RuntimeError("vLLM did not come up in 600 s")


def stop(proc: subprocess.Popen) -> None:
    wsl("p=$(cat /root/vllm.pid); kill $p; while kill -0 $p 2>/dev/null; do sleep 1; done")
    proc.wait()


async def judged(label: str, tasks: list, temperature: float, repair: bool) -> list[dict]:
    proposer = propose.make_proposer(URL, "policy", temperature=temperature, repair=repair)
    route = propose.run_info(URL, "policy", False, temperature)
    d = await run.run_tasks(label, "grpo", tasks, proposer, split=tasks[0]["split"], route=route, concurrency=8)
    return [json.loads(l) for l in (d / "results.jsonl").read_text(encoding="utf-8").splitlines()[1:]]


def advantages(results: list[dict]) -> list[float] | None:
    """Group-relative: the judge's reward minus the group's mean. None when every reply in the
    group scored alike, since such a group carries no signal about which reply was better."""
    rewards = [float(r["model_success"]) for r in results]
    mean = sum(rewards) / len(rewards)
    return None if all(x == mean for x in rewards) else [x - mean for x in rewards]


# A failed measurement is not a judgement of the reply: such attempts leave the group entirely.
UNMEASURED = {"judge_unavailable", "executor_error", "model_timeout", "model_unreachable"}


def batch(results: list[dict]) -> tuple[list[dict], float, int, int]:
    groups, dropped = defaultdict(list), 0
    for r in results:
        if r["trace"].get("error_code") in UNMEASURED or not r["trace"].get("replies"):
            dropped += 1
            continue
        groups[r["task_id"]].append(r)
    rows, signal = [], 0
    for tid, rs in groups.items():
        adv = advantages(rs)
        if adv is None:
            continue
        signal += 1
        for r, a in zip(rs, adv):
            reply = r["trace"]["replies"][0]
            if reply["content"]:
                rows.append({"task_id": tid, "attempt": r["attempt"], "advantage": a,
                             "messages": [{"role": "system", "content": propose.SYSTEM},
                                          {"role": "user", "content": r["trace"]["prompt"]}],
                             "completion": reply["content"], "finished": reply["finish_reason"] == "stop"})
    measured = [r for rs in groups.values() for r in rs]
    return rows, sum(r["model_success"] for r in measured) / max(1, len(measured)), signal, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="WSL path of the adapter to start from")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--out", default="/root/adapters/grpo")
    args = ap.parse_args()
    if propose.reachable(URL):
        print("a server already answers on :8000; stop it first (this tool starts and stops its own)")
        return 2
    if wsl(f"test -e {args.out}").returncode == 0:
        print(f"{args.out} exists; choose a fresh --out so no recorded adapter is overwritten")
        return 2
    tasks = load_tasks()
    train = [t for t in tasks.values() if t["split"] == "train"]
    dev = [t for t in tasks.values() if t["split"] == "dev"]
    log, adapter = [], args.start
    for k in range(args.iterations + 1):
        proc = serve(adapter)
        try:
            async def score_and_sample():   # one event loop: the shared Guardrails rail is bound to it
                d = await judged(f"grpo dev {k}", dev, 0, True)
                s = await judged(f"grpo samples {k}", train * args.group, 1.0, False) if k < args.iterations else None
                return d, s
            dev_res, res = asyncio.run(score_and_sample())
            entry = {"iteration": k, "adapter": adapter,
                     "dev_success": sum(r["model_success"] for r in dev_res), "dev_n": len(dev_res)}
        finally:
            stop(proc)
        rows = None
        if k < args.iterations:
            rows, reward, signal, dropped = batch(res)
            entry.update(train_reward=round(reward, 4), groups_with_signal=signal, groups=len(train),
                         unmeasured=dropped)
        if rows == []:
            entry["stalled"] = "no group had both an accepted and a rejected reply"
        elif rows:
            bfile = ROOT / "runs" / f"grpo-batch-{Path(args.out).name}-{k}.jsonl"
            bfile.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            nxt = f"{args.out}/iter-{k + 1}"
            out = wsl(f"set -o pipefail; cd {wsl_path(ROOT)} && /opt/vllm/bin/python tools/grpo_step.py --adapter-in {adapter} "
                      f"--batch {wsl_path(bfile)} --out {nxt} 2>/root/grpo-step.log | tail -1")
            if out.returncode:
                raise RuntimeError(f"grpo_step failed; see /root/grpo-step.log\n{out.stderr[-500:]}")
            entry["step"] = json.loads(out.stdout)
            adapter = nxt
        log.append(entry)
        print(json.dumps(entry))
        best = max(log, key=lambda e: (e["dev_success"], -e["iteration"]))   # dev alone; a tie keeps fewer updates
        MANIFEST.write_text(json.dumps({"start": args.start, "group": args.group, "reward": "scene.model_success",
                                        "iterations": log, "selected": best["adapter"]}, indent=2) + "\n",
                            encoding="utf-8")
        if "stalled" in entry:
            break
    print(f"selected {best['adapter']} (dev {best['dev_success']}/{best['dev_n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
