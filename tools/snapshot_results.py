"""snapshot_results.py: copy the run results behind the README into results/, and print each run's
accepted and model-success counts so the README can be checked against them.

    python tools/snapshot_results.py RUN_DIR=NAME ...      # copy, then summarize
    python tools/snapshot_results.py                        # summarize what is in results/
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scene  # noqa: E402

OUT = ROOT / "results"


def summarize(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])["header"]
    rows = [json.loads(x) for x in lines[1:]]
    accepted = sum((r.get("verdict") or {}).get("grade") == "accepted" for r in rows)
    success = sum(scene.model_success(r.get("verdict"), r.get("receipt")) for r in rows)
    fails = [r["task_id"] for r in rows if (r.get("verdict") or {}).get("grade") != "accepted"]
    return (f"{path.parent.name:<34} {header.get('split') or header.get('kind'):<8} "
            f"accepted {accepted}/{len(rows)}  model success {success}/{len(rows)}  failed {fails}")


def main() -> int:
    for arg in sys.argv[1:]:
        src, name = arg.split("=", 1)
        dst = OUT / name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(src) / "results.jsonl", dst / "results.jsonl")
    for f in sorted(OUT.glob("*/results.jsonl")):
        print(summarize(f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
