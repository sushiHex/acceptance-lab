"""Print which checks fail for each wrong control, so a fail control is seen to fail for its reason."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controls  # noqa: E402

tasks = [json.loads(l) for l in (controls.ROOT / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
base = Path(tempfile.mkdtemp(prefix="why-"))
try:
    for t in tasks:
        if t["id"] not in controls.WRONG:
            continue
        v = controls.run_attempt(t, controls.WRONG[t["id"]], base)
        fails = [f"{c['name']}({c['observed'][:40]})" for c in v["checks"] if c["status"] != "pass"]
        print(f"{t['id']:<17} {v['grade']:<8} {'; '.join(fails)}")
finally:
    shutil.rmtree(base, ignore_errors=True)
