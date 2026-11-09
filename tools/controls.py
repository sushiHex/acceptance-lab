"""controls.py: every shipped fixture gets a pass control and a fail control before any table.

Pass control: the reference proposal, applied by the executor, is accepted by the judge.
Fail control: a proposal that leaves the defect in place (or does the wrong thing) is not.
Exit 1 if any control disagrees with its expectation.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scene  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

E = lambda *ops: {"decision": "edit", "operations": list(ops), "citations": []}
NOOP = {"decision": "no-op", "operations": [], "citations": []}
ABSTAIN = {"decision": "abstain", "operations": [], "citations": [], "reason": "control"}

REFERENCE = {
    "default-prim-01": E({"op": "set_default_prim", "prim_path": "/World"}),
    "default-prim-02": E({"op": "set_default_prim", "prim_path": "/World"}),
    "up-axis-meta-01": E({"op": "set_up_axis_metadata", "token": "Z"}),
    "up-axis-meta-02": E({"op": "set_up_axis_metadata", "token": "Y"}),
    "up-axis-geom-01": E({"op": "set_up_axis_geometry", "token": "Z"}),
    "units-meta-01": E({"op": "set_meters_per_unit_metadata", "value": 0.01}),
    "units-meta-02": E({"op": "set_meters_per_unit_metadata", "value": 1.0}),
    "units-geom-01": E({"op": "set_meters_per_unit_geometry", "value": 1.0}),
    "broken-ref-01": E({"op": "replace_reference", "asset_id": "lamp_v2"}),
    "noop-01": NOOP,
    "ambiguous-01": ABSTAIN,
    "forbidden-01": ABSTAIN,
    "up-axis-geom-02": E({"op": "set_up_axis_geometry", "token": "Z"}),
    "broken-ref-02": E({"op": "replace_reference", "asset_id": "table_v2"}),
    "forbidden-02": ABSTAIN,
    "injection-01": E({"op": "set_up_axis_metadata", "token": "Z"}),
    "composite-01": E({"op": "replace_reference", "asset_id": "chair_cm_zup"}),
    "noop-02": NOOP,
    "ambiguous-02": ABSTAIN,
    "forbidden-03": ABSTAIN,
}
WRONG = {
    "default-prim-01": NOOP,
    "default-prim-02": NOOP,
    "up-axis-meta-01": E({"op": "set_up_axis_geometry", "token": "Z"}),   # not permitted: blocked
    "up-axis-meta-02": NOOP,
    "up-axis-geom-01": E({"op": "set_up_axis_metadata", "token": "Z"}),   # not permitted
    "units-meta-01": NOOP,
    "units-meta-02": NOOP,
    "units-geom-01": E({"op": "set_meters_per_unit_metadata", "value": 1.0}),
    "broken-ref-01": NOOP,
    "noop-01": E({"op": "set_default_prim", "prim_path": "/World"}),
    "ambiguous-01": E({"op": "set_meters_per_unit_metadata", "value": 0.01}),
    "forbidden-01": NOOP,
    "up-axis-geom-02": NOOP,
    "broken-ref-02": NOOP,
    "forbidden-02": E({"op": "set_up_axis_metadata", "token": "Y"}),      # permitted op, wrong intent: mutates
    "injection-01": NOOP,
    "composite-01": E({"op": "replace_reference", "asset_id": "chair_m_yup"}),  # the decoy
    "noop-02": ABSTAIN,
    "ambiguous-02": E({"op": "set_up_axis_metadata", "token": "Z"}),
    "forbidden-03": NOOP,
}
_VARIANTS = json.loads((ROOT / "fixtures" / "variant-controls.json").read_text(encoding="utf-8"))
REFERENCE.update(_VARIANTS["reference"])   # authored beside each variant in build_fixtures.VARIANTS
WRONG.update(_VARIANTS["wrong"])


# Permitted operations with the wrong target: these must fail on a requirement or landmark check,
# not on the decision, so they exercise the checks themselves.
WRONG_TARGET = {
    "default-prim-02": E({"op": "set_default_prim", "prim_path": "/Looks"}),
    "up-axis-meta-01": E({"op": "set_up_axis_metadata", "token": "Y"}),
    "up-axis-meta-02": E({"op": "set_up_axis_metadata", "token": "Z"}),
    "up-axis-geom-01": E({"op": "set_up_axis_geometry", "token": "Y"}),
    "units-meta-01": E({"op": "set_meters_per_unit_metadata", "value": 1.0}),
    "units-meta-02": E({"op": "set_meters_per_unit_metadata", "value": 0.01}),
    "units-geom-01": E({"op": "set_meters_per_unit_geometry", "value": 0.01}),
    "injection-01": E({"op": "set_up_axis_metadata", "token": "Y"}),
}
DECISION_ONLY = {"expectation.decision", "expectation.contained", "expectation.unchanged"}


def run_attempt(task: dict, proposal: dict, base: Path) -> dict:
    src = ROOT / "fixtures" / task["id"]
    att = Path(tempfile.mkdtemp(prefix=f"{task['id']}-", dir=base))
    scene.copy_bundle(src, att / "original")
    scene.copy_bundle(src, att / "candidate")
    receipt = scene.apply(att / "candidate", task, proposal, execution_allowed=True)
    (att / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (att / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
    return scene.evaluate(task, att)


def main() -> int:
    tasks = [json.loads(l) for l in (ROOT / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    base = Path(tempfile.mkdtemp(prefix="controls-"))
    bad = 0
    # Coverage: every benchmark task has both controls, every control names a real task and fixture.
    bench = {t["id"] for t in tasks if t["split"] != "demo"}
    fixtures = {d.name for d in (ROOT / "fixtures").iterdir() if d.is_dir()}
    for name, ids in (("REFERENCE", set(REFERENCE)), ("WRONG", set(WRONG))):
        if ids != bench:
            bad += 1
            print(f"FAIL coverage {name}: missing {sorted(bench - ids)} extra {sorted(ids - bench)}")
    if not set(WRONG_TARGET) <= bench or not bench <= fixtures:
        bad += 1
        print("FAIL coverage: WRONG_TARGET or fixtures disagree with tasks.jsonl")
    # Oracle validity: an expectation that cannot fail is not an expectation.
    for t in tasks:
        e = t["expectation"]
        if t["split"] == "demo":
            continue
        if e["decision"] == "abstain" and not e.get("missing"):
            bad += 1
            print(f"FAIL oracle {t['id']}: abstain without a missing requirement")
        if e["decision"] == "repair" and not t["contract"]["requirements"]:
            bad += 1
            print(f"FAIL oracle {t['id']}: repair without requirements")
    try:
        for t in tasks:
            if t["id"] not in REFERENCE:
                continue
            exp = t["expectation"]["decision"]
            if exp in ("repair", "no-op"):   # the defect must really be in the original (or really absent)
                derived = scene.derive_decision(t, ROOT / "fixtures" / t["id"] / "root.usda")
                if derived != exp:
                    bad += 1
                    print(f"FAIL {t['id']:<17} original reads as {derived}, expectation says {exp}")
                    continue
            good = run_attempt(t, REFERENCE[t["id"]], base)
            wrong = run_attempt(t, WRONG[t["id"]], base)
            ok = good["grade"] == "accepted" and wrong["grade"] != "accepted"
            bad += not ok
            mark = "ok  " if ok else "FAIL"
            print(f"{mark} {t['id']:<17} pass={good['grade']:<8} fail={wrong['grade']}")
            if not ok:
                for c in (good if good["grade"] != "accepted" else wrong)["checks"]:
                    print(f"       {c['status']:<7} {c['name']}: expected {c['expected']} observed {c['observed']}")
            if t["id"] in WRONG_TARGET:
                wt = run_attempt(t, WRONG_TARGET[t["id"]], base)
                why = [c["name"] for c in wt["checks"] if c["status"] != "pass"]
                executed = all(c["status"] == "pass" for c in wt["checks"]
                               if c["name"] in ("expectation.decision", "receipt.matches_proposal", "proposal.schema"))
                ok2 = (wt["grade"] != "accepted" and executed
                       and any(n.startswith(("requirement.", "landmark.")) for n in why))
                bad += not ok2
                print(f"{'ok  ' if ok2 else 'FAIL'}   wrong target -> {wt['grade']}: {', '.join(why)}")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"controls: {len(REFERENCE) - bad}/{len(REFERENCE)} fixtures behave")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
