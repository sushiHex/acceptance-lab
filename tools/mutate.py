"""mutate.py: break the code on purpose and confirm the guarding test goes red. Restores every file.

    python tools/mutate.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (file, original text, mutated text, test that must fail)
MUTATIONS = [
    ("scene.py",
     '                                 "fail" if tidy else "pass"))',
     '                                 "pass"))',
     "test_repair_that_tidies_is_not_accepted"),
    ("tools/grpo.py",
     "    return None if all(x == mean for x in rewards) else [x - mean for x in rewards]",
     "    return rewards",
     "test_grpo_reward_is_group_relative_judge_success"),
    ("propose.py",
     "\"properties\": {\"decision\": {\"enum\": sorted(scene.DECISIONS)},",
     "\"properties\": {\"decision\": {\"enum\": sorted(scene.DECISIONS - {\"abstain\"})},",
     "test_decoding_schema_matches_the_contract_vocabulary"),
    ("run.py",
     'return {"id": task["id"], "request": task["request"], "contract": task["contract"],',
     'return {**{k: v for k, v in task.items() if k != "expectation"}, "request": task["request"], "contract": task["contract"],',
     "test_visible_excludes_oracle"),
    ("scene.py",
     "    if not execution_allowed or not all(v[\"allowed\"] for v in verdicts):",
     "    if False:",
     "test_rejected_edit_preserves_original"),
    ("protocols.py",
     "        if state[\"finalized\"]:\n            raise ValueError(\"attempt already finalized\")",
     "        if False:\n            raise ValueError(\"attempt already finalized\")",
     "test_finalize_once_per_attempt"),
    ("protocols.py",
     '    if manifest.get("attempt") != attempt_id:',
     '    if False:',
     "test_verdict_binds_attempt"),
    ("scene.py",
     "            why = _lexical_reject(p)",
     "            why = None",
     "test_scene_asset_paths_confined"),
    ("scene.py",
     "            if bundle_root not in target.parents and target != bundle_root:",
     "            if not leaf and bundle_root not in target.parents and target != bundle_root:",
     "test_audit_leaves_stay_confined"),
    ("audit.py",
     '                           "fail" if unknown else "pass")]',
     '                           "pass")]',
     "test_audit_leaves_stay_confined"),
    ("audit.py",
     "for lo, hi in zip(ext, real) for a, b in zip(lo, hi)):\n            return None",
     "for lo, hi in zip(ext, real) for a, b in zip(lo, hi)):\n            continue",
     "test_audit_leaves_stay_confined"),
    ("audit.py",
     "        shape = UsdGeom.Boundable(prim)\n",
     "        shape = UsdGeom.PointBased(prim)\n",
     "test_audit_stage_checks_can_fail"),
    ("audit.py",
     "            return None           # an extent with no points behind it",
     "            continue",
     "test_audit_stage_checks_can_fail"),
    ("audit.py",
     "    if not start or not start.IsActive():",
     "    if not start:",
     "test_audit_stage_checks_can_fail"),
    ("audit.py",
     "    if UsdGeom.Imageable(start).ComputeVisibility() == UsdGeom.Tokens.invisible:",
     "    if False:",
     "test_audit_stage_checks_can_fail"),
    ("audit.py",
     "    if any(lo > hi for lo, hi in zip(want[\"min\"], want[\"max\"])):",
     "    if False:",
     "test_audit_stage_checks_can_fail"),
    ("audit.py",
     '                                         "pass" if have == want else "fail"))',
     '                                         "pass"))',
     "test_audit_stage_checks_can_fail"),
    ("audit.py",
     '            have = pr.GetInfo("timeCodesPerSecond") if pr.HasInfo("timeCodesPerSecond") else None',
     '            have = pr.GetInfo("timeCodesPerSecond")',
     "test_audit_stage_checks_can_fail"),
    ("audit.py",
     '    return scene._check(name, box, scene._fmt(r), "pass" if ok else "fail")',
     '    return scene._check(name, box, scene._fmt(r), "pass")',
     "test_audit_stage_checks_can_fail"),
    ("audit.py",
     "    return scene._extent_range(stage, prim_path, canonical=True)",
     "    return scene._extent_range(stage, prim_path, canonical=False)",
     "test_audit_stage_checks_can_fail"),
    ("scene.py",
     '    return all(o.get("status") == "permitted" for o in ops)',
     '    return True',
     "test_containment_is_not_model_success"),
    ("report.py",
     '    ok = lambda r: (r.get("verdict") or {}).get("grade") == "accepted"',
     '    ok = lambda r: (r.get("verdict") or {}).get("grade") != "failed"',
     "test_missing_verdict_is_never_success"),
    ("run.py",
     "import scene\n",
     "import scene\nimport nemoguardrails  # noqa\n",
     "test_baseline_without_model_credentials"),
]


def run_test(name: str) -> bool:
    r = subprocess.run([sys.executable, "-m", "unittest", f"test_lab.Boundaries.{name}"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    bad = 0
    for test in sorted({m[3] for m in MUTATIONS}):   # a test already red would count every change as caught
        if not run_test(test):
            print(f"RED      {test}  (fails before any change)")
            bad += 1
    if bad:
        return 1
    for f, old, new, test in MUTATIONS:
        path = ROOT / f
        src = path.read_text(encoding="utf-8")
        if old not in src:
            print(f"SKIP {test}: mutation anchor not found in {f}")
            bad += 1
            continue
        try:
            path.write_text(src.replace(old, new, 1), encoding="utf-8", newline="\n")
            survived = run_test(test)
        finally:
            path.write_text(src, encoding="utf-8", newline="\n")
        print(f"{'SURVIVED' if survived else 'killed  '} {test}  ({f})")
        bad += survived
    print(f"mutations: {len(MUTATIONS) - bad}/{len(MUTATIONS)} killed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
