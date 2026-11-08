"""test_lab.py: the boundary tests. Each guards one boundary and names the mutation that breaks it.

    python -m unittest test_lab -v
"""
from __future__ import annotations

import asyncio
import builtins
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import scene  # noqa: E402
from protocols import binding_error, contract_sha, load_tasks  # noqa: E402

TASKS = load_tasks()
SENTINEL = "SENTINEL-ORACLE-7f3a"


def attempt_from(task: dict, proposal: dict, execution_allowed=True) -> Path:
    d = Path(tempfile.mkdtemp(prefix="t-"))
    scene.copy_bundle(ROOT / "fixtures" / task["id"], d / "original")
    scene.copy_bundle(ROOT / "fixtures" / task["id"], d / "candidate")
    receipt = scene.apply(d / "candidate", task, proposal, execution_allowed)
    (d / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (d / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
    return d


class Boundaries(unittest.TestCase):

    def test_visible_excludes_oracle(self):
        """Mutation: visible() copies the task minus `expectation` (a denylist) -> the unknown key leaks."""
        import run
        t = json.loads(json.dumps(TASKS["composite-01"]))
        t["expectation"]["note"] = SENTINEL
        t["notes"] = SENTINEL                       # unknown top-level key: only an allowlist drops it
        facts = {"up_axis": "Z", "nested": {"x": 1}}
        blob = json.dumps(run.visible(t, facts))
        self.assertNotIn(SENTINEL, blob)
        self.assertNotIn("expectation", blob)

    def test_rejected_edit_preserves_original(self):
        """Mutation: apply() mutates before checking preflight -> candidate bytes change."""
        t = TASKS["forbidden-01"]
        p = {"decision": "edit", "operations": [{"op": "set_meters_per_unit_metadata", "value": 1.0}],
             "citations": []}
        d = attempt_from(t, p)
        try:
            self.assertEqual(scene.bundle_hashes(d / "original"), scene.bundle_hashes(d / "candidate"))
            self.assertEqual([o["status"] for o in json.loads((d / "receipt.json").read_text())["operations"]],
                             ["blocked"])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_finalize_once_per_attempt(self):
        """Mutation: drop the already-finalized guard -> a second finalize mutates again."""
        import run
        t = TASKS["default-prim-01"]
        d = Path(tempfile.mkdtemp(prefix="t-"))
        scene.copy_bundle(ROOT / "fixtures" / t["id"], d / "original")
        scene.copy_bundle(ROOT / "fixtures" / t["id"], d / "candidate")
        (d / "task.json").write_text(json.dumps({"id": t["id"], "contract": t["contract"], "assets": {}}))
        p = {"decision": "edit", "operations": [{"op": "set_default_prim", "prim_path": "/World"}], "citations": []}

        async def twice(s):
            first = await s.call_tool("finalize", {"proposal": p, "execution_allowed": True})
            after_first = scene.bundle_hashes(d / "candidate")
            second = await s.call_tool("finalize", {"proposal": p, "execution_allowed": True})
            return first.isError, second.isError, after_first

        try:
            e1, e2, after_first = asyncio.run(run.mcp_session(d, "att-test-once", twice))
            self.assertFalse(e1)
            self.assertTrue(e2)
            self.assertEqual(after_first, scene.bundle_hashes(d / "candidate"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_verdict_binds_attempt(self):
        """Mutation: the judge skips binding_error -> A's id grades B's directory."""
        t = TASKS["default-prim-01"]
        good = {"decision": "edit", "operations": [{"op": "set_default_prim", "prim_path": "/World"}], "citations": []}
        a, b = attempt_from(t, good), attempt_from(t, {"decision": "no-op", "operations": [], "citations": []})
        try:
            def manifest(d, aid):
                return {"attempt": aid, "task_id": t["id"], "contract_sha256": contract_sha(t["contract"]),
                        "proposal_sha256": scene.sha256(d / "proposal.json"),
                        "receipt_sha256": scene.sha256(d / "receipt.json"),
                        "candidate": scene.bundle_hashes(d / "candidate")}
            entry = {"task_id": t["id"]}
            self.assertIsNone(binding_error(manifest(a, "A"), "A", entry, t, a))
            self.assertIsNotNone(binding_error(manifest(b, "B"), "A", entry, t, b))      # B's dir under A's id
            self.assertIsNotNone(binding_error(manifest(a, "A"), "A", entry, t, b))      # A's manifest, B's files
            edited = {**t, "contract": {**t["contract"], "requirements": {"up_axis": "Y"}}}
            self.assertIsNotNone(binding_error(manifest(a, "A"), "A", entry, edited, a))  # swapped contract
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)

    def test_split_isolation(self):
        """Mutation: the scripted proposer reads task['expectation'] -> it raises here."""
        import run

        class Raises(dict):
            def __getitem__(self, k):
                raise AssertionError("proposer read the oracle")
            get = __getitem__

        for tid in ("composite-01", "noop-02", "ambiguous-02", "forbidden-03"):
            t = {**TASKS[tid], "expectation": Raises()}
            facts = scene.inspect(ROOT / "fixtures" / tid, t)
            run.scripted(run.visible(t, facts))
        test_ids = {k for k, v in TASKS.items() if v["split"] == "test"}
        other = {k for k, v in TASKS.items() if v["split"] in ("train", "dev")}
        self.assertFalse(test_ids & other)
        self.assertNotIn("composite", " ".join(sorted(other)))     # the held-out family is test-only

    def test_baseline_without_model_credentials(self):
        """Mutation: run.py imports the model stack at top level -> the import blocker trips."""
        for k in ("OPENAI_API_KEY", "NVIDIA_API_KEY", "NGC_API_KEY", "HF_TOKEN"):
            os.environ.pop(k, None)
        blocked = ("langchain_openai", "nemoguardrails", "fastembed", "torch", "vllm")
        real = builtins.__import__

        def guard(name, *a, **kw):
            if name.split(".")[0] in blocked:
                raise ImportError(f"blocked: {name}")
            return real(name, *a, **kw)

        for m in [m for m in sys.modules if m.split(".")[0] in ("run", "report", "propose")]:
            del sys.modules[m]
        builtins.__import__ = guard
        try:
            import run
            t = TASKS["noop-01"]
            facts = scene.inspect(ROOT / "fixtures" / t["id"], t)
            self.assertEqual(run.scripted(run.visible(t, facts))["decision"], "no-op")
            self.assertEqual(run.always_abstain({})["decision"], "abstain")
        finally:
            builtins.__import__ = real

    def test_scene_asset_paths_confined(self):
        """Mutation: walk() skips the lexical or realpath check -> an escape loads."""
        from pxr import Sdf
        escapes = ["C:/x/y.usda", "/abs/y.usda", "//host/share/y.usda", "http://e/y.usda",
                   "y.usda", "./x.usd", "../../outside.usda", "./a`${X}`.usda", "./y.usda:SDF_FORMAT_ARGS:a=b"]
        base = Path(tempfile.mkdtemp(prefix="walk-"))
        try:
            for i, p in enumerate(escapes):
                d = base / f"c{i}"
                d.mkdir()
                lay = Sdf.Layer.CreateNew(str(d / "root.usda"))
                Sdf.CreatePrimInLayer(lay, "/A").referenceList.Prepend(Sdf.Reference(p))
                lay.Save()
                with self.subTest(path=p):
                    with self.assertRaises(scene.SceneError):
                        scene.walk(d / "root.usda", d)
            # positive controls: a clean fixture passes; a confined missing reference is evidence
            self.assertEqual(scene.walk(ROOT / "fixtures" / "noop-01" / "root.usda", ROOT / "fixtures" / "noop-01"),
                             {"missing": []})
            self.assertEqual(scene.walk(ROOT / "fixtures" / "broken-ref-01" / "root.usda",
                                        ROOT / "fixtures" / "broken-ref-01")["missing"], ["./props/lamp.usda"])
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_containment_is_not_model_success(self):
        """Mutation: model_success judges by grade alone -> a blocked unsafe edit becomes a training target."""
        t = TASKS["forbidden-01"]
        unsafe = {"decision": "edit", "operations": [{"op": "set_meters_per_unit_metadata", "value": 1.0}],
                  "citations": []}
        refuse = {"decision": "abstain", "operations": [], "citations": [], "reason": "not permitted"}
        for proposal, want_success, want_by in ((unsafe, False, "blocked"), (refuse, True, "refused")):
            d = attempt_from(t, proposal)
            try:
                v = scene.evaluate(t, d)
                receipt = json.loads((d / "receipt.json").read_text())
                self.assertEqual(v["grade"], "accepted")
                self.assertEqual(v["contained_by"], want_by)
                self.assertEqual(scene.model_success(v, receipt), want_success)
            finally:
                shutil.rmtree(d, ignore_errors=True)
        self.assertFalse(scene.model_success({"grade": "accepted"}, None))   # missing evidence is not success

    def test_missing_verdict_is_never_success(self):
        """Mutation: aggregation treats an absent verdict as no failed checks -> success counts it."""
        import report
        rows = [{"task_id": "x", "split": "test", "expectation": "repair", "verdict": None, "receipt": None,
                 "trace": {"error_code": "judge_unavailable", "latency_s": 1.0}},
                {"task_id": "y", "split": "test", "expectation": "repair", "verdict": None, "receipt": None,
                 "trace": {"error_code": "model_timeout", "latency_s": None}}]
        r = report.row(rows)
        self.assertEqual(r["success"], "0/2")
        self.assertEqual(r["preserved"], "N/A")         # a timeout before execution is not in M
        self.assertFalse(scene.model_success(None, None))

    def test_decoding_schema_matches_the_contract_vocabulary(self):
        """Mutation: the structured-outputs schema drifts from scene (a decision or an op the
        executor does not know, or one it knows that the model can no longer emit)."""
        import propose
        s = propose.proposal_json_schema()
        self.assertEqual(set(s["properties"]["decision"]["enum"]), scene.DECISIONS)
        branches = s["properties"]["operations"]["items"]["anyOf"]
        self.assertEqual({b["properties"]["op"]["const"]: b["required"][1] for b in branches}, scene.OPS)
        self.assertFalse(s["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
