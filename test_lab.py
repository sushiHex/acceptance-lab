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

    def test_audit_leaves_stay_confined(self):
        """Mutation: walk() lets a leaf extension skip the realpath check -> a texture path escapes."""
        from pxr import Sdf
        import audit
        base = Path(tempfile.mkdtemp(prefix="leaf-"))
        try:
            def bundle(name, texture):
                d = base / name / "assets"
                d.mkdir(parents=True)
                lay = Sdf.Layer.CreateNew(str(d / "root.usda"))
                lay.pseudoRoot.SetInfo("upAxis", "Z")
                prim = Sdf.CreatePrimInLayer(lay, "/A")
                prim.specifier = Sdf.SpecifierDef
                Sdf.AttributeSpec(prim, "file", Sdf.ValueTypeNames.Asset).default = Sdf.AssetPath(texture)
                lay.Save()
                return d
            inside = bundle("in", "./t.png")
            (inside / "t.png").write_bytes(b"not a layer")
            with self.assertRaises(scene.SceneError):       # the judge's walk still refuses textures
                scene.walk(inside / "root.usda", inside)
            self.assertEqual(scene.walk(inside / "root.usda", inside, audit.LEAF_EXT), {"missing": []})
            outside = bundle("out", "../t.png")
            (outside.parent / "t.png").write_bytes(b"outside the bundle")
            with self.assertRaises(scene.SceneError):
                scene.walk(outside / "root.usda", outside, audit.LEAF_EXT)
            contract = {"scene": "assets/root.usda", "bundle": "assets", "requirements": {"up_axis": "Z"}}
            self.assertEqual(audit.audit(contract, inside.parent)["grade"], "accepted")
            self.assertEqual(audit.audit(contract, outside.parent)["grade"], "error")
            contract["requirements"]["frames_per_second"] = 24   # a key nothing checks must fail, not vanish
            self.assertEqual(audit.audit(contract, inside.parent)["grade"], "failed")
            contract["requirements"] = {}                         # nothing asked is not acceptance
            self.assertEqual(audit.audit(contract, inside.parent)["grade"], "failed")
            elsewhere = {"scene": "assets/root.usda", "bundle": "other", "requirements": {"up_axis": "Z"}}
            (inside.parent / "other").mkdir()                     # the scene must sit inside its bundle
            self.assertEqual(audit.audit(elsewhere, inside.parent)["grade"], "error")
            (inside / "root.usda").write_text("not usd", encoding="utf-8")   # unreadable is an error, not a crash
            self.assertEqual(audit.audit(contract, inside.parent)["grade"], "error")
            layered = bundle("layer", "./t.png")                  # a texture suffix must not smuggle a layer
            shutil.copy(ROOT / "fixtures" / "noop-01" / "root.usda", layered / "t.png")
            lay = Sdf.Layer.FindOrOpen(str(layered / "root.usda"))
            lay.subLayerPaths.append("./t.png")
            lay.Save()
            smuggled = {"scene": "assets/root.usda", "bundle": "assets", "requirements": {"up_axis": "Z"}}
            self.assertNotEqual(audit.audit(smuggled, layered.parent)["grade"], "accepted")
            from pxr import Usd, UsdGeom                          # a stale extent must not vouch for a bound
            shapes = base / "shapes" / "assets"
            shapes.mkdir(parents=True)
            st = Usd.Stage.CreateNew(str(shapes / "root.usda"))
            UsdGeom.SetStageUpAxis(st, "Z")
            UsdGeom.SetStageMetersPerUnit(st, 1.0)
            UsdGeom.Cube.Define(st, "/A/Box").GetSizeAttr().Set(4.0)
            mesh = UsdGeom.Mesh.Define(st, "/A/Tri")
            mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
            mesh.GetExtentAttr().Set([(0, 0, 0), (1, 1, 0)])
            st.GetRootLayer().Save()
            inside_box = {"scene": "assets/root.usda", "bundle": "assets", "requirements": {
                "contains_m": {"prim": "/A", "min": [-1, -1, -1], "max": [1, 1, 1]}}}
            self.assertEqual(audit.audit(inside_box, shapes.parent)["grade"], "accepted")
            mesh.GetExtentAttr().Set([(-50, -50, -50), (50, 50, 50)])
            st.GetRootLayer().Save()
            del st
            self.assertEqual(audit.audit(inside_box, shapes.parent)["grade"], "failed")
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_audit_stage_checks_can_fail(self):
        """Mutation: an audit-only check (time-code rate, contained region) reports pass whatever
        it observed, or reads a fallback or unconverted value -> a scene off its contract is accepted."""
        from pxr import Usd, UsdGeom
        import audit
        base = Path(tempfile.mkdtemp(prefix="stage-"))
        try:
            def bundle(name, meters_per_unit, size, rate):
                d = base / name / "assets"
                d.mkdir(parents=True)
                st = Usd.Stage.CreateNew(str(d / "root.usda"))
                UsdGeom.SetStageUpAxis(st, "Z")
                UsdGeom.SetStageMetersPerUnit(st, meters_per_unit)
                if rate:
                    st.SetTimeCodesPerSecond(rate)
                UsdGeom.Cube.Define(st, "/A/Box").GetSizeAttr().Set(size)
                st.GetRootLayer().Save()
                return d.parent

            def grade(root, **requirements):
                return audit.audit({"scene": "assets/root.usda", "bundle": "assets",
                                    "requirements": requirements}, root)["grade"]

            def box(half, prim="/A"):
                return {"prim": prim, "min": [-half] * 3, "max": [half] * 3}

            meters = bundle("m", 1.0, 4.0, 24)                    # a 4 m cube: reaches 2 m each way
            self.assertEqual(grade(meters, time_codes_per_second=24, contains_m=box(1)), "accepted")
            self.assertEqual(grade(meters, time_codes_per_second=60), "failed")
            self.assertEqual(grade(meters, contains_m=box(3)), "failed")
            self.assertEqual(grade(meters, contains_m=box(1, "/Missing")), "failed")
            unstated = bundle("unstated", 1.0, 4.0, None)         # the schema fallback is not a stated rate
            self.assertEqual(grade(unstated, time_codes_per_second=24), "failed")
            centimeters = bundle("cm", 0.01, 400.0, 24)           # the same cube authored in centimeters
            self.assertEqual(grade(centimeters, contains_m=box(1)), "accepted")
            self.assertEqual(grade(centimeters, contains_m=box(3)), "failed")
            reversed_box = {"prim": "/A", "min": [1] * 3, "max": [-1] * 3}   # contains nothing, so shows nothing
            self.assertEqual(grade(meters, contains_m=reversed_box), "failed")

            def edited(name, edit):
                root = bundle(name, 1.0, 4.0, 24)
                st = Usd.Stage.Open(str(root / "assets" / "root.usda"))
                edit(st, UsdGeom.Cube.Get(st, "/A/Box"))
                st.GetRootLayer().Save()
                return root

            lying = edited("lying", lambda st, c: (c.GetSizeAttr().Set(0.1),      # a small shape, a large claim
                                                   c.GetExtentAttr().Set([(-50,) * 3, (50,) * 3])))
            self.assertEqual(grade(lying, contains_m=box(1)), "failed")
            inactive = edited("inactive", lambda st, c: c.GetPrim().SetActive(False))
            self.assertEqual(grade(inactive, contains_m=box(1, "/A/Box")), "failed")
            hidden = edited("hidden", lambda st, c: c.MakeInvisible())
            self.assertEqual(grade(hidden, contains_m=box(1, "/A/Box")), "failed")

            def animate(st, c):                                   # no default points: nothing to check an extent against
                mesh = UsdGeom.Mesh.Define(st, "/A/Tri")
                mesh.GetPointsAttr().Set([(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0)], 1.0)
                mesh.GetExtentAttr().Set([(-50,) * 3, (50,) * 3])
            self.assertEqual(grade(edited("animated", animate), contains_m=box(3)), "failed")
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

    def test_repair_that_tidies_is_not_accepted(self):
        """Mutation: drop expectation.minimal -> a repair that also edits a field no requirement
        names is accepted, and training rewards tidying."""
        t = TASKS["default-prim-06"]
        sdp = {"op": "set_default_prim", "prim_path": "/World"}
        tidy = scene.evaluate(t, attempt_from(t, {"decision": "edit", "citations": [], "operations": [
            sdp, {"op": "set_up_axis_metadata", "token": "Z"}]}))
        exact = scene.evaluate(t, attempt_from(t, {"decision": "edit", "citations": [], "operations": [sdp]}))
        self.assertNotEqual(tidy["grade"], "accepted")
        self.assertEqual(exact["grade"], "accepted")

    def test_grpo_reward_is_group_relative_judge_success(self):
        """Mutation: advantages() returns the raw reward -> a group the judge scored all alike
        trains, and a contained (blocked) reply counts as rewarded."""
        sys.path.insert(0, str(ROOT / "tools"))
        from grpo import advantages
        r = lambda ok: {"model_success": ok}
        self.assertIsNone(advantages([r(True)] * 4))
        self.assertIsNone(advantages([r(False)] * 4))
        adv = advantages([r(True), r(False), r(False), r(False)])
        self.assertAlmostEqual(sum(adv), 0.0)
        self.assertEqual([a > 0 for a in adv], [True, False, False, False])

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
