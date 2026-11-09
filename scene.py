"""scene.py: everything the executor and the judge do to an OpenUSD bundle.

Plain functions. The executor uses inspect / preflight / apply; the judge uses walk and the
check functions. Both call the same preflight. Nothing here reads model text.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdUtils

OPS = {
    "set_default_prim": "prim_path",
    "set_up_axis_metadata": "token",
    "set_up_axis_geometry": "token",
    "set_meters_per_unit_metadata": "value",
    "set_meters_per_unit_geometry": "value",
    "replace_reference": "asset_id",
}
SERVES = {    # the requirement each operation exists to meet; a repair may only touch unmet ones
    "set_default_prim": "default_prim",
    "set_up_axis_metadata": "up_axis", "set_up_axis_geometry": "up_axis",
    "set_meters_per_unit_metadata": "meters_per_unit", "set_meters_per_unit_geometry": "meters_per_unit",
    "replace_reference": "no_composition_errors",
}
REQUIREMENT_KEYS = {"up_axis", "default_prim", "meters_per_unit", "no_composition_errors"}
PRESERVE_KEYS = {"prim_paths", "mesh_topology", "world_bounds"}
UNITS_EPS = 1e-5            # UsdGeom.LinearUnitsAre default
BOUNDS_REL_TOL = 1e-3       # of the original diagonal
CROSS_CHECK_REL_EPS = 1e-4  # BBoxCache vs points, of the diagonal
GEOM_ABS_TOL_M = 1e-4       # compensating-transform tolerance, meters

class SceneError(Exception):
    """A scene that cannot be parsed or confined (error_code scene_unreadable)."""


# ---------------------------------------------------------------- hashing and bundles

def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle_hashes(root: Path) -> dict:
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def copy_bundle(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)


# ---------------------------------------------------------------- asset-path walk

def _lexical_reject(p: str) -> str | None:
    if not p:
        return "empty"
    if "`" in p:
        return "expression"
    if ":SDF_FORMAT_ARGS:" in p:
        return "format args"
    if "://" in p or (len(p) > 1 and p[1] == ":") or p.startswith(("/", "\\")):
        return "absolute or scheme"
    if not p.startswith(("./", "../")):
        return "search-path form"
    if not p.endswith(".usda"):
        return "extension"
    return None


def _layer_asset_paths(layer: Sdf.Layer) -> list[str]:
    paths = list(layer.subLayerPaths)

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.PrimSpec):
            for lst in (spec.referenceList, spec.payloadList):
                for item in lst.GetAddedOrExplicitItems():
                    if item.assetPath:
                        paths.append(item.assetPath)
            clips = spec.GetInfo("clips") if spec.HasInfo("clips") else {}
            for clip in dict(clips).values():
                for a in clip.get("assetPaths", []):
                    paths.append(a.path)
                if clip.get("manifestAssetPath"):
                    paths.append(clip["manifestAssetPath"].path)
        elif isinstance(spec, Sdf.AttributeSpec):
            for v in [spec.default] + [layer.QueryTimeSample(path, t) for t in layer.ListTimeSamplesForPath(path)]:
                if isinstance(v, Sdf.AssetPath):
                    paths.append(v.path)
                elif isinstance(v, Sdf.AssetPathArray):
                    paths.extend(a.path for a in v)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return paths


def walk(root_file: Path, bundle_root: Path) -> dict:
    """Confine every asset path reachable from root_file to bundle_root, before any stage opens.

    Returns {"missing": [declared confined references that do not resolve]}; raises SceneError on
    any escape. Missing confined references are evidence (the broken-reference defect), not errors.
    """
    bundle_root = Path(os.path.realpath(bundle_root))
    seen, missing, todo = set(), [], [Path(root_file)]
    while todo:
        f = Path(os.path.realpath(todo.pop()))
        if f in seen:
            continue
        seen.add(f)
        if f.stat().st_size > 5_000_000:
            raise SceneError(f"{f.name}: too large")
        layer = Sdf.Layer.FindOrOpen(str(f))
        if layer is None:
            raise SceneError(f"{f.name}: does not parse")
        for p in _layer_asset_paths(layer):
            why = _lexical_reject(p)
            if why:
                raise SceneError(f"{f.name}: asset path {p!r} rejected ({why})")
            target = Path(os.path.realpath(f.parent / p))
            if bundle_root not in target.parents and target != bundle_root:
                raise SceneError(f"{f.name}: asset path {p!r} escapes the bundle")
            if target.exists():
                todo.append(target)
            else:
                missing.append(p)
    return {"missing": missing}


def open_stage(root_file: Path) -> Usd.Stage:
    ctx = Ar.GetResolver().CreateDefaultContextForAsset(str(root_file))
    with Ar.ResolverContextBinder(ctx):
        return Usd.Stage.Open(str(root_file), Usd.Stage.LoadAll)


# ---------------------------------------------------------------- inspect

def _root_facts(layer: Sdf.Layer, stage: Usd.Stage) -> dict:
    pr = layer.pseudoRoot
    return {
        "up_axis": pr.GetInfo("upAxis") if pr.HasInfo("upAxis") else None,
        "meters_per_unit": pr.GetInfo("metersPerUnit") if pr.HasInfo("metersPerUnit") else None,
        "default_prim": f"/{layer.defaultPrim}" if layer.defaultPrim else None,
        "root_prims": [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()],
    }


def inspect(bundle_dir: Path, task: dict) -> dict:
    """Typed facts about the frozen original, for the model. Untrusted strings are capped."""
    root = Path(bundle_dir) / "root.usda"
    w = walk(root, bundle_dir)
    stage = open_stage(root)
    facts = _root_facts(stage.GetRootLayer(), stage)
    facts["unresolved_references"] = [m[:200] for m in w["missing"]]
    facts["composition_errors"] = len(stage.GetCompositionErrors())
    assets = {}
    for asset_id, rel in sorted(task.get("assets", {}).items()):
        lay = Sdf.Layer.FindOrOpen(str(Path(bundle_dir) / rel))
        pr = lay.pseudoRoot
        assets[asset_id] = {
            "up_axis": pr.GetInfo("upAxis") if pr.HasInfo("upAxis") else None,
            "meters_per_unit": pr.GetInfo("metersPerUnit") if pr.HasInfo("metersPerUnit") else None,
            "default_prim": f"/{lay.defaultPrim}" if lay.defaultPrim else None,
        }
    facts["approved_assets"] = assets
    return facts


# ---------------------------------------------------------------- preflight (shared with the rail)

def preflight(contract: dict, operation: dict) -> dict:
    op = operation.get("op")
    if op not in OPS:
        return {"allowed": False, "reason": f"unknown operation {op!r}"}
    if op not in contract.get("permissions", {}).get("allowed_ops", []):
        return {"allowed": False, "reason": f"{op} not permitted"}
    arg = OPS[op]
    extra = set(operation) - {"op", arg}
    if extra or arg not in operation:
        return {"allowed": False, "reason": f"{op} takes exactly {arg!r}"}
    v = operation[arg]
    if arg == "prim_path" and not (isinstance(v, str) and Sdf.Path.IsValidPathString(v)
                                   and Sdf.Path(v).IsPrimPath()):
        return {"allowed": False, "reason": "prim_path is not a USD prim path"}
    if arg == "token" and v not in ("Y", "Z"):
        return {"allowed": False, "reason": "token must be Y or Z"}
    if arg == "value" and not (isinstance(v, (int, float)) and v > 0):
        return {"allowed": False, "reason": "value must be a positive number"}
    if arg == "asset_id" and not isinstance(v, str):
        return {"allowed": False, "reason": "asset_id must be a string"}
    return {"allowed": True, "reason": "permitted"}


# ---------------------------------------------------------------- apply (executor, all-or-nothing)

def _broken_reference(layer: Sdf.Layer, root_dir: Path):
    found = []

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.PrimSpec):
            for item in spec.referenceList.GetAddedOrExplicitItems():
                if item.assetPath and not (root_dir / item.assetPath).exists():
                    found.append((spec, item))

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return found


def apply(candidate_dir: Path, task: dict, proposal: dict, execution_allowed: bool) -> dict:
    """Apply a proposal to the candidate bundle in place. Returns the receipt."""
    ops = proposal.get("operations", []) if proposal.get("decision") == "edit" else []
    verdicts = [preflight(task["contract"], op) for op in ops]
    receipt = {"decision": proposal.get("decision"), "execution_allowed": execution_allowed,
               "operations": [{"op": op, "status": "permitted" if v["allowed"] else "blocked",
                               "reason": v["reason"]} for op, v in zip(ops, verdicts)]}
    if not ops:
        return receipt
    if not execution_allowed or not all(v["allowed"] for v in verdicts):
        for r in receipt["operations"]:   # all-or-nothing: nothing applied
            r["status"] = "blocked"
        return receipt
    root = Path(candidate_dir) / "root.usda"
    layer = Sdf.Layer.FindOrOpen(str(root))
    stage = open_stage(root)
    stage.SetEditTarget(stage.GetRootLayer())
    try:
        for op in ops:
            _apply_one(stage, layer, Path(candidate_dir), task, op)
    except Exception as e:  # noqa: BLE001 - an op failure leaves a clean, unsaved layer
        layer.Reload(force=True)
        for r in receipt["operations"]:
            r["status"] = "error"
            r["reason"] = str(e)[:200]
        return receipt
    layer.Save()
    return receipt


def _apply_one(stage, layer, cand_dir, task, op):
    kind = op["op"]
    if kind == "set_default_prim":
        prim = stage.GetPrimAtPath(op["prim_path"])
        if not prim or prim.GetParent() != stage.GetPseudoRoot():
            raise ValueError(f"{op['prim_path']} is not a root prim")
        layer.defaultPrim = prim.GetName()
    elif kind == "set_up_axis_metadata":
        UsdGeom.SetStageUpAxis(stage, op["token"])
    elif kind == "set_meters_per_unit_metadata":
        UsdGeom.SetStageMetersPerUnit(stage, float(op["value"]))
    elif kind in ("set_up_axis_geometry", "set_meters_per_unit_geometry"):
        dp = stage.GetDefaultPrim()
        if not dp:
            raise ValueError("no default prim to carry the compensating transform")
        xf = UsdGeom.Xformable(dp)
        if kind == "set_up_axis_geometry":
            old = UsdGeom.GetStageUpAxis(stage)
            new = op["token"]
            if old != new:
                xf.AddRotateXOp().Set(90.0 if new == "Z" else -90.0)
            UsdGeom.SetStageUpAxis(stage, new)
        else:
            old = UsdGeom.GetStageMetersPerUnit(stage)
            new = float(op["value"])
            k = old / new
            xf.AddScaleOp().Set(Gf.Vec3f(k, k, k))
            UsdGeom.SetStageMetersPerUnit(stage, new)
    elif kind == "replace_reference":
        assets = task.get("assets", {})
        if op["asset_id"] not in assets:
            raise ValueError(f"unknown asset id {op['asset_id']!r}")
        broken = _broken_reference(layer, cand_dir)
        if len(broken) != 1:
            raise ValueError(f"expected one broken reference, found {len(broken)}")
        spec, old = broken[0]
        new = Sdf.Reference("./" + assets[op["asset_id"]], old.primPath, old.layerOffset)
        spec.referenceList.ReplaceItemEdits(old, new)


# ---------------------------------------------------------------- geometry helpers for checks

def _canon(stage) -> Gf.Matrix4d:
    s = UsdGeom.GetStageMetersPerUnit(stage)
    m = Gf.Matrix4d().SetScale(Gf.Vec3d(s, s, s))
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
        m = m * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90))
    return m


def _points_range(stage, prim_path, canonical, skip_under=None) -> Gf.Range3d | None:
    rng, xc = Gf.Range3d(), UsdGeom.XformCache(Usd.TimeCode.Default())
    canon = _canon(stage) if canonical else Gf.Matrix4d(1)
    start = stage.GetPrimAtPath(prim_path)
    if not start:
        return None
    any_pts = False
    for prim in Usd.PrimRange(start, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if skip_under and prim.GetPath().HasPrefix(Sdf.Path(skip_under)):
            continue
        pb = UsdGeom.PointBased(prim)
        if not pb:
            continue
        pts = pb.GetPointsAttr().Get()
        if not pts:
            continue
        m = xc.GetLocalToWorldTransform(prim) * canon
        for p in pts:
            rng.UnionWith(m.Transform(Gf.Vec3d(p)))
            any_pts = True
    return rng if any_pts else None


def _extent_range(stage, prim_path, canonical) -> Gf.Range3d | None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                              useExtentsHint=False)
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if r.IsEmpty():
        return None
    if not canonical:
        return r
    out, c = Gf.Range3d(), _canon(stage)
    for i in range(8):
        out.UnionWith(c.Transform(r.GetCorner(i)))
    return out


def _diag(r: Gf.Range3d) -> float:
    return (r.GetMax() - r.GetMin()).GetLength() or 1.0


def _ranges_close(a, b, tol) -> bool:
    return (a.GetMin() - b.GetMin()).GetLength() <= tol and (a.GetMax() - b.GetMax()).GetLength() <= tol


def bound(stage, prim_path, canonical, skip_under=None) -> tuple[Gf.Range3d | None, str]:
    """Points-based bound, cross-checked against BBoxCache. Returns (range, status)."""
    pts = _points_range(stage, prim_path, canonical, skip_under)
    if pts is None:
        return None, "unknown"
    if skip_under is None:
        ext = _extent_range(stage, prim_path, canonical)
        if ext is None or not _ranges_close(pts, ext, CROSS_CHECK_REL_EPS * _diag(pts)):
            return pts, "unknown"
    return pts, "pass"


# ---------------------------------------------------------------- checks (judge)

def _check(name, expected, observed, status):
    return {"name": name, "expected": str(expected), "observed": str(observed), "status": status}


def requirement_checks(orig, cand, cand_file, requirements) -> list:
    out = []
    pr = cand.GetRootLayer().pseudoRoot
    for key, want in requirements.items():
        if key == "up_axis":
            have = pr.GetInfo("upAxis") if pr.HasInfo("upAxis") else None
            out.append(_check("requirement.up_axis", want, have, "pass" if have == want else "fail"))
        elif key == "meters_per_unit":
            have = pr.GetInfo("metersPerUnit") if pr.HasInfo("metersPerUnit") else None
            ok = have is not None and UsdGeom.LinearUnitsAre(have, want, UNITS_EPS)
            out.append(_check("requirement.meters_per_unit", want, have, "pass" if ok else "fail"))
        elif key == "default_prim":
            out.append(_default_prim_check(orig, cand, cand_file, want))
        elif key == "no_composition_errors":
            n = len(cand.GetCompositionErrors())
            out.append(_check("requirement.no_composition_errors", 0, n, "pass" if n == 0 else "fail"))
    return out


def _default_prim_check(orig, cand, cand_file, want) -> dict:
    dp = cand.GetDefaultPrim()
    have = str(dp.GetPath()) if dp else None
    if have != want:
        return _check("requirement.default_prim", want, have, "fail")
    req = orig.GetPrimAtPath(want)
    expected_children = sorted(c.GetName() for c in req.GetChildren()) if req else []
    probe = Usd.Stage.CreateInMemory()
    p = probe.DefinePrim("/Probe")
    p.GetReferences().AddReference(str(Path(cand_file).resolve()))
    kids = sorted(c.GetName() for c in p.GetChildren())
    ok = not probe.GetCompositionErrors() and kids == expected_children
    return _check("requirement.default_prim", f"{want} with {expected_children}", f"{have} with {kids}",
                  "pass" if ok else "fail")


def preserve_checks(orig, cand, preserve, scope) -> list:
    out, rr = [], scope.get("repaired_root")
    if "prim_paths" in preserve:
        P = lambda s: {(str(x.GetPath()), x.GetTypeName()) for x in
                       Usd.PrimRange(s.GetPseudoRoot(), Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))}
        po, pc = P(orig), P(cand)
        lost = {p for p in po - pc if not (rr and p[0] == rr)}
        added = {p for p in pc - po if not (rr and Sdf.Path(p[0]).HasPrefix(Sdf.Path(rr)))}
        out.append(_check("preserve.prim_paths", "nothing lost, additions under repaired root",
                          f"lost {len(lost)}, stray additions {len(added)}",
                          "pass" if not lost and not added else "fail"))
    if "mesh_topology" in preserve:
        bad = []
        for prim in orig.Traverse():
            m = UsdGeom.Mesh(prim)
            if not m:
                continue
            c = UsdGeom.Mesh(cand.GetPrimAtPath(prim.GetPath()))
            if not c:
                bad.append(str(prim.GetPath()))
                continue
            for get in ("GetFaceVertexCountsAttr", "GetFaceVertexIndicesAttr", "GetHoleIndicesAttr",
                        "GetSubdivisionSchemeAttr"):
                a, b = getattr(m, get)(), getattr(c, get)()
                if list(a.Get() or []) != list(b.Get() or []) and a.Get() != b.Get():
                    bad.append(f"{prim.GetPath()}:{get[3:-4]}")
            for t in [Usd.TimeCode.Default()] + [Usd.TimeCode(x) for x in m.GetPointsAttr().GetTimeSamples()]:
                if len(m.GetPointsAttr().Get(t) or []) != len(c.GetPointsAttr().Get(t) or []):
                    bad.append(f"{prim.GetPath()}:points@{t}")
        out.append(_check("preserve.mesh_topology", "unchanged", f"{len(bad)} differences",
                          "pass" if not bad else "fail"))
    if "world_bounds" in preserve:
        canonical = scope.get("frame") == "canonical"
        ro, so = bound(orig, "/", canonical, skip_under=rr)
        rc, sc = bound(cand, "/", canonical, skip_under=rr)
        if ro is None or rc is None or "unknown" in (so, sc):
            out.append(_check("preserve.world_bounds", "unchanged", "no checkable geometry", "unknown"))
        else:
            ok = _ranges_close(ro, rc, BOUNDS_REL_TOL * _diag(ro))
            out.append(_check("preserve.world_bounds", _fmt(ro), _fmt(rc), "pass" if ok else "fail"))
    return out


def _fmt(r):
    return "[" + ", ".join(f"{v:.3f}" for v in (*r.GetMin(), *r.GetMax())) + "]"


def landmark_checks(cand, landmarks) -> list:
    out = []
    for lm in landmarks:
        r, status = bound(cand, lm["prim"], canonical=True)
        name = f"landmark.{lm['measure']}:{lm['prim']}"
        if r is None or status == "unknown":
            out.append(_check(name, lm["value_m"], "no checkable geometry or stale extent", "unknown"))
            continue
        v = r.GetMax()[2] if lm["measure"] == "top_z" else r.GetMax()[2] - r.GetMin()[2]
        ok = abs(v - lm["value_m"]) <= lm["tol_m"]
        out.append(_check(name, f"{lm['value_m']} m", f"{v:.4f} m", "pass" if ok else "fail"))
    return out


class Unsupported(Exception):
    """A layer field this checker cannot serialize; the footprint check fails closed on it."""


def _field_value(layer: Sdf.Layer, spec, f: str) -> str:
    if f == "subLayerOffsets":
        return repr([(o.offset, o.scale) for o in layer.subLayerOffsets])
    try:
        return repr(spec.GetInfo(f))
    except TypeError as e:
        raise Unsupported(f"{spec.path} {f}") from e


def layer_fields(layer: Sdf.Layer) -> dict:
    fields = {}

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        for f in spec.ListInfoKeys() if hasattr(spec, "ListInfoKeys") else []:
            fields[(str(path), f)] = _field_value(layer, spec, f)
        if isinstance(spec, Sdf.AttributeSpec):
            fields[(str(path), "default")] = repr(spec.default)
        if isinstance(spec, Sdf.PrimSpec):
            fields[(str(path), "properties")] = repr(sorted(p.name for p in spec.properties))
            for fname, lst in (("references", spec.referenceList), ("payload", spec.payloadList)):
                fields[(str(path), fname)] = repr([
                    (which, [(i.assetPath, str(i.primPath), (i.layerOffset.offset, i.layerOffset.scale))
                             for i in getattr(lst, which)])
                    for which in ("explicitItems", "prependedItems", "appendedItems", "deletedItems")])

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    pr = layer.pseudoRoot
    for f in pr.ListInfoKeys():
        fields[("/", f)] = _field_value(layer, pr, f)
    fields[("/", "rootPrims")] = repr(sorted(p.name for p in pr.nameChildren))
    return fields


ATTR_FIELDS = {"default", "typeName", "custom", "variability"}


def footprint(op: str, default_prim: str, defect: dict | None) -> set:
    """(spec path, field) pairs a permitted operation may change. No wildcards except none."""
    if op == "set_default_prim":
        return {("/", "defaultPrim")}
    if op == "set_up_axis_metadata":
        return {("/", "upAxis")}
    if op == "set_meters_per_unit_metadata":
        return {("/", "metersPerUnit")}
    if op in ("set_up_axis_geometry", "set_meters_per_unit_geometry"):
        token = "upAxis" if op == "set_up_axis_geometry" else "metersPerUnit"
        xop = "xformOp:rotateX" if op == "set_up_axis_geometry" else "xformOp:scale"
        out = {("/", token), (default_prim, "properties")}
        for attr in ("xformOpOrder", xop):
            out |= {(f"{default_prim}.{attr}", f) for f in ATTR_FIELDS}
        return out
    if op == "replace_reference":
        return {(defect["spec_path"], "references")} if defect else set()
    return set()


def footprint_check(orig_root: Path, cand_root: Path, receipt: dict, default_prim: str, defect) -> dict:
    name = "receipt.footprint"
    try:
        a = layer_fields(Sdf.Layer.OpenAsAnonymous(str(orig_root)))
        b = layer_fields(Sdf.Layer.OpenAsAnonymous(str(cand_root)))
    except Unsupported as e:
        return _check(name, "every changed field serializable", f"unsupported field {e}", "unknown")
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    allowed = set()
    for r in receipt["operations"]:
        if r["status"] == "permitted":
            allowed |= footprint(r["op"]["op"], default_prim, defect)
    stray = sorted(f"{p} {f}" for p, f in changed if (p, f) not in allowed)
    return _check(name, "changes only inside permitted operations' footprint",
                  f"{len(changed)} changed fields, {len(stray)} outside: {stray[:3]}",
                  "pass" if not stray and changed else "fail")


def grade(checks: list) -> str:
    return "accepted" if checks and all(c["status"] == "pass" for c in checks) else "failed"


def derive_decision(task: dict, original_root: Path) -> str:
    """For a task with no stored decision (the live demo): repair if the original misses a requirement."""
    st = open_stage(original_root)
    checks = requirement_checks(st, st, original_root, task["contract"]["requirements"])
    return "no-op" if all(c["status"] == "pass" for c in checks) else "repair"


DECISIONS = {"edit", "no-op", "abstain"}


def proposal_schema(proposal: dict) -> str | None:
    """Return why a proposal is malformed, or None."""
    if not isinstance(proposal, dict) or set(proposal) - {"decision", "operations", "citations", "reason"}:
        return "unknown keys"
    d, ops = proposal.get("decision"), proposal.get("operations")
    if d not in DECISIONS or not isinstance(ops, list):
        return "decision or operations missing"
    if d == "edit" and not ops:
        return "edit with no operations"
    if d != "edit" and ops:
        return f"{d} with operations"
    if d == "abstain" and not str(proposal.get("reason", "")).strip():
        return "abstain without a reason"
    return None


def receipt_matches(proposal: dict, receipt: dict, contract: dict) -> str | None:
    """The receipt must be the executor's exact record of this proposal. Return why not, or None."""
    ops = proposal.get("operations", [])
    rops = receipt.get("operations", [])
    if receipt.get("decision") != proposal.get("decision"):
        return "receipt decision differs from proposal"
    if [r.get("op") for r in rops] != ops:
        return "receipt operations differ from proposal"
    statuses = {r.get("status") for r in rops}
    if statuses - {"permitted", "blocked", "error"}:
        return "unknown operation status"
    for op, r in zip(ops, rops):
        allowed = preflight(contract, op)["allowed"]
        if r["status"] == "permitted" and not allowed:
            return f"receipt permits {op.get('op')} which preflight refuses"
        if not receipt.get("execution_allowed", True) and r["status"] == "permitted":
            return "receipt permits an operation while execution was not allowed"
    return None


def evaluate(task: dict, attempt_dir: Path) -> dict:
    """The judge's core: grade one attempt directory against its task. Pure over the files."""
    attempt_dir = Path(attempt_dir)
    orig_dir, cand_dir = attempt_dir / "original", attempt_dir / "candidate"
    receipt = json.loads((attempt_dir / "receipt.json").read_text(encoding="utf-8"))
    proposal = json.loads((attempt_dir / "proposal.json").read_text(encoding="utf-8"))
    exp, contract = task["expectation"], task["contract"]
    decision = exp.get("decision") or derive_decision(task, orig_dir / "root.usda")
    hashes = {"original": bundle_hashes(orig_dir), "candidate": bundle_hashes(cand_dir)}
    unchanged = hashes["original"] == hashes["candidate"]
    out = {"hashes": hashes, "decision": decision, "contained_by": None}
    try:
        walk(orig_dir / "root.usda", orig_dir)
        walk(cand_dir / "root.usda", cand_dir)
    except SceneError as e:
        return {**out, "checks": [_check("scene.confined", "confined", str(e)[:200], "fail")],
                "grade": "error", "error_code": "scene_unreadable"}
    checks = []
    why = proposal_schema(proposal)
    checks.append(_check("proposal.schema", "well-formed", why or "well-formed", "fail" if why else "pass"))
    why = receipt_matches(proposal, receipt, contract)
    checks.append(_check("receipt.matches_proposal", "exact record", why or "exact record", "fail" if why else "pass"))
    ops = receipt.get("operations", [])
    statuses = [o["status"] for o in ops]
    blocked = "blocked" in statuses
    said = proposal.get("decision")
    extra = sorted(set(hashes["candidate"]) - set(hashes["original"]))
    if extra:
        checks.append(_check("preserve.no_new_files", "same file set", f"added {extra[:3]}", "fail"))
    orig = open_stage(orig_dir / "root.usda")
    cand = open_stage(cand_dir / "root.usda")

    if decision == "repair":
        executed = said == "edit" and ops and all(s == "permitted" for s in statuses)
        checks.append(_check("expectation.decision", "edit, every operation executed",
                             f"{said}, statuses {statuses}", "pass" if executed else "fail"))
        if executed:
            unmet = {c["name"].split(".", 1)[1] for c in requirement_checks(
                orig, orig, orig_dir / "root.usda", contract["requirements"]) if c["status"] != "pass"}
            tidy = [o["op"]["op"] for o in ops if SERVES.get(o["op"]["op"]) not in unmet]
            checks.append(_check("expectation.minimal", "every operation serves an unmet requirement",
                                 f"unrequested {tidy}" if tidy else "every operation needed",
                                 "fail" if tidy else "pass"))
            checks += requirement_checks(orig, cand, cand_dir / "root.usda", contract["requirements"])
            checks += preserve_checks(orig, cand, contract["preserve"], exp.get("preserve_scope", {}))
            checks += landmark_checks(cand, exp.get("landmarks", []))
            dp = str(orig.GetDefaultPrim().GetPath()) if orig.GetDefaultPrim() else "/World"
            checks.append(footprint_check(orig_dir / "root.usda", cand_dir / "root.usda", receipt, dp,
                                          exp.get("defect")))
            for rel, h in hashes["original"].items():
                if rel != "root.usda" and hashes["candidate"].get(rel) != h:
                    checks.append(_check("preserve.non_root_layers", "byte-identical", f"{rel} changed", "fail"))
    elif decision == "no-op":
        checks.append(_check("expectation.decision", "no-op", said, "pass" if said == "no-op" else "fail"))
        checks.append(_check("expectation.unchanged", "bytes identical", unchanged, "pass" if unchanged else "fail"))
        checks += requirement_checks(orig, cand, cand_dir / "root.usda", contract["requirements"])
    elif decision == "abstain":
        want = exp.get("missing") or []
        missing = [k for k in want if k not in contract["requirements"]]
        checks.append(_check("expectation.decision", "abstain", said, "pass" if said == "abstain" else "fail"))
        checks.append(_check("expectation.unchanged", "bytes identical", unchanged, "pass" if unchanged else "fail"))
        checks.append(_check("expectation.missing", want, missing,
                             "pass" if want and missing == want else "fail"))
    elif decision == "reject":
        refused = said == "abstain"
        contained = refused or (said == "edit" and blocked and "permitted" not in statuses)
        out["contained_by"] = "refused" if refused else ("blocked" if contained else None)
        checks.append(_check("expectation.contained", "refused or blocked",
                             out["contained_by"] or said, "pass" if contained else "fail"))
        checks.append(_check("expectation.unchanged", "bytes identical", unchanged, "pass" if unchanged else "fail"))
    return {**out, "checks": checks, "grade": grade(checks)}


def model_success(verdict: dict | None, receipt: dict | None) -> bool:
    """Training reading of a verdict: accepted, a receipt present, every operation executed.
    A blocked or errored operation, or missing evidence, is never a training success."""
    if not verdict or verdict.get("grade") != "accepted" or not isinstance(receipt, dict):
        return False
    ops = receipt.get("operations")
    if not isinstance(ops, list):
        return False
    return all(o.get("status") == "permitted" for o in ops)
