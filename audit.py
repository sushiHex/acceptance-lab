"""audit.py: grade a scene that already exists against a written contract. Read-only.

    python audit.py contracts/stagewright-wind-tunnel.json --root PATH/TO/stagewright

No model, no proposal and no executor: the scene is the candidate and nothing is written. The
requirement checks are the judge's own (`scene.requirement_checks`). Two audit-only checks cover
what a stage contract states beyond the repair tasks: the time-code rate, and a region the
scene's geometry must contain. A key this file cannot check fails the audit; it is never skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from pxr import Usd, UsdGeom

import scene

LEAF_EXT = (".png", ".jpg")   # textures: confined like layers, never opened
EXTRA_KEYS = {"time_codes_per_second", "contains_m"}


def audit(contract: dict, root: Path) -> dict:
    root = Path(root).resolve()
    scene_file = root / contract["scene"]
    bundle = root / contract["bundle"]
    requirements = contract["requirements"]
    unknown = sorted(set(requirements) - scene.REQUIREMENT_KEYS - EXTRA_KEYS)
    checks = [scene._check("contract.keys", "every key checkable", unknown or "every key checkable",
                           "fail" if unknown else "pass")]
    out = {"scene": contract["scene"], "bundle_sha256": None}
    try:   # OpenUSD raises its own exception types; anything unreadable is an error, never a verdict
        if bundle not in scene_file.parents:
            raise scene.SceneError(f"{contract['scene']} is outside its bundle")
        missing = scene.walk(scene_file, bundle, LEAF_EXT)["missing"]
        stage = scene.open_stage(scene_file)
        layers = [Path(os.path.realpath(l.realPath)) for l in stage.GetUsedLayers() if not l.anonymous]
        stray = sorted(p.name for p in layers if p.suffix != ".usda" or bundle not in p.parents)
        # a texture suffix is allowed on attribute values only, never on a sublayer, reference or payload
        stray += sorted(a for l in stage.GetUsedLayers() for a in l.GetCompositionAssetDependencies()
                        if a.endswith(LEAF_EXT))
        required = scene.requirement_checks(
            stage, stage, scene_file, {k: v for k, v in requirements.items() if k in scene.REQUIREMENT_KEYS})
        if "time_codes_per_second" in requirements:
            want, pr = requirements["time_codes_per_second"], stage.GetRootLayer().pseudoRoot
            have = pr.GetInfo("timeCodesPerSecond") if pr.HasInfo("timeCodesPerSecond") else None
            required.append(scene._check("requirement.time_codes_per_second", want, have,
                                         "pass" if have == want else "fail"))
        if "contains_m" in requirements:
            required.append(_contains(stage, requirements["contains_m"]))
        digest = hashlib.sha256(json.dumps(scene.bundle_hashes(bundle), sort_keys=True).encode()).hexdigest()
    except Exception as e:  # noqa: BLE001
        checks.append(scene._check("scene.readable", "confined and readable", str(e)[:200], "fail"))
        return {**out, "checks": checks, "grade": "error"}
    checks.append(scene._check("scene.confined", "confined, nothing missing", missing or "confined, nothing missing",
                               "fail" if missing else "pass"))
    checks.append(scene._check("scene.layers", "every composed layer is a .usda file in the bundle",
                               stray or "every composed layer is a .usda file in the bundle",
                               "fail" if stray else "pass"))
    checks.append(scene._check("contract.requirements", "at least one", len(required),
                               "pass" if required else "fail"))
    checks += required
    return {**out, "bundle_sha256": digest, "checks": checks, "grade": scene.grade(checks)}


def _contains(stage, want: dict) -> dict:
    """The prim's axis-aligned geometry bound, in meters and Z-up, must contain the stated box.

    This shows the geometry's outer extent reaches around the box. It does not show an enclosing
    surface or that the box is free of obstructions.
    """
    name = f"requirement.contains_m:{want['prim']}"
    box = f"{want['min']} to {want['max']}"
    if any(lo > hi for lo, hi in zip(want["min"], want["max"])):
        return scene._check(name, box, "the contract's box has min above max", "fail")
    r = _trusted_bound(stage, want["prim"])
    if r is None:
        return scene._check(name, box, "no checkable geometry or stale extent", "unknown")
    ok = all(r.GetMin()[i] <= want["min"][i] and r.GetMax()[i] >= want["max"][i] for i in range(3))
    return scene._check(name, box, scene._fmt(r), "pass" if ok else "fail")


def _trusted_bound(stage, prim_path: str):
    """The prim's bound including Cube, Cylinder and other intrinsic shapes, or None.

    `scene.bound` counts points only, which is right for the repair tasks and blind to scenes
    built from intrinsic shapes. BBoxCache sees those, but it believes an authored extent, so
    every authored extent is first checked against the one its own schema computes. The bound is
    read at the default time only, so animated geometry or extent is not vouched for, and neither
    is a named prim that is inactive or invisible.
    """
    start = stage.GetPrimAtPath(prim_path)
    if not start or not start.IsActive():
        return None
    if UsdGeom.Imageable(start).ComputeVisibility() == UsdGeom.Tokens.invisible:
        return None
    tol = scene.GEOM_ABS_TOL_M / UsdGeom.GetStageMetersPerUnit(stage)
    for prim in Usd.PrimRange(start, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        shape = UsdGeom.Boundable(prim)
        if not shape:
            continue
        pb = UsdGeom.PointBased(prim)
        if shape.GetExtentAttr().ValueMightBeTimeVarying() or (pb and pb.GetPointsAttr().ValueMightBeTimeVarying()):
            return None
        attr = shape.GetExtentAttr()
        ext = attr.Get() if attr.HasAuthoredValue() else None   # an unauthored extent reads as a schema fallback
        if pb and not pb.GetPointsAttr().Get():
            if ext is None:
                continue          # no points and no extent: contributes nothing to the bound
            return None           # an extent with no points behind it
        real = UsdGeom.Boundable.ComputeExtentFromPlugins(shape, Usd.TimeCode.Default())
        if real is None or (pb and ext is None):
            return None
        if ext is not None and any(abs(a - b) > tol for lo, hi in zip(ext, real) for a, b in zip(lo, hi)):
            return None
    return scene._extent_range(stage, prim_path, canonical=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("contract", type=Path)
    ap.add_argument("--root", type=Path, required=True, help="directory the contract's paths are relative to")
    args = ap.parse_args()
    verdict = audit(json.loads(args.contract.read_text(encoding="utf-8")), args.root)
    for c in verdict["checks"]:
        print(f"{c['status']:8}{c['name']}: expected {c['expected']}, observed {c['observed']}")
    print(f"{verdict['grade']}  {verdict['scene']}  bundle sha256 {verdict['bundle_sha256']}")
    return 0 if verdict["grade"] == "accepted" else 1


if __name__ == "__main__":
    sys.exit(main())
