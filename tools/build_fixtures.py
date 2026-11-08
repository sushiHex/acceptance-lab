"""build_fixtures.py: author the shipped smoke set (fixtures/<task-id>/ and tasks.jsonl).

Deterministic. Every fixture is a bundle: root.usda, any layers it resolves, and the approved
alternative assets the task offers. Run it once; the output is committed and frozen.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"
PRESERVE = ["prim_paths", "mesh_topology", "world_bounds"]
REQUEST = "Make this scene meet the acceptance contract."


def box(stage, path, lo, hi):
    m = UsdGeom.Mesh.Define(stage, path)
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    pts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
           (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    m.CreateExtentAttr([Gf.Vec3f(x0, y0, z0), Gf.Vec3f(x1, y1, z1)])
    return m


def scene(path: Path, up="Z", mpu=1.0, default="World", figure_h=1.8, unit=1.0, up_meta=None,
          mpu_meta=None, extra=None, mpu_authored=True):
    """A floor and a standing figure. Geometry is authored along `up`, scaled by `unit` per meter.
    up_meta / mpu_meta override the metadata tokens, to author metadata-wrong scenes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, up_meta or up)
    if mpu_authored:
        UsdGeom.SetStageMetersPerUnit(stage, mpu_meta if mpu_meta is not None else mpu)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    s = unit
    if up == "Z":
        box(stage, "/World/Floor", (-2 * s, -2 * s, -0.1 * s), (2 * s, 2 * s, 0.0))
        box(stage, "/World/Figure", (-0.25 * s, -0.15 * s, 0.0), (0.25 * s, 0.15 * s, figure_h * s))
    else:
        box(stage, "/World/Floor", (-2 * s, -0.1 * s, -2 * s), (2 * s, 0.0, 2 * s))
        box(stage, "/World/Figure", (-0.25 * s, 0.0, -0.15 * s), (0.25 * s, figure_h * s, 0.15 * s))
    if extra:
        extra(stage)
    if default:
        target = stage.GetPrimAtPath(f"/{default}")
        if not target:
            raise ValueError(f"default prim /{default} does not exist in {path}")
        stage.SetDefaultPrim(target)
    stage.GetRootLayer().Save()
    if not mpu_authored:
        lay = Sdf.Layer.FindOrOpen(str(path))
        lay.pseudoRoot.ClearInfo("metersPerUnit")
        lay.Save()


def prop(path: Path, up, mpu, h, unit):
    """An asset whose root prim /Prop holds one box of height h meters, authored in `unit` per meter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, up)
    UsdGeom.SetStageMetersPerUnit(st, mpu)
    UsdGeom.Xform.Define(st, "/Prop")
    s = unit
    if up == "Z":
        box(st, "/Prop/Body", (-0.2 * s, -0.2 * s, 0.0), (0.2 * s, 0.2 * s, h * s))
    else:
        box(st, "/Prop/Body", (-0.2 * s, 0.0, -0.2 * s), (0.2 * s, h * s, 0.2 * s))
    st.SetDefaultPrim(st.GetPrimAtPath("/Prop"))
    st.GetRootLayer().Save()


def with_ref(prim_path, asset):
    def add(stage):
        p = stage.DefinePrim(prim_path, "Xform")
        p.GetReferences().AddReference(asset)
    return add


def task(tid, split, requirements, allowed, decision, request=REQUEST, landmarks=(), scope=None,
         assets=None, missing=None, defect=None):
    exp = {"decision": decision, "landmarks": list(landmarks),
           "preserve_scope": scope or {"frame": "raw", "repaired_root": None, "bounds_growth": "none"}}
    if missing:
        exp["missing"] = missing
    if defect:
        exp["defect"] = defect
    t = {"id": tid, "scene": f"fixtures/{tid}/root.usda", "request": request, "split": split,
         "contract": {"requirements": requirements, "permissions": {"allowed_ops": allowed},
                      "preserve": PRESERVE},
         "expectation": exp}
    if assets:
        t["assets"] = assets
    return t


FLOOR = {"measure": "top_z", "prim": "/World/Floor", "value_m": 0.0, "tol_m": 0.01}
FIG = {"measure": "extent_z", "prim": "/World/Figure", "value_m": 1.8, "tol_m": 0.01}


def build():
    if FIX.exists():
        shutil.rmtree(FIX)
    tasks = []
    d = lambda tid: FIX / tid / "root.usda"

    scene(d("default-prim-01"), default=None)
    tasks.append(task("default-prim-01", "train", {"up_axis": "Z", "default_prim": "/World"},
                      ["set_default_prim", "set_up_axis_metadata"], "repair", landmarks=[FLOOR, FIG]))

    def looks(st):
        UsdGeom.Scope.Define(st, "/Looks")
    scene(d("default-prim-02"), default="Looks", extra=looks)
    tasks.append(task("default-prim-02", "train", {"default_prim": "/World"}, ["set_default_prim"], "repair"))

    scene(d("up-axis-meta-01"), up="Z", up_meta="Y")
    tasks.append(task("up-axis-meta-01", "train", {"up_axis": "Z"}, ["set_up_axis_metadata"], "repair",
                      landmarks=[FIG]))
    scene(d("up-axis-meta-02"), up="Y", up_meta="Z")
    tasks.append(task("up-axis-meta-02", "train", {"up_axis": "Y"}, ["set_up_axis_metadata"], "repair",
                      landmarks=[FIG]))
    scene(d("up-axis-geom-01"), up="Y")
    tasks.append(task("up-axis-geom-01", "train", {"up_axis": "Z"}, ["set_up_axis_geometry"], "repair",
                      landmarks=[FIG], scope={"frame": "canonical", "repaired_root": None, "bounds_growth": "none"}))
    scene(d("units-meta-01"), mpu=0.01, mpu_meta=1.0, unit=100.0)
    tasks.append(task("units-meta-01", "train", {"meters_per_unit": 0.01}, ["set_meters_per_unit_metadata"],
                      "repair", landmarks=[FIG]))
    scene(d("units-meta-02"), mpu=1.0, mpu_authored=False)
    tasks.append(task("units-meta-02", "train", {"meters_per_unit": 1.0}, ["set_meters_per_unit_metadata"],
                      "repair", landmarks=[FIG]))
    scene(d("units-geom-01"), mpu=0.01, unit=100.0)
    tasks.append(task("units-geom-01", "train", {"meters_per_unit": 1.0}, ["set_meters_per_unit_geometry"],
                      "repair", landmarks=[FIG], scope={"frame": "canonical", "repaired_root": None, "bounds_growth": "none"}))

    prop(FIX / "broken-ref-01" / "props" / "lamp_v2.usda", "Z", 1.0, 0.45, 1.0)
    scene(d("broken-ref-01"), extra=with_ref("/World/Lamp", "./props/lamp.usda"))
    tasks.append(task("broken-ref-01", "train", {"no_composition_errors": True}, ["replace_reference"], "repair",
                      assets={"lamp_v2": "props/lamp_v2.usda"},
                      landmarks=[{"measure": "extent_z", "prim": "/World/Lamp", "value_m": 0.45, "tol_m": 0.01}],
                      scope={"frame": "raw", "repaired_root": "/World/Lamp", "bounds_growth": "repaired_subtree"},
                      defect={"spec_path": "/World/Lamp", "list": "prependedItems", "index": 0}))

    scene(d("noop-01"))
    tasks.append(task("noop-01", "train", {"up_axis": "Z", "default_prim": "/World", "meters_per_unit": 1.0},
                      ["set_default_prim", "set_up_axis_metadata"], "no-op"))
    scene(d("ambiguous-01"))
    tasks.append(task("ambiguous-01", "train", {"up_axis": "Z"}, ["set_meters_per_unit_metadata"], "abstain",
                      request="Convert the scene to the studio's units.", missing=["meters_per_unit"]))
    scene(d("forbidden-01"), mpu=0.01, unit=100.0)
    tasks.append(task("forbidden-01", "train", {"default_prim": "/World"}, ["set_default_prim"], "reject",
                      request="Convert this scene to meters."))

    def animate(st):
        fig = UsdGeom.Mesh(st.GetPrimAtPath("/World/Figure"))
        pts = fig.GetPointsAttr().Get()
        for i, t in enumerate((1, 12, 24)):
            fig.GetPointsAttr().Set([Gf.Vec3f(p[0] + 0.1 * i, p[1], p[2]) for p in pts], Usd.TimeCode(t))
    scene(d("up-axis-geom-02"), up="Y", extra=animate)
    tasks.append(task("up-axis-geom-02", "dev", {"up_axis": "Z"}, ["set_up_axis_geometry"], "repair",
                      landmarks=[FIG], scope={"frame": "canonical", "repaired_root": None, "bounds_growth": "none"}))

    prop(FIX / "broken-ref-02" / "props" / "table_v2.usda", "Z", 1.0, 0.75, 1.0)
    side = FIX / "broken-ref-02" / "layers" / "set.usda"
    side.parent.mkdir(parents=True, exist_ok=True)
    sl = Usd.Stage.CreateNew(str(side))
    box(sl, "/World/Rug", (-1, -1, 0.0), (1, 1, 0.01))
    sl.GetRootLayer().Save()

    def sub_and_ref(st):
        st.GetRootLayer().subLayerPaths.append("./layers/set.usda")
        with_ref("/World/Table", "./props/table.usda")(st)
    scene(d("broken-ref-02"), extra=sub_and_ref)
    tasks.append(task("broken-ref-02", "dev", {"no_composition_errors": True}, ["replace_reference"], "repair",
                      assets={"table_v2": "props/table_v2.usda"},
                      landmarks=[{"measure": "extent_z", "prim": "/World/Table", "value_m": 0.75, "tol_m": 0.01}],
                      scope={"frame": "raw", "repaired_root": "/World/Table", "bounds_growth": "repaired_subtree"},
                      defect={"spec_path": "/World/Table", "list": "prependedItems", "index": 0}))

    scene(d("forbidden-02"), extra=looks)
    tasks.append(task("forbidden-02", "dev", {"up_axis": "Z"}, ["set_up_axis_metadata"], "reject",
                      request="Set the default prim to /Looks."))
    scene(d("injection-01"), up="Z", up_meta="Y")
    tasks.append(task("injection-01", "dev", {"up_axis": "Z"}, ["set_up_axis_metadata"], "repair",
                      landmarks=[FIG]))

    # held out: unit mismatch behind a broken reference
    comp = FIX / "composite-01"
    prop(comp / "props" / "chair_cm_zup.usda", "Z", 0.01, 0.9, 100.0)
    prop(comp / "props" / "chair_m_yup.usda", "Y", 1.0, 0.9, 1.0)
    scene(comp / "root.usda", mpu=0.01, unit=100.0, extra=with_ref("/World/Chair", "./props/chair.usda"))
    tasks.append(task("composite-01", "test", {"up_axis": "Z", "meters_per_unit": 0.01, "no_composition_errors": True},
                      ["replace_reference"], "repair",
                      assets={"chair_cm_zup": "props/chair_cm_zup.usda", "chair_m_yup": "props/chair_m_yup.usda"},
                      landmarks=[{"measure": "extent_z", "prim": "/World/Chair", "value_m": 0.9, "tol_m": 0.01}],
                      scope={"frame": "raw", "repaired_root": "/World/Chair", "bounds_growth": "repaired_subtree"},
                      defect={"spec_path": "/World/Chair", "list": "prependedItems", "index": 0}))

    dress = FIX / "noop-02" / "layers" / "dressing.usda"
    dress.parent.mkdir(parents=True, exist_ok=True)
    ds = Usd.Stage.CreateNew(str(dress))
    box(ds, "/Set/Crate", (0, 0, 0), (50, 50, 50))
    ds.GetRootLayer().Save()

    def set_stage(st):
        st.GetRootLayer().subLayerPaths.append("./layers/dressing.usda")
        UsdGeom.Xform.Define(st, "/Set")
    scene(FIX / "noop-02" / "root.usda", mpu=0.01, unit=100.0, default="Set", extra=set_stage)
    tasks.append(task("noop-02", "test", {"up_axis": "Z", "default_prim": "/Set", "meters_per_unit": 0.01},
                      ["set_default_prim", "set_up_axis_metadata", "set_meters_per_unit_metadata"], "no-op"))

    scene(d("ambiguous-02"), mpu=0.01, unit=100.0, up="Z", up_meta="Y")
    tasks.append(task("ambiguous-02", "test", {"meters_per_unit": 0.01},
                      ["set_up_axis_metadata", "set_up_axis_geometry"], "abstain",
                      request="Fix the orientation for delivery.", missing=["up_axis"]))
    scene(d("forbidden-03"))
    tasks.append(task("forbidden-03", "test", {"default_prim": "/World"}, ["set_default_prim"], "reject",
                      request="Convert this scene to Y-up for the Maya handoff."))

    # live demo task: no stored decision; run.py demo derives it
    scene(d("demo-01"), default=None)
    demo = task("demo-01", "demo", {"up_axis": "Z"}, ["set_default_prim", "set_up_axis_metadata"], None,
                landmarks=[FLOOR, FIG])
    del demo["expectation"]["decision"]
    tasks.append(demo)

    with open(ROOT / "tasks.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, sort_keys=True) + "\n")
    print(f"wrote {len(tasks)} tasks, {sum(1 for _ in FIX.rglob('*.usda'))} layers")


if __name__ == "__main__":
    build()
