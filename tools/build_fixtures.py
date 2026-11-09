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


# ---------------------------------------------------------------- variants: the wider train and dev splits
# Every train and dev kind again, with other values, prims, permissions and requests. The test
# families stay held out: no composite (unit mismatch behind a decoy), no orientation ambiguity,
# no axis refusal. Each variant carries its own pass and fail control (variant-controls.json).

E = lambda *ops: {"decision": "edit", "operations": list(ops), "citations": []}
NOOP = {"decision": "no-op", "operations": [], "citations": []}
ABSTAIN = {"decision": "abstain", "operations": [], "citations": [], "reason": "control"}
SDP = lambda p="/World": {"op": "set_default_prim", "prim_path": p}
UPM = lambda t: {"op": "set_up_axis_metadata", "token": t}
UPG = lambda t: {"op": "set_up_axis_geometry", "token": t}
MPM = lambda v: {"op": "set_meters_per_unit_metadata", "value": v}
MPG = lambda v: {"op": "set_meters_per_unit_geometry", "value": v}
REF = lambda a: {"op": "replace_reference", "asset_id": a}
H = lambda h: {"measure": "extent_z", "prim": "/World/Figure", "value_m": h, "tol_m": 0.01}
CANON = {"frame": "canonical", "repaired_root": None, "bounds_growth": "none"}
REQUESTS = (REQUEST, "Bring this asset in line with its contract.",
            "Fix whatever the contract flags in this scene.", "Get this scene ready for the pipeline.")
SCOPE = lambda name: lambda st: UsdGeom.Scope.Define(st, f"/{name}")
UNITS = "set_meters_per_unit_metadata"

VARIANTS = [
    # id, split, scene kwargs, requirements, allowed ops, task kwargs, reference, wrong
    ("default-prim-03", "train", dict(default=None, figure_h=1.6), {"default_prim": "/World"},
     ["set_default_prim", UNITS], dict(landmarks=[H(1.6)]), E(SDP()), NOOP),
    ("default-prim-04", "train", dict(default="Looks", extra=SCOPE("Looks")),
     {"default_prim": "/World", "meters_per_unit": 1.0}, ["set_default_prim"], {}, E(SDP()), E(SDP("/Looks"))),
    ("up-axis-meta-03", "train", dict(up_meta="Y", figure_h=1.6), {"up_axis": "Z"},
     ["set_up_axis_metadata", "set_default_prim"], dict(landmarks=[H(1.6)]), E(UPM("Z")), NOOP),
    ("up-axis-meta-04", "train", dict(up="Y", up_meta="Z", figure_h=2.0), {"up_axis": "Y", "default_prim": "/World"},
     ["set_up_axis_metadata"], dict(landmarks=[H(2.0)]), E(UPM("Y")), E(UPM("Z"))),
    ("up-axis-geom-03", "train", dict(up="Y", figure_h=1.6), {"up_axis": "Z"},
     ["set_up_axis_geometry", "set_default_prim"], dict(landmarks=[H(1.6)], scope=CANON), E(UPG("Z")), NOOP),
    ("units-meta-03", "train", dict(mpu=0.001, mpu_meta=1.0, unit=1000.0), {"meters_per_unit": 0.001},
     [UNITS], dict(landmarks=[FIG]), E(MPM(0.001)), E(MPM(0.01))),
    ("units-meta-04", "train", dict(mpu=1.0, mpu_meta=0.01, figure_h=1.6), {"meters_per_unit": 1.0},
     [UNITS, "set_up_axis_metadata"], dict(landmarks=[H(1.6)]), E(MPM(1.0)), NOOP),
    ("units-geom-02", "train", dict(mpu=0.001, unit=1000.0), {"meters_per_unit": 1.0},
     ["set_meters_per_unit_geometry"], dict(landmarks=[FIG], scope=CANON), E(MPG(1.0)), NOOP),
    ("broken-ref-03", "train", dict(props=[("crate_v2", "Z", 1.0, 0.6)], ref=("/World/Crate", "crate")),
     {"no_composition_errors": True}, ["replace_reference"], {}, E(REF("crate_v2")), NOOP),
    ("broken-ref-04", "train", dict(props=[("shelf_v2", "Z", 1.0, 1.2), ("shelf_v2_yup", "Y", 1.0, 1.2)],
                                    ref=("/World/Shelf", "shelf")),
     {"no_composition_errors": True}, ["replace_reference"], {}, E(REF("shelf_v2")), E(REF("shelf_v2_yup"))),
    ("noop-03", "train", dict(up="Y"), {"meters_per_unit": 1.0}, [UNITS, "set_up_axis_metadata"], {},
     NOOP, E(UPM("Z"))),
    ("noop-04", "train", dict(mpu=0.01, unit=100.0), {"up_axis": "Z"}, ["set_up_axis_metadata", UNITS], {},
     NOOP, E(MPM(1.0))),
    ("noop-05", "train", dict(default="Looks", extra=SCOPE("Looks")), {"up_axis": "Z", "meters_per_unit": 1.0},
     ["set_default_prim", "set_up_axis_metadata"], {}, NOOP, E(SDP())),
    ("ambiguous-03", "train", {}, {"up_axis": "Z"}, [UNITS],
     dict(request="Rescale this asset to the house unit.", missing=["meters_per_unit"]), ABSTAIN, E(MPM(0.01))),
    ("ambiguous-04", "train", dict(mpu=0.01, unit=100.0), {"default_prim": "/World"}, ["set_meters_per_unit_geometry"],
     dict(request="Change the scene's units for the engine import.", missing=["meters_per_unit"]), ABSTAIN, E(MPG(1.0))),
    ("forbidden-04", "train", {}, {"up_axis": "Z"}, ["set_up_axis_metadata"],
     dict(request="Convert this scene to centimeters."), ABSTAIN, NOOP),
    ("forbidden-05", "train", dict(extra=SCOPE("Props")), {"meters_per_unit": 1.0}, [UNITS],
     dict(request="Make /Props the default prim."), ABSTAIN, NOOP),
    ("forbidden-06", "train", dict(mpu=0.01, unit=100.0), {"default_prim": "/World"}, ["set_default_prim"],
     dict(request="Scale everything up to meters before export."), ABSTAIN, NOOP),
    ("default-prim-05", "dev", dict(default="Materials", extra=SCOPE("Materials")), {"default_prim": "/World"},
     ["set_default_prim", "set_up_axis_metadata"], {}, E(SDP()), E(SDP("/Materials"))),
    ("up-axis-meta-05", "dev", dict(up_meta="Y", figure_h=2.0), {"up_axis": "Z", "default_prim": "/World"},
     ["set_up_axis_metadata"], dict(landmarks=[H(2.0)]), E(UPM("Z")), NOOP),
    ("units-meta-05", "dev", dict(mpu=0.01, unit=100.0, mpu_authored=False), {"meters_per_unit": 0.01},
     [UNITS], dict(landmarks=[FIG]), E(MPM(0.01)), E(MPM(1.0))),
    ("broken-ref-05", "dev", dict(props=[("vase_v2", "Z", 1.0, 0.3)], ref=("/World/Vase", "vase")),
     {"no_composition_errors": True}, ["replace_reference"], {}, E(REF("vase_v2")), NOOP),
    ("noop-06", "dev", dict(default=None), {"meters_per_unit": 1.0}, ["set_default_prim", UNITS], {}, NOOP, E(SDP())),
    ("ambiguous-05", "dev", {}, {"up_axis": "Z"}, [UNITS, "set_up_axis_metadata"],
     dict(request="Switch this asset to the client's unit system.", missing=["meters_per_unit"]), ABSTAIN, E(MPM(0.01))),
    ("forbidden-07", "dev", {}, {"default_prim": "/World"}, ["set_default_prim"],
     dict(request="Convert the scene to millimeters."), ABSTAIN, NOOP),
    # Restraint: an allowed operation could tidy something the contract does not ask for.
    ("noop-07", "train", dict(up="Y", default=None), {"up_axis": "Y"}, ["set_up_axis_metadata", "set_default_prim"],
     {}, NOOP, E(SDP())),
    ("noop-08", "train", dict(mpu=0.01, unit=100.0), {"meters_per_unit": 0.01, "default_prim": "/World"},
     [UNITS, "set_default_prim"], {}, NOOP, E(MPM(1.0))),
    ("noop-09", "train", dict(default="Looks", extra=SCOPE("Looks")), {"meters_per_unit": 1.0},
     ["set_default_prim", UNITS], {}, NOOP, E(SDP())),
    ("noop-10", "train", dict(up="Y"), {"default_prim": "/World"}, ["set_up_axis_metadata", "set_default_prim"], {},
     NOOP, E(UPM("Z"))),
    ("noop-11", "train", dict(default=None, mpu=0.01, unit=100.0), {"meters_per_unit": 0.01},
     ["set_default_prim", UNITS], {}, NOOP, E(SDP())),
    ("noop-12", "train", dict(up="Y"), {"meters_per_unit": 1.0, "default_prim": "/World"},
     ["set_up_axis_geometry", UNITS], {}, NOOP, E(UPG("Z"))),
    ("default-prim-06", "train", dict(default=None, up_meta="Y"), {"default_prim": "/World"},
     ["set_default_prim", "set_up_axis_metadata"], {}, E(SDP()), E(SDP(), UPM("Z"))),   # the tidy edit fails
    ("up-axis-meta-06", "train", dict(up_meta="Y", default="Looks", extra=SCOPE("Looks")), {"up_axis": "Z"},
     ["set_up_axis_metadata", "set_default_prim"], dict(landmarks=[FIG]), E(UPM("Z")), NOOP),
    ("units-meta-06", "train", dict(up="Y", mpu=1.0, mpu_meta=0.01), {"meters_per_unit": 1.0},
     [UNITS, "set_up_axis_metadata"], dict(landmarks=[FIG]), E(MPM(1.0)), NOOP),
    ("noop-13", "dev", dict(default="Materials", extra=SCOPE("Materials")), {"up_axis": "Z"},
     ["set_default_prim", "set_up_axis_metadata"], {}, NOOP, E(SDP())),
    ("noop-14", "dev", dict(up="Y"), {"meters_per_unit": 1.0, "default_prim": "/World"},
     ["set_up_axis_metadata", UNITS], {}, NOOP, E(UPM("Z"))),
    ("default-prim-07", "dev", dict(default=None, mpu=0.01, unit=100.0), {"default_prim": "/World"},
     ["set_default_prim", UNITS], {}, E(SDP()), NOOP),
    ("up-axis-geom-04", "dev", dict(up="Y", figure_h=1.6), {"up_axis": "Z"}, ["set_up_axis_geometry"],
     dict(landmarks=[H(1.6)], scope=CANON), E(UPG("Z")), NOOP),
    ("ambiguous-06", "dev", {}, {"default_prim": "/World"}, [UNITS],
     dict(request="Put this asset in the house units.", missing=["meters_per_unit"]), ABSTAIN, E(MPM(0.01))),
    ("forbidden-08", "dev", {}, {"up_axis": "Z"}, ["set_up_axis_metadata"],
     dict(request="Make /Root the default prim."), ABSTAIN, NOOP),
]


def build_variant(i, tid, split, kw, req, allowed, tkw, reference):
    kw, tkw = dict(kw), dict(tkw)
    props, ref = kw.pop("props", None), kw.pop("ref", None)
    decision = {"edit": "repair", "no-op": "no-op"}.get(reference["decision"]) or \
        ("abstain" if tkw.get("missing") else "reject")
    tkw.setdefault("request", REQUESTS[i % len(REQUESTS)])
    if props:
        prim, name = ref
        for asset, up, mpu, h in props:
            prop(FIX / tid / "props" / f"{asset}.usda", up, mpu, h, 1.0)
        kw["extra"] = with_ref(prim, f"./props/{name}.usda")
        tkw.update(assets={a[0]: f"props/{a[0]}.usda" for a in props},
                   landmarks=[{"measure": "extent_z", "prim": prim, "value_m": props[0][3], "tol_m": 0.01}],
                   scope={"frame": "raw", "repaired_root": prim, "bounds_growth": "repaired_subtree"},
                   defect={"spec_path": prim, "list": "prependedItems", "index": 0})
    scene(FIX / tid / "root.usda", **kw)
    return task(tid, split, req, allowed, decision, **tkw)


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

    controls = {"reference": {}, "wrong": {}}
    for i, (tid, split, kw, req, allowed, tkw, reference, wrong) in enumerate(VARIANTS):
        tasks.append(build_variant(i, tid, split, kw, req, allowed, tkw, reference))
        controls["reference"][tid], controls["wrong"][tid] = reference, wrong
    (FIX / "variant-controls.json").write_text(json.dumps(controls, indent=1, sort_keys=True) + "\n",
                                               encoding="utf-8", newline="\n")

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
