"""Probe 4: the pxr APIs the judge and executor depend on, in the pinned usd-exchange build."""
import os
import tempfile

from pxr import Ar, Sdf, Usd, UsdGeom, UsdUtils

ok = True


def report(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


print("pxr", Usd.GetVersion())

# 1. ReplaceItemEdits lives on the ListEditorProxy (primSpec.referenceList).
lay = Sdf.Layer.CreateAnonymous()
spec = Sdf.CreatePrimInLayer(lay, "/A")
spec.referenceList.Prepend(Sdf.Reference("./old.usda"))
replaced = spec.referenceList.ReplaceItemEdits(Sdf.Reference("./old.usda"), Sdf.Reference("./new.usda"))
items = [r.assetPath for r in spec.referenceList.prependedItems]
# Judge by the resulting list, not the return value (it returns None in 0.26.8).
print(f"INFO  ReplaceItemEdits returned {replaced!r}")
report("ReplaceItemEdits on referenceList", items == ["./new.usda"], items)
report("list-op flavour stays prepend", not spec.referenceList.isExplicit, "")

# 2. PrimAllPrimsPredicate includes an inactive prim.
st = Usd.Stage.CreateInMemory()
st.DefinePrim("/X").SetActive(False)
paths = [str(p.GetPath()) for p in Usd.PrimRange(st.GetPseudoRoot(), Usd.PrimAllPrimsPredicate)]
report("PrimAllPrimsPredicate lists inactive /X", "/X" in paths, paths)

# 3. ComputeAllDependencies names a sublayer; 4. search-path resolution against cwd.
d = tempfile.mkdtemp(prefix="probe-pxr-")
with open(os.path.join(d, "dep.usda"), "w", newline="\n") as f:
    f.write("#usda 1.0\n")
with open(os.path.join(d, "root.usda"), "w", newline="\n") as f:
    f.write("#usda 1.0\n(\n    subLayers = [@./dep.usda@]\n)\n")
layers, assets, unresolved = UsdUtils.ComputeAllDependencies(os.path.join(d, "root.usda"))
names = [os.path.basename(l.realPath) for l in layers]
report("ComputeAllDependencies names dep.usda", "dep.usda" in names, names)

old = os.getcwd()
os.chdir(d)
try:
    resolved = str(Ar.GetResolver().Resolve("dep.usda"))
finally:
    os.chdir(old)
report("search-path form resolves against cwd (the risk the walk closes)", bool(resolved), repr(resolved))

# 5. BBoxCache with useExtentsHint=False vs authored extent (smith 13's claim).
st2 = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(st2, UsdGeom.Tokens.z)
m = UsdGeom.Mesh.Define(st2, "/M")
m.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
m.CreateFaceVertexCountsAttr([3])
m.CreateFaceVertexIndicesAttr([0, 1, 2])
m.CreateExtentAttr([(0, 0, 0), (5, 5, 5)])
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
rng = cache.ComputeWorldBound(m.GetPrim()).ComputeAlignedRange()
mx = tuple(rng.GetMax())
print(f"INFO  BBoxCache max with stale extent 5 and points to 1: {mx}")
report("points cross-check needed (BBoxCache trusts authored extent)", mx[0] > 1.5, "if FAIL, the cross-check is belt and braces only")

# 6. Does an open-and-save of an unmodified hand-written .usda round-trip byte-identically?
src = (
    '#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
    'def Xform "World"\n{\n    def Mesh "Floor"\n    {\n        int[] faceVertexCounts = [3]\n'
    '        int[] faceVertexIndices = [0, 1, 2]\n        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n    }\n}\n\n'
)
p = os.path.join(d, "fixture.usda")
with open(p, "w", newline="\n") as f:
    f.write(src)
before = open(p, "rb").read()
lay2 = Sdf.Layer.FindOrOpen(p)
lay2.SetDirty = None  # no-op marker; we force a save path below
exported = lay2.ExportToString()
report("ExportToString of an untouched fixture equals the file", exported.encode() == before,
       "if FAIL, freeze fixtures as exported text so diffs stay small")
Sdf.Layer.FindOrOpen(p).Export(os.path.join(d, "resaved.usda"))
after = open(os.path.join(d, "resaved.usda"), "rb").read()
report("Export round-trip byte-identical", after == before, f"{len(before)} vs {len(after)} bytes")

# 7. A one-token edit after freezing as exported text yields a small diff.
frozen = os.path.join(d, "frozen.usda")
with open(frozen, "w", newline="\n") as f:
    f.write(exported)
lay3 = Sdf.Layer.FindOrOpen(frozen)
st3 = Usd.Stage.Open(lay3)
UsdGeom.SetStageUpAxis(st3, UsdGeom.Tokens.y)
lay3.Save()
import difflib
diff = [l for l in difflib.unified_diff(exported.splitlines(), open(frozen).read().splitlines(), lineterm="")
        if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
report("one-token edit on frozen text diffs to 2 lines", len(diff) == 2, diff)

print("PROBE 4", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
