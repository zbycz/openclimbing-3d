#!/usr/bin/env python3
# Generates v3/kaggle/tiles/korno-v3-tiles.ipynb - a streaming LOD pyramid over the full 17M mesh.
import json
import pathlib

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md(
    """
# Korno v3 — a streaming multiresolution pyramid

v2 ships one 90 MB glTF: the browser downloads all of it before drawing anything, and the mesh has
to be decimated from 17 050 230 triangles to 2.5 M to fit. This builds the Sketchfab-style
alternative instead — the model is cut into a **tile pyramid**, and the viewer downloads only the
tiles inside the frustum, only to the depth the screen can actually resolve.

The wall has a planar UV frame already (v2's orthophoto parametrisation), which makes the spatial
index obvious: a **quadtree over UV**, exactly like a map tile pyramid.

```
level 0   2 x 1 tiles     coarse, ~33k triangles each      -> first paint
level 1   4 x 2
level 2   8 x 4
level 3  16 x 8
level 4  32 x 16 tiles    the ORIGINAL triangles, undecimated, all 17 050 230 of them
```

Leaves are exact: every triangle of the full mesh ends up in exactly one leaf, sorted by the UV of
its centroid. Each parent is its four children merged and decimated back to one tile's budget, so
cost per tile is constant and the pyramid is built bottom-up in one pass.

Three details make it work:

* **UV is analytic.** `to_uv` is a plane projection, so texture coordinates are recomputed from
  vertex positions after every decimation rather than carried through it — which is just as well,
  since Open3D's quadric decimation drops UVs.
* **Skirts, not stitching.** Neighbouring tiles rendered at different levels leave cracks. Every cut
  edge gets a flange extruded backwards along the view axis, deep enough to cover the level's own
  geometric error. It is what Cesium and Google Earth do, and it needs no agreement between tiles.
* **Real geometric error.** Each node measures the 99th-percentile distance from the original
  triangles in its extent to its own decimated surface. That number is what the viewer turns into a
  screen-space error, so refinement follows measured deviation rather than a guess about level.

Nothing is compressed or quantised here — full float32 positions and normals, so this is the
top-quality reference. Compression is a separate, later question.
"""
)

code(
    r"""
import os, sys, io, json, time, glob, struct, shutil, subprocess

CFG = dict(
    INPUT_DIR = "/kaggle/input",
    OUT       = "/kaggle/working",
    TILES     = "/kaggle/working/tiles",

    BASE      = (2, 1),   # tiles at level 0
    LEVELS    = 5,        # 0..4, so the leaf grid is 32 x 16
    TEX_MAX   = 512,      # per-tile texture, capped; finer tiles keep their native atlas crop
    TEX_Q     = 90,
    ERR_PCT   = 99.0,     # percentile of the sampled deviation used as the tile's geometric error
    ERR_SAMPLES = 40000,
    SKIRT_MUL = 3.0,      # skirt depth = this * the tile's geometric error (floored, see below)
    SKIRT_MIN = 0.004,    # ~1 cm at this scene scale
)
os.makedirs(CFG["TILES"], exist_ok=True)
T0 = time.time()
REPORT = {"cfg": {k: v for k, v in CFG.items()}, "timings": {}}

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open3d", "--no-warn-conflicts"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pycolmap==4.1.1"])
import numpy as np
import open3d as o3d
import pycolmap
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
print("open3d", o3d.__version__, "| cpus", os.cpu_count())


def stamp(name, t):
    REPORT["timings"][name] = round(time.time() - t, 1)
    print(f"[{name}] {REPORT['timings'][name]} s   (total {round(time.time() - T0)} s)", flush=True)


def find(pattern):
    hits = sorted(glob.glob(os.path.join(CFG["INPUT_DIR"], "**", pattern), recursive=True))
    assert hits, f"{pattern} not found"
    return hits


MESH_PLY = find("korno_v2_mesh.ply")[0]
SPARSE = os.path.dirname(find("cameras.bin")[0])
GLB = find("korno_v2_desmudged.glb")[0]
print("mesh", round(os.path.getsize(MESH_PLY) / 1e6, 1), "MB | glb", round(os.path.getsize(GLB) / 1e6, 1), "MB")
"""
)

md("## 1. The mesh, the UV frame and the photo texture — all as v2 built them")

code(
    r"""
t = time.time()
mesh = o3d.io.read_triangle_mesh(MESH_PLY)
tri_cluster, cluster_n, _ = mesh.cluster_connected_triangles()
tri_cluster, cluster_n = np.asarray(tri_cluster), np.asarray(cluster_n)
mesh.remove_triangles_by_mask(cluster_n[tri_cluster] < max(20000, 0.02 * cluster_n.max()))
mesh.remove_unreferenced_vertices()
mesh.remove_degenerate_triangles()
mesh.compute_vertex_normals()
V = np.asarray(mesh.vertices).astype(np.float32)
F = np.asarray(mesh.triangles).astype(np.int32)
N = np.asarray(mesh.vertex_normals).astype(np.float32)
print("mesh:", len(F), "triangles,", len(V), "vertices")
REPORT["mesh"] = {"triangles": int(len(F)), "vertices": int(len(V))}

rec = pycolmap.Reconstruction(SPARSE)
IMS = [i for i in rec.images.values() if i.has_pose]
view = np.mean([i.viewing_direction() / np.linalg.norm(i.viewing_direction()) for i in IMS], 0)
view /= np.linalg.norm(view)
up = -np.mean([i.cam_from_world().rotation.matrix()[1] for i in IMS], 0)
up /= np.linalg.norm(up)
right = np.cross(view, up); right /= np.linalg.norm(right)
up = np.cross(right, view)
ORIGIN = np.asarray(mesh.vertices).mean(0)


def to_uv(X):
    d = X - ORIGIN
    return np.stack([d @ right, d @ up], 1)


uvV = to_uv(np.asarray(mesh.vertices))
UV_LO = np.percentile(uvV, 0.2, axis=0)
UV_SPAN = np.percentile(uvV, 99.8, axis=0) - UV_LO
UV = ((uvV - UV_LO) / UV_SPAN).astype(np.float32)      # [0,1] over the wall, v pointing up
VIEW = view.astype(np.float32)
print("uv span", UV_SPAN.round(3))
stamp("mesh", t)
"""
)

code(
    r"""
t = time.time()
# the photo-baked atlas lives inside korno_v2_desmudged.glb - pull the JPEG straight out of it
with open(GLB, "rb") as fh:
    assert struct.unpack("<I", fh.read(4))[0] == 0x46546C67
    fh.read(8)
    jlen, _ = struct.unpack("<II", fh.read(8))
    gltf = json.loads(fh.read(jlen))
    blen, _ = struct.unpack("<II", fh.read(8))
    blob = fh.read(blen)
bv = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
ATLAS = np.asarray(Image.open(io.BytesIO(blob[bv["byteOffset"]:bv["byteOffset"] + bv["byteLength"]])).convert("RGB"))
del blob
ATLAS = ATLAS[::-1]                    # glTF stores v downwards; keep it v-up like UV
AH, AW = ATLAS.shape[:2]
print("atlas", AW, "x", AH)
REPORT["atlas"] = [int(AW), int(AH)]
stamp("atlas", t)
"""
)

md(
    """
## 2. Sort every triangle into a leaf

By the UV of its centroid, so each of the 17 M triangles lands in exactly one of the 32 × 16 leaves
and the leaf level is the original mesh, losslessly redistributed.
"""
)

code(
    r"""
t = time.time()
GX = CFG["BASE"][0] * 2 ** (CFG["LEVELS"] - 1)
GY = CFG["BASE"][1] * 2 ** (CFG["LEVELS"] - 1)
cu = (UV[F[:, 0], 0] + UV[F[:, 1], 0] + UV[F[:, 2], 0]) / 3.0
cv = (UV[F[:, 0], 1] + UV[F[:, 1], 1] + UV[F[:, 2], 1]) / 3.0
gi = np.clip((cu * GX).astype(np.int32), 0, GX - 1)
gj = np.clip((cv * GY).astype(np.int32), 0, GY - 1)
key = (gj.astype(np.int64) * GX + gi).astype(np.int64)
del cu, cv, gi, gj
order = np.argsort(key, kind="stable")
kk = key[order]
del key
starts = np.searchsorted(kk, np.arange(GX * GY), side="left")
ends = np.searchsorted(kk, np.arange(GX * GY), side="right")
counts = ends - starts
print(f"leaf grid {GX} x {GY} = {GX * GY} cells")
print(f"triangles per leaf: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()},"
      f" empty {int((counts == 0).sum())}")
REPORT["leaf_grid"] = [int(GX), int(GY)]
REPORT["leaf_tris"] = {"min": int(counts.min()), "median": int(np.median(counts)),
                       "max": int(counts.max()), "empty": int((counts == 0).sum())}
stamp("partition", t)
"""
)

md(
    """
## 3. Build the pyramid bottom-up

A node holds a *core* mesh (what gets decimated and what its parent is built from) and, on disk, that
core plus skirts. The skirt is added last so it never feeds into the next level up.
"""
)

code(
    r"""
BUDGET = int(np.median(counts[counts > 0]))    # every tile lands on ~the same triangle count,
                                              # so each level up is 4x coarser per unit area
print("tile budget:", BUDGET, "triangles")
REPORT["budget"] = BUDGET


def submesh(face_idx):
    f = F[face_idx]
    uniq, inv = np.unique(f, return_inverse=True)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V[uniq].astype(np.float64)),
        o3d.utility.Vector3iVector(inv.reshape(-1, 3).astype(np.int32)))
    m.vertex_normals = o3d.utility.Vector3dVector(N[uniq].astype(np.float64))
    return m


# leaf: the original triangles; otherwise the four children merged and decimated back
def core_of(level, i, j, children):
    if level == CFG["LEVELS"] - 1:
        c = starts[j * GX + i], ends[j * GX + i]
        if c[1] <= c[0]:
            return None
        return submesh(order[c[0]:c[1]])
    m = o3d.geometry.TriangleMesh()
    for ch in children:
        m += ch
    if len(m.triangles) == 0:
        return None
    # `+=` concatenates without welding: the children share their seam vertices exactly, and left
    # unwelded those seams stay boundaries all the way up and the decimator cannot cross them
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    if len(m.triangles) > BUDGET:
        m = m.simplify_quadric_decimation(BUDGET)
    m.compute_vertex_normals()
    return m


# 99th-percentile distance from the original triangles under this node to the node's own surface
def geometric_error(m, level, i, j):
    if level == CFG["LEVELS"] - 1:
        return 0.0
    step = 2 ** (CFG["LEVELS"] - 1 - level)
    cells = [(j * step + b) * GX + (i * step + a) for a in range(step) for b in range(step)]
    idx = np.concatenate([order[starts[c]:ends[c]] for c in cells]) if cells else np.empty(0, np.int64)
    if not len(idx):
        return 0.0
    if len(idx) > CFG["ERR_SAMPLES"]:
        idx = idx[np.linspace(0, len(idx) - 1, CFG["ERR_SAMPLES"]).astype(np.int64)]
    tri = V[F[idx]]
    w = np.random.default_rng(0).dirichlet((1, 1, 1), len(idx)).astype(np.float32)
    pts = (tri * w[:, :, None]).sum(1)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    d = sc.compute_distance(o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)).numpy()
    return float(np.percentile(d, CFG["ERR_PCT"]))
"""
)

code(
    r"""
# extrude the cut edges backwards so a neighbour at a coarser level cannot show a crack
def with_skirt(m, depth, level, i, j):
    Vm = np.asarray(m.vertices)
    Fm = np.asarray(m.triangles)
    Nm = np.asarray(m.vertex_normals)
    if not len(Fm):
        return Vm, Fm, Nm
    e = np.sort(np.concatenate([Fm[:, [0, 1]], Fm[:, [1, 2]], Fm[:, [2, 0]]]), axis=1)
    uniq, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    bnd = uniq[cnt == 1]
    if not len(bnd):
        return Vm, Fm, Nm
    # Only skirt edges lying on this tile's own cut line - the wall's ragged outer rim is real
    # geometry and must not grow a flange. The cut lines are the nominal grid, known exactly, so the
    # test is against those rather than against the tile's actual extent.
    uvm = (to_uv(Vm) - UV_LO) / UV_SPAN
    gxl = CFG["BASE"][0] * 2 ** level
    gyl = CFG["BASE"][1] * 2 ** level
    r0 = np.array([i / gxl, j / gyl])
    r1 = np.array([(i + 1) / gxl, (j + 1) / gyl])
    mid = uvm[bnd].mean(1)
    eps = 2.0 * np.median(np.linalg.norm(uvm[bnd[:, 0]] - uvm[bnd[:, 1]], axis=1))
    near = (np.abs(mid - r0) < eps).any(1) | (np.abs(mid - r1) < eps).any(1)
    bnd = bnd[near]
    if not len(bnd):
        return Vm, Fm, Nm
    ring = np.unique(bnd)
    remap = np.full(len(Vm), -1, np.int64)
    remap[ring] = np.arange(len(ring)) + len(Vm)
    Vs = np.vstack([Vm, Vm[ring] + VIEW.astype(np.float64) * depth])
    Ns = np.vstack([Nm, Nm[ring]])
    a, b = bnd[:, 0], bnd[:, 1]
    quads = np.stack([np.stack([a, b, remap[b]], 1), np.stack([a, remap[b], remap[a]], 1)]).reshape(-1, 3)
    return Vs, np.vstack([Fm, quads]).astype(np.int64), Ns
"""
)

code(
    r"""
def write_tile(level, i, j, m, err):
    Vm, Fm, Nm = with_skirt(m, max(CFG["SKIRT_MIN"], CFG["SKIRT_MUL"] * err), level, i, j)
    uvm = ((to_uv(Vm) - UV_LO) / UV_SPAN).astype(np.float32)
    lo = uvm.min(0)
    hi = uvm.max(0)
    span = np.maximum(hi - lo, 1e-6)
    uvl = ((uvm - lo) / span).astype(np.float32)

    d = os.path.join(CFG["TILES"], str(level))
    os.makedirs(d, exist_ok=True)
    small = len(Vm) <= 65535
    idx = Fm.astype(np.uint16 if small else np.uint32)
    with open(os.path.join(d, f"{i}_{j}.bin"), "wb") as fh:
        fh.write(struct.pack("<4sIIII", b"KRN3", 1, len(Vm), len(Fm), 2 if small else 4))
        fh.write(Vm.astype(np.float32).tobytes())
        fh.write(Nm.astype(np.float32).tobytes())
        fh.write(uvl.tobytes())
        fh.write(idx.tobytes())

    x0 = int(np.clip(np.floor(lo[0] * AW), 0, AW - 1)); x1 = int(np.clip(np.ceil(hi[0] * AW), x0 + 1, AW))
    y0 = int(np.clip(np.floor(lo[1] * AH), 0, AH - 1)); y1 = int(np.clip(np.ceil(hi[1] * AH), y0 + 1, AH))
    crop = ATLAS[y0:y1, x0:x1]
    img = Image.fromarray(crop[::-1])                      # back to v-down for the JPEG
    if max(img.size) > CFG["TEX_MAX"]:
        s = CFG["TEX_MAX"] / max(img.size)
        img = img.resize((max(1, round(img.size[0] * s)), max(1, round(img.size[1] * s))), Image.LANCZOS)
    img.save(os.path.join(d, f"{i}_{j}.jpg"), quality=CFG["TEX_Q"], subsampling=0)

    # what one texel of this tile covers in the world; the viewer refines on whichever is worse
    texel = float(span[0] * UV_SPAN[0] / img.size[0])
    err = max(err, texel)

    bmin, bmax = Vm.min(0), Vm.max(0)
    return {"level": level, "i": i, "j": j,
            "error": round(err, 5), "texel": round(texel, 5),
            "verts": int(len(Vm)), "tris": int(len(Fm)),
            # the atlas rect the texture actually covers, so the crop and the uvs cannot drift apart
            "uv": [round(float(lo[0]), 6), round(float(lo[1]), 6),
                   round(float(span[0]), 6), round(float(span[1]), 6)],
            "min": [round(float(x), 4) for x in bmin],
            "max": [round(float(x), 4) for x in bmax]}
"""
)

code(
    r"""
t = time.time()
NODES = {}
built = 0


def build(level, i, j):
    global built
    kids = []
    if level < CFG["LEVELS"] - 1:
        for a in (0, 1):
            for b in (0, 1):
                c = build(level + 1, i * 2 + a, j * 2 + b)
                if c is not None:
                    kids.append(c)
    m = core_of(level, i, j, kids)
    if m is None or len(m.triangles) == 0:
        return None
    err = geometric_error(m, level, i, j)
    NODES[(level, i, j)] = write_tile(level, i, j, m, err)
    built += 1
    if built % 100 == 0:
        print(f"  {built} tiles ({time.time() - t:.0f} s)", flush=True)
    return m


for i in range(CFG["BASE"][0]):
    for j in range(CFG["BASE"][1]):
        build(0, i, j)
print(f"{len(NODES)} tiles written")
stamp("pyramid", t)
"""
)

md("## 4. Index")

code(
    r"""
t = time.time()
tiles = {}
for (lv, i, j), n in NODES.items():
    kids = [f"{lv+1}/{i*2+a}_{j*2+b}" for a in (0, 1) for b in (0, 1)
            if (lv + 1, i * 2 + a, j * 2 + b) in NODES]
    n = dict(n); n["children"] = kids
    tiles[f"{lv}/{i}_{j}"] = n

index = {
    "format": "korno-v3-tiles/1",
    "note": "positions are in the same coordinate frame as korno_v2.glb",
    "levels": CFG["LEVELS"],
    "base": list(CFG["BASE"]),
    "roots": [f"0/{i}_{j}" for i in range(CFG["BASE"][0]) for j in range(CFG["BASE"][1])
              if (0, i, j) in NODES],
    "tiles": tiles,
}
with open(os.path.join(CFG["OUT"], "index.json"), "w") as fh:
    json.dump(index, fh)
shutil.copy(os.path.join(CFG["OUT"], "index.json"), os.path.join(CFG["TILES"], "index.json"))

by_level = {}
for (lv, _, _), n in NODES.items():
    d = by_level.setdefault(lv, {"tiles": 0, "tris": 0, "bytes": 0, "err": []})
    d["tiles"] += 1; d["tris"] += n["tris"]; d["err"].append(n["error"])
total = 0
for lv in sorted(by_level):
    d = by_level[lv]
    p = os.path.join(CFG["TILES"], str(lv))
    d["bytes"] = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p))
    total += d["bytes"]
    print(f"  L{lv}: {d['tiles']:5d} tiles, {d['tris']:9d} tris, "
          f"{d['bytes'] / 1e6:8.1f} MB, error p50 {np.median(d['err']):.4f}")
    d["err"] = round(float(np.median(d["err"])), 5)
print(f"TOTAL {total / 1e6:.1f} MB in {sum(v['tiles'] for v in by_level.values())} tiles")
REPORT["levels"] = by_level
REPORT["total_mb"] = round(total / 1e6, 1)
REPORT["total_s"] = round(time.time() - T0)
with open(os.path.join(CFG["OUT"], "tiles_report.json"), "w") as fh:
    json.dump(REPORT, fh, indent=1)
stamp("index", t)
print("TOTAL", REPORT["total_s"], "s")
"""
)


nb = {
    "cells": CELLS,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11.13"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = pathlib.Path(__file__).with_name("korno-v3-tiles.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(CELLS), "cells")
