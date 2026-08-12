#!/usr/bin/env python3
"""Generates kaggle/mesh/korno-mesh.ipynb from the cell sources below."""
import json
import pathlib

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


md(
    """
# Korno rockface — gaussian splat to a textured mesh

Takes `korno_full.ply` (3.7 M gaussians) from the
[korno-gaussian-splat](https://www.kaggle.com/code/pavelzbytovsk/korno-gaussian-splat) kernel and turns it
into an ordinary textured triangle mesh, in three size classes for the web.

Everything here is CPU work — Poisson reconstruction, KD-tree lookups and quadric decimation — so this
runs on a plain CPU kernel; a GPU would sit idle.

**The haze.** A gaussian splat carries a lot of low-opacity gaussians that only make sense from the
training views; orbiting away from them you look straight through the fog. The filter below keeps
gaussians that are opaque, compact, and *seen by many cameras*, which leaves the rock and the vegetation
growing on it and drops the haze.

| output | what |
|---|---|
| `korno_mesh_100mb.glb` | full detail |
| `korno_mesh_50mb.glb` | balanced |
| `korno_mesh_10mb.glb` | fast loading |
| `korno_mesh_full.ply` | the undecimated mesh with vertex colours |
"""
)

code(
    r"""
import io, os, sys, json, time, glob, shutil, subprocess, zipfile

CFG = dict(
    INPUT_DIR   = "/kaggle/input",
    WORK        = "/kaggle/temp/mesh",
    OUT         = "/kaggle/working",

    # --- keep only gaussians that describe real surface ---
    MIN_OPACITY = 0.30,     # the haze is low-opacity
    MIN_SEEN    = 8,        # ... and is seen by few cameras
    MAX_SCALE   = 0.10,     # ... and is often a big soft blob
    VOXEL       = 0.004,    # averages out centre noise before reconstruction

    POISSON_DEPTH   = 11,
    DENSITY_QUANTILE = 0.02,   # trim the extrapolated skirt Poisson invents
    TAUBIN_ITERS    = 12,      # removes the "popcorn" without shrinking the shape

    # target size -> (budget MB, texture width)
    TARGETS = [("100mb", 100, 8192), ("50mb", 50, 4096), ("10mb", 10, 2048)],
    TEX_K   = 8,            # colour = inverse-distance blend of K nearest gaussians
    JPEG_Q  = 90,           # trimesh embeds PNG unless the PIL image is already a JPEG
)
os.makedirs(CFG["WORK"], exist_ok=True)
os.makedirs(CFG["OUT"], exist_ok=True)
REPORT = {"cfg": {k: v for k, v in CFG.items()}, "timings": {}}
T0 = time.time()


def run(cmd, throttle=0.0, tag="run", check=True, **kw):
    t0 = time.time(); last = -1e9
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    p = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, errors="replace", **kw)
    tail = []
    for line in p.stdout:
        line = line.rstrip("\n").split("\r")[-1]
        if not line.strip():
            continue
        tail.append(line); del tail[:-40]
        now = time.time()
        if throttle <= 0 or now - last >= throttle:
            last = now; print(f"[{tag} {now - t0:6.0f}s] {line}", flush=True)
    rc = p.wait()
    print(f"[{tag}] exit={rc} after {time.time() - t0:.0f}s", flush=True)
    if rc != 0:
        print("\n".join(tail), flush=True)
        if check:
            raise RuntimeError(f"{tag} failed with exit code {rc}")
    return rc


def stamp(name, t):
    REPORT["timings"][name] = round(time.time() - t, 1)
    print(f"### {name}: {REPORT['timings'][name]}s (total {time.time() - T0:.0f}s)", flush=True)
"""
)

md("## 1. Dependencies and inputs")

code(
    r"""
t = time.time()
run([sys.executable, "-m", "pip", "install", "-q", "open3d", "trimesh", "plyfile",
     "--no-warn-conflicts"], tag="pip", throttle=15)
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pycolmap==4.1.1"], tag="pip2")
import numpy as np
import open3d as o3d
import trimesh
import pycolmap
from plyfile import PlyData
from scipy.spatial import cKDTree
from PIL import Image
print("open3d", o3d.__version__, "| trimesh", trimesh.__version__, "| numpy", np.__version__)
print("cpus", os.cpu_count())
stamp("setup", t)
"""
)

code(
    r"""
# the gaussian splat kernel's output is attached as a kernel data source
def find(pattern):
    hits = sorted(glob.glob(os.path.join(CFG["INPUT_DIR"], "**", pattern), recursive=True))
    assert hits, f"{pattern} not found under {CFG['INPUT_DIR']}"
    return hits[0]

PLY_IN = find("korno_full.ply")
ZIP_IN = find("korno_colmap.zip")
print(PLY_IN, round(os.path.getsize(PLY_IN) / 1e6, 1), "MB")
print(ZIP_IN, round(os.path.getsize(ZIP_IN) / 1e6, 1), "MB")

SPARSE = os.path.join(CFG["WORK"], "colmap")
if not os.path.isdir(SPARSE):
    with zipfile.ZipFile(ZIP_IN) as zf:
        zf.extractall(SPARSE)
rec = pycolmap.Reconstruction(os.path.join(SPARSE, "0"))
IMS = [i for i in rec.images.values() if i.has_pose]
CAM = list(rec.cameras.values())[0]
print(rec.summary())
"""
)

md("## 2. Load the gaussians and drop the haze")

code(
    r"""
t = time.time()
v = PlyData.read(PLY_IN)["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
opacity = (1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))).astype(np.float32)
SH_C0 = 0.28209479177387814
color = np.clip(0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1).astype(np.float32)
quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
quat /= np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12
del v
n = len(xyz)
print("gaussians", n)
stamp("load_ply", t)
"""
)

code(
    r"""
t = time.time()
fx, fy, cx, cy = CAM.params[:4]
W, H = CAM.width, CAM.height

# how many training cameras actually see each gaussian - real surface is seen by many, haze by few
seen = np.zeros(n, np.int16)
for im in IMS:
    cw = im.cam_from_world()
    c = xyz @ cw.rotation.matrix().T.astype(np.float32) + cw.translation.astype(np.float32)
    z = c[:, 2]
    ok = z > 0.05
    zz = np.where(ok, z, 1.0)
    u = c[:, 0] / zz * fx + cx
    w_ = c[:, 1] / zz * fy + cy
    seen += (ok & (u >= 0) & (u < W) & (w_ >= 0) & (w_ < H)).astype(np.int16)

keep = (opacity > CFG["MIN_OPACITY"]) & (seen >= CFG["MIN_SEEN"]) & (scale.max(1) < CFG["MAX_SCALE"])
print(f"kept {keep.sum()} / {n}  ({100 * keep.mean():.1f}%)")
REPORT["filter"] = {"gaussians": int(n), "kept": int(keep.sum()),
                    "seen_median": float(np.median(seen))}

P, C, Q, S = xyz[keep], color[keep], quat[keep], scale[keep]
del xyz, color, quat, scale, opacity, seen
stamp("filter", t)
"""
)

md("## 3. Poisson surface reconstruction")

code(
    r"""
t = time.time()
# surface normal = the gaussian's thinnest axis, flipped to face the cameras
w_, x_, y_, z_ = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
R = np.empty((len(Q), 3, 3), np.float32)
R[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_); R[:, 0, 1] = 2 * (x_ * y_ - w_ * z_); R[:, 0, 2] = 2 * (x_ * z_ + w_ * y_)
R[:, 1, 0] = 2 * (x_ * y_ + w_ * z_); R[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_); R[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
R[:, 2, 0] = 2 * (x_ * z_ - w_ * y_); R[:, 2, 1] = 2 * (y_ * z_ + w_ * x_); R[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)
N = R[np.arange(len(R)), :, S.argmin(1)]
N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
CAMPOS = np.array([i.projection_center() for i in IMS], np.float32)
nearest = CAMPOS[cKDTree(CAMPOS).query(P, k=1)[1]]
N[np.einsum("ij,ij->i", N, nearest - P) < 0] *= -1
del R, Q, S

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
pcd.normals = o3d.utility.Vector3dVector(N.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float64))
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.voxel_down_sample(CFG["VOXEL"])
pcd.normalize_normals()
print("reconstruction input points", len(pcd.points))
stamp("normals", t)
"""
)

code(
    r"""
t = time.time()
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=CFG["POISSON_DEPTH"], scale=1.1, n_threads=os.cpu_count())
dens = np.asarray(dens)
mesh.remove_vertices_by_mask(dens < np.quantile(dens, CFG["DENSITY_QUANTILE"]))
mesh.remove_unreferenced_vertices()
tri_cluster, cluster_n, _ = mesh.cluster_connected_triangles()
tri_cluster, cluster_n = np.asarray(tri_cluster), np.asarray(cluster_n)
mesh.remove_triangles_by_mask(cluster_n[tri_cluster] < max(5000, 0.05 * cluster_n.max()))
mesh.remove_unreferenced_vertices()
mesh = mesh.filter_smooth_taubin(number_of_iterations=CFG["TAUBIN_ITERS"])
mesh.compute_vertex_normals()
print("mesh tris", len(mesh.triangles), "verts", len(mesh.vertices))
REPORT["mesh"] = {"triangles": len(mesh.triangles), "vertices": len(mesh.vertices)}
o3d.io.write_triangle_mesh(os.path.join(CFG["OUT"], "korno_mesh_full.ply"), mesh)
stamp("poisson", t)
"""
)

md(
    """
## 4. Texture

The face is near-planar (about 12 × 6 units wide, 2.6 deep), so a single planar projection along the mean
camera direction is a better UV map than an atlas — no seams, no charts, and the texture is effectively an
orthophoto of the wall. Each texel takes an inverse-distance blend of the nearest gaussian colours.
"""
)

code(
    r"""
t = time.time()
V = np.asarray(mesh.vertices)
F = np.asarray(mesh.triangles)

view = np.mean([i.viewing_direction() / np.linalg.norm(i.viewing_direction()) for i in IMS], 0)
view /= np.linalg.norm(view)
up = -np.mean([i.cam_from_world().rotation.matrix()[1] for i in IMS], 0)
up /= np.linalg.norm(up)
right = np.cross(view, up); right /= np.linalg.norm(right)
up = np.cross(right, view)
ORIGIN = V.mean(0)


def to_uv(X):
    d = X - ORIGIN
    return np.stack([d @ right, d @ up], 1)


uvV = to_uv(V)
UV_LO = np.percentile(uvV, 0.2, axis=0)
UV_HI = np.percentile(uvV, 99.8, axis=0)
UV_SPAN = UV_HI - UV_LO
print("uv span", UV_SPAN.round(2))
TREE = cKDTree(to_uv(np.asarray(pcd.points)))
PCOL = np.asarray(pcd.colors).astype(np.float32)


def bake(width):
    height = int(round(width * UV_SPAN[1] / UV_SPAN[0] / 4)) * 4
    gx = UV_LO[0] + (np.arange(width) + 0.5) / width * UV_SPAN[0]
    gy = UV_LO[1] + (np.arange(height) + 0.5) / height * UV_SPAN[1]
    tex = np.empty((height, width, 3), np.uint8)
    rows = max(1, 2_000_000 // width)          # chunked: 8192px wide would otherwise need many GB
    for y0 in range(0, height, rows):
        y1 = min(height, y0 + rows)
        GX, GY = np.meshgrid(gx, gy[y0:y1])
        dist, idx = TREE.query(np.stack([GX.ravel(), GY.ravel()], 1), k=CFG["TEX_K"], workers=-1)
        wgt = 1.0 / np.maximum(dist, 1e-4) ** 2
        wgt /= wgt.sum(1, keepdims=True)
        col = (PCOL[idx] * wgt[..., None]).sum(1)
        tex[y0:y1] = (np.clip(col, 0, 1) * 255).astype(np.uint8).reshape(y1 - y0, width, 3)
    # round-trip through JPEG: trimesh embeds the image verbatim, and a PNG here costs ~5x more
    buf = io.BytesIO()
    Image.fromarray(tex[::-1]).save(buf, format="JPEG", quality=CFG["JPEG_Q"])
    img = Image.open(buf)
    img.jpeg_bytes = buf.getbuffer().nbytes
    return img


stamp("uv", t)
"""
)

md("## 5. Three size classes")

code(
    r"""
t = time.time()
BYTES = 1024 * 1024


def export(path, o3d_mesh, texture):
    V = np.asarray(o3d_mesh.vertices)
    F = np.asarray(o3d_mesh.triangles)
    uv = (to_uv(V) - UV_LO) / UV_SPAN
    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    tm.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0))
    tm.export(path, include_normals=True)
    return os.path.getsize(path)


REPORT["outputs"] = {}
for name, budget_mb, tex_w in CFG["TARGETS"]:
    texture = bake(tex_w)
    # ~26 B per triangle (position+normal+uv per vertex plus indices) seeds the search close
    geom_budget = budget_mb * BYTES - texture.jpeg_bytes
    target_tris = min(len(mesh.triangles), max(20000, int(geom_budget / 26)))
    path = os.path.join(CFG["OUT"], f"korno_mesh_{name}.glb")
    print(f"{name}: texture {texture.size} = {texture.jpeg_bytes / BYTES:.1f} MB, "
          f"seed {target_tris} tris", flush=True)
    for attempt in range(4):
        simplified = (mesh if target_tris >= len(mesh.triangles)
                      else mesh.simplify_quadric_decimation(int(target_tris)))
        simplified.compute_vertex_normals()
        size = export(path, simplified, texture)
        mb = size / BYTES
        print(f"  {name} try{attempt}: {len(simplified.triangles)} tris -> {mb:.1f} MB", flush=True)
        if mb <= budget_mb or target_tris <= 20000:
            break
        # scale the triangle budget by how far over we are, leaving room for the texture
        target_tris = max(20000, int(len(simplified.triangles) * (budget_mb / mb) * 0.95))
    REPORT["outputs"][name] = {"file": os.path.basename(path), "mb": round(mb, 1),
                               "triangles": len(simplified.triangles),
                               "vertices": len(simplified.vertices), "texture": texture.size}
    print(f"{name}: {mb:.1f} MB, {len(simplified.triangles)} tris, texture {texture.size}", flush=True)
stamp("export", t)
"""
)

code(
    r"""
REPORT["viewer"] = {
    "center": [round(float(x), 3) for x in (UV_LO + UV_SPAN / 2) @ np.stack([right, up]) + ORIGIN],
    "cameraUp": [round(float(x), 5) for x in up],
    "viewDir": [round(float(x), 5) for x in view],
    "extent": [round(float(x), 3) for x in UV_SPAN],
}
REPORT["total_seconds"] = round(time.time() - T0, 1)
with open(os.path.join(CFG["OUT"], "mesh_report.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2, default=str)

for f in sorted(os.listdir(CFG["OUT"])):
    print(f"{os.path.getsize(os.path.join(CFG['OUT'], f)) / 1e6:10.1f} MB  {f}")
print(json.dumps(REPORT["timings"], indent=1))
print("TOTAL", REPORT["total_seconds"], "s")
"""
)


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(__file__).with_name("korno-mesh.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(CELLS), "cells")
