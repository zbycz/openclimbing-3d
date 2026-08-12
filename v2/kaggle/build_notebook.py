#!/usr/bin/env python3
"""Generates v2/kaggle/korno-v2-mvs.ipynb from the cell sources below."""
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
# Korno rockface v2 — classic dense photogrammetry

Straight from the 103 drone photos to a textured mesh, with **no gaussian splatting anywhere**:
COLMAP SfM → CUDA dense stereo (`patch_match_stereo`) → depth-map fusion → Poisson mesh → texture.

Where v1 inferred a surface from the centres of 1.4 M gaussians, this measures depth **per pixel** in
every image and triangulates it across views. That is what puts the cracks and edges in.

### Getting CUDA COLMAP onto Kaggle

`patch_match_stereo` is CUDA-only — no CPU fallback — and there is no pip wheel, so it comes from
conda-forge via micromamba. That package has three traps, all found the hard way:

* it links `libfaiss.so` and `libOpenImageIO.so` without declaring either dependency;
* it was built against faiss **1.10**, while the current 1.14 has an incompatible ABI;
* faiss must be the **openblas** variant, or colmap wants Intel MKL runtime libs that aren't there.

`colmap=4.1.1=cuda*` + `libfaiss=1.10.0=*openblas*` + `openimageio` is the combination that works.
COLMAP 4.x also renamed the option namespaces (`--FeatureExtraction.use_gpu`, not `--SiftExtraction.use_gpu`).

| output | what |
|---|---|
| `korno_v2.glb` | the textured mesh for the web |
| `korno_v2_light.glb` | small version for phones |
| `korno_v2_cameras.json` | camera poses, for the viewer's camera overlay |
| `korno_v2_mesh.ply` | full-resolution mesh, undecimated |
| `korno_v2_dense.ply` | the fused dense point cloud |
"""
)

code(
    r"""
import io, os, sys, json, time, glob, shutil, subprocess

CFG = dict(
    INPUT_DIR = "/kaggle/input",
    WORK      = "/kaggle/temp/v2",
    OUT       = "/kaggle/working",
    MAMBA     = "/kaggle/temp/mm",

    SFM_WIDTH   = 3200,     # images are downscaled to this for feature extraction
    MAX_FEATURES = 16384,   # rock is textured, so more features pay off
    STEREO_MAX  = 2600,     # dense stereo resolution - the main quality/time dial
    SRC_VIEWS   = 10,       # source views per reference image; cost is linear in this

    FUSION_MIN_PIXELS = 4,  # a point must be seen consistently by this many views
    POISSON_DEPTH = 13,     # 2^13 grid over the scene ~ 5 mm cells at this scale
    POISSON_TRIM  = 10,

    # (name, budget MB, texture width)
    TARGETS = [("", 110, 8192), ("_light", 12, 2048)],
    TEX_K   = 6,
    JPEG_Q  = 92,
)
os.makedirs(CFG["WORK"], exist_ok=True)
os.makedirs(CFG["OUT"], exist_ok=True)
REPORT = {"cfg": dict(CFG), "timings": {}}
T0 = time.time()


def run(cmd, throttle=0.0, tag="run", check=True, shell=False, **kw):
    t0 = time.time(); last = -1e9
    print(f"\n$ {cmd if shell else ' '.join(map(str, cmd))}", flush=True)
    p = subprocess.Popen(cmd if shell else list(map(str, cmd)), shell=shell,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1, errors="replace", **kw)
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
        print("--- tail ---"); print("\n".join(tail), flush=True)
        if check:
            raise RuntimeError(f"{tag} failed rc={rc}")
    return rc


def stamp(name, t):
    REPORT["timings"][name] = round(time.time() - t, 1)
    print(f"### {name}: {REPORT['timings'][name]}s (total {time.time() - T0:.0f}s)", flush=True)


run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv"], tag="gpu", check=False)
run(["free", "-g"], tag="mem", check=False)
print("cpus", os.cpu_count())
"""
)

md("## 1. CUDA COLMAP")

code(
    r"""
t = time.time()
MAMBA = CFG["MAMBA"]
if not os.path.exists(f"{MAMBA}/bin/colmap"):
    run("curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /kaggle/temp bin/micromamba",
        tag="micromamba", shell=True)
    run(f"/kaggle/temp/bin/micromamba create -y -p {MAMBA} -c conda-forge "
        f"'colmap=4.1.1=cuda*' 'libfaiss=1.10.0=*openblas*' openimageio",
        tag="mamba", throttle=15, shell=True)

MM = f"/kaggle/temp/bin/micromamba run -p {MAMBA}"


# run a colmap subcommand inside the micromamba env
def cm(*args, **kw):
    return run(MM + " colmap " + " ".join(str(a) for a in args), shell=True, **kw)


run(f"{MM} bash -c 'ldd {MAMBA}/bin/colmap | grep \"not found\" || echo ALL_LIBS_RESOLVED'",
    tag="ldd", shell=True, check=False)
cm("-h", tag="colmap", check=False)

run([sys.executable, "-m", "pip", "install", "-q", "open3d", "trimesh", "plyfile", "--no-warn-conflicts"],
    tag="pip", throttle=15)
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pycolmap==4.1.1"], tag="pip2")
import numpy as np
import open3d as o3d
import trimesh
import pycolmap
from scipy.spatial import cKDTree
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
stamp("setup", t)
"""
)

md("## 2. Images")

code(
    r"""
t = time.time()
from concurrent.futures import ThreadPoolExecutor

SRC = sorted(p for p in glob.glob(os.path.join(CFG["INPUT_DIR"], "**", "*"), recursive=True)
             if p.lower().endswith((".jpg", ".jpeg")) and os.path.isfile(p))
print("source photos:", len(SRC), SRC[:1])
assert SRC, "no photos found"

IMG = os.path.join(CFG["WORK"], "images")
os.makedirs(IMG, exist_ok=True)


def prep(path):
    dst = os.path.join(IMG, os.path.basename(path))
    if os.path.exists(dst):
        return dst
    im = Image.open(path)
    exif = im.info.get("exif")
    w = CFG["SFM_WIDTH"]
    im = im.convert("RGB").resize((w, round(im.size[1] * w / im.size[0])), Image.LANCZOS)
    im.save(dst, quality=96, **({"exif": exif} if exif else {}))
    return dst


with ThreadPoolExecutor(os.cpu_count()) as ex:
    list(ex.map(prep, SRC))
W0, H0 = Image.open(SRC[0]).size
W1, H1 = Image.open(os.path.join(IMG, os.listdir(IMG)[0])).size
print(f"{len(os.listdir(IMG))} images: {W0}x{H0} -> {W1}x{H1}")
REPORT["images"] = {"count": len(SRC), "source": [W0, H0], "sfm": [W1, H1]}
stamp("prepare", t)
"""
)

md("## 3. Structure from motion (GPU)")

code(
    r"""
DB = os.path.join(CFG["WORK"], "db.db")
SPARSE = os.path.join(CFG["WORK"], "sparse")
os.makedirs(SPARSE, exist_ok=True)

t = time.time()
if not os.path.exists(DB):
    cm("feature_extractor", "--database_path", DB, "--image_path", IMG,
       "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "SIMPLE_RADIAL",
       "--FeatureExtraction.use_gpu", 1,
       "--FeatureExtraction.max_image_size", CFG["SFM_WIDTH"],
       "--SiftExtraction.max_num_features", CFG["MAX_FEATURES"],
       tag="extract", throttle=10)
    stamp("sfm_extract", t)

    t = time.time()
    cm("exhaustive_matcher", "--database_path", DB, "--FeatureMatching.use_gpu", 1,
       tag="match", throttle=20)
    stamp("sfm_match", t)

t = time.time()
if not os.path.isdir(os.path.join(SPARSE, "0")):
    cm("mapper", "--database_path", DB, "--image_path", IMG, "--output_path", SPARSE,
       tag="mapper", throttle=30)
stamp("sfm_map", t)

shutil.copytree(os.path.join(SPARSE, "0"), os.path.join(CFG["OUT"], "sparse"), dirs_exist_ok=True)
rec = pycolmap.Reconstruction(os.path.join(SPARSE, "0"))
print(rec.summary())
REPORT["sfm"] = {"registered": rec.num_reg_images(), "points3D": rec.num_points3D(),
                 "mean_reproj_error": round(rec.compute_mean_reprojection_error(), 3)}
assert rec.num_reg_images() >= 0.8 * len(SRC), f"only {rec.num_reg_images()}/{len(SRC)} registered"
"""
)

md("## 4. Dense stereo — one depth map per photo (the slow part)")

code(
    r"""
DENSE = os.path.join(CFG["WORK"], "dense")
t = time.time()
if not os.path.isdir(os.path.join(DENSE, "images")):
    cm("image_undistorter", "--image_path", IMG, "--input_path", os.path.join(SPARSE, "0"),
       "--output_path", DENSE, "--output_type", "COLMAP",
       "--max_image_size", CFG["STEREO_MAX"],
       "--num_patch_match_src_images", CFG["SRC_VIEWS"],
       tag="undistort", throttle=10)
stamp("undistort", t)
print("undistorted:", len(os.listdir(os.path.join(DENSE, "images"))),
      Image.open(glob.glob(os.path.join(DENSE, "images", "*"))[0]).size)
"""
)

code(
    r"""
t = time.time()
cm("patch_match_stereo", "--workspace_path", DENSE, "--workspace_format", "COLMAP",
   "--PatchMatchStereo.max_image_size", CFG["STEREO_MAX"],
   "--PatchMatchStereo.geom_consistency", "true",
   "--PatchMatchStereo.cache_size", 24,
   tag="stereo", throttle=60)
stamp("stereo", t)
"""
)

code(
    r"""
t = time.time()
FUSED = os.path.join(CFG["OUT"], "korno_v2_dense.ply")
cm("stereo_fusion", "--workspace_path", DENSE, "--workspace_format", "COLMAP",
   "--input_type", "geometric", "--output_path", FUSED,
   "--StereoFusion.min_num_pixels", CFG["FUSION_MIN_PIXELS"],
   tag="fusion", throttle=30)
print("dense cloud:", round(os.path.getsize(FUSED) / 1e6, 1), "MB")
stamp("fusion", t)
"""
)

md("## 5. Mesh")

code(
    r"""
t = time.time()
MESH_PLY = os.path.join(CFG["OUT"], "korno_v2_mesh.ply")
rc = cm("poisson_mesher", "--input_path", FUSED, "--output_path", MESH_PLY,
        "--PoissonMeshing.depth", CFG["POISSON_DEPTH"],
        "--PoissonMeshing.trim", CFG["POISSON_TRIM"],
        "--PoissonMeshing.num_threads", os.cpu_count(),
        tag="poisson", throttle=30, check=False)


# COLMAP's bundled PoissonRecon can segfault in cleanup after writing a valid mesh
def ply_is_complete(path):
    if not os.path.exists(path):
        return False
    header, nv, nf = b"", 0, 0
    with open(path, "rb") as fh:
        while b"end_header" not in header:
            chunk = fh.readline()
            if not chunk:
                return False
            header += chunk
        for line in header.split(b"\n"):
            if line.startswith(b"element vertex"):
                nv = int(line.split()[-1])
            elif line.startswith(b"element face"):
                nf = int(line.split()[-1])
        body = len(header)
    # x,y,z,value floats + rgb bytes per vertex; int count + 3 int indices per face
    expect = body + nv * (4 * 4 + 3) + nf * (4 + 3 * 4)
    actual = os.path.getsize(path)
    print(f"ply check: {nv} verts, {nf} faces, expect {expect} bytes, have {actual}")
    return actual >= expect


if rc != 0 and ply_is_complete(MESH_PLY):
    print(f"poisson_mesher exited {rc} but wrote a complete mesh - continuing", flush=True)
elif rc != 0:
    print("poisson_mesher produced no usable mesh, falling back to open3d", flush=True)
    pcd = o3d.io.read_point_cloud(FUSED)
    om, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=CFG["POISSON_DEPTH"] - 1, scale=1.1, n_threads=os.cpu_count())
    dens = np.asarray(dens)
    om.remove_vertices_by_mask(dens < np.quantile(dens, CFG["POISSON_TRIM"] / 100.0))
    o3d.io.write_triangle_mesh(MESH_PLY, om)
    del pcd, om
print("raw mesh:", round(os.path.getsize(MESH_PLY) / 1e6, 1), "MB")
stamp("poisson", t)
"""
)

code(
    r"""
t = time.time()
mesh = o3d.io.read_triangle_mesh(MESH_PLY)
print("loaded", len(mesh.triangles), "triangles")
# Poisson closes the surface over empty space; keep only the big connected pieces
tri_cluster, cluster_n, _ = mesh.cluster_connected_triangles()
tri_cluster, cluster_n = np.asarray(tri_cluster), np.asarray(cluster_n)
mesh.remove_triangles_by_mask(cluster_n[tri_cluster] < max(20000, 0.02 * cluster_n.max()))
mesh.remove_unreferenced_vertices()
mesh.remove_degenerate_triangles()
mesh.compute_vertex_normals()
print("cleaned:", len(mesh.triangles), "triangles,", len(mesh.vertices), "vertices")
REPORT["mesh"] = {"triangles": len(mesh.triangles), "vertices": len(mesh.vertices)}
stamp("clean", t)
"""
)

md(
    """
## 6. Texture

The face is near-planar, so one planar projection along the mean camera direction beats an atlas
unwrap — no seams, no charts, and the texture comes out as an orthophoto of the wall. Texel colours
are an inverse-distance blend of the nearest dense-cloud points, which carry the photo colours
averaged over every view that saw them.
"""
)

code(
    r"""
t = time.time()
from plyfile import PlyData

dense = PlyData.read(FUSED)["vertex"]
DP = np.stack([dense["x"], dense["y"], dense["z"]], 1).astype(np.float32)
DC = (np.stack([dense["red"], dense["green"], dense["blue"]], 1).astype(np.float32) / 255.0)
del dense
print("dense points for texturing:", len(DP))

IMS = [i for i in rec.images.values() if i.has_pose]
view = np.mean([i.viewing_direction() / np.linalg.norm(i.viewing_direction()) for i in IMS], 0)
view /= np.linalg.norm(view)
up = -np.mean([i.cam_from_world().rotation.matrix()[1] for i in IMS], 0)
up /= np.linalg.norm(up)
right = np.cross(view, up); right /= np.linalg.norm(right)
up = np.cross(right, view)
V0 = np.asarray(mesh.vertices)
ORIGIN = V0.mean(0)


def to_uv(X):
    d = X - ORIGIN
    return np.stack([d @ right, d @ up], 1)


uvV = to_uv(V0)
UV_LO = np.percentile(uvV, 0.2, axis=0)
UV_HI = np.percentile(uvV, 99.8, axis=0)
UV_SPAN = UV_HI - UV_LO
print("uv span", UV_SPAN.round(2))
TREE = cKDTree(to_uv(DP))


def bake(width):
    height = int(round(width * UV_SPAN[1] / UV_SPAN[0] / 4)) * 4
    gx = UV_LO[0] + (np.arange(width) + 0.5) / width * UV_SPAN[0]
    gy = UV_LO[1] + (np.arange(height) + 0.5) / height * UV_SPAN[1]
    tex = np.empty((height, width, 3), np.uint8)
    rows = max(1, 2_000_000 // width)
    for y0 in range(0, height, rows):
        y1 = min(height, y0 + rows)
        GX, GY = np.meshgrid(gx, gy[y0:y1])
        dist, idx = TREE.query(np.stack([GX.ravel(), GY.ravel()], 1), k=CFG["TEX_K"], workers=-1)
        wgt = 1.0 / np.maximum(dist, 1e-5) ** 2
        wgt /= wgt.sum(1, keepdims=True)
        col = (DC[idx] * wgt[..., None]).sum(1)
        tex[y0:y1] = (np.clip(col, 0, 1) * 255).astype(np.uint8).reshape(y1 - y0, width, 3)
    buf = io.BytesIO()
    Image.fromarray(tex[::-1]).save(buf, format="JPEG", quality=CFG["JPEG_Q"])
    img = Image.open(buf)
    img.jpeg_bytes = buf.getbuffer().nbytes
    return img


stamp("uv", t)
"""
)

md("## 7. Export")

code(
    r"""
t = time.time()
BYTES = 1024 * 1024


def export(path, m, texture):
    V = np.asarray(m.vertices)
    F = np.asarray(m.triangles)
    uv = (to_uv(V) - UV_LO) / UV_SPAN
    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    tm.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0))
    tm.export(path, include_normals=True)
    return os.path.getsize(path)


REPORT["outputs"] = {}
for suffix, budget_mb, tex_w in CFG["TARGETS"]:
    texture = bake(tex_w)
    target = min(len(mesh.triangles),
                 max(20000, int((budget_mb * BYTES - texture.jpeg_bytes) / 26)))
    path = os.path.join(CFG["OUT"], f"korno_v2{suffix}.glb")
    print(f"korno_v2{suffix}: texture {texture.size} = {texture.jpeg_bytes / BYTES:.1f} MB, "
          f"seed {target} tris", flush=True)
    for attempt in range(4):
        small = mesh if target >= len(mesh.triangles) else mesh.simplify_quadric_decimation(int(target))
        small.compute_vertex_normals()
        mb = export(path, small, texture) / BYTES
        print(f"  try{attempt}: {len(small.triangles)} tris -> {mb:.1f} MB", flush=True)
        if mb <= budget_mb or target <= 20000:
            break
        target = max(20000, int(len(small.triangles) * (budget_mb / mb) * 0.95))
    REPORT["outputs"][f"korno_v2{suffix}.glb"] = {
        "mb": round(mb, 1), "triangles": len(small.triangles), "texture": list(texture.size)}
stamp("export", t)
"""
)

code(
    r"""
# camera poses for the viewer's overlay
cams = []
for im in sorted(IMS, key=lambda i: i.name):
    cw = im.cam_from_world()
    R = cw.rotation.matrix()
    cam = rec.cameras[im.camera_id]
    fx = float(cam.params[0])
    cams.append({
        "name": im.name,
        "position": [round(float(x), 4) for x in im.projection_center()],
        # world-from-camera basis: right, down, forward
        "right":   [round(float(x), 5) for x in R[0]],
        "down":    [round(float(x), 5) for x in R[1]],
        "forward": [round(float(x), 5) for x in R[2]],
        "hfov": round(float(2 * np.arctan(cam.width / (2 * fx))), 5),
        "aspect": round(float(cam.width / cam.height), 4),
    })
centre = (UV_LO + UV_SPAN / 2) @ np.stack([right, up]) + ORIGIN
with open(os.path.join(CFG["OUT"], "korno_v2_cameras.json"), "w") as fh:
    json.dump({"cameras": cams,
               "up": [round(float(x), 5) for x in up],
               "viewDir": [round(float(x), 5) for x in view],
               "centre": [round(float(x), 4) for x in centre],
               "extent": [round(float(x), 3) for x in UV_SPAN]}, fh)
print("cameras exported:", len(cams))

REPORT["total_seconds"] = round(time.time() - T0, 1)
with open(os.path.join(CFG["OUT"], "v2_report.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2, default=str)
for f in sorted(os.listdir(CFG["OUT"])):
    print(f"{os.path.getsize(os.path.join(CFG['OUT'], f)) / 1e6:10.1f} MB  {f}")
print(json.dumps(REPORT["timings"], indent=1))
print("TOTAL", REPORT["total_seconds"], "s")
"""
)


nb = {
    "cells": CELLS,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11.13"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = pathlib.Path(__file__).with_name("korno-v2-mvs.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(CELLS), "cells")
