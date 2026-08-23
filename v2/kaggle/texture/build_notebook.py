#!/usr/bin/env python3
# Generates v2/kaggle/texture/korno-v2-retexture.ipynb - re-bakes the v2 texture from the photos.
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
# Korno v2 — texture from the photos, not from the point cloud

`korno_v2.glb` is sharp low on the wall and smeared above it. The geometry is not the problem: the
texture is baked from the **colours of the dense points**, so a texel is only as sharp as the local
point density. Below h ≈ 1.55 the dense points are 0.0029 units apart against a texel of 0.00184, so
every texel gets its own measurement. Above it they are 0.011 apart — one measurement smeared over
six texels — because the drone never flew above the top of the wall and dense stereo accepted far
fewer depths up there.

This kernel keeps the same geometry and the same planar UV parametrisation, and replaces only the
bake: every texel is projected into the **original 8000 × 4500 photos** and sampled there. Texture
detail then depends on what the photos saw, not on how many depth points survived fusion.

```
texel -> ray along the orthophoto axis -> surface point + normal      (position map)
      -> project into all 102 posed photos
      -> reject: behind the camera, outside the frame, too grazing, occluded (per-camera depth map)
      -> score by sampling density (cos / d^2), keep the best three, blend softly
      -> sample the photo bilinearly at full resolution
```

The old point-cloud bake is still computed — as the photometric reference for per-photo exposure
gains, as the fill for texels no photo saw, and as the before/after comparison.

Outputs `korno_v2_desmudged.glb` and `korno_v2_desmudged_light.glb`; the original models are not
touched.
"""
)

code(
    r"""
import os, sys, io, json, time, glob, copy, subprocess

CFG = dict(
    INPUT_DIR = "/kaggle/input",
    OUT       = "/kaggle/working",

    TEX_W       = 8192,     # same atlas as korno_v2.glb, so the comparison is like for like
    TILE        = 512,      # atlas tile for the assignment pass
    DEPTH_W     = 1024,     # per-photo depth map used for the occlusion test
    TOP_K       = 3,        # photos blended per texel
    BLEND_SHARP = 8.0,      # w = (s/s_best)^SHARP - a narrow transition, not a general blur
    MIN_COS     = 0.15,     # ignore views more grazing than this
    BORDER      = 0.03,     # fade out this fraction of the frame at the photo edges
    OCCL_REL    = 1.004,    # depth-map tolerance, relative
    OCCL_ABS    = 3.0,      # ... and absolute, in texels
    GAIN_CLIP   = [0.85, 1.18],
    TEX_K       = 6,        # reference bake: nearest dense points per texel, as in the v2 kernel
    JPEG_Q      = 95,
    # (suffix, budget MB, texture width)
    TARGETS     = [("_desmudged", 92, 8192), ("_desmudged_light", 12, 2048)],
)
os.makedirs(CFG["OUT"], exist_ok=True)
T0 = time.time()
REPORT = {"cfg": CFG, "timings": {}}

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open3d", "trimesh", "plyfile", "--no-warn-conflicts"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pycolmap==4.1.1"])
import numpy as np
import open3d as o3d
import pycolmap
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
Image.MAX_IMAGE_PIXELS = None
print("open3d", o3d.__version__, "| trimesh", trimesh.__version__, "| cpus", os.cpu_count())


def stamp(name, t):
    REPORT["timings"][name] = round(time.time() - t, 1)
    print(f"[{name}] {REPORT['timings'][name]} s   (total {round(time.time() - T0)} s)", flush=True)


def find(pattern):
    hits = sorted(glob.glob(os.path.join(CFG["INPUT_DIR"], "**", pattern), recursive=True))
    assert hits, f"{pattern} not found"
    return hits


MESH_PLY = find("korno_v2_mesh.ply")[0]
DENSE_PLY = find("korno_v2_dense.ply")[0]
SPARSE = os.path.dirname(find("cameras.bin")[0])
PHOTOS = {os.path.basename(p): p for p in find("*.JPG") if os.path.isfile(p)}
print("mesh  ", MESH_PLY, round(os.path.getsize(MESH_PLY) / 1e6, 1), "MB")
print("dense ", DENSE_PLY, round(os.path.getsize(DENSE_PLY) / 1e6, 1), "MB")
print("sparse", SPARSE, "| photos", len(PHOTOS))
"""
)

md("## 1. The same mesh and the same UV frame as `korno_v2.glb`")

code(
    r"""
t = time.time()
mesh = o3d.io.read_triangle_mesh(MESH_PLY)
print("loaded", len(mesh.triangles), "triangles")
# identical cleaning to the v2 kernel, so the geometry of the two models matches
tri_cluster, cluster_n, _ = mesh.cluster_connected_triangles()
tri_cluster, cluster_n = np.asarray(tri_cluster), np.asarray(cluster_n)
mesh.remove_triangles_by_mask(cluster_n[tri_cluster] < max(20000, 0.02 * cluster_n.max()))
mesh.remove_unreferenced_vertices()
mesh.remove_degenerate_triangles()
mesh.compute_vertex_normals()
print("cleaned:", len(mesh.triangles), "triangles,", len(mesh.vertices), "vertices")
REPORT["mesh"] = {"triangles": len(mesh.triangles), "vertices": len(mesh.vertices)}
stamp("mesh", t)
"""
)

code(
    r"""
t = time.time()
rec = pycolmap.Reconstruction(SPARSE)
IMS = [i for i in rec.images.values() if i.has_pose]
print(rec.summary())
print("posed images:", len(IMS))

# the orthophoto frame, reproduced exactly from the v2 kernel
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
TEX_W = CFG["TEX_W"]
TEX_H = int(round(TEX_W * UV_SPAN[1] / UV_SPAN[0] / 4)) * 4
TEXEL = float(UV_SPAN[0] / TEX_W)
EXTENT = float(np.linalg.norm(V0.max(0) - V0.min(0)))
print("uv span", UV_SPAN.round(3), "-> atlas", TEX_W, "x", TEX_H, "| texel", round(TEXEL, 5))
REPORT["atlas"] = {"width": TEX_W, "height": TEX_H, "texel": round(TEXEL, 6),
                   "uv_span": [round(float(x), 3) for x in UV_SPAN]}
stamp("uv_frame", t)
"""
)

md(
    """
## 2. Position map

The atlas is an orthophoto: dropping the `view` component is what `to_uv` does, so a texel is a
*line* through the scene along `view`. Casting that line and taking the first hit gives the surface
point the texel stands for, and the primitive normal there. Cast against the full-resolution mesh —
the parametrisation is a function of world position, so it does not depend on how far the exported
copy is later decimated.
"""
)

code(
    r"""
t = time.time()
tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(tmesh)
print("BVH built in", round(time.time() - t, 1), "s", flush=True)

gx = (UV_LO[0] + (np.arange(TEX_W) + 0.5) / TEX_W * UV_SPAN[0]).astype(np.float32)
gy = (UV_LO[1] + (np.arange(TEX_H) + 0.5) / TEX_H * UV_SPAN[1]).astype(np.float32)
BACK = ORIGIN - view * (EXTENT * 1.5)

POS = np.zeros((TEX_H, TEX_W, 3), np.float32)
NRM = np.zeros((TEX_H, TEX_W, 3), np.float32)
HIT = np.zeros((TEX_H, TEX_W), bool)
rows = max(1, 4_000_000 // TEX_W)
tc = time.time()
for y0 in range(0, TEX_H, rows):
    y1 = min(TEX_H, y0 + rows)
    GX, GY = np.meshgrid(gx, gy[y0:y1])
    org = (BACK[None, :] + GX.ravel()[:, None] * right[None, :].astype(np.float32)
                         + GY.ravel()[:, None] * up[None, :].astype(np.float32)).astype(np.float32)
    dirs = np.tile(view.astype(np.float32), (len(org), 1))
    res = scene.cast_rays(o3d.core.Tensor(np.hstack([org, dirs]), dtype=o3d.core.Dtype.Float32))
    th = res["t_hit"].numpy()
    ok = np.isfinite(th)
    P = org + dirs * np.where(ok, th, 0.0)[:, None]
    n = res["primitive_normals"].numpy()
    POS[y0:y1] = P.reshape(y1 - y0, TEX_W, 3)
    NRM[y0:y1] = n.reshape(y1 - y0, TEX_W, 3)
    HIT[y0:y1] = ok.reshape(y1 - y0, TEX_W)
    if (y0 // rows) % 4 == 0:
        done = y1 / TEX_H
        print(f"  position map {done * 100:5.1f}%  ({time.time() - tc:.0f} s)", flush=True)

# orient the normals towards the cameras
flip = np.einsum("ijk,k->ij", NRM, view.astype(np.float32)) > 0
NRM[flip] *= -1
print(f"{HIT.sum()} / {HIT.size} texels land on the mesh ({100 * HIT.mean():.1f}%)")
REPORT["position_map"] = {"texels": int(HIT.size), "on_surface": int(HIT.sum())}
stamp("position_map", t)
"""
)

md(
    """
## 3. Depth map per photo

Occlusion has to be tested or a texel on a hidden face will happily sample whatever the photo shows
in front of it. One depth map per photo at 1024 px answers that with a lookup instead of a ray, and
1024 px is ample: a texel is 0.00184 units and a depth-map pixel covers a few centimetres.
"""
)

code(
    r"""
t = time.time()
CAMS, DEPTH, CENTRE, ROT = {}, {}, {}, {}
DW = CFG["DEPTH_W"]
for k, im in enumerate(IMS):
    cam = rec.cameras[im.camera_id]
    dh = int(round(DW * cam.height / cam.width))
    u = (np.arange(DW) + 0.5) * cam.width / DW
    v = (np.arange(dh) + 0.5) * cam.height / dh
    U, V = np.meshgrid(u, v)
    rays_cam = np.asarray(cam.cam_ray_from_img(np.stack([U.ravel(), V.ravel()], 1)))
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
    R = im.cam_from_world().rotation.matrix()
    C = im.projection_center()
    rays_world = (rays_cam @ R).astype(np.float32)
    org = np.tile(C.astype(np.float32), (len(rays_world), 1))
    th = scene.cast_rays(o3d.core.Tensor(np.hstack([org, rays_world]),
                                         dtype=o3d.core.Dtype.Float32))["t_hit"].numpy()
    DEPTH[im.name] = th.reshape(dh, DW).astype(np.float32)
    CENTRE[im.name] = C.astype(np.float64)
    ROT[im.name] = R.astype(np.float64)
    CAMS[im.name] = cam
    if (k + 1) % 20 == 0:
        print(f"  depth maps {k + 1}/{len(IMS)}", flush=True)
print("depth maps:", len(DEPTH), "at", DW, "px")
stamp("depth_maps", t)
"""
)

md(
    """
## 4. Reference bake — the old method

Kept for three jobs: the photometric anchor the per-photo gains are fitted to, the fill for texels no
photo saw, and the before/after comparison.
"""
)

code(
    r"""
t = time.time()
from plyfile import PlyData

dense = PlyData.read(DENSE_PLY)["vertex"]
DP = np.stack([dense["x"], dense["y"], dense["z"]], 1).astype(np.float32)
DC = (np.stack([dense["red"], dense["green"], dense["blue"]], 1).astype(np.float32) / 255.0)
del dense
print("dense points:", len(DP))

TREE = cKDTree(to_uv(DP))
REF = np.empty((TEX_H, TEX_W, 3), np.uint8)
rows = max(1, 2_000_000 // TEX_W)
for y0 in range(0, TEX_H, rows):
    y1 = min(TEX_H, y0 + rows)
    GX, GY = np.meshgrid(gx, gy[y0:y1])
    dist, idx = TREE.query(np.stack([GX.ravel(), GY.ravel()], 1), k=CFG["TEX_K"], workers=-1)
    wgt = 1.0 / np.maximum(dist, 1e-5) ** 2
    wgt /= wgt.sum(1, keepdims=True)
    col = (DC[idx] * wgt[..., None]).sum(1)
    REF[y0:y1] = (np.clip(col, 0, 1) * 255).astype(np.uint8).reshape(y1 - y0, TEX_W, 3)
del TREE, DP, DC
stamp("reference_bake", t)
"""
)

md(
    """
## 5. Which photo sees each texel

Scored by sampling density — `cos(angle to the surface) / distance²` is proportional to how many
photo pixels land on the texel — with a fade at the frame edges so a photo does not end abruptly
mid-wall. The best three are kept and blended with `(s / s_best)^8`, which is the best view almost
everywhere and a narrow cross-fade where two photos are equally good.
"""
)

code(
    r"""
t = time.time()
NAMES = [im.name for im in IMS]
NC = len(NAMES)
PHOTO_WH = {}
for n in NAMES:
    with Image.open(PHOTOS[n]) as im0:
        PHOTO_WH[n] = im0.size
FULL = {}
for n in NAMES:
    c = CAMS[n]
    W, H = PHOTO_WH[n]
    c2 = pycolmap.Camera(model=c.model.name, width=c.width, height=c.height, params=list(c.params))
    c2.rescale(W, H)          # f, cx, cy scale; the radial term is in normalised coords already
    FULL[n] = c2
print("photo size", PHOTO_WH[NAMES[0]], "-> intrinsics", np.round(FULL[NAMES[0]].params, 1))

BEST = np.full((TEX_H, TEX_W, CFG["TOP_K"]), -1, np.int16)
WGT = np.zeros((TEX_H, TEX_W, CFG["TOP_K"]), np.float32)
TILE = CFG["TILE"]
occl_abs = CFG["OCCL_ABS"] * TEXEL
tiles = 0
tc = time.time()
for y0 in range(0, TEX_H, TILE):
    for x0 in range(0, TEX_W, TILE):
        y1, x1 = min(TEX_H, y0 + TILE), min(TEX_W, x0 + TILE)
        hit = HIT[y0:y1, x0:x1].ravel()
        if not hit.any():
            continue
        P = POS[y0:y1, x0:x1].reshape(-1, 3)[hit].astype(np.float64)
        N = NRM[y0:y1, x0:x1].reshape(-1, 3)[hit].astype(np.float64)
        lo, hi = P.min(0), P.max(0)
        corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
        s1 = np.zeros(len(P), np.float32); s2 = np.zeros_like(s1); s3 = np.zeros_like(s1)
        i1 = np.full(len(P), -1, np.int16); i2 = i1.copy(); i3 = i1.copy()
        for ci, name in enumerate(NAMES):
            R, C, cam, camf = ROT[name], CENTRE[name], CAMS[name], FULL[name]
            # cheap reject: does the tile's bounding box fall in this frame at all?
            Kc = (corners - C) @ R.T
            if (Kc[:, 2] <= 0).all():
                continue
            q = np.asarray(cam.img_from_cam(np.where(Kc[:, 2:] > 1e-6, Kc, [0.0, 0.0, 1.0])))
            mx = cam.width * 0.25, cam.height * 0.25
            if (q[:, 0] < -mx[0]).all() or (q[:, 0] > cam.width + mx[0]).all() \
               or (q[:, 1] < -mx[1]).all() or (q[:, 1] > cam.height + mx[1]).all():
                continue

            Xc = (P - C) @ R.T
            z = Xc[:, 2]
            front = z > 1e-3
            if not front.any():
                continue
            d = np.linalg.norm(Xc, axis=1)
            to_cam = (C - P) / np.maximum(d, 1e-9)[:, None]
            cos = np.einsum("ij,ij->i", N, to_cam)
            good = front & (cos > CFG["MIN_COS"])
            if not good.any():
                continue
            uv = np.full((len(P), 2), -1e9)
            uv[good] = np.asarray(cam.img_from_cam(Xc[good]))
            fx = uv[:, 0] / cam.width
            fy = uv[:, 1] / cam.height
            b = CFG["BORDER"]
            fade = (np.clip(fx / b, 0, 1) * np.clip((1 - fx) / b, 0, 1)
                    * np.clip(fy / b, 0, 1) * np.clip((1 - fy) / b, 0, 1))
            good &= fade > 0
            if not good.any():
                continue
            dm = DEPTH[name]
            di = np.clip((fx * dm.shape[1]).astype(np.int32), 0, dm.shape[1] - 1)
            dj = np.clip((fy * dm.shape[0]).astype(np.int32), 0, dm.shape[0] - 1)
            zbuf = dm[dj, di]
            good &= ~(np.isfinite(zbuf) & (d > zbuf * CFG["OCCL_REL"] + occl_abs))
            s = np.where(good, cos / np.maximum(d, 1e-6) ** 2 * fade, 0.0).astype(np.float32)
            if not (s > 0).any():
                continue

            m1 = s > s1
            s3 = np.where(m1, s2, s3); i3 = np.where(m1, i2, i3)
            s2 = np.where(m1, s1, s2); i2 = np.where(m1, i1, i2)
            s1 = np.where(m1, s, s1);  i1 = np.where(m1, np.int16(ci), i1)
            m2 = ~m1 & (s > s2)
            s3 = np.where(m2, s2, s3); i3 = np.where(m2, i2, i3)
            s2 = np.where(m2, s, s2);  i2 = np.where(m2, np.int16(ci), i2)
            m3 = ~m1 & ~m2 & (s > s3)
            s3 = np.where(m3, s, s3);  i3 = np.where(m3, np.int16(ci), i3)

        S = np.stack([s1, s2, s3], 1)
        I = np.stack([i1, i2, i3], 1)
        top = np.maximum(S[:, :1], 1e-30)
        w = np.where(S > 0, (S / top) ** CFG["BLEND_SHARP"], 0.0)
        tot = w.sum(1, keepdims=True)
        w = np.where(tot > 0, w / np.maximum(tot, 1e-30), 0.0)
        bt = np.full((y1 - y0) * (x1 - x0) * CFG["TOP_K"], -1, np.int16).reshape(-1, CFG["TOP_K"])
        wt = np.zeros_like(bt, np.float32)
        bt[hit] = I
        wt[hit] = w
        BEST[y0:y1, x0:x1] = bt.reshape(y1 - y0, x1 - x0, CFG["TOP_K"])
        WGT[y0:y1, x0:x1] = wt.reshape(y1 - y0, x1 - x0, CFG["TOP_K"])
        tiles += 1
        if tiles % 20 == 0:
            print(f"  tiles {tiles} ({time.time() - tc:.0f} s)", flush=True)

covered = (BEST[..., 0] >= 0)
print(f"{covered.sum()} / {covered.size} texels have a photo ({100 * covered.mean():.1f}%)")
print(f"of the texels on the surface: {100 * covered[HIT].mean():.1f}%")
REPORT["coverage"] = {"texels_with_photo": int(covered.sum()),
                      "pct_of_surface": round(100 * float(covered[HIT].mean()), 2)}
del NRM
stamp("assignment", t)
"""
)

md(
    """
## 6. Exposure gains

The flight took twenty minutes with auto-exposure, so neighbouring photos disagree slightly about
how bright the same rock is — which shows up as patchwork at the view boundaries. Each photo gets one
RGB gain fitted so its texels match the reference bake, which is an average over every view that saw
them and therefore a stable anchor. PIL's `draft` decodes the JPEG at 1/8 scale, so this costs
seconds rather than a second full pass over 103 photos.
"""
)

code(
    r"""
t = time.time()
flat_best = BEST[..., 0].ravel()
GAIN = {}
rng = np.random.default_rng(0)
for ci, name in enumerate(NAMES):
    idx = np.flatnonzero(flat_best == ci)
    if len(idx) < 20000:
        GAIN[name] = np.ones(3, np.float32); continue
    idx = rng.choice(idx, 200000, replace=False) if len(idx) > 200000 else idx
    with Image.open(PHOTOS[name]) as im0:
        im0.draft("RGB", (im0.size[0] // 8, im0.size[1] // 8))
        small = np.asarray(im0.convert("RGB"))
    sh, sw = small.shape[:2]
    cam, R, C = CAMS[name], ROT[name], CENTRE[name]
    P = POS.reshape(-1, 3)[idx].astype(np.float64)
    Xc = (P - C) @ R.T
    uv = np.asarray(cam.img_from_cam(Xc))
    px = np.clip((uv[:, 0] / cam.width * sw).astype(np.int32), 0, sw - 1)
    py = np.clip((uv[:, 1] / cam.height * sh).astype(np.int32), 0, sh - 1)
    got = small[py, px].astype(np.float32)
    want = REF.reshape(-1, 3)[idx].astype(np.float32)
    g = want.mean(0) / np.maximum(got.mean(0), 1e-3)
    GAIN[name] = np.clip(g, CFG["GAIN_CLIP"][0], CFG["GAIN_CLIP"][1]).astype(np.float32)
g = np.array(list(GAIN.values()))
print("gains: min", g.min(0).round(3), "max", g.max(0).round(3), "mean", g.mean(0).round(3))
print("clipped at a bound:", int(((g <= CFG["GAIN_CLIP"][0] + 1e-6) | (g >= CFG["GAIN_CLIP"][1] - 1e-6)).any(1).sum()), "photos")
REPORT["gains"] = {"mean": [round(float(x), 4) for x in g.mean(0)],
                   "min": [round(float(x), 4) for x in g.min(0)],
                   "max": [round(float(x), 4) for x in g.max(0)]}
stamp("gains", t)
"""
)

md("## 7. Sample the photos")

code(
    r"""
t = time.time()
ACC = np.zeros((TEX_H * TEX_W, 3), np.float32)
WSUM = np.zeros(TEX_H * TEX_W, np.float32)
BESTF = BEST.reshape(-1, CFG["TOP_K"])
WGTF = WGT.reshape(-1, CFG["TOP_K"])
POSF = POS.reshape(-1, 3)
CHUNK = 4_000_000

for ci, name in enumerate(NAMES):
    sel = np.flatnonzero((BESTF == ci).any(1))
    if not len(sel):
        continue
    with Image.open(PHOTOS[name]) as im0:
        img = np.asarray(im0.convert("RGB"))
    ih, iw = img.shape[:2]
    cam, R, C, gain = FULL[name], ROT[name], CENTRE[name], GAIN[name]
    for c0 in range(0, len(sel), CHUNK):
        idx = sel[c0:c0 + CHUNK]
        w = np.where(BESTF[idx] == ci, WGTF[idx], 0.0).sum(1)
        keep = w > 1e-4
        if not keep.any():
            continue
        idx, w = idx[keep], w[keep].astype(np.float32)
        Xc = (POSF[idx].astype(np.float64) - C) @ R.T
        uv = np.asarray(cam.img_from_cam(Xc))
        x, y = uv[:, 0], uv[:, 1]
        ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= iw - 1.001) & (y >= 0) & (y <= ih - 1.001)
        if not ok.any():
            continue
        idx, w, x, y = idx[ok], w[ok], x[ok], y[ok]
        x0 = x.astype(np.int32); y0 = y.astype(np.int32)
        fx = (x - x0).astype(np.float32)[:, None]; fy = (y - y0).astype(np.float32)[:, None]
        c00 = img[y0, x0].astype(np.float32); c10 = img[y0, x0 + 1].astype(np.float32)
        c01 = img[y0 + 1, x0].astype(np.float32); c11 = img[y0 + 1, x0 + 1].astype(np.float32)
        col = (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy
        # idx is unique within a chunk, so a plain += is safe and much faster than np.add.at
        ACC[idx] += col * gain * w[:, None]
        WSUM[idx] += w
    del img
    if (ci + 1) % 10 == 0:
        print(f"  photos {ci + 1}/{NC} ({time.time() - t:.0f} s)", flush=True)

TEX = REF.reshape(-1, 3).astype(np.float32).copy()
have = WSUM > 1e-4
TEX[have] = ACC[have] / WSUM[have][:, None]
TEX = np.clip(TEX, 0, 255).astype(np.uint8).reshape(TEX_H, TEX_W, 3)
print(f"{have.sum()} texels from photos, {(~have).sum()} filled from the reference bake")
REPORT["texels_from_photos"] = int(have.sum())
del ACC, WSUM, BESTF, WGTF, BEST, WGT
stamp("sampling", t)
"""
)

md(
    """
## 8. Did it work?

The claim is that the smear is a bake problem, so the test is local contrast: the mean absolute
Laplacian of the luma, per horizontal band of the atlas. Bands low on the wall should barely move —
the point cloud was already denser than the texture there — and bands above the coverage line should
jump.
"""
)

code(
    r"""
t = time.time()


def sharpness(img, bands=12):
    lum = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
    lap = np.abs(4 * lum[1:-1, 1:-1] - lum[:-2, 1:-1] - lum[2:, 1:-1] - lum[1:-1, :-2] - lum[1:-1, 2:])
    h = lap.shape[0] // bands
    return [float(lap[i * h:(i + 1) * h].mean()) for i in range(bands)]


sr, sn = sharpness(REF), sharpness(TEX)
print(" band (0 = foot of the wall)  reference   photos    gain")
rows = []
for i, (a, b) in enumerate(zip(sr, sn)):
    # the atlas is stored bottom-up; print top-down so it reads like the wall
    rows.append({"band": i, "reference": round(a, 2), "photos": round(b, 2), "gain": round(b / max(a, 1e-6), 2)})
for r in reversed(rows):
    print(f"   {r['band']:2d}   {r['reference']:14.2f} {r['photos']:10.2f} {r['gain']:8.2f}x")
REPORT["sharpness_bands_bottom_up"] = rows
REPORT["sharpness_overall"] = {"reference": round(float(np.mean(sr)), 3),
                               "photos": round(float(np.mean(sn)), 3),
                               "gain": round(float(np.mean(sn) / np.mean(sr)), 3)}
print("overall", REPORT["sharpness_overall"])

# a few before/after crops, top band first
crops = []
for frac in (0.86, 0.70, 0.52, 0.30):        # high on the wall first
    cy = int(TEX_H * frac); cx = TEX_W // 2
    a = REF[cy:cy + 460, cx:cx + 700][::-1]   # the atlas is stored bottom-up
    b = TEX[cy:cy + 460, cx:cx + 700][::-1]
    if a.shape[:2] == (460, 700):
        crops.append(np.hstack([a, np.full((460, 8, 3), 255, np.uint8), b]))
if crops:
    sheet = np.vstack([np.vstack([c, np.full((8, c.shape[1], 3), 255, np.uint8)]) for c in crops])
    Image.fromarray(sheet).save(os.path.join(CFG["OUT"], "compare_texture.jpg"),
                                quality=92, subsampling=0)
    print("wrote compare_texture.jpg — left half is the old bake, right half the photos")
stamp("metrics", t)
"""
)

md("## 9. Export")

code(
    r"""
t = time.time()
BYTES = 1024 * 1024
_pil_save = Image.Image.save


def _save_hq(self, fp, format=None, **kw):
    # trimesh's gltf exporter re-encodes at PIL's default quality 75 with 4:2:0 subsampling
    if (format or self.format) == "JPEG":
        kw.setdefault("quality", CFG["JPEG_Q"])
        kw.setdefault("subsampling", 0)
    return _pil_save(self, fp, format=format, **kw)


Image.Image.save = _save_hq


def as_jpeg(arr, width):
    img = Image.fromarray(arr[::-1])          # glTF v runs down the image
    if width != img.size[0]:
        img = img.resize((width, int(round(width * img.size[1] / img.size[0] / 4)) * 4), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=CFG["JPEG_Q"], subsampling=0)
    out = Image.open(buf)
    out.jpeg_bytes = buf.getbuffer().nbytes
    return out


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
    texture = as_jpeg(TEX, tex_w)
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
    del small
stamp("export", t)

REPORT["total_s"] = round(time.time() - T0)
with open(os.path.join(CFG["OUT"], "retexture_report.json"), "w") as fh:
    json.dump(REPORT, fh, indent=1)
for f in sorted(os.listdir(CFG["OUT"])):
    print(f"{os.path.getsize(os.path.join(CFG['OUT'], f)) / 1e6:10.1f} MB  {f}")
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
out = pathlib.Path(__file__).with_name("korno-v2-retexture.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(CELLS), "cells")
