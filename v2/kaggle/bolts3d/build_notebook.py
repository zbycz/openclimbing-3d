#!/usr/bin/env python3
"""Generates v2/kaggle/bolts3d/korno-bolts-3d.ipynb — 2D bolt detections onto the 3D model."""
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
# Korno — bolts onto the 3D model

Turns per-photo 2D detections into 3D bolt positions, **without touching the reconstructed models**.
The result is one small JSON that the viewer overlays; `korno_v2.glb` and the splat stay byte-identical.

How it works:

1. every detection is a pixel in a photo whose camera pose SfM already solved, so it defines a ray
   from that camera into the scene (`cam_ray_from_img` handles the lens distortion);
2. the ray is intersected with the reconstructed mesh — where it lands is where the bolt is;
3. the same bolt is detected in many overlapping photos, so the landing points are clustered. A cluster
   confirmed by several independent views is a real bolt; a one-off is almost always a false positive.

That last step is what makes this worth doing in 3D: multi-view agreement is a much stronger filter than
any confidence threshold on a single photo.
"""
)

code(
    r"""
import os, sys, json, time, glob, subprocess, zipfile
from collections import defaultdict

CFG = dict(
    INPUT_DIR = "/kaggle/input",
    OUT       = "/kaggle/working",

    MIN_SCORE   = 0.25,   # keep everything the detector emitted - agreement between
                          # views filters far better than any single-photo threshold
    MAX_RAY_LEN = 40.0,   # scene units; rays that hit nothing this far are dropped
    CLUSTER_R   = 0.045,  # ~14 cm at this scene scale - bolts are never closer than that
    MIN_VIEWS   = 2,      # a bolt must be seen from at least this many photos
)
os.makedirs(CFG["OUT"], exist_ok=True)
T0 = time.time()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open3d", "--no-warn-conflicts"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pycolmap==4.1.1"])
import numpy as np
import open3d as o3d
import pycolmap
print("open3d", o3d.__version__, "| cpus", os.cpu_count())


def find(pattern, must=True):
    hits = sorted(glob.glob(os.path.join(CFG["INPUT_DIR"], "**", pattern), recursive=True))
    if must:
        assert hits, f"{pattern} not found"
    return hits


MESH_PLY = find("korno_v2_mesh.ply")[0]
SPARSE = os.path.dirname(find("cameras.bin")[0])
BOLT_JSONS = find("*.JPG.json")
print("mesh   ", MESH_PLY, round(os.path.getsize(MESH_PLY) / 1e6, 1), "MB")
print("sparse ", SPARSE)
print("bolt jsons:", len(BOLT_JSONS))
"""
)

md("## 1. Cameras and detections")

code(
    r"""
t = time.time()
rec = pycolmap.Reconstruction(SPARSE)
IMS = {im.name: im for im in rec.images.values() if im.has_pose}
print(rec.summary())
print("posed images:", len(IMS))

dets = []
kept = dropped_score = no_pose = 0
for path in BOLT_JSONS:
    d = json.load(open(path))
    name = d["image"]
    im = IMS.get(name)
    if im is None:
        no_pose += 1
        continue
    cam = rec.cameras[im.camera_id]
    for b in d["detections"]:
        if b["score"] < CFG["MIN_SCORE"]:
            dropped_score += 1
            continue
        # detections are normalised, so they map onto the SfM image size directly
        dets.append((name, b["cx"] * cam.width, b["cy"] * cam.height, b["score"]))
        kept += 1
print(f"detections: {kept} kept, {dropped_score} below score, {no_pose} photos without a pose")
"""
)

md("## 2. Ray-cast every detection onto the mesh")

code(
    r"""
t = time.time()
mesh = o3d.io.read_triangle_mesh(MESH_PLY)
print("mesh", len(mesh.triangles), "triangles")
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
print("BVH built in", round(time.time() - t, 1), "s", flush=True)

by_image = defaultdict(list)
for name, u, v, score in dets:
    by_image[name].append((u, v, score))

origins, dirs, meta = [], [], []
for name, items in by_image.items():
    im = IMS[name]
    cam = rec.cameras[im.camera_id]
    cw = im.cam_from_world()
    R = cw.rotation.matrix()
    centre = im.projection_center()
    pts = np.array([[u, v] for u, v, _ in items], dtype=np.float64)
    rays_cam = np.asarray(cam.cam_ray_from_img(pts))          # unit-ish rays in camera frame
    rays_world = rays_cam @ R                                  # R is world->cam, so R^T applied on the right
    for (u, v, score), rw in zip(items, rays_world):
        origins.append(centre)
        dirs.append(rw / np.linalg.norm(rw))
        meta.append((name, score, u, v))

rays = np.hstack([np.asarray(origins, np.float32), np.asarray(dirs, np.float32)])
hits = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
dist = hits["t_hit"].numpy()
ok = np.isfinite(dist) & (dist < CFG["MAX_RAY_LEN"])
pts3d = np.asarray(origins)[ok] + np.asarray(dirs)[ok] * dist[ok, None]
meta = [m for m, k in zip(meta, ok) if k]
print(f"{ok.sum()} / {len(ok)} rays hit the mesh")
stamp = time.time() - t
print("raycast", round(stamp, 1), "s", flush=True)
"""
)

md("## 3. Cluster across views")

code(
    r"""
t = time.time()
from scipy.spatial import cKDTree

# greedy clustering by 3D proximity, strongest detection first
order = np.argsort([-m[1] for m in meta])
tree = cKDTree(pts3d)
taken = np.zeros(len(pts3d), bool)
clusters = []
for i in order:
    if taken[i]:
        continue
    idx = [j for j in tree.query_ball_point(pts3d[i], CFG["CLUSTER_R"]) if not taken[j]]
    if not idx:
        continue
    taken[idx] = True
    views = {}
    for j in idx:
        name, score, u, v = meta[j]
        # one vote per photo - the strongest detection in it
        if name not in views or score > views[name][0]:
            views[name] = (score, j)
    member = [j for _, j in views.values()]
    clusters.append({
        "position": pts3d[member].mean(0),
        "views": sorted(views.keys()),
        "scores": [round(float(s), 4) for s, _ in views.values()],
        "spread": float(np.linalg.norm(pts3d[member] - pts3d[member].mean(0), axis=1).max()) if len(member) > 1 else 0.0,
    })

print(f"{len(clusters)} clusters from {len(pts3d)} rays")
strong = [c for c in clusters if len(c["views"]) >= CFG["MIN_VIEWS"]]
print(f"{len(strong)} confirmed by >= {CFG['MIN_VIEWS']} views "
      f"({len(clusters) - len(strong)} single-view rejected)")
hist = defaultdict(int)
for c in clusters:
    hist[min(len(c["views"]), 10)] += 1
print("views per cluster:", dict(sorted(hist.items())))
print("cluster", round(time.time() - t, 1), "s")
"""
)

code(
    r"""
bolts = []
for i, c in enumerate(sorted(strong, key=lambda c: -len(c["views"]))):
    bolts.append({
        "id": i,
        "position": [round(float(x), 4) for x in c["position"]],
        "views": len(c["views"]),
        "score": round(float(np.mean(c["scores"])), 4),
        "best_score": round(float(max(c["scores"])), 4),
        "spread": round(c["spread"], 4),
        "photos": c["views"][:12],
    })

out = {
    "model": "openclimbing-bolts-v1",
    "count": len(bolts),
    "params": {k: CFG[k] for k in ("MIN_SCORE", "CLUSTER_R", "MIN_VIEWS", "MAX_RAY_LEN")},
    "note": "positions are in the same coordinate frame as korno_v2.glb",
    "bolts": bolts,
}
with open(os.path.join(CFG["OUT"], "korno_v2_bolts.json"), "w") as fh:
    json.dump(out, fh, indent=1)

# a rejected-detections file too, so the threshold can be reviewed without re-running
weak = [{"position": [round(float(x), 4) for x in c["position"]],
         "views": len(c["views"]), "score": round(float(max(c["scores"])), 4)}
        for c in clusters if len(c["views"]) < CFG["MIN_VIEWS"]]
with open(os.path.join(CFG["OUT"], "korno_v2_bolts_rejected.json"), "w") as fh:
    json.dump({"count": len(weak), "bolts": weak}, fh)

print(f"wrote {len(bolts)} bolts")
for b in bolts[:15]:
    print(f"  #{b['id']:3d} {b['position']} views={b['views']:2d} score={b['score']:.2f} spread={b['spread']:.3f}")
for f in sorted(os.listdir(CFG["OUT"])):
    print(f"{os.path.getsize(os.path.join(CFG['OUT'], f)) / 1e3:10.1f} kB  {f}")
print("TOTAL", round(time.time() - T0), "s")
"""
)


nb = {
    "cells": CELLS,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11.13"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = pathlib.Path(__file__).with_name("korno-bolts-3d.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(CELLS), "cells")
