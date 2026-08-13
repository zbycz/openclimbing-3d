#!/usr/bin/env python3
"""Generates v2/kaggle/routes/korno-routes-3d.ipynb — OSM route lines onto the 3D model."""
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
# Korno — climbing routes onto the 3D model

openclimbing.org stores each route as a polyline drawn **on a photo**: tag `wikimedia_commons` says which
Commons image, `wikimedia_commons:path` gives the points as normalised `x,y|x,y|…`. Those are 2D
annotations on a picture — this turns them into 3D lines on the reconstruction.

The link between the two worlds is that one of the Commons photos of this crag is **the same shot as one
of the drone photos**, whose camera pose SfM already solved. So:

1. find which drone photo the Commons photo is, by matching image content (SIFT + RANSAC homography);
2. the homography maps a point in the Commons photo to the same point in the drone photo;
3. from there it is the same ray-cast as the bolts — camera → pixel → ray → mesh.

Nothing about the reconstructed models changes; the output is one more metadata JSON.
"""
)

code(
    r"""
import os, sys, json, time, glob, re, subprocess, urllib.request, urllib.parse, sqlite3
from collections import defaultdict

CFG = dict(
    INPUT_DIR = "/kaggle/input",
    WORK      = "/kaggle/temp/routes",
    OUT       = "/kaggle/working",
    EXPORT_API = "https://openclimbing.org/api/climbing-tiles/export",
    CRAG_NAME  = "Korno",
    UA = "openclimbing-3d/1.0 (zbytovsky@gmail.com)",

    MATCH_WIDTH   = 1600,   # both images are compared at this width
    SIFT_FEATURES = 4000,
    RATIO         = 0.75,   # Lowe ratio test
    MIN_INLIERS   = 40,     # below this the two photos are not the same shot
    MAX_RAY_LEN   = 40.0,
)
os.makedirs(CFG["WORK"], exist_ok=True)
os.makedirs(CFG["OUT"], exist_ok=True)
T0 = time.time()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open3d", "--no-warn-conflicts"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pycolmap==4.1.1"])
import numpy as np
import cv2
import open3d as o3d
import pycolmap
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
print("opencv", cv2.__version__, "| open3d", o3d.__version__, "| cpus", os.cpu_count())


def fetch(url, dest, data=None):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest
    req = urllib.request.Request(url, data=data, headers={"User-Agent": CFG["UA"]},
                                 method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as fh:
        fh.write(r.read())
    return dest


def find(pattern):
    hits = sorted(glob.glob(os.path.join(CFG["INPUT_DIR"], "**", pattern), recursive=True))
    assert hits, f"{pattern} not found"
    return hits


MESH_PLY = find("korno_v2_mesh.ply")[0]
SPARSE = os.path.dirname(find("cameras.bin")[0])
DRONE = [p for p in find("*.JPG") if os.path.isfile(p)]
print("mesh", MESH_PLY, "| sparse", SPARSE, "| drone photos", len(DRONE))
"""
)

md("## 1. Routes from the production database")

code(
    r"""
t = time.time()
DB = os.path.join(CFG["WORK"], "openclimbing.sqlite")
fetch(CFG["EXPORT_API"], DB, data=b"")
print("export", round(os.path.getsize(DB) / 1e6, 1), "MB")

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
crag = con.execute(
    'SELECT * FROM climbing_features WHERE type=\'crag\' AND ("nameRaw"=? OR name=?)',
    (CFG["CRAG_NAME"], CFG["CRAG_NAME"])).fetchone()
print("crag:", crag["nameRaw"], "| osm", crag["osmType"], crag["osmId"],
      "| routes", crag["routeCount"], "| with photo", crag["routesWithPhoto"])

refs = [m["ref"] for m in json.loads(crag["members"]) if m["type"] == "node"]
qs = ",".join("?" * len(refs))
rows = con.execute(
    f'SELECT "nameRaw","gradeTxt","osmId",tags FROM climbing_features '
    f'WHERE "osmType"=\'node\' AND "osmId" IN ({qs})', refs).fetchall()

# path format per openclimbing's pathUtils.ts: "x,y|x,y" with an optional letter on y
# encoding the point type, and a trailing colon meaning the line to the next point is dotted
BOLT_CODES = {"B": "bolt", "A": "anchor", "P": "piton", "S": "sling", "U": "unfinished"}


def parse_path(raw):
    out = []
    for seg in raw.split("|"):
        seg = seg.strip()
        if not seg:
            continue
        dotted = ":" in seg
        seg = seg[:-1] if dotted else seg
        if "," not in seg:
            continue
        xs, ys = seg.split(",", 1)
        kind = None
        if ys and ys[-1] in BOLT_CODES:
            kind, ys = BOLT_CODES[ys[-1]], ys[:-1]
        try:
            x, y = float(xs), float(ys)
        except ValueError:
            continue
        out.append({"x": x, "y": y, "type": kind, "dotted_after": dotted})
    return out


routes = defaultdict(list)
kinds = defaultdict(int)
for r in rows:
    tags = json.loads(r["tags"])
    img, path = tags.get("wikimedia_commons"), tags.get("wikimedia_commons:path")
    if not img or not path:
        continue
    pts = parse_path(path)
    for q in pts:
        kinds[q["type"]] += 1
    if len(pts) < 2:
        continue
    routes[img].append({"name": r["nameRaw"] or tags.get("name"),
                        "grade": r["gradeTxt"], "osmId": r["osmId"],
                        "path": pts, "url": tags.get("website")})
print("point types:", dict(kinds))
for img, rs in sorted(routes.items(), key=lambda kv: -len(kv[1])):
    print(f"  {len(rs):3d} routes on {img}")
print("stage", round(time.time() - t, 1), "s")
"""
)

code(
    r"""
t = time.time()
# resolve the Commons titles to file URLs and download them
titles = list(routes.keys())
api = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo"
       "&iiprop=url|size&titles=" + urllib.parse.quote("|".join(titles)))
info = json.load(open(fetch(api, os.path.join(CFG["WORK"], "commons.json"))))
commons = {}
for p in info["query"]["pages"].values():
    ii = (p.get("imageinfo") or [{}])[0]
    if not ii.get("url"):
        print("  no imageinfo for", p["title"]); continue
    dest = os.path.join(CFG["WORK"], re.sub(r"[^A-Za-z0-9.]+", "_", p["title"]))
    fetch(ii["url"], dest)
    commons[p["title"]] = dest
    print(f"  {p['title']}  {ii['width']}x{ii['height']}  {os.path.getsize(dest)/1e6:.1f} MB")
print("stage", round(time.time() - t, 1), "s")
"""
)

md(
    """
## 2. Which drone photo is the Commons photo?

SIFT features on both, Lowe ratio test, then RANSAC for a homography. Keypoints are converted to
normalised `[0,1]` coordinates first, so the homography maps *fractions of one image to fractions of the
other* and image size drops out of the problem entirely — which is what makes the route paths (also
normalised) map straight through it.
"""
)

code(
    r"""
t = time.time()
sift = cv2.SIFT_create(nfeatures=CFG["SIFT_FEATURES"])
matcher = cv2.BFMatcher()


def features(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    s = CFG["MATCH_WIDTH"] / w
    img = cv2.resize(img, (CFG["MATCH_WIDTH"], round(h * s)), interpolation=cv2.INTER_AREA)
    kp, des = sift.detectAndCompute(img, None)
    # normalised coordinates: the homography then knows nothing about resolution
    pts = np.array([[k.pt[0] / img.shape[1], k.pt[1] / img.shape[0]] for k in kp], np.float32)
    return pts, des


def match(des_a, pts_a, des_b, pts_b):
    if des_a is None or des_b is None or len(des_a) < 10 or len(des_b) < 10:
        return None, 0
    pairs = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < CFG["RATIO"] * n.distance]
    if len(good) < 10:
        return None, len(good)
    src = np.array([pts_a[m.queryIdx] for m in good], np.float32).reshape(-1, 1, 2)
    dst = np.array([pts_b[m.trainIdx] for m in good], np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.004)   # 0.4% of image size
    return H, int(mask.sum()) if mask is not None else 0


drone_feats = {}
for i, p in enumerate(DRONE):
    drone_feats[os.path.basename(p)] = features(p)
    if (i + 1) % 25 == 0:
        print(f"  features {i+1}/{len(DRONE)}", flush=True)
print("drone features done", round(time.time() - t, 1), "s", flush=True)
"""
)

code(
    r"""
t = time.time()
matches = {}
for title, path in commons.items():
    pts_c, des_c = features(path)
    best = []
    for name, (pts_d, des_d) in drone_feats.items():
        H, inl = match(des_c, pts_c, des_d, pts_d)
        if inl:
            best.append((inl, name, H))
    best.sort(key=lambda x: -x[0])
    top = best[:3]
    print(f"\n{title}")
    for inl, name, _ in top:
        print(f"   {inl:5d} inliers  {name}")
    if top and top[0][0] >= CFG["MIN_INLIERS"]:
        inl, name, H = top[0]
        runner = top[1][0] if len(top) > 1 else 0
        matches[title] = {"drone": name, "inliers": inl, "H": H.tolist(), "runner_up": runner}
        print(f"   -> MATCH {name} ({inl} inliers, next best {runner})")
    else:
        print(f"   -> no match (best {top[0][0] if top else 0} < {CFG['MIN_INLIERS']})")
print("\nmatching", round(time.time() - t, 1), "s")
"""
)

md("## 3. Project the route paths onto the mesh")

code(
    r"""
t = time.time()
rec = pycolmap.Reconstruction(SPARSE)
IMS = {im.name: im for im in rec.images.values() if im.has_pose}
mesh = o3d.io.read_triangle_mesh(MESH_PLY)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
print("mesh", len(mesh.triangles), "triangles | posed images", len(IMS))

out_routes = []
for title, m in matches.items():
    name = m["drone"]
    if name not in IMS:
        print("drone photo has no pose:", name); continue
    im = IMS[name]
    cam = rec.cameras[im.camera_id]
    R = im.cam_from_world().rotation.matrix()
    centre = im.projection_center()
    H = np.array(m["H"])

    for route in routes[title]:
        p = np.array([[q["x"], q["y"]] for q in route["path"]], np.float64)   # normalised, Commons photo
        q = cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)   # -> normalised in the drone photo
        inside = (q[:, 0] > -0.02) & (q[:, 0] < 1.02) & (q[:, 1] > -0.02) & (q[:, 1] < 1.02)
        pix = np.stack([q[:, 0] * cam.width, q[:, 1] * cam.height], 1)
        rays = np.asarray(cam.cam_ray_from_img(pix)) @ R
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        arr = np.hstack([np.tile(centre, (len(rays), 1)).astype(np.float32), rays.astype(np.float32)])
        hit = scene.cast_rays(o3d.core.Tensor(arr, dtype=o3d.core.Dtype.Float32))["t_hit"].numpy()
        ok = np.isfinite(hit) & (hit < CFG["MAX_RAY_LEN"]) & inside
        if ok.sum() < 2:
            continue
        pts3d = np.tile(centre, (len(rays), 1))[ok] + rays[ok] * hit[ok, None]
        out_routes.append({
            "name": route["name"], "grade": route["grade"], "osmId": route["osmId"],
            "url": route["url"], "photo": title, "drone_photo": name,
            "points": [[round(float(x), 4) for x in p] for p in pts3d],
            "points_2d": [[round(float(a), 4), round(float(b), 4)] for a, b in q[ok]],
            "types": [route["path"][i]["type"] for i in np.nonzero(ok)[0]],
            "dotted_after": [route["path"][i]["dotted_after"] for i in np.nonzero(ok)[0]],
            "dropped": int((~ok).sum()),
        })

print(f"\n{len(out_routes)} routes projected")
for r in sorted(out_routes, key=lambda r: -len(r["points"]))[:20]:
    print(f"   {str(r['name'])[:28]:28s} {str(r['grade']):8s} {len(r['points'])} pts"
          f"{'  (' + str(r['dropped']) + ' dropped)' if r['dropped'] else ''}")
print("project", round(time.time() - t, 1), "s")
"""
)

code(
    r"""
out = {
    "crag": CFG["CRAG_NAME"],
    "source": "openclimbing.org export -> OSM wikimedia_commons:path",
    "note": "positions are in the same coordinate frame as korno_v2.glb",
    "matches": {k: {"drone": v["drone"], "inliers": v["inliers"], "runner_up": v["runner_up"]}
                for k, v in matches.items()},
    "count": len(out_routes),
    "routes": sorted(out_routes, key=lambda r: (r["name"] or "")),
}
with open(os.path.join(CFG["OUT"], "korno_v2_routes.json"), "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)

# a visual check: the matched drone photo with the route paths drawn where they landed
for title, m in matches.items():
    src = [p for p in DRONE if os.path.basename(p) == m["drone"]][0]
    img = cv2.imread(src)
    for r in out_routes:
        if r["drone_photo"] != m["drone"]:
            continue
        pts = (np.array(r["points_2d"]) * [img.shape[1], img.shape[0]]).astype(np.int32)
        cv2.polylines(img, [pts], False, (0, 90, 255), 6, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 9, (0, 220, 255), -1)
        cv2.putText(img, str(r["name"]), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3, cv2.LINE_AA)
    h = 1600
    img = cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(CFG["OUT"], "routes_on_" + m["drone"]), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("preview written for", m["drone"])

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
out = pathlib.Path(__file__).with_name("korno-routes-3d.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(CELLS), "cells")
