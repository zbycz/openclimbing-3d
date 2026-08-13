# v2 — classic dense photogrammetry

A second, independent reconstruction of the same rock face. It starts from the
[original photos](https://www.kaggle.com/datasets/pavelzbytovsk/korno-rockface-photogrammetry) and uses
**no gaussian splatting and none of v1's outputs**.

**Viewer: https://zbycz.github.io/openclimbing-3d/v2/** — with a checkbox that draws the position and
frustum of every photo.

## Pipeline

```
103 photos (8000×4500)
  → COLMAP SfM, GPU SIFT at 3200 px
  → image_undistorter at 2600 px
  → patch_match_stereo   (CUDA dense stereo — one depth map per photo)
  → stereo_fusion        (depth maps → dense coloured point cloud)
  → poisson_mesher       (depth 13)
  → planar-UV texture bake → glTF
```

The difference from v1 is where the geometry comes from. v1 inferred a surface from the centres of
1.4 M gaussians; here every pixel of every photo gets a depth estimate that is checked against the other
views, so cracks and edges are measured rather than implied.

`kaggle/build_notebook.py` generates the notebook:

```bash
python3 v2/kaggle/build_notebook.py
kaggle kernels push -p v2/kaggle
kaggle kernels logs pavelzbytovsk/korno-v2-mvs --follow
```

## Result

| | |
|---|---|
| registered | 102 / 103 photos, 172 952 sparse points, 1.35 px reprojection error |
| dense cloud | 5 249 666 measured points (one depth estimate per pixel, cross-checked between views) |
| raw mesh | 17 050 230 triangles |
| `korno_v2.glb` | 90 MiB — 3 000 000 triangles, 8192 × 4500 texture |
| `korno_v2_light.glb` | 11 MiB — 380 000 triangles, 2048 × 1124 texture |
| runtime | 2 h 21 m on one P100 (80 min of that is dense stereo) |

## Getting CUDA COLMAP onto Kaggle

`patch_match_stereo` is CUDA-only — there is no CPU fallback — and there is no pip wheel, so COLMAP comes
from conda-forge through micromamba. That package has three traps, each of which fails differently:

| symptom | cause | fix |
|---|---|---|
| `libfaiss.so: cannot open shared object file` | colmap links faiss but doesn't declare it | install `libfaiss` explicitly |
| `undefined symbol: faiss::IndexIVFFlat...` | built against faiss 1.10, current is 1.14 | pin `libfaiss=1.10.0` |
| `libmkl_intel_lp64.so.2: cannot open...` | solver picked faiss's MKL variant | pin `=*openblas*` |
| `libOpenImageIO.so.3.1 => not found` | also linked, also undeclared | install `openimageio` |

A fifth trap is in COLMAP itself: its bundled PoissonRecon **segfaults** (`SIGSEGV`, exit 139) at depth 13
on this cloud — but only *after* writing a complete mesh. The notebook therefore tolerates a non-zero exit
and validates the PLY by comparing its byte size against the vertex and face counts in its own header,
falling back to Open3D's Poisson only if the file really is truncated.

Two more things bite on the way out. GitHub rejects any file over **100 MiB** and Pages does not resolve
Git LFS pointers, so the web model is fitted to a 92 MiB budget. And trimesh's glTF exporter claims not to
re-encode JPEGs but calls `img.save(format="JPEG")` anyway, landing on PIL's default quality 75 with 4:2:0
chroma subsampling — on rock that discards precisely the high-frequency contrast that shows cracks, so the
notebook forces quality 95 at 4:4:4 (3x the texture data for the same pixel count).

The working combination is `colmap=4.1.1=cuda*` + `libfaiss=1.10.0=*openblas*` + `openimageio`.
COLMAP 4.x also renamed its option namespaces — it is `--FeatureExtraction.use_gpu`, not
`--SiftExtraction.use_gpu`.

`kaggle/probe/` is the throwaway notebook that established all of this by running the whole pipeline on
8 images, so a mistake cost two minutes instead of three hours. It is also what measured
`patch_match_stereo` at 15.5 s/image at 1200 px on a P100, which is how the real run was sized:
cost scales with pixels and with source views per image, so 2600 px × 10 source views ≈ 3 h for 102 photos.

## Cost of the GPU

Feature extraction and matching are the same algorithm as v1, just on the GPU instead of 4 CPU cores:

| stage | v1 (CPU, 2400 px) | v2 (GPU, 3200 px) |
|---|---|---|
| feature extraction | 597 s | 12 s |
| matching | 1046 s | 183 s |


## Bolts (metadata only)

The reconstructed models are **not modified**. Bolt positions live in their own JSON that the viewer
overlays, exactly like the camera poses — `korno_v2.glb` and `korno_v2_light.glb` stay byte-identical.

```
103 photos → YOLOv8-nano ONNX (openclimbing-bolts-ai) → 152 detections
           → ray from each camera through each detection → intersect the mesh
           → cluster the landing points across views → 19 bolts
```

| | |
|---|---|
| `bolts_2d/<photo>.json` | per-photo detections, normalised boxes |
| `korno_v2_bolts.json` | 19 bolts confirmed by ≥2 photos |
| `korno_v2_bolts_rejected.json` | 94 single-view candidates, kept for review |

Two kernels, both CPU: [`kaggle/bolts`](kaggle/bolts) (~23 min) and [`kaggle/bolts3d`](kaggle/bolts3d)
(~4 min). Splitting them means the projection can be re-tuned without repeating the inference.

**Multi-view agreement is the real filter.** The detector is out of its training domain here — it learnt
from ground-level photos where a bolt fills 30–100 px, while these drone frames show them at 12–34 px, so
scores are low (median 0.30) and false positives are common. Thresholding on score alone would throw away
real bolts and keep noise. Instead every detection is kept and the question becomes geometric: do several
independent photos put a bolt in the *same place in 3D*? 94 of 113 candidates were seen once and dropped;
the 19 that survived agree to within 0.001–0.028 units across up to 7 views.

That the survivors form two parallel vertical lines with plausible clip spacing — while the rejects are
scattered uniformly over the wall — is a good sign the geometry is right, and it is not something a
per-photo confidence threshold could have discovered.


## Climbing routes (metadata only)

openclimbing.org stores each route as a polyline drawn **on a photo** — `wikimedia_commons` names the
Commons image, `wikimedia_commons:path` holds the points as normalised `x,y|x,y|…`. The crag "Korno"
(OSM relation 17486978) has 58 routes over 5 photos.

The bridge is that the crag's first Commons photo is **the same shot as one of the drone photos**:
`Tomáškův lom - Korno.jpg` = `DJI_20260811211418_0016_D.JPG`, established by SIFT + RANSAC with
**4000 inliers** — the SIFT feature cap, so effectively every feature matched, against 1861 for the next
best (a neighbouring frame with genuine overlap). The other four photos are phone shots from the ground
and correctly matched nothing (9 inliers, well under the threshold of 40).

Keypoints are normalised to `[0,1]` before the homography is fitted, so it maps *fractions of one image to
fractions of the other* — resolution drops out, and the route paths (also normalised) map straight through
it. From there it is the same ray-cast as the bolts.

| | |
|---|---|
| `korno_v2_routes.json` | 13 routes as 3D polylines, with names, grades and point types |
| `routes_preview.jpg` | the matched drone photo with the paths drawn on it, for checking |

The path format is decoded per openclimbing's own `pathUtils.ts`: a trailing letter on the `y` value is the
point type (`A` anchor, `B` bolt, `P` piton, `S` sling, `U` unfinished) and a trailing colon means the line
to the next point is dotted. Reading those as plain numbers fails — 13 of the points here end in `A`.
