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

## Getting CUDA COLMAP onto Kaggle

`patch_match_stereo` is CUDA-only — there is no CPU fallback — and there is no pip wheel, so COLMAP comes
from conda-forge through micromamba. That package has three traps, each of which fails differently:

| symptom | cause | fix |
|---|---|---|
| `libfaiss.so: cannot open shared object file` | colmap links faiss but doesn't declare it | install `libfaiss` explicitly |
| `undefined symbol: faiss::IndexIVFFlat...` | built against faiss 1.10, current is 1.14 | pin `libfaiss=1.10.0` |
| `libmkl_intel_lp64.so.2: cannot open...` | solver picked faiss's MKL variant | pin `=*openblas*` |
| `libOpenImageIO.so.3.1 => not found` | also linked, also undeclared | install `openimageio` |

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
