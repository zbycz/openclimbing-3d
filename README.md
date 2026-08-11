# openclimbing-3d

3D Gaussian Splatting of the Korno rock face, reconstructed from 103 drone photos.

**Live viewer: https://zbycz.github.io/openclimbing-3d/**

| | |
|---|---|
| photos | [korno-rockface-photogrammetry](https://www.kaggle.com/datasets/pavelzbytovsk/korno-rockface-photogrammetry) (Kaggle dataset, 103 × 8000×4500) |
| pipeline | [`kaggle/korno-gaussian-splat.ipynb`](kaggle/korno-gaussian-splat.ipynb) — COLMAP SfM → 3DGS training on a Kaggle P100 |
| viewer | [`index.html`](index.html) — [@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D) |

## Notebook

`kaggle/korno-gaussian-splat.ipynb` is generated from `kaggle/build_notebook.py`:

```bash
python3 kaggle/build_notebook.py
kaggle kernels push -p kaggle
kaggle kernels logs pavelzbytovsk/korno-gaussian-splat --follow
```

It produces three artifacts:

| file | what |
|---|---|
| `korno_full.ply` | full-quality 3DGS point cloud (every gaussian, SH degree 3) |
| `korno_web.splat` | ~70 MB web build, importance-pruned — the one committed here |
| `korno_colmap.zip` | camera poses + sparse points, so training can be repeated without redoing SfM |
