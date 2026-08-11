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

| file | size | what |
|---|---|---|
| `korno_full.ply` | 922 MB | full-quality 3DGS point cloud — 3 718 543 gaussians, SH degree 3 |
| `korno_web.splat` | 73 MB | web build — the 2 293 760 most significant gaussians (98.4 % of screen coverage), committed here |
| `korno_colmap.zip` | 27 MB | camera poses + sparse points, so training can be repeated without redoing SfM |

## Result

| | |
|---|---|
| registered | 102 / 103 images, 114 008 sparse points, 1.23 px mean reprojection error |
| model | 3 718 543 gaussians, 30 000 iterations |
| hardware | Tesla P100 (sm_60), 4 CPU cores — 1 h 55 m end to end |

Kaggle's stock PyTorch is built for `sm_70`+ and cannot run on a P100 at all, so the notebook
detects the mismatch and installs a Pascal-capable build (2.6.0+cu124) before importing torch.
Training also has to be kept inside 16 GB of VRAM — see `DENSIFY_GRAD_THRESHOLD` /
`DENSIFY_UNTIL_ITER` in the config cell.
