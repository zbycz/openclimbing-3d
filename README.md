# openclimbing-3d

3D Gaussian Splatting of the Korno rock face, reconstructed from 103 drone photos.

**Live viewer: https://zbycz.github.io/openclimbing-3d/**

Two independent reconstructions of the same wall:

* **v1** (this folder) — 3D Gaussian Splatting, then a mesh derived from it.
* **[v2](v2/)** — classic dense MVS photogrammetry straight from the photos, no splatting.
  [viewer](https://zbycz.github.io/openclimbing-3d/v2/)

| | |
|---|---|
| photos | [korno-rockface-photogrammetry](https://www.kaggle.com/datasets/pavelzbytovsk/korno-rockface-photogrammetry) (Kaggle dataset, 103 × 8000×4500) |
| splat | [`kaggle/splat/`](kaggle/splat) — COLMAP SfM → 3DGS training on a Kaggle P100 |
| mesh | [`kaggle/mesh/`](kaggle/mesh) — splat → textured glTF on a Kaggle CPU kernel |
| viewer | [`index.html`](index.html) — [@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D) |

## Notebooks

Each notebook is generated from its `build_notebook.py`, so the source stays diffable:

```bash
python3 kaggle/splat/build_notebook.py && kaggle kernels push -p kaggle/splat
python3 kaggle/mesh/build_notebook.py  && kaggle kernels push -p kaggle/mesh
kaggle kernels logs pavelzbytovsk/korno-gaussian-splat --follow
```

### 1. `kaggle/splat` — gaussian splatting (GPU, P100)

Produces three artifacts:

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

### 2. `kaggle/mesh` — textured mesh (CPU, ~10 min)

Turns `korno_full.ply` into an ordinary triangle mesh. All CPU work, so it runs on a plain CPU kernel.

| file | size | triangles | texture |
|---|---|---|---|
| `korno_mesh_10mb.glb` | 9.8 MB | 366 k | 2048 × 984 |
| `korno_mesh_50mb.glb` | 47.6 MB | 1.73 M | 4096 × 1964 |
| `korno_mesh_100mb.glb` | 95.4 MB | 3.40 M | 8192 × 3932 |

The published site is the splat only; the meshes are committed but nothing links to them. To look at one,
open `mesh.html?f=korno_mesh_50mb.glb` directly.

Two decisions worth knowing about:

**Dropping the haze.** A splat carries many low-opacity gaussians that only resolve from the training
views; orbiting away you look straight through fog. Keeping gaussians that are opaque *and seen by at
least 8 cameras* leaves 37 % of them — the rock and the vegetation on it — and drops the haze.

**Planar UVs instead of an atlas.** The face is ~13 × 6 units wide but only 2.6 deep, so one planar
projection along the mean camera direction beats an atlas unwrap: no seams, no charts, and the texture
is effectively an orthophoto. Because the texture is then independent of triangle count, the 10 MB model
still looks close to the 100 MB one at overview zoom.
