#!/usr/bin/env python3
"""Generates kaggle/korno-gaussian-splat.ipynb from the cell sources below."""
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
# Korno rockface — 3D Gaussian Splatting (Kaggle P100)

Drone photos -> COLMAP SfM -> 3D Gaussian Splatting -> web-ready splat.

Outputs written to `/kaggle/working`:

| file | what |
|---|---|
| `korno_full.ply` | full-quality 3DGS point cloud (all gaussians, SH degree 3) |
| `korno_web.splat` | ~70 MB web viewer file (`.splat`, importance-pruned) |
| `korno_colmap.zip` | SfM result (camera poses + sparse points), for re-training |
| `run_report.json` | timings, counts, versions, viewer camera hints |
"""
)

code(
    r"""
import os, sys, json, time, glob, shutil, subprocess, math

CFG = dict(
    INPUT_DIR     = "/kaggle/input",       # every image found below this path is used
    WORK          = "/kaggle/temp/gs",     # scratch, NOT part of notebook output
    OUT           = "/kaggle/working",     # notebook artifacts
    SFM_WIDTH     = 2400,                  # images are downscaled to this width for SfM + training
    MAX_FEATURES  = 8192,                  # SIFT features per image
    ITERATIONS    = 30000,                 # 3DGS training iterations
    SAVE_ITERATIONS = [7000, 15000, 30000],  # intermediate saves survive a late OOM
    # defaults (0.0002 / 15000) densify past what a 16 GB P100 holds for this scene
    DENSIFY_GRAD_THRESHOLD = 0.0003,
    DENSIFY_UNTIL_ITER     = 13000,
    WEB_TARGET_MB = 70,                    # size budget for the web splat
    FOCAL_35MM    = 26.0,                  # EXIF 35mm-equivalent focal, used as COLMAP prior
    GS_REPO       = "https://github.com/graphdeco-inria/gaussian-splatting.git",
)
os.makedirs(CFG["WORK"], exist_ok=True)
os.makedirs(CFG["OUT"], exist_ok=True)
REPORT = {"cfg": CFG, "timings": {}}
T0 = time.time()


# streams (optionally throttled) subprocess output so the kaggle log stays readable
def run(cmd, cwd=None, env=None, throttle=0.0, tag="run", check=True):
    t0 = time.time()
    last = -1e9
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    p = subprocess.Popen(
        list(map(str, cmd)), cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace",
    )
    tail = []
    for line in p.stdout:
        line = line.rstrip("\n").split("\r")[-1]
        if not line.strip():
            continue
        tail.append(line)
        del tail[:-60]
        now = time.time()
        if throttle <= 0 or now - last >= throttle:
            last = now
            print(f"[{tag} {now - t0:6.0f}s] {line}", flush=True)
    rc = p.wait()
    dt = time.time() - t0
    print(f"[{tag}] exit={rc} after {dt:.0f}s", flush=True)
    if rc != 0:
        print("--- last output ---")
        print("\n".join(tail), flush=True)
        if check:
            raise RuntimeError(f"{tag} failed with exit code {rc}")
    return rc


def stamp(name, t):
    REPORT["timings"][name] = round(time.time() - t, 1)
    print(f"### {name}: {REPORT['timings'][name]}s (total {time.time() - T0:.0f}s)", flush=True)
"""
)

md("## 1. Environment")

code(
    r"""
# nothing may import torch before the compatibility check in the next cell
def smi(query):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return out.splitlines()


run(["nvidia-smi"], tag="gpu", check=False)
GPU = smi("name")[0]
try:
    CC = tuple(int(x) for x in smi("compute_cap")[0].split("."))
except Exception:
    CC = (7, 0)
ARCH = f"sm_{CC[0]}{CC[1]}"
print("gpu:", GPU, "| compute capability:", ARCH, "| cpus:", os.cpu_count())
run(["nvcc", "--version"], tag="nvcc", check=False)
run(["free", "-g"], tag="mem", check=False)
run(["df", "-h", "/kaggle/temp", "/kaggle/working"], tag="disk", check=False)
run(["bash", "-c", "ls -la /kaggle/input/*/ | head -20; find /kaggle/input -type f | wc -l"],
    tag="input", check=False)
REPORT["env"] = {"gpu": GPU, "arch": ARCH, "cpus": os.cpu_count(), "python": sys.version.split()[0]}
"""
)

md(
    """
## 2. PyTorch that can actually talk to a P100

Kaggle's stock PyTorch is compiled for `sm_70`+ only, while the P100 is Pascal (`sm_60`) — every CUDA
kernel would fail with *"no kernel image is available for execution on the device"*. So roll PyTorch
back to the newest release whose official wheels still ship Pascal kernels.
"""
)

code(
    r"""
t = time.time()


def torch_probe():
    out = subprocess.run(
        [sys.executable, "-c",
         "import json, torch; print('PROBE' + json.dumps("
         "{'v': torch.__version__, 'a': torch.cuda.get_arch_list()}))"],
        capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("PROBE"):
            return json.loads(line[5:])
    print(out.stdout[-2000:], out.stderr[-2000:])
    return {"v": None, "a": []}


info = torch_probe()
print("stock torch:", info["v"], info["a"])
for tv, vv, cu in [("2.6.0", "0.21.0", "cu124"), ("2.5.1", "0.20.1", "cu121")]:
    if ARCH in info["a"]:
        break
    print(f"--- {ARCH} not supported, installing torch {tv} ({cu}) ---", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", f"torch=={tv}", f"torchvision=={vv}",
         "--index-url", f"https://download.pytorch.org/whl/{cu}",
         "--extra-index-url", "https://pypi.org/simple"],
        tag=f"torch=={tv}", throttle=20, check=False)
    info = torch_probe()
    print("->", info["v"], info["a"])
assert ARCH in info["a"], f"no available torch build supports {ARCH}"
REPORT["env"]["torch"] = info["v"]
stamp("torch_setup", t)
"""
)

md("## 3. Dependencies + build the CUDA rasterizer (fail fast before the long steps)")

code(
    r"""
t = time.time()
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pycolmap==4.1.1", "plyfile"],
    tag="pip", throttle=10)
import numpy as np
import pycolmap
print("pycolmap", pycolmap.__version__, "numpy", np.__version__)

GS = os.path.join(CFG["WORK"], "gaussian-splatting")
if not os.path.isdir(GS):
    run(["git", "clone", "--depth", "1", CFG["GS_REPO"], GS], tag="clone", throttle=5)
    # only the CUDA extensions - never SIBR_viewers, it is huge and useless here
    run(["git", "submodule", "update", "--init", "--recursive",
         "submodules/diff-gaussian-rasterization", "submodules/simple-knn", "submodules/fused-ssim"],
        cwd=GS, tag="submodules", throttle=5)
REPORT["gs_commit"] = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=GS, capture_output=True, text=True).stdout.strip()
print("gaussian-splatting @", REPORT["gs_commit"])

env = dict(os.environ)
env["TORCH_CUDA_ARCH_LIST"] = f"{CC[0]}.{CC[1]}"          # P100 -> 6.0
env["MAX_JOBS"] = str(os.cpu_count())
for sub in ["diff-gaussian-rasterization", "simple-knn", "fused-ssim"]:
    path = os.path.join(GS, "submodules", sub)
    if not os.path.isdir(path):
        print("skip (not in repo):", sub)
        continue
    # fused-ssim is optional - 3DGS falls back to its own SSIM implementation
    run([sys.executable, "-m", "pip", "install", "-q", path], env=env, tag=f"build:{sub}",
        throttle=20, check=(sub != "fused-ssim"))
stamp("setup", t)
"""
)

code(
    r"""
# Smoke test: does the rasterizer actually run on this GPU (P100 = sm_60)?
# Runs out-of-process so it leaves no CUDA context behind - the P100's 16 GB are all needed later.
SMOKE = r'''
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

dev = "cuda"
N, H, W = 64, 64, 64
defaults = dict(
    image_height=H, image_width=W, tanfovx=0.5, tanfovy=0.5,
    bg=torch.zeros(3, device=dev), scale_modifier=1.0,
    viewmatrix=torch.eye(4, device=dev), projmatrix=torch.eye(4, device=dev),
    sh_degree=0, campos=torch.zeros(3, device=dev),
    prefiltered=False, debug=False, antialiasing=False,
)
# the settings tuple gained/lost fields between releases - build it by field name
settings = GaussianRasterizationSettings(**{f: defaults[f] for f in GaussianRasterizationSettings._fields})
rasterizer = GaussianRasterizer(raster_settings=settings)

torch.manual_seed(0)
means3D = torch.randn(N, 3, device=dev) * 0.2 + torch.tensor([0., 0., 3.], device=dev)
means3D.requires_grad_(True)
means2D = torch.zeros_like(means3D, requires_grad=True)
image = rasterizer(
    means3D=means3D, means2D=means2D, shs=None,
    colors_precomp=torch.rand(N, 3, device=dev),
    opacities=torch.rand(N, 1, device=dev),
    scales=torch.full((N, 3), 0.05, device=dev),
    rotations=torch.tensor([[1., 0., 0., 0.]], device=dev).repeat(N, 1),
    cov3D_precomp=None,
)[0]
image.sum().backward()
print("rasterizer OK -", torch.cuda.get_device_name(0), "| image", tuple(image.shape),
      "| rendered sum", float(image.sum()), "| grad ok", bool(means3D.grad.abs().sum() > 0))
'''
smoke_py = os.path.join(CFG["WORK"], "smoke_test.py")
with open(smoke_py, "w") as fh:
    fh.write(SMOKE)
run([sys.executable, smoke_py], tag="smoke")
"""
)

md("## 4. Prepare images (downscale, keep EXIF)")

code(
    r"""
t = time.time()
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

SRC = sorted(p for p in glob.glob(os.path.join(CFG["INPUT_DIR"], "**", "*"), recursive=True)
             if p.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(p))
print("source images:", len(SRC), SRC[:2])
assert SRC, f"no input images found under {CFG['INPUT_DIR']}"

IMG_DIR = os.path.join(CFG["WORK"], "input_images")
os.makedirs(IMG_DIR, exist_ok=True)


def prep(path):
    dst = os.path.join(IMG_DIR, os.path.basename(path))
    if os.path.exists(dst):
        return dst
    im = Image.open(path)
    exif = im.info.get("exif")
    w = CFG["SFM_WIDTH"]
    h = round(im.size[1] * w / im.size[0])
    im = im.convert("RGB").resize((w, h), Image.LANCZOS)
    im.save(dst, quality=95, exif=exif) if exif else im.save(dst, quality=95)
    return dst


with ThreadPoolExecutor(os.cpu_count()) as ex:
    prepared = list(ex.map(prep, SRC))
W, H = Image.open(prepared[0]).size
print(f"prepared {len(prepared)} images at {W}x{H}")
REPORT["images"] = {"count": len(prepared), "width": W, "height": H,
                    "source_size": Image.open(SRC[0]).size}
stamp("prepare_images", t)
"""
)

md("## 5. Structure-from-Motion (COLMAP via pycolmap, CPU)")

code(
    r"""
t = time.time()
DB = os.path.join(CFG["WORK"], "colmap.db")
SPARSE = os.path.join(CFG["WORK"], "sparse")
os.makedirs(SPARSE, exist_ok=True)

if not os.path.exists(DB):
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    # single shared camera, focal initialised from the EXIF 35mm equivalent
    f = W * CFG["FOCAL_35MM"] / 36.0
    reader.camera_params = f"{f},{W / 2},{H / 2},0.0"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = CFG["MAX_FEATURES"]
    print("focal prior:", round(f, 1), "px")
    pycolmap.extract_features(DB, IMG_DIR, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction,
                              device=pycolmap.Device.cpu)
    stamp("sfm_extract", t)

    t = time.time()
    pycolmap.match_exhaustive(DB, device=pycolmap.Device.cpu)
    stamp("sfm_match", t)

t = time.time()
recs = pycolmap.incremental_mapping(DB, IMG_DIR, SPARSE)
stamp("sfm_map", t)

best = max(recs.values(), key=lambda r: r.num_reg_images())
print(best.summary())
REPORT["sfm"] = {
    "models": len(recs),
    "registered": best.num_reg_images(),
    "points3D": best.num_points3D(),
    "mean_reproj_error": round(best.compute_mean_reprojection_error(), 3),
    "camera": str(list(best.cameras.values())[0].params.tolist()),
}
if best.num_reg_images() < len(prepared):
    print(f"WARNING: {len(prepared) - best.num_reg_images()} images could not be registered")
assert best.num_reg_images() >= max(20, 0.4 * len(prepared)), \
    f"reconstruction too small: {best.num_reg_images()}/{len(prepared)} images"
BEST_DIR = os.path.join(SPARSE, str([k for k, v in recs.items() if v is best][0]))
"""
)

md("## 6. Undistort into a 3DGS-ready dataset")

code(
    r"""
t = time.time()
SCENE = os.path.join(CFG["WORK"], "scene")
if not os.path.isdir(os.path.join(SCENE, "images")):
    shutil.rmtree(SCENE, ignore_errors=True)
    pycolmap.undistort_images(SCENE, BEST_DIR, IMG_DIR, output_type="COLMAP")
    # 3DGS expects sparse/0/*.bin
    tmp = os.path.join(CFG["WORK"], "_sparse0")
    shutil.move(os.path.join(SCENE, "sparse"), tmp)
    os.makedirs(os.path.join(SCENE, "sparse"))
    shutil.move(tmp, os.path.join(SCENE, "sparse", "0"))
print(sorted(os.listdir(SCENE)))
print("undistorted images:", len(os.listdir(os.path.join(SCENE, "images"))))
und = pycolmap.Reconstruction(os.path.join(SCENE, "sparse", "0"))
print(und.summary())
stamp("undistort", t)
"""
)

md("## 7. Train 3D Gaussian Splatting")

code(
    r"""
t = time.time()
MODEL = os.path.join(CFG["WORK"], "model")
env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
# 16 GB fills up fast on a scene this size; expandable segments reclaim the fragmentation
# that otherwise strands several GB, and the images live in host RAM instead of VRAM.
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
rc = run([sys.executable, "train.py",
          "-s", SCENE, "-m", MODEL,
          "--iterations", CFG["ITERATIONS"],
          "--save_iterations", *[str(i) for i in CFG["SAVE_ITERATIONS"]],
          "--test_iterations", "-1",
          "--disable_viewer",
          "--data_device", "cpu",
          "--densify_grad_threshold", CFG["DENSIFY_GRAD_THRESHOLD"],
          "--densify_until_iter", CFG["DENSIFY_UNTIL_ITER"]],
         cwd=GS, env=env, tag="train", throttle=30, check=False)
stamp("train", t)
run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"], tag="gpu", check=False)

# even if training died late (OOM), export whatever the last saved iteration was
saved = sorted(int(d.split("_")[1]) for d in os.listdir(os.path.join(MODEL, "point_cloud")))
assert saved, "training produced no point cloud"
ITER = saved[-1]
if rc != 0:
    print(f"WARNING: train.py exited {rc}; falling back to iteration {ITER}")
REPORT["train"] = {"exit_code": rc, "iterations": ITER, "saved": saved}
PLY = os.path.join(MODEL, "point_cloud", f"iteration_{ITER}", "point_cloud.ply")
print(PLY, round(os.path.getsize(PLY) / 1e6, 1), "MB")
"""
)

md("## 8. Export artifacts")

code(
    r"""
t = time.time()
import numpy as np
from plyfile import PlyData

v = PlyData.read(PLY)["vertex"]
n = len(v["x"])
print("gaussians:", n)

xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
rot /= np.linalg.norm(rot, axis=1, keepdims=True) + 1e-12
opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
SH_C0 = 0.28209479177387814
rgb = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)

# importance = how much of the image a gaussian actually paints
importance = opacity * scales.prod(1)
order = np.argsort(-importance)

keep = min(n, int(CFG["WEB_TARGET_MB"] * 1024 * 1024 // 32))
sel = order[:keep]
print(f"web splat keeps {keep}/{n} gaussians ({100 * keep / n:.1f}%)")

buf = np.zeros((keep, 32), dtype=np.uint8)
buf[:, 0:12] = xyz[sel].view(np.uint8).reshape(keep, 12)
buf[:, 12:24] = scales[sel].view(np.uint8).reshape(keep, 12)
buf[:, 24:27] = np.clip(rgb[sel] * 255, 0, 255).astype(np.uint8)
buf[:, 27] = np.clip(opacity[sel] * 255, 0, 255).astype(np.uint8)
buf[:, 28:32] = np.clip(rot[sel] * 128 + 128, 0, 255).astype(np.uint8)

WEB = os.path.join(CFG["OUT"], "korno_web.splat")
buf.tofile(WEB)
FULL = os.path.join(CFG["OUT"], "korno_full.ply")
shutil.copy(PLY, FULL)
REPORT["export"] = {
    "gaussians": int(n), "web_gaussians": int(keep),
    "full_ply_mb": round(os.path.getsize(FULL) / 1e6, 1),
    "web_splat_mb": round(os.path.getsize(WEB) / 1e6, 1),
}
print(REPORT["export"])
stamp("export", t)
"""
)

code(
    r"""
# camera hints for the web viewer + the SfM archive
posed = [im for im in und.images.values() if im.has_pose]
view = np.mean([im.viewing_direction() / np.linalg.norm(im.viewing_direction()) for im in posed], 0)
view = view / np.linalg.norm(view)
up = -np.mean([im.cam_from_world().rotation.matrix()[1] for im in posed], 0)
up = up / np.linalg.norm(up)

# frame the dense core of the scene, measured along the viewer camera's own right/up axes
lo, hi = np.percentile(xyz[sel], [5, 95], axis=0)
center = (lo + hi) / 2
right = np.cross(view, up); right /= np.linalg.norm(right)
cam_up = np.cross(right, view)
half_w = float(np.abs((hi - lo) / 2 @ np.abs(right)))
half_h = float(np.abs((hi - lo) / 2 @ np.abs(cam_up)))
VFOV, ASPECT, MARGIN = np.radians(50), 1.6, 1.15
dist = MARGIN * max(half_h / np.tan(VFOV / 2), half_w / (ASPECT * np.tan(VFOV / 2)))
eye = center - view * dist
REPORT["viewer"] = {
    "cameraUp": [round(float(x), 5) for x in up],
    "initialCameraPosition": [round(float(x), 3) for x in eye],
    "initialCameraLookAt": [round(float(x), 3) for x in center],
    "sceneRadius": round(float(np.linalg.norm(hi - lo) / 2), 3),
    "splatCount": int(keep),
}
print(json.dumps(REPORT["viewer"], indent=1))

shutil.make_archive(os.path.join(CFG["OUT"], "korno_colmap"), "zip",
                    os.path.join(SCENE, "sparse"))
REPORT["total_seconds"] = round(time.time() - T0, 1)
with open(os.path.join(CFG["OUT"], "run_report.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
with open(os.path.join(CFG["OUT"], "viewer-config.json"), "w") as fh:
    json.dump(REPORT["viewer"], fh, indent=2)

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

out = pathlib.Path(__file__).with_name("korno-gaussian-splat.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(CELLS), "cells")
