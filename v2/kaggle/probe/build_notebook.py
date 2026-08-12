#!/usr/bin/env python3
"""Generates the throwaway probe notebook that validates the CUDA COLMAP toolchain.

Runs the complete MVS pipeline on a handful of images so a failure costs minutes, not hours:
does conda-forge's CUDA colmap load kernels on this GPU, and how fast is patch_match_stereo?
"""
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
# v2 probe — does CUDA COLMAP work here?

`patch_match_stereo` (COLMAP's dense stereo) is **CUDA-only**, there is no CPU fallback, so the whole v2
pipeline stands or falls on conda-forge's `colmap=*=cuda*` loading kernels on this GPU. This runs the
complete pipeline on 8 images to find out in minutes instead of hours, and measures seconds-per-image so
the real run can be sized.
"""
)

code(
    r"""
import os, sys, time, glob, subprocess, shutil

WORK = "/kaggle/temp/probe"
MAMBA = "/kaggle/temp/mm"
N_IMAGES = 8
STEREO_MAX = 1200
os.makedirs(WORK, exist_ok=True)
T0 = time.time()


def run(cmd, throttle=0.0, tag="run", check=True, env=None, shell=False):
    t0 = time.time(); last = -1e9
    print(f"\n$ {cmd if shell else ' '.join(map(str, cmd))}", flush=True)
    p = subprocess.Popen(cmd if shell else list(map(str, cmd)), shell=shell, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1, errors="replace")
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


run(["nvidia-smi"], tag="gpu", check=False)
run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv"], tag="gpu", check=False)
print("cpus", os.cpu_count())
"""
)

md("## Install CUDA COLMAP from conda-forge")

code(
    r"""
t = time.time()
if not os.path.exists(f"{MAMBA}/bin/colmap"):
    run("curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /kaggle/temp bin/micromamba",
        tag="micromamba", shell=True)
    # conda-forge's colmap 4.1.1 links libfaiss and libOpenImageIO without declaring either, and it was
    # built against faiss 1.10 - the current 1.14 has a different ABI. Pin all three explicitly.
    run(f"/kaggle/temp/bin/micromamba create -y -p {MAMBA} -c conda-forge "
        f"'colmap=4.1.1=cuda*' 'libfaiss=1.10.0=*openblas*' openimageio",
        tag="mamba-install", throttle=10, shell=True)

MM = f"/kaggle/temp/bin/micromamba run -p {MAMBA}"


# run a colmap subcommand inside the micromamba env
def cm(*args, **kw):
    return run(MM + " colmap " + " ".join(str(a) for a in args), shell=True, **kw)


# any remaining unresolved library shows up here rather than 6 hours in
run(f"{MM} bash -c 'ldd {MAMBA}/bin/colmap | grep \"not found\" || echo ALL_LIBS_RESOLVED'",
    tag="ldd", shell=True, check=False)
cm("-h", tag="colmap", check=False)
print(f"### install: {time.time() - t:.0f}s", flush=True)
"""
)

md("## Full pipeline on 8 images")

code(
    r"""
from PIL import Image
t = time.time()
SRC = sorted(p for p in glob.glob("/kaggle/input/**/*", recursive=True)
             if p.lower().endswith((".jpg", ".jpeg")) and os.path.isfile(p))
print("dataset images:", len(SRC))
assert SRC, "no images found"
# consecutive frames overlap, so take a contiguous run
pick = SRC[40:40 + N_IMAGES]
IMG = f"{WORK}/images"
os.makedirs(IMG, exist_ok=True)
for p in pick:
    dst = os.path.join(IMG, os.path.basename(p))
    if os.path.exists(dst):
        continue
    im = Image.open(p)
    exif = im.info.get("exif")
    w = 2000
    im = im.convert("RGB").resize((w, round(im.size[1] * w / im.size[0])), Image.LANCZOS)
    im.save(dst, quality=95, **({"exif": exif} if exif else {}))
print("prepared", len(os.listdir(IMG)), "at", Image.open(os.path.join(IMG, os.listdir(IMG)[0])).size)
print(f"### prepare: {time.time() - t:.0f}s", flush=True)
"""
)

code(
    r"""
DB = f"{WORK}/db.db"
SPARSE = f"{WORK}/sparse"
DENSE = f"{WORK}/dense"
os.makedirs(SPARSE, exist_ok=True)

t = time.time()
cm("feature_extractor", "--database_path", DB, "--image_path", IMG,
   "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "SIMPLE_RADIAL",
   "--FeatureExtraction.use_gpu", 1, tag="extract", throttle=5)
print(f"### extract: {time.time() - t:.0f}s   <-- proves GPU SIFT", flush=True)

t = time.time()
cm("exhaustive_matcher", "--database_path", DB, "--FeatureMatching.use_gpu", 1, tag="match", throttle=5)
print(f"### match: {time.time() - t:.0f}s", flush=True)

t = time.time()
cm("mapper", "--database_path", DB, "--image_path", IMG, "--output_path", SPARSE,
   tag="mapper", throttle=10)
print(f"### mapper: {time.time() - t:.0f}s", flush=True)
run(["ls", "-la", f"{SPARSE}/0"], tag="ls", check=False)
"""
)

code(
    r"""
t = time.time()
cm("image_undistorter", "--image_path", IMG, "--input_path", f"{SPARSE}/0",
   "--output_path", DENSE, "--output_type", "COLMAP", "--max_image_size", STEREO_MAX,
   tag="undistort", throttle=5)
print(f"### undistort: {time.time() - t:.0f}s", flush=True)

# the moment of truth: CUDA-only dense stereo
t = time.time()
rc = cm("patch_match_stereo", "--workspace_path", DENSE,
        "--PatchMatchStereo.max_image_size", STEREO_MAX,
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.num_iterations", 5,
        tag="stereo", throttle=15, check=False)
dt = time.time() - t
n_img = len(os.listdir(f"{DENSE}/images"))
print(f"### patch_match_stereo: exit={rc} {dt:.0f}s for {n_img} images "
      f"= {dt / max(n_img, 1):.1f}s/image at {STEREO_MAX}px", flush=True)
assert rc == 0, "CUDA dense stereo failed on this GPU"
"""
)

code(
    r"""
t = time.time()
cm("stereo_fusion", "--workspace_path", DENSE, "--workspace_format", "COLMAP",
   "--input_type", "geometric", "--output_path", f"{WORK}/fused.ply", tag="fusion", throttle=10)
size = os.path.getsize(f"{WORK}/fused.ply")
print(f"### fusion: {time.time() - t:.0f}s -> fused.ply {size / 1e6:.1f} MB", flush=True)

from plyfile import PlyData
pts = PlyData.read(f"{WORK}/fused.ply")["vertex"]
print("dense points:", len(pts["x"]), "| properties:", [p.name for p in pts.properties])
print(f"\nTOTAL {time.time() - T0:.0f}s")
print("\nEXTRAPOLATION for 102 images:")
print(f"  patch_match_stereo at {STEREO_MAX}px: {102 * dt / max(n_img,1) / 60:.0f} min")
print(f"  ... at 2000px (~2.8x pixels): {102 * dt / max(n_img,1) * 2.8 / 60:.0f} min")
"""
)


nb = {
    "cells": CELLS,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11.13"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = pathlib.Path(__file__).with_name("korno-v2-probe.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(CELLS), "cells")
