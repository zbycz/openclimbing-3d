#!/usr/bin/env python3
"""Generates v2/kaggle/bolts/korno-bolts-infer.ipynb — YOLO bolt detection on every photo."""
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
# Korno — bolt detection on every photo

Runs [openclimbing-bolts-ai](https://github.com/zbycz/openclimbing-bolts-ai)'s YOLOv8-nano ONNX model over
all 103 drone photos and writes one JSON per photo.

Bolts are only a few pixels across, so the model is fed **native-resolution 1024 px tiles** rather than a
downscaled whole image — 60 tiles per 8000×4500 photo, overlapping by 20 % so a bolt on a seam is still
seen whole. Detections are merged back with NMS in full-image coordinates.

The threshold is deliberately loose (0.25). Filtering is better done later, when a detection can be checked
against the other views that see the same piece of rock.

Output: `bolts/<photo>.json` with normalised boxes, plus a combined `bolts_all.json`.
"""
)

code(
    r"""
import os, sys, json, time, glob, subprocess, urllib.request

CFG = dict(
    INPUT_DIR = "/kaggle/input",
    OUT       = "/kaggle/working",
    MODEL_URL = "https://github.com/zbycz/openclimbing-bolts-ai/raw/main/models/openclimbing-bolts-v1.onnx",

    TILE    = 1024,     # the model's fixed input size
    OVERLAP = 0.20,
    CONF    = 0.25,
    NMS_IOU = 0.45,
)
os.makedirs(os.path.join(CFG["OUT"], "bolts"), exist_ok=True)
T0 = time.time()
print("cpus", os.cpu_count())
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"])

import numpy as np
import onnxruntime as ort
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

MODEL = "/kaggle/temp/bolts.onnx"
if not os.path.exists(MODEL):
    os.makedirs("/kaggle/temp", exist_ok=True)
    urllib.request.urlretrieve(CFG["MODEL_URL"], MODEL)
print("model", round(os.path.getsize(MODEL) / 1e6, 1), "MB")

opts = ort.SessionOptions()
opts.intra_op_num_threads = os.cpu_count()
sess = ort.InferenceSession(MODEL, opts, providers=["CPUExecutionProvider"])
IN_NAME = sess.get_inputs()[0].name
OUT_NAME = sess.get_outputs()[0].name
print("input", sess.get_inputs()[0].shape, "output", sess.get_outputs()[0].shape)
print("classes", sess.get_modelmeta().custom_metadata_map.get("names"))
"""
)

code(
    r"""
TILE = CFG["TILE"]
STRIDE = round(TILE * (1 - CFG["OVERLAP"]))


def tile_pos(size):
    if size <= TILE:
        return [0]
    pos = list(range(0, size - TILE + 1, STRIDE))
    if pos[-1] != size - TILE:
        pos.append(size - TILE)
    return pos


def nms(boxes, thr):
    # boxes: (cx, cy, w, h, score) in pixels, highest score wins
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep, dead = [], [False] * len(boxes)
    for i, a in enumerate(boxes):
        if dead[i]:
            continue
        keep.append(a)
        ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
        for j in range(i + 1, len(boxes)):
            if dead[j]:
                continue
            b = boxes[j]
            bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
            iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            ih = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter = iw * ih
            union = a[2] * a[3] + b[2] * b[3] - inter
            if union > 0 and inter / union > thr:
                dead[j] = True
    return keep


def detect(path):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    raw = []
    for ty in tile_pos(H):
        for tx in tile_pos(W):
            tw, th = min(TILE, W - tx), min(TILE, H - ty)
            # pad short edge tiles with grey rather than rescaling - the model expects 1:1 pixels
            canvas = Image.new("RGB", (TILE, TILE), (128, 128, 128))
            canvas.paste(img.crop((tx, ty, tx + tw, ty + th)), (0, 0))
            tensor = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
            out = sess.run([OUT_NAME], {IN_NAME: tensor})[0][0]      # (5, N)
            hit = out[4] >= CFG["CONF"]
            for cx, cy, bw, bh, sc in out[:, hit].T:
                if cx > tw or cy > th:      # detected in the grey padding
                    continue
                raw.append((tx + float(cx), ty + float(cy), float(bw), float(bh), float(sc)))
    img.close()
    return nms(raw, CFG["NMS_IOU"]), W, H
"""
)

code(
    r"""
PHOTOS = sorted(p for p in glob.glob(os.path.join(CFG["INPUT_DIR"], "**", "*"), recursive=True)
                if p.lower().endswith((".jpg", ".jpeg")) and os.path.isfile(p))
print("photos:", len(PHOTOS))
assert PHOTOS, "no photos found"

summary, total = [], 0
t0 = time.time()
for i, path in enumerate(PHOTOS):
    name = os.path.basename(path)
    boxes, W, H = detect(path)
    total += len(boxes)
    rec = {
        "image": name, "width": W, "height": H,
        "model": "openclimbing-bolts-v1",
        "conf_threshold": CFG["CONF"],
        # normalised so the JSON survives any later rescaling of the photo
        "detections": [{"cx": round(b[0] / W, 6), "cy": round(b[1] / H, 6),
                        "w": round(b[2] / W, 6), "h": round(b[3] / H, 6),
                        "score": round(b[4], 4)} for b in boxes],
    }
    with open(os.path.join(CFG["OUT"], "bolts", name + ".json"), "w") as fh:
        json.dump(rec, fh)
    summary.append(rec)
    done = i + 1
    if done % 5 == 0 or done == len(PHOTOS):
        rate = (time.time() - t0) / done
        print(f"  [{done}/{len(PHOTOS)}] {name} -> {len(boxes)} bolts "
              f"(total {total}, {rate:.1f}s/photo, eta {rate * (len(PHOTOS) - done) / 60:.0f} min)",
              flush=True)

with open(os.path.join(CFG["OUT"], "bolts_all.json"), "w") as fh:
    json.dump({"model": "openclimbing-bolts-v1", "conf_threshold": CFG["CONF"],
               "tile": TILE, "overlap": CFG["OVERLAP"], "nms_iou": CFG["NMS_IOU"],
               "photos": summary}, fh)

counts = sorted(len(r["detections"]) for r in summary)
print(f"\n{total} detections in {len(PHOTOS)} photos")
print(f"per photo: min {counts[0]}, median {counts[len(counts) // 2]}, max {counts[-1]}")
print(f"photos with none: {sum(1 for c in counts if c == 0)}")
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
out = pathlib.Path(__file__).with_name("korno-bolts-infer.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(CELLS), "cells")
