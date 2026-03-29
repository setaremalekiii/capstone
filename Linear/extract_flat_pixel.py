#!/usr/bin/env python3
"""
===========================
Option 1 — Raw flattened pixels as features.

Resizes every labeled chromosome image to IMG_SIZE x IMG_SIZE,
flattens to a 1D vector, and saves to features.npz.

Usage
-----
python extract_features_pixels.py \
    --in_dir  /path/to/preprocessed_train \
    --labels  labels.json \
    --out_npz pixel_features.npz \
    [--img_size 64]

Output
------
pixel_features.npz:
    X      : (N, img_size*img_size) float32  — flattened grayscale pixels
    labels : (N,)                  int8
    keys   : (N,)                  object
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def pad_to_square(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    if h == w:
        return gray
    diff = abs(h - w)
    if h < w:
        pad_top = diff // 2
        gray = cv2.copyMakeBorder(gray, pad_top, diff - pad_top, 0, 0,
                                   cv2.BORDER_CONSTANT, value=255)
    else:
        pad_left = diff // 2
        gray = cv2.copyMakeBorder(gray, 0, 0, pad_left, diff - pad_left,
                                   cv2.BORDER_CONSTANT, value=255)
    return gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",   required=True)
    ap.add_argument("--labels",   required=True)
    ap.add_argument("--out_npz",  default="pixel_features.npz")
    ap.add_argument("--img_size", type=int, default=64)
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_npz = Path(args.out_npz)
    size    = args.img_size

    with open(args.labels, "r") as f:
        raw_labels = json.load(f)
    print(f"[INFO] {len(raw_labels)} labeled entries | img_size={size}x{size}")

    X_list, lbl_list, key_list = [], [], []
    skipped = 0

    for rel_key, flip_label in raw_labels.items():
        img_path = str(in_dir / rel_key)
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"[SKIP] {rel_key}: unreadable")
            skipped += 1
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = pad_to_square(gray)
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

        x = gray.astype(np.float32).flatten() / 255.0   # (size*size,)
        X_list.append(x)
        lbl_list.append(int(flip_label))
        key_list.append(rel_key)

    if not X_list:
        print("[ERROR] No features extracted.")
        sys.exit(1)

    X      = np.stack(X_list,  axis=0)
    labels = np.array(lbl_list, dtype=np.int8)
    keys   = np.array(key_list, dtype=object)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_npz), X=X, labels=labels, keys=keys)

    print(f"[DONE] {len(X)} samples  (skipped {skipped})")
    print(f"       feature_dim={X.shape[1]}")
    print(f"       keep(0)={int((labels==0).sum())}  flip(1)={int((labels==1).sum())}")
    print(f"       Saved → {out_npz}")


if __name__ == "__main__":
    main()