#!/usr/bin/env python3
"""
extract_features_pca.py
========================
Option 2 — PCA on flattened pixels.

Resizes images, flattens them, fits PCA to reduce dimensionality,
then saves the PCA-projected features to features.npz.

This is the most consistent with your CFP (which proposed PCA as the
baseline method) and is interpretable — the top PCA components can
be visualized as "eigenfaces" of chromosomes.

Usage
-----
python extract_features_pca.py \
    --in_dir    /path/to/preprocessed_train \
    --labels    labels.json \
    --out_npz   pca_features.npz \
    --out_pca   pca_model.npz \
    [--img_size 64] \
    [--n_components 32]

Output
------
pca_features.npz:
    X      : (N, n_components) float32
    labels : (N,)              int8
    keys   : (N,)              object

pca_model.npz:
    components  : (n_components, img_size*img_size)
    mean        : (img_size*img_size,)
    n_components: scalar
    img_size    : scalar
    explained_variance_ratio: (n_components,)
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


def fit_pca(X: np.ndarray, n_components: int):
    """Minimal PCA using SVD — no sklearn needed."""
    mean = X.mean(axis=0)
    Xc   = X - mean
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]               # (n_components, D)
    X_proj     = Xc @ components.T               # (N, n_components)

    total_var  = (S ** 2).sum()
    var_ratio  = (S[:n_components] ** 2) / total_var

    return X_proj.astype(np.float32), mean, components, var_ratio


def project_pca(X: np.ndarray, mean: np.ndarray, components: np.ndarray):
    return ((X - mean) @ components.T).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",        required=True)
    ap.add_argument("--labels",        required=True)
    ap.add_argument("--out_npz",       default="pca_features.npz")
    ap.add_argument("--out_pca",       default="pca_model.npz")
    ap.add_argument("--img_size",      type=int, default=64)
    ap.add_argument("--n_components",  type=int, default=32)
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    size    = args.img_size
    n_comp  = args.n_components

    with open(args.labels, "r") as f:
        raw_labels = json.load(f)
    print(f"[INFO] {len(raw_labels)} labeled entries | img_size={size} | n_components={n_comp}")

    raw_X, lbl_list, key_list = [], [], []
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

        raw_X.append(gray.astype(np.float32).flatten() / 255.0)
        lbl_list.append(int(flip_label))
        key_list.append(rel_key)

    if not raw_X:
        print("[ERROR] No features extracted.")
        sys.exit(1)

    X_raw  = np.stack(raw_X, axis=0)       # (N, size*size)
    labels = np.array(lbl_list, dtype=np.int8)
    keys   = np.array(key_list, dtype=object)

    print(f"[INFO] Fitting PCA on {len(X_raw)} samples, raw_dim={X_raw.shape[1]} ...")
    X_pca, pca_mean, pca_components, var_ratio = fit_pca(X_raw, n_comp)

    print(f"[INFO] Explained variance by component:")
    cumvar = 0.0
    for i, v in enumerate(var_ratio):
        cumvar += v
        print(f"       PC{i+1:02d}: {v*100:5.2f}%  (cumulative {cumvar*100:5.2f}%)")

    # save features
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_npz), X=X_pca, labels=labels, keys=keys)

    # save PCA model for use at inference
    out_pca = Path(args.out_pca)
    np.savez(
        str(out_pca),
        components=pca_components.astype(np.float32),
        mean=pca_mean.astype(np.float32),
        n_components=np.array([n_comp]),
        img_size=np.array([size]),
        explained_variance_ratio=var_ratio.astype(np.float32),
    )

    print(f"\n[DONE] {len(X_pca)} samples  (skipped {skipped})")
    print(f"       feature_dim={X_pca.shape[1]}")
    print(f"       keep(0)={int((labels==0).sum())}  flip(1)={int((labels==1).sum())}")
    print(f"       Features → {out_npz}")
    print(f"       PCA model → {out_pca}")


if __name__ == "__main__":
    main()