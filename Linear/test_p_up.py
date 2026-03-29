#!/usr/bin/env python3
"""
p_up_inference.py
=================
Unified inference script — works with orientation models trained from
ANY of the 3 feature options (pixels, PCA, CNN).

Pass --mode to select which feature extractor to use at inference.

Usage
-----
# Option 1 — pixels
python p_up_inference.py \
    --mode     pixels \
    --in_dir   /path/to/images \
    --out_dir  /path/to/output \
    --model    orientation_model.npz \
    [--img_size 64]

# Option 2 — PCA
python p_up_inference.py \
    --mode     pca \
    --in_dir   /path/to/images \
    --out_dir  /path/to/output \
    --model    orientation_model.npz \
    --pca      pca_model.npz

# Option 3 — CNN
python p_up_inference.py \
    --mode     cnn \
    --in_dir   /path/to/images \
    --out_dir  /path/to/output \
    --model    orientation_model.npz \
    [--backbone resnet50]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

EXTS      = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FLIP_AXIS = 0   # top↔bottom


# ── shared utils ─────────────────────────────────────────────────────────── #

def pad_to_square(gray):
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


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def load_model(model_path):
    d = np.load(model_path)
    return d["W"].astype(np.float64), float(d["b"][0])


def decide(x_feat, W, b, threshold):
    logit     = float(np.dot(x_feat, W) + b)
    flip_prob = sigmoid(logit)
    return flip_prob > threshold, flip_prob


def safe_imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, img):
        raise RuntimeError(f"cv2.imwrite failed: {path}")


# ── feature extractors ────────────────────────────────────────────────────── #

def extract_pixels(img_path, img_size):
    bgr  = cv2.imread(img_path)
    if bgr is None:
        raise RuntimeError(f"Could not read: {img_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = pad_to_square(gray)
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float64).flatten() / 255.0


def extract_pca(img_path, img_size, pca_mean, pca_components):
    x_raw = extract_pixels(img_path, img_size)
    return ((x_raw - pca_mean) @ pca_components.T).astype(np.float64)


def build_cnn_extractor(backbone_name, device):
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T

    name = backbone_name.lower()
    if name == "resnet18":
        m   = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Identity()
    elif name == "resnet50":
        m   = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Identity()
    elif name == "efficientnet_b0":
        m   = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    for p in m.parameters():
        p.requires_grad = False
    m.eval().to(device)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract(img_path):
        import torch
        bgr  = cv2.imread(img_path)
        if bgr is None:
            raise RuntimeError(f"Could not read: {img_path}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = pad_to_square(gray)
        rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        t    = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = m(t).squeeze().cpu().numpy()
        return feat.astype(np.float64)

    return extract


# ─────────────────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",      required=True, choices=["pixels", "pca", "cnn"])
    ap.add_argument("--in_dir",    required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--model",     required=True, help="orientation_model.npz")
    ap.add_argument("--pca",       default=None,  help="pca_model.npz (required for --mode pca)")
    ap.add_argument("--backbone",  default="resnet50",
                    choices=["resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--img_size",  type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    W, b = load_model(args.model)
    print(f"[INFO] Model loaded | mode={args.mode} | threshold={args.threshold}")

    # ── build feature extractor ───────────────────────────────────────────────
    if args.mode == "pixels":
        size = args.img_size
        def get_features(p):
            return extract_pixels(str(p), size)

    elif args.mode == "pca":
        if args.pca is None:
            print("[ERROR] --pca pca_model.npz is required for --mode pca")
            sys.exit(1)
        pca_data   = np.load(args.pca)
        pca_mean   = pca_data["mean"].astype(np.float64)
        pca_comps  = pca_data["components"].astype(np.float64)
        size       = int(pca_data["img_size"][0])
        print(f"[INFO] PCA loaded | img_size={size} | n_components={pca_comps.shape[0]}")
        def get_features(p):
            return extract_pca(str(p), size, pca_mean, pca_comps)

    elif args.mode == "cnn":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading pretrained {args.backbone} | device={device}")
        cnn_extract = build_cnn_extractor(args.backbone, device)
        def get_features(p):
            return cnn_extract(str(p))

    # ── collect files ─────────────────────────────────────────────────────────
    if args.recursive:
        files = sorted(p for p in in_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in EXTS)
    else:
        files = sorted(p for p in in_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in EXTS)

    if not files:
        print(f"[ERROR] No images found in {in_dir}")
        sys.exit(1)

    print(f"[INFO] Processing {len(files)} images ...\n")
    wrote = skipped = flipped_count = 0

    for p in files:
        try:
            rel = p.relative_to(in_dir).as_posix()
        except ValueError:
            rel = p.name

        orig_bgr = cv2.imread(str(p))
        if orig_bgr is None:
            print(f"[SKIP] {rel}: unreadable")
            skipped += 1
            continue

        try:
            feat = get_features(p)
        except Exception as e:
            print(f"[SKIP] {rel}: feature extraction failed → {e}")
            skipped += 1
            continue

        do_flip, flip_prob = decide(feat, W, b, args.threshold)

        final_bgr = cv2.flip(orig_bgr, FLIP_AXIS) if do_flip else orig_bgr.copy()
        if do_flip:
            flipped_count += 1

        # preserve subdirectory structure
        try:
            out_rel = p.relative_to(in_dir)
        except ValueError:
            out_rel = Path(p.name)
        out_path = out_dir / out_rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            safe_imwrite(str(out_path), final_bgr)
        except Exception as e:
            print(f"[FAIL] {rel}: {e}")
            skipped += 1
            continue

        wrote += 1

        meta = {
            "file":      rel,
            "mode":      args.mode,
            "flip_prob": round(flip_prob, 6),
            "threshold": args.threshold,
            "flipped":   bool(do_flip),
            "flip_axis": FLIP_AXIS,
            "output":    str(out_path),
        }
        meta_path = out_path.with_name(out_path.stem + "__meta.json")
        with open(str(meta_path), "w") as f:
            json.dump(meta, f, indent=2)

        status = "FLIPPED" if do_flip else "kept   "
        print(f"[OK] {rel:55s}  p_flip={flip_prob:.3f}  {status}")

    print(f"\n{'─'*60}")
    print(f"Done.  {wrote} written  |  {flipped_count} flipped  |  {skipped} skipped")
    print(f"Output → {out_dir}")


if __name__ == "__main__":
    main()