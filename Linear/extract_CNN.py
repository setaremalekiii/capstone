#!/usr/bin/env python3
"""
extract_features_cnn.py
========================
Option 3 — Pretrained CNN (ResNet50) feature extraction.

Passes each chromosome image through a frozen pretrained ResNet50
(trained on ImageNet) and saves the 2048-dim embedding from the
global average pooling layer as features.

No training of the CNN at all — weights are frozen and downloaded
automatically from torchvision.

Usage
-----
python extract_features_cnn.py \
    --in_dir  /path/to/preprocessed_train \
    --labels  labels.json \
    --out_npz cnn_features.npz \
    [--backbone resnet50]   # or resnet18, efficientnet_b0

Output
------
cnn_features.npz:
    X      : (N, feature_dim) float32   — CNN embeddings
    labels : (N,)             int8
    keys   : (N,)             object
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


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


def build_backbone(name: str) -> tuple:
    """
    Returns (model, feature_dim) with the classification head removed.
    Model outputs a flat feature vector per image.
    """
    name = name.lower()

    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        dim = m.fc.in_features
        m.fc = nn.Identity()

    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        dim = m.fc.in_features
        m.fc = nn.Identity()

    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        dim = m.classifier[1].in_features
        m.classifier = nn.Identity()

    else:
        raise ValueError(f"Unknown backbone: {name}. Choose resnet18, resnet50, or efficientnet_b0.")

    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m, dim


# ImageNet normalization — required for pretrained torchvision models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def image_to_rgb_tensor(img_path: str) -> torch.Tensor:
    """Load grayscale chromosome, pad to square, convert to 3-channel RGB tensor."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise RuntimeError(f"Could not read: {img_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = pad_to_square(gray)
    # Convert grayscale → RGB by stacking 3 channels
    rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return transform(rgb)   # (3, 224, 224)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",   required=True)
    ap.add_argument("--labels",   required=True)
    ap.add_argument("--out_npz",  default="cnn_features.npz")
    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    in_dir  = Path(args.in_dir)
    out_npz = Path(args.out_npz)

    with open(args.labels, "r") as f:
        raw_labels = json.load(f)
    print(f"[INFO] {len(raw_labels)} labeled entries")

    print(f"[INFO] Loading pretrained {args.backbone} (frozen) ...")
    model, feat_dim = build_backbone(args.backbone)
    model.to(device)
    print(f"[INFO] Feature dim = {feat_dim} | device = {device}")

    # ── load all images ───────────────────────────────────────────────────────
    tensors, lbl_list, key_list = [], [], []
    skipped = 0

    for rel_key, flip_label in raw_labels.items():
        img_path = str(in_dir / rel_key)
        try:
            t = image_to_rgb_tensor(img_path)
        except RuntimeError as e:
            print(f"[SKIP] {rel_key}: {e}")
            skipped += 1
            continue

        tensors.append(t)
        lbl_list.append(int(flip_label))
        key_list.append(rel_key)

    if not tensors:
        print("[ERROR] No images loaded.")
        sys.exit(1)

    # ── extract features in batches ───────────────────────────────────────────
    print(f"[INFO] Extracting features from {len(tensors)} images in batches of {args.batch_size} ...")
    X_list = []
    bs = args.batch_size

    with torch.no_grad():
        for i in range(0, len(tensors), bs):
            batch = torch.stack(tensors[i:i+bs]).to(device)   # (B, 3, 224, 224)
            feats = model(batch)                                # (B, feat_dim)
            X_list.append(feats.cpu().numpy())
            if (i // bs) % 5 == 0:
                print(f"  batch {i//bs + 1}/{(len(tensors)-1)//bs + 1}")

    X      = np.concatenate(X_list, axis=0).astype(np.float32)  # (N, feat_dim)
    labels = np.array(lbl_list, dtype=np.int8)
    keys   = np.array(key_list, dtype=object)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_npz), X=X, labels=labels, keys=keys)

    print(f"\n[DONE] {len(X)} samples  (skipped {skipped})")
    print(f"       backbone={args.backbone}  feature_dim={feat_dim}")
    print(f"       keep(0)={int((labels==0).sum())}  flip(1)={int((labels==1).sum())}")
    print(f"       Saved → {out_npz}")


if __name__ == "__main__":
    main()