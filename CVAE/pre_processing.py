"""
ChromosomeDataset with preprocessing tuned for latent-factor discovery (angle + p/q morphology)

Key changes vs your version:
- Removed the w>h auto-rotate (so pose/angle variation is preserved instead of partially normalized)
- Disabled random horizontal flip by default (flips can scramble angle sign); you can enable it and we’ll track it
- Padding value defaults to 0 (often closer to microscopy background than 114)
- Optional: returns a simple estimated orientation angle (PCA on foreground mask) to help you probe which z dims encode angle
- More robust label indexing (handles class1..class24 vs class0..class23)
"""

import os
import re
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision.utils import save_image


class ChromosomeDataset(Dataset):
    def __init__(
        self,
        img_paths,
        target_size=(64, 64),
        transform=False,
        num_classes=24,
        # preprocessing toggles
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid=(8, 8),
        use_sharpen=True,
        # augmentation toggles (default: do NOT augment away angle)
        aug_flip=False,
        aug_brightness=True,
        brightness_range=(0.8, 1.2),
        # letterbox
        pad_value=0,
        # outputs
        return_meta=False,
        compute_angle_meta=True,
    ):
        self.img_paths = list(img_paths)
        self.target_size = tuple(target_size)
        self.transform = transform
        self.num_classes = num_classes

        self.use_clahe = use_clahe
        self.clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=tuple(clahe_tile_grid),
        )
        self.use_sharpen = use_sharpen

        self.aug_flip = aug_flip
        self.aug_brightness = aug_brightness
        self.brightness_range = brightness_range

        self.pad_value = pad_value

        self.return_meta = return_meta
        self.compute_angle_meta = compute_angle_meta

        self.to_tensor = ToTensor()

        raw_labels = self.extract_labels(self.img_paths)
        self.labels = self._normalize_label_ids(raw_labels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        meta = {}

        # ---- Optional augmentation (kept minimal so angle remains learnable) ----
        if self.transform:
            img, aug_meta = self.random_augment(img)
            meta.update(aug_meta)

        # ---- Contrast/sharpness (can help, but keep toggles for ablation) ----
        if self.use_clahe:
            img = self.clahe.apply(img)

        if self.use_sharpen:
            gaussian_3 = cv2.GaussianBlur(img, (3, 3), 0.5)
            img = cv2.addWeighted(img, 2.0, gaussian_3, -1.0, 0)

        # ---- Meta: estimated orientation angle (useful for probing latent dims) ----
        if self.return_meta and self.compute_angle_meta:
            meta["angle_deg_est"] = float(self.estimate_angle_deg(img))

        # ---- Letterbox resize + pad (NO auto-rotate; preserve pose) ----
        img_letterbox = self.apply_letterbox(img, pad_value=self.pad_value)

        img_tensor = self.to_tensor(img_letterbox)  # [1,H,W], float in [0,1]

        # ---- One-hot label ----
        label = int(self.labels[idx])
        label_t = torch.tensor(label, dtype=torch.long)
        one_hot = F.one_hot(label_t, num_classes=self.num_classes).float()

        if self.return_meta:
            meta["path"] = path
            meta["label_id"] = label
            return img_tensor, one_hot, meta

        return img_tensor, one_hot

    # -------------------- Augmentation --------------------
    def random_augment(self, img):
        meta = {"flip_applied": False, "brightness_alpha": 1.0}

        if self.aug_flip and (torch.rand(1).item() < 0.5):
            img = cv2.flip(img, 1)
            meta["flip_applied"] = True

        if self.aug_brightness:
            lo, hi = self.brightness_range
            alpha = lo + (hi - lo) * torch.rand(1).item()
            img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            meta["brightness_alpha"] = float(alpha)

        return img, meta

    # -------------------- Letterbox --------------------
    def apply_letterbox(self, img, pad_value=0):
        h, w = img.shape[:2]
        target_w, target_h = self.target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=int(pad_value),
        )
        return padded

    # -------------------- Angle estimation (PCA on foreground mask) --------------------
    @staticmethod
    def estimate_angle_deg(img_uint8):
        """
        Returns an estimated major-axis orientation angle in degrees.
        Uses Otsu threshold to get foreground, then PCA on (x,y) coords.
        Angle is in [-90, 90] approximately (symmetry means 180° flips are equivalent).
        """
        # Otsu threshold (foreground as white)
        _, bw = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ys, xs = np.nonzero(bw)
        if len(xs) < 50:
            return 0.0  # too few pixels; fallback

        coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        coords -= coords.mean(axis=0, keepdims=True)

        cov = (coords.T @ coords) / max(len(coords) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, np.argmax(eigvals)]  # principal direction (x,y)

        angle = np.degrees(np.arctan2(v[1], v[0]))  # atan2(y, x)
        # map to [-90, 90]
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
        return angle

    # -------------------- Labels --------------------
    @staticmethod
    def extract_labels(img_paths):
        labels = []
        for path in img_paths:
            filename = os.path.basename(path)
            match = re.search(r"class(\d+)", filename)
            if not match:
                raise ValueError(f"Could not extract class_id from {path}")
            labels.append(int(match.group(1)))
        return labels

    def _normalize_label_ids(self, labels):
        """
        Handles common cases:
        - class0..class23  -> keep
        - class1..class24  -> shift down by 1
        """
        mn, mx = min(labels), max(labels)

        if mn == 0 and mx == self.num_classes - 1:
            return labels

        if mn == 1 and mx == self.num_classes:
            return [x - 1 for x in labels]

        raise ValueError(
            f"Unexpected label range: min={mn}, max={mx}. "
            f"Expected 0..{self.num_classes-1} or 1..{self.num_classes}."
        )


if __name__ == "__main__":
    MAIN_DIR = "/scratch/st-li1210-1/pearl/karyotype-detector/"
    cropped_box_path = os.path.join(MAIN_DIR, "data", "cropped_v2")
    train_dir = os.path.join(cropped_box_path, "train")
    paths = glob.glob(os.path.join(train_dir, "*.jpg"))

    ds = ChromosomeDataset(
        paths,
        target_size=(64, 64),
        transform=True,
        # important defaults for your goal:
        aug_flip=False,          # keep angle consistent
        aug_brightness=True,     # ok
        use_clahe=True,
        use_sharpen=True,
        pad_value=0,             # background-like padding
        return_meta=True,        # so you can log angle estimate
        compute_angle_meta=True,
    )

    x, y, meta = ds[3]
    print(meta)

    out_path = os.path.join(MAIN_DIR, "debug_sample.png")
    save_image(x, out_path, normalize=True)
    print("Saved:", out_path)
