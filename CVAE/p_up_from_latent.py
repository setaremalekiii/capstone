import os
import json
import math
from pathlib import Path
import glob
import re

import cv2
import numpy as np
import torch

# ---------- USER CONFIG ----------
# Input original (unprobed) chromosome crops
IN_DIR = r"/scratch/st-li1210-1/pearl/karyotype-detector/data/preprocessed_train"

# Output: flipped originals saved here
OUT_DIR = r"/arc/project/st-li1210-1/pearl/karyotype-detector/models/CVAE/output_p_up"

# Output: 3 probed recon images saved per chromosome here (separate folder per image)
RECON_OUT_DIR = r"/arc/project/st-li1210-1/pearl/karyotype-detector/models/CVAE/probe_recons_by_image"

# Your probe map file (JSON). See expected format below.
PROBE_MAP_PATH = r"/arc/project/st-li1210-1/pearl/karyotype-detector/models/CVAE/probe_map.json"

# Model files
WEIGHTS_PATH = r"/arc/project/st-li1210-1/pearl/karyotype-detector/models/CVAE/best_32.pth"
IMG_SIZE = (64, 64)         # must match your CVAE training
LATENT_DIM = 32       # must match your CVAE training

# Flip axis on ORIGINAL:
# 0 = vertical (top/bottom)  <-- usually correct for "p arm up"
# 1 = horizontal (left/right)
FLIP_AXIS = 0

# Save debug overlays
SAVE_DEBUG_TEXT = True

# Image extensions
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Torch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- EXPECTED probe_map.json FORMAT ----------
"""
Either per-image:
{
  "103064_chrom12": [{"dim": 7, "value": -2.0}, {"dim": 12, "value": 1.5}, {"dim": 19, "value": 2.0}],
  "103065_chrom03": [{"dim": 7, "value": -2.0}, {"dim": 12, "value": 1.5}, {"dim": 19, "value": 2.0}]
}

Or global default + optional overrides:
{
  "_default": [{"dim": 7, "value": -2.0}, {"dim": 12, "value": 1.5}, {"dim": 19, "value": 2.0}],
  "103064_chrom12": [{"dim": 7, "value": -2.5}, {"dim": 12, "value": 1.0}, {"dim": 19, "value": 2.0}]
}
"""


# ---------- IMPORT YOUR MODEL ----------
# Adjust these imports to match your repo.
# This assumes you have ConvCVAE.py with class ConvCVAE inside.
from ConvCVAE import ConvCVAE


# ---------- MASK + P-UP DETECTION (from earlier) ----------
def largest_component(th_255: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(th_255, connectivity=8)
    if num <= 1:
        return np.zeros(th_255.shape, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (labels == idx).astype(np.uint8)
def chrom_key_from_base(base: str) -> str:
    m = re.search(r"class(\d+)", base)
    if not m:
        raise ValueError(f"Could not find 'class#' in filename base: {base}")
    cls = int(m.group(1))
    if cls == 23:           # change if X is a different class id
        return "chromX"
    return f"chrom{cls}"

def segment_chromosome_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, th     = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def clean(thr):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)
        return thr

    th_inv = clean(th_inv)
    th = clean(th)

    m1 = largest_component(th_inv)
    m2 = largest_component(th)

    h, w = gray.shape
    a1, a2 = m1.sum(), m2.sum()

    def ok(a):
        frac = a / float(h * w)
        return 0.005 < frac < 0.6

    if ok(a1) and not ok(a2): return m1
    if ok(a2) and not ok(a1): return m2
    return m1 if a1 >= a2 else m2

def crop_to_mask(bgr: np.ndarray, mask01: np.ndarray, pad: int = 5):
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return None, None
    h, w = mask01.shape
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(y0 - pad, 0); x0 = max(x0 - pad, 0)
    y1 = min(y1 + pad, h - 1); x1 = min(x1 + pad, w - 1)
    return bgr[y0:y1+1, x0:x1+1], mask01[y0:y1+1, x0:x1+1]

def estimate_p_is_top(mask01: np.ndarray) -> bool:
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return True

    y_min, y_max = ys.min(), ys.max()
    height = (y_max - y_min + 1)

    widths = np.full(height, np.inf, dtype=np.float32)
    for y in range(y_min, y_max + 1):
        row_x = xs[ys == y]
        if row_x.size > 0:
            widths[y - y_min] = float(row_x.max() - row_x.min())

    lo = int(0.2 * height)
    hi = int(0.8 * height)
    lo = max(lo, 0); hi = max(hi, lo + 1)

    mid = widths[lo:hi]
    if np.all(np.isinf(mid)):
        k = int(np.argmin(widths))
    else:
        k = lo + int(np.argmin(mid))

    y_c = y_min + k
    top_extent = y_c - y_min
    bottom_extent = y_max - y_c
    return top_extent < bottom_extent

def p_up_from_bgr(bgr: np.ndarray) -> tuple[bool, bool]:
    """Returns (ok, p_is_top)."""
    mask01 = segment_chromosome_mask(bgr)
    cropped, cropped_mask = crop_to_mask(bgr, mask01, pad=5)
    if cropped is None:
        return False, True
    return True, estimate_p_is_top(cropped_mask)

def majority_vote_3(v0: bool, v1: bool, v2: bool) -> bool:
    return (v0 + v1 + v2) >= 2


# ---------- CVAE PROBING ----------
def load_image_tensor(path: str, img_size) -> torch.Tensor:
    """
    Loads an image as 1x1xHxW float tensor in [0,1].
    img_size can be int or (H,W).
    """
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if isinstance(img_size, int):
        h = w = img_size
    else:
        h, w = img_size  # (H, W)

    # cv2.resize expects (width, height)
    gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

    x = gray.astype(np.float32) / 255.0
    x = torch.from_numpy(x)[None, None, :, :]  # 1x1xHxW
    return x

@torch.no_grad()
def recon_with_dim_set(model, x: torch.Tensor, dim: int, value: float) -> torch.Tensor:
    """
    Encodes x -> z (use mu), sets z[dim] = value, decodes -> recon.
    Requires model.encode() and model.decode().
    """
    if not (hasattr(model, "encode") and hasattr(model, "decode")):
        raise RuntimeError("Your model must implement encode() and decode() for this probing script.")

    mu, logvar = model.encode(x)
    z = mu.clone()
    if dim < 0 or dim >= z.shape[1]:
        raise ValueError(f"dim {dim} out of range for latent size {z.shape[1]}")
    z[:, dim] = float(value)
    recon = model.decode(z)
    return recon

def tensor_to_uint8_img(recon: torch.Tensor) -> np.ndarray:
    """
    recon: 1x1xHxW or 1xHxW or HxW
    returns HxW uint8
    """
    r = recon.detach().cpu()
    while r.ndim > 2:
        r = r[0]
    r = torch.clamp(r, 0.0, 1.0).numpy()
    return (r * 255.0).astype(np.uint8)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_probe_map(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def get_probe_triplet(probe_map: dict, base: str):
    if base in probe_map:
        trip = probe_map[base]
    elif "_default" in probe_map:
        trip = probe_map["_default"]
    else:
        raise KeyError(f"No probe entry for '{base}' and no '_default' in probe_map.")
    if not isinstance(trip, list) or len(trip) != 3:
        raise ValueError(f"Probe entry for '{base}' must be a list of 3 items like {{dim, value}}.")
    return [(int(t["dim"]), float(t["value"])) for t in trip]


# ---------- MAIN PIPELINE ----------
def main():
    ensure_dir(RECON_OUT_DIR)

    probe_map = load_probe_map(PROBE_MAP_PATH)   # load once, before the loop

    files = sorted([f for f in os.listdir(IN_DIR) if f.lower().endswith(EXTS)])
    for fname in files:
        base = os.path.splitext(fname)[0]        # base exists here

    chrom_key = chrom_key_from_base(base)    # now this works
    probes = get_probe_triplet(probe_map, chrom_key)

    # Load model
    model = ConvCVAE(latent_dim=LATENT_DIM, img_size=IMG_SIZE) if "img_size" in ConvCVAE.__init__.__code__.co_varnames else ConvCVAE(latent_dim=LATENT_DIM)
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    # support either full state dict or wrapped dict
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(DEVICE)

    files = sorted([f for f in os.listdir(IN_DIR) if f.lower().endswith(EXTS)])
    if not files:
        print(f"No images found in IN_DIR: {IN_DIR}")
        return

    wrote = 0
    for fname in files:
        in_path = os.path.join(IN_DIR, fname)
        base = os.path.splitext(fname)[0]

        try:
            probes = get_probe_triplet(probe_map, base)  # [(dim,val), (dim,val), (dim,val)]
        except Exception as e:
            print(f"[SKIP] {fname}: probe map issue -> {e}")
            continue

        # Load original image (for final flipping)
        orig_bgr = cv2.imread(in_path)
        if orig_bgr is None:
            print(f"[SKIP] {fname}: could not read original")
            continue

        # Prepare model input tensor
        try:
            x = load_image_tensor(in_path, IMG_SIZE).to(DEVICE)
        except Exception as e:
            print(f"[SKIP] {fname}: input load issue -> {e}")
            continue

        # Folder to save the 3 recon images for this chromosome/image
        recon_folder = os.path.join(RECON_OUT_DIR, base)
        ensure_dir(recon_folder)

        recon_paths = []
        recon_votes = []

        # Generate 3 probed reconstructions
        for idx, (d, v) in enumerate(probes):
            try:
                recon = recon_with_dim_set(model, x, d, v)
            except Exception as e:
                print(f"[SKIP] {fname}: recon failed for probe {idx} (d={d}, v={v}) -> {e}")
                recon = None

            if recon is None:
                recon_votes.append(True)  # default vote
                continue

            recon_u8 = tensor_to_uint8_img(recon)
            recon_bgr = cv2.cvtColor(recon_u8, cv2.COLOR_GRAY2BGR)

            # p-up on recon
            ok, p_is_top = p_up_from_bgr(recon_bgr)
            recon_votes.append(p_is_top if ok else True)

            tag = f"probe{idx}_d{d}_v{v}".replace(".", "p").replace("-", "m")
            out_recon_path = os.path.join(recon_folder, f"{base}__{tag}.png")
            cv2.imwrite(out_recon_path, recon_bgr)
            recon_paths.append(out_recon_path)

        # Need exactly 3 votes
        if len(recon_votes) != 3:
            print(f"[SKIP] {fname}: did not produce 3 votes")
            continue

        final_p_up = majority_vote_3(recon_votes[0], recon_votes[1], recon_votes[2])

        # Apply flip to ORIGINAL (unprobed)
        final_bgr = orig_bgr.copy()
        if not final_p_up:
            final_bgr = cv2.flip(final_bgr, FLIP_AXIS)

        if SAVE_DEBUG_TEXT:
            txt = f"vote_p_up={final_p_up} votes={recon_votes}"
            cv2.putText(final_bgr, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(final_bgr, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(OUT_DIR, base + ".png")
        cv2.imwrite(out_path, final_bgr)
        wrote += 1

        # Save metadata log
        meta = {
            "base": base,
            "original_path": in_path,
            "probes": [{"dim": d, "value": v, "recon_path": rp if i < len(recon_paths) else None, "p_up": bool(recon_votes[i])}
                       for i, ((d, v), rp) in enumerate(zip(probes, recon_paths + [None]*3))],
            "votes": [bool(v) for v in recon_votes],
            "final_vote_p_up": bool(final_p_up),
            "flip_axis": FLIP_AXIS,
            "output_path": out_path,
        }
        with open(os.path.join(recon_folder, f"{base}__meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] {fname} -> final_p_up={final_p_up} votes={recon_votes}  saved={os.path.basename(out_path)}")

    print(f"\nDone. Wrote {wrote}/{len(files)} flipped originals to:\n{OUT_DIR}")
    print(f"Recons saved per-image under:\n{RECON_OUT_DIR}")


if __name__ == "__main__":
    main()