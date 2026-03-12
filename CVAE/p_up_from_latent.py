"""
p_up_orient.py
==============
Uses a trained ConvCVAE to orient chromosome crops so the p-arm faces up.

HOW IT WORKS
------------
1. For each chromosome image, look up its 3 "probe" latent dimensions from
   probe_map.json.  These dims were identified because manipulating them
   *shrinks the centromere*, making the p/q boundary clearly visible in the
   reconstruction.
2. For each probe dim, encode the image → take the mean z, force z[dim]=value,
   decode → get a "clearer" reconstruction.
3. Run the p-is-top geometric detector on all 3 reconstructions.
4. Majority vote (≥2/3): if p is NOT on top → flip the ORIGINAL image
   vertically (FLIP_AXIS=0).
5. Save the (possibly flipped) original + metadata JSON + the 3 recon images.

REQUIREMENTS
------------
- ConvCVAE.py must be importable (run from the CVAE folder, or add it to PYTHONPATH)
- Your ConvCVAE.encode(x, y) and .decode(z, y) must accept a label tensor y
  of shape (B, num_classes).  If your model is unconditional (no y), set
  LABEL_DIM = 0 below.
"""

import os
import json
import re
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from ConvCVAE import ConvCVAE


# ─────────────────────────── USER CONFIG ────────────────────────────────── #

IN_DIR        = r"/scratch/st-li1210-1/pearl/karyotype-detector/data/preprocessed_train"
OUT_DIR       = r"/arc/project/st-li1210-1/pearl/karyotype-detector/models/CVAE/output_p_up"
RECON_OUT_DIR = r"/arc/project/st-li1210-1/pearl/karyotype-detector/models/CVAE/probe_recons_by_image"
PROBE_MAP_PATH= r"/arc/project/st-li1210-1/pearl/karyotype-detector/models/CVAE/probe_map.json"
WEIGHTS_PATH  = r"/arc/project/st-li1210-1/pearl/karyotype-detector/models/CVAE/best_32.pth"

IMG_SIZE    = (64, 64)   # (H, W) — must match training
LATENT_DIM  = 32         # must match training

# Set to 0 if your CVAE is unconditional (no class label input).
# Set to the number of chromosome classes (e.g. 24) if it IS conditional.
# The script will also try to auto-detect this from the model weights.
LABEL_DIM   = None       # None = auto-detect

# 0 = flip top↔bottom (correct for "p arm up")
# 1 = flip left↔right
FLIP_AXIS = 0

SAVE_DEBUG_TEXT = False
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If the primary output dirs are not writable (e.g. ARC quota), fall back here
FALLBACK_OUT_ROOT = r"/scratch/st-li1210-1/pearl/karyotype-detector/cvae_outputs"

# ──────────────────────────────────────────────────────────────────────────── #


# ═══════════════════════════ I/O HELPERS ════════════════════════════════════ #

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def is_writable_dir(p: str) -> bool:
    return os.path.isdir(p) and os.access(p, os.W_OK)


def pick_output_dir(primary: str, fallback_subdir: str) -> str:
    try:
        ensure_dir(primary)
    except Exception:
        pass
    if is_writable_dir(primary):
        return primary
    fallback = os.path.join(FALLBACK_OUT_ROOT, fallback_subdir)
    ensure_dir(fallback)
    if not is_writable_dir(fallback):
        raise RuntimeError(
            f"Neither primary nor fallback output dir is writable:\n"
            f"  {primary}\n  {fallback}"
        )
    print(f"[WARN] Primary output not writable, using fallback:\n  {fallback}")
    return fallback


def safe_imwrite(path: str, img: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


# ═══════════════════════════ PROBE MAP ══════════════════════════════════════ #

def load_probe_map(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_probe_triplet(
    probe_map: Dict[str, Any], key: str
) -> List[Tuple[int, float]]:
    """
    Returns [(dim, value), (dim, value), (dim, value)] for the given key.
    Falls back to '_default' if the key is absent.
    """
    entry = probe_map.get(key) or probe_map.get("_default")
    if entry is None:
        raise KeyError(f"No probe entry for '{key}' and no '_default' in probe_map.")
    if not isinstance(entry, list) or len(entry) != 3:
        raise ValueError(
            f"Probe entry for '{key}' must be a list of exactly 3 dicts "
            f"like {{\"dim\": int, \"value\": float}}."
        )
    return [(int(t["dim"]), float(t["value"])) for t in entry]


# ═══════════════════════════ CLASS / CONDITIONING ════════════════════════════ #

def class_id_from_base(base: str) -> int:
    """
    Parses the chromosome class id from a filename like
    '103064_class12_crop3' → 12.
    X chromosome (class 23 by convention) → key 'chromX'.
    """
    m = re.search(r"class(\d+)", base)
    if not m:
        raise ValueError(f"Could not find 'class<N>' in filename: '{base}'")
    return int(m.group(1))


def chrom_key_from_class_id(cls_id: int) -> str:
    return "chromX" if cls_id == 23 else f"chrom{cls_id}"


def infer_label_dim(model: nn.Module, img_channels: int = 1) -> int:
    """
    Auto-detect label_dim by inspecting the first Conv2d in the encoder.
    label_dim = encoder_input_channels − image_channels.
    Returns 0 if unconditional.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            diff = m.in_channels - img_channels
            return max(0, diff)
    return 0


def make_y_onehot(cls_id: int, label_dim: int, device: str) -> Optional[torch.Tensor]:
    """Returns a (1, label_dim) one-hot tensor, or None if label_dim==0."""
    if label_dim == 0:
        return None
    if not (0 <= cls_id < label_dim):
        raise ValueError(
            f"class_id={cls_id} is out of range for label_dim={label_dim}"
        )
    y = torch.zeros(1, label_dim, device=device)
    y[0, cls_id] = 1.0
    return y


def expand_y_to_spatial(y: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(B, C) → (B, C, H, W) by tiling."""
    return y[:, :, None, None].expand(y.shape[0], y.shape[1], H, W)


# ═══════════════════════════ P-ARM DETECTION ════════════════════════════════ #

def largest_component(th_255: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(th_255, connectivity=8)
    if num <= 1:
        return np.zeros(th_255.shape, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (labels == idx).astype(np.uint8)


def segment_chromosome_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, th     = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)

    def clean(t):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        t = cv2.morphologyEx(t, cv2.MORPH_OPEN,  k, iterations=1)
        t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, k, iterations=2)
        return t

    m1 = largest_component(clean(th_inv))
    m2 = largest_component(clean(th))

    h, w = gray.shape
    a1, a2 = m1.sum(), m2.sum()

    def ok(a):
        return 0.005 < (a / float(h * w)) < 0.6

    if ok(a1) and not ok(a2): return m1
    if ok(a2) and not ok(a1): return m2
    return m1 if a1 >= a2 else m2


def crop_to_mask(
    bgr: np.ndarray, mask01: np.ndarray, pad: int = 5
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return None, None
    h, w = mask01.shape
    y0 = max(ys.min() - pad, 0);  x0 = max(xs.min() - pad, 0)
    y1 = min(ys.max() + pad, h - 1); x1 = min(xs.max() + pad, w - 1)
    return bgr[y0:y1+1, x0:x1+1], mask01[y0:y1+1, x0:x1+1]


def estimate_p_is_top(mask01: np.ndarray) -> bool:
    """
    Finds the narrowest horizontal cross-section (centromere) of the
    chromosome mask.  If it's in the upper half → short arm (p) is on top.
    """
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return True

    y_min, y_max = int(ys.min()), int(ys.max())
    height = y_max - y_min + 1

    widths = np.full(height, np.inf, dtype=np.float32)
    for y in range(y_min, y_max + 1):
        row_x = xs[ys == y]
        if row_x.size > 0:
            widths[y - y_min] = float(row_x.max() - row_x.min())

    # Only search the middle 60 % to avoid end-cap noise
    lo = max(int(0.2 * height), 0)
    hi = max(int(0.8 * height), lo + 1)
    mid = widths[lo:hi]

    if np.all(np.isinf(mid)):
        k = int(np.argmin(widths))
    else:
        k = lo + int(np.argmin(mid))

    y_centromere = y_min + k
    top_extent    = y_centromere - y_min
    bottom_extent = y_max - y_centromere
    return top_extent < bottom_extent   # p-arm is the shorter arm


def p_up_from_bgr(bgr: np.ndarray) -> Tuple[bool, bool]:
    """Returns (detection_ok, p_is_top)."""
    mask01 = segment_chromosome_mask(bgr)
    cropped, cropped_mask = crop_to_mask(bgr, mask01, pad=5)
    if cropped is None:
        return False, True          # fallback: assume already correct
    return True, estimate_p_is_top(cropped_mask)


# ═══════════════════════════ CVAE PROBING ════════════════════════════════════ #

def pad_to_square(gray: np.ndarray) -> np.ndarray:
    """Pad a grayscale image to square with white (255) so aspect ratio is preserved."""
    h, w = gray.shape
    if h == w:
        return gray
    diff = abs(h - w)
    if h < w:
        # image is wider than tall — pad top and bottom
        pad_top = diff // 2
        pad_bot = diff - pad_top
        gray = cv2.copyMakeBorder(gray, pad_top, pad_bot, 0, 0,
                                   cv2.BORDER_CONSTANT, value=255)
    else:
        # image is taller than wide — pad left and right
        pad_left = diff // 2
        pad_right = diff - pad_left
        gray = cv2.copyMakeBorder(gray, 0, 0, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=255)
    return gray


def load_image_tensor(path: str, img_size: Tuple[int, int]) -> torch.Tensor:
    """
    Loads a grayscale image as (1, 1, H, W) float32 tensor in [0, 1].
    Pads to square first so the chromosome aspect ratio is preserved
    when resized to img_size — otherwise tall thin chromosomes get squashed
    into a square and the centromere position is distorted.
    """
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = pad_to_square(gray)          # ← preserve aspect ratio
    H, W = img_size
    gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None, :, :]
    return x  # (1, 1, H, W)


@torch.no_grad()
def recon_with_dim_set(
    model: nn.Module,
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    dim: int,
    value: float,
) -> torch.Tensor:
    """
    Encode x (with optional label y) → mu → force mu[dim]=value → decode.

    Handles both:
      • Conditional  CVAE: encode(x, y),  decode(z, y)
      • Unconditional CVAE: encode(x),     decode(z)

    Also handles spatial-y models where y must be expanded to (B,C,H,W).
    """
    B, _, H, W = x.shape

    # ── encode ──────────────────────────────────────────────────────────────
    enc_result = _try_encode(model, x, y, H, W)

    if isinstance(enc_result, (tuple, list)):
        if len(enc_result) == 2:
            mu, logvar = enc_result
        elif len(enc_result) >= 3:
            # some architectures return (z_sample, mu, logvar)
            _, mu, logvar = enc_result[0], enc_result[1], enc_result[2]
        else:
            raise RuntimeError(f"encode() returned {len(enc_result)} values; expected 2 or 3.")
    else:
        raise RuntimeError(f"encode() did not return a tuple/list: {type(enc_result)}")

    # ── probe ────────────────────────────────────────────────────────────────
    z = mu.clone()
    if not (0 <= dim < z.shape[1]):
        raise ValueError(f"Probe dim {dim} is out of range (latent_dim={z.shape[1]})")
    z[:, dim] = float(value)

    # ── decode ───────────────────────────────────────────────────────────────
    recon = _try_decode(model, z, y, H, W)
    return recon


def _try_encode(model, x, y, H, W):
    """Try encode with y (vector), then spatial y, then without y."""
    if y is not None:
        try:
            return model.encode(x, y)
        except (TypeError, RuntimeError):
            pass
        try:
            return model.encode(x, expand_y_to_spatial(y, H, W))
        except (TypeError, RuntimeError):
            pass
    # unconditional fallback
    return model.encode(x)


def _try_decode(model, z, y, H, W):
    """Try decode with y (vector), then spatial y, then without y."""
    if y is not None:
        try:
            return model.decode(z, y)
        except (TypeError, RuntimeError):
            pass
        try:
            return model.decode(z, expand_y_to_spatial(y, H, W))
        except (TypeError, RuntimeError):
            pass
    return model.decode(z)


def tensor_to_uint8(recon: torch.Tensor) -> np.ndarray:
    """(1,1,H,W) or any leading singleton dims → (H,W) uint8."""
    r = recon.detach().cpu()
    while r.ndim > 2:
        r = r[0]
    return (torch.clamp(r, 0.0, 1.0).numpy() * 255.0).astype(np.uint8)


# ═══════════════════════════ MAIN ════════════════════════════════════════════ #

def main():
    out_dir    = pick_output_dir(OUT_DIR,       "output_p_up")
    recon_root = pick_output_dir(RECON_OUT_DIR, "probe_recons_by_image")

    probe_map = load_probe_map(PROBE_MAP_PATH)

    # ── load model ───────────────────────────────────────────────────────────
    model = ConvCVAE(latent_dim=LATENT_DIM, img_size=IMG_SIZE)
    ckpt  = torch.load(WEIGHTS_PATH, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(DEVICE)

    # ── resolve label_dim ─────────────────────────────────────────────────
    global LABEL_DIM
    if LABEL_DIM is None:
        LABEL_DIM = infer_label_dim(model, img_channels=1)
    print(f"[INFO] device={DEVICE} | latent_dim={LATENT_DIM} | label_dim={LABEL_DIM}")
    print(f"[INFO] out_dir      = {out_dir}")
    print(f"[INFO] recon_root   = {recon_root}")

    # ── collect files ────────────────────────────────────────────────────────
    files = sorted(
        f for f in os.listdir(IN_DIR) if f.lower().endswith(EXTS)
    )
    if not files:
        print(f"[ERROR] No images found in IN_DIR: {IN_DIR}")
        return

    wrote = 0

    for fname in files:
        in_path = os.path.join(IN_DIR, fname)
        base    = os.path.splitext(fname)[0]

        # ── parse class & probes ─────────────────────────────────────────
        try:
            cls_id    = class_id_from_base(base)
            chrom_key = chrom_key_from_class_id(cls_id)
            probes    = get_probe_triplet(probe_map, chrom_key)
        except Exception as e:
            print(f"[SKIP] {fname}: class/probe-map issue → {e}")
            continue

        # ── load original ────────────────────────────────────────────────
        orig_bgr = cv2.imread(in_path)
        if orig_bgr is None:
            print(f"[SKIP] {fname}: could not read image")
            continue

        # ── build model inputs ───────────────────────────────────────────
        try:
            x = load_image_tensor(in_path, IMG_SIZE).to(DEVICE)
            y = make_y_onehot(cls_id, LABEL_DIM, DEVICE)
        except Exception as e:
            print(f"[SKIP] {fname}: input build issue → {e}")
            continue

        # ── run 3 probe reconstructions ──────────────────────────────────
        recon_folder = os.path.join(recon_root, base)
        ensure_dir(recon_folder)

        votes: List[Optional[bool]] = []
        saved_recon_paths: List[str] = []

        for probe_idx, (dim, val) in enumerate(probes):
            try:
                recon   = recon_with_dim_set(model, x, y, dim, val)
                recon_u8 = tensor_to_uint8(recon)
                recon_bgr = cv2.cvtColor(recon_u8, cv2.COLOR_GRAY2BGR)

                ok, p_is_top = p_up_from_bgr(recon_bgr)
                votes.append(p_is_top if ok else None)

                # Save the reconstruction image
                tag      = f"probe{probe_idx}_d{dim}_v{val}".replace(".", "p").replace("-", "m")
                rpath    = os.path.join(recon_folder, f"{base}__{tag}.png")
                safe_imwrite(rpath, recon_bgr)
                saved_recon_paths.append(rpath)

            except Exception as e:
                print(f"  [WARN] {fname} probe {probe_idx} (dim={dim}, val={val}) failed → {e}")
                votes.append(None)
                saved_recon_paths.append("")

        # ── majority vote ────────────────────────────────────────────────
        # Only count probes that produced a valid detection
        valid_votes = [v for v in votes if v is not None]

        if len(valid_votes) < 2:
            print(f"[SKIP] {fname}: only {len(valid_votes)}/3 valid votes — skipping")
            continue

        # p_is_top=True means no flip needed; False means we need to flip.
        p_up_count  = sum(valid_votes)
        final_p_up  = p_up_count >= (len(valid_votes) / 2.0 + 0.5)  # strict majority

        # ── apply flip to ORIGINAL ───────────────────────────────────────
        final_bgr = orig_bgr.copy()
        if not final_p_up:
            final_bgr = cv2.flip(final_bgr, FLIP_AXIS)

        # ── optional debug overlay ───────────────────────────────────────
        if SAVE_DEBUG_TEXT:
            label = f"{chrom_key}  p_up={final_p_up}  votes={votes}"
            cv2.putText(final_bgr, label, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,   0,   0), 2, cv2.LINE_AA)
            cv2.putText(final_bgr, label, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)

        # ── save final image ─────────────────────────────────────────────
        out_path = os.path.join(out_dir, base + ".png")
        try:
            safe_imwrite(out_path, final_bgr)
        except Exception as e:
            print(f"[FAIL] {fname}: could not write output → {e}")
            continue

        wrote += 1

        # ── save metadata ─────────────────────────────────────────────────
        meta = {
            "base":           base,
            "chrom_key":      chrom_key,
            "class_id":       cls_id,
            "original_path":  in_path,
            "probes": [
                {"dim": int(d), "value": float(v),
                 "vote": votes[i],
                 "recon_path": saved_recon_paths[i] if i < len(saved_recon_paths) else ""}
                for i, (d, v) in enumerate(probes)
            ],
            "valid_votes":       valid_votes,
            "final_vote_p_up":   bool(final_p_up),
            "flip_axis":         FLIP_AXIS,
            "flipped":           not final_p_up,
            "output_path":       out_path,
        }
        with open(os.path.join(recon_folder, f"{base}__meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[OK] {fname:50s}  {chrom_key:8s}  "
            f"votes={votes}  final_p_up={final_p_up}  "
            f"{'FLIPPED' if not final_p_up else 'kept   '}"
        )

    print(f"\n{'─'*60}")
    print(f"Done.  Wrote {wrote}/{len(files)} images to:\n  {out_dir}")
    print(f"Recon images + metadata JSON per image under:\n  {recon_root}")


if __name__ == "__main__":
    main()