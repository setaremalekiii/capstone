import os
import cv2
import numpy as np

IN_DIR        = r"/scratch/st-li1210-1/pearl/karyotype-detector/data/all_classes"
OUT_DIR       = r"/arc/project/st-li1210-1/pearl/karyotype-detector/models/CVAE/step1"
os.makedirs(OUT_DIR, exist_ok=True)

# Flip axis:
#   0 = vertical flip (top/bottom)  <-- use this for "p arm up"
#   1 = horizontal flip (left/right)
FLIP_AXIS = 0

# Save debug overlays (text + optional mask)
SAVE_DEBUG = False
SAVE_MASKS = False
MASK_DIR = os.path.join(OUT_DIR, "_masks")
if SAVE_MASKS:
    os.makedirs(MASK_DIR, exist_ok=True)

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def ensure_portrait(bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    If the image is wider than tall (landscape), rotate 90 degrees clockwise
    so the long axis is always vertical (portrait).
    Returns (bgr, was_rotated).
    """
    h, w = bgr.shape[:2]
    if w > h:
        # cv2.ROTATE_90_CLOCKWISE: columns become rows, width becomes height
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        return bgr, True
    return bgr, False


def segment_chromosome_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask for the chromosome (largest connected component).
    Assumes a mostly light background with a darker chromosome.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def clean(th):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
        return th

    th1 = clean(th1)
    th2 = clean(th2)

    mask1 = largest_component(th1)
    mask2 = largest_component(th2)

    h, w = gray.shape
    a1 = mask1.sum()
    a2 = mask2.sum()

    def ok(a):
        frac = a / float(h * w)
        return 0.005 < frac < 0.6

    if ok(a1) and not ok(a2):
        return mask1
    if ok(a2) and not ok(a1):
        return mask2
    return mask1 if a1 >= a2 else mask2


def largest_component(th_255: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(th_255, connectivity=8)
    if num <= 1:
        return np.zeros(th_255.shape, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (labels == idx).astype(np.uint8)


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
    """
    Uses the minimum width row as a centromere proxy, then compares top vs
    bottom extents. Returns True if p-arm is on top.
    """
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return True

    y_min, y_max = ys.min(), ys.max()
    height = y_max - y_min + 1

    widths = np.full(height, np.inf, dtype=np.float32)
    for y in range(y_min, y_max + 1):
        row_x = xs[ys == y]
        if row_x.size > 0:
            widths[y - y_min] = float(row_x.max() - row_x.min())

    lo = max(int(0.2 * height), 0)
    hi = max(int(0.8 * height), lo + 1)
    mid_widths = widths[lo:hi]

    if np.all(np.isinf(mid_widths)):
        k = int(np.argmin(widths))
    else:
        k = lo + int(np.argmin(mid_widths))

    y_c = y_min + k
    top_extent    = y_c - y_min
    bottom_extent = y_max - y_c

    return top_extent < bottom_extent


def process_one(path: str):
    bgr = cv2.imread(path)
    if bgr is None:
        return None, "could not read"

    # Step 1: make sure the image is portrait (h > w)
    bgr, was_rotated = ensure_portrait(bgr)

    # Step 2: segment and crop to chromosome
    mask01 = segment_chromosome_mask(bgr)
    cropped, cropped_mask = crop_to_mask(bgr, mask01, pad=5)
    if cropped is None:
        return None, "empty mask"

    # Step 3: detect p-arm and flip if needed
    p_is_top = estimate_p_is_top(cropped_mask)
    out = cropped.copy()
    if not p_is_top:
        out = cv2.flip(out, FLIP_AXIS)

    if SAVE_DEBUG:
        txt = f"p_up={p_is_top}  rotated={was_rotated}"
        cv2.putText(out, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if SAVE_MASKS:
        m = (cropped_mask * 255).astype(np.uint8)
        base = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(MASK_DIR, base + "_mask.png"), m)

    return out, (p_is_top, was_rotated)


def main():
    files = sorted(f for f in os.listdir(IN_DIR) if f.lower().endswith(EXTS))

    ok_count = 0
    for f in files:
        in_path  = os.path.join(IN_DIR, f)
        base     = os.path.splitext(f)[0]
        out_path = os.path.join(OUT_DIR, base + ".png")

        out, info = process_one(in_path)
        if out is None:
            print(f"[SKIP] {f}: {info}")
            continue

        cv2.imwrite(out_path, out)
        ok_count += 1
        p_is_top, was_rotated = info
        print(
            f"[OK] {f:50s}  p_up={p_is_top}  "
            f"{'rotated90 ' if was_rotated else 'no-rotate '}"
        )

    print(f"\nDone. Wrote {ok_count}/{len(files)} images to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()