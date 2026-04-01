import cv2
import numpy as np
import os
import glob
from segment_anything import sam_model_registry, SamPredictor

# =========================
# Config
# =========================
sam = sam_model_registry["vit_b"](checkpoint="sam_b.pt")
predictor = SamPredictor(sam)

imgs_dir   = "train_cropped"
labels_dir = "Labels"
out_dir    = "crops_hybrid"

os.makedirs(out_dir, exist_ok=True)


# =========================
# Classical backup method
# (this is the simpler earlier version)
# =========================
def segment_chromosome_classical(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    h, w = mask.shape
    cx_roi = w / 2.0
    cy_roi = h / 2.0

    best_idx = None
    best_score = -1e18

    for lab in range(1, num_labels):
        x, y, bw, bh, area = stats[lab]

        if area < 20:
            continue

        comp_cx = x + bw / 2.0
        comp_cy = y + bh / 2.0

        dist2 = (comp_cx - cx_roi) ** 2 + (comp_cy - cy_roi) ** 2
        score = area - 0.15 * dist2

        if score > best_score:
            best_score = score
            best_idx = lab

    if best_idx is None:
        return np.zeros_like(mask, dtype=np.uint8)

    mask_main = np.zeros_like(mask, dtype=np.uint8)
    mask_main[labels == best_idx] = 255

    k_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_main = cv2.morphologyEx(mask_main, cv2.MORPH_CLOSE, k_final, iterations=1)

    return mask_main


# =========================
# SAM method
# =========================
def segment_with_sam(img_bgr, box_xyxy, predictor, pad=6):
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy

    x1 = max(0, int(np.floor(x1)) - pad)
    y1 = max(0, int(np.floor(y1)) - pad)
    x2 = min(W - 1, int(np.ceil(x2)) + pad)
    y2 = min(H - 1, int(np.ceil(y2)) + pad)

    box_padded = np.array([x1, y1, x2, y2], dtype=np.float32)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    masks, scores, _ = predictor.predict(
        box=box_padded[None, :],
        multimask_output=True
    )

    best_idx = np.argmax(scores)
    mask_bool = masks[best_idx].astype(bool)

    # Optional tiny gap repair
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask_u8


# =========================
# Broken-mask detector
# =========================
def is_broken_mask(mask, min_component_area=40):
    """
    Returns True if mask appears split into multiple meaningful pieces.
    Tiny specks are ignored.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return True

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    big_components = 0
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            big_components += 1

    return big_components > 1


def tight_crop_with_mask(roi_bgr, mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None

    white = np.full_like(roi_bgr, 255)
    result = white.copy()
    result[mask > 0] = roi_bgr[mask > 0]

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    return result[y0:y1 + 1, x0:x1 + 1]


# =========================
# Main loop
# =========================
for img_path in glob.glob(os.path.join(imgs_dir, "*.*")):
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_dir, base + ".txt")

    if not os.path.exists(label_path):
        print(f"[skip] no label for {base}")
        continue

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[skip] could not read {img_path}")
        continue

    H, W = img_bgr.shape[:2]

    boxes_list = []
    classes = []

    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) != 5:
                continue

            cls = int(parts[0])
            xc, yc, w_norm, h_norm = parts[1:]

            bw = w_norm * W
            bh = h_norm * H
            x1 = xc * W - bw / 2
            y1 = yc * H - bh / 2
            x2 = xc * W + bw / 2
            y2 = yc * H + bh / 2

            boxes_list.append([x1, y1, x2, y2])
            classes.append(cls)

    if not boxes_list:
        print(f"[skip] no boxes in {label_path}")
        continue

    boxes = np.array(boxes_list, dtype=np.float32)
    classes = np.array(classes, dtype=int)

    chrom_counters = {}

    for i, (box, cls) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = box

        # Small ROI for classical fallback
        pad_roi = 6
        rx1 = max(0, int(np.floor(x1)) - pad_roi)
        ry1 = max(0, int(np.floor(y1)) - pad_roi)
        rx2 = min(W - 1, int(np.ceil(x2)) + pad_roi)
        ry2 = min(H - 1, int(np.ceil(y2)) + pad_roi)

        if rx2 <= rx1 or ry2 <= ry1:
            print(f"[skip] invalid box for {base} box {i}")
            continue

        roi = img_bgr[ry1:ry2 + 1, rx1:rx2 + 1]
        if roi.size == 0:
            print(f"[skip] empty roi for {base} box {i}")
            continue

        # 1) SAM primary
        sam_mask_full = segment_with_sam(img_bgr, box, predictor, pad=6)

        # Convert SAM full-image mask to ROI-local mask
        sam_mask_roi = sam_mask_full[ry1:ry2 + 1, rx1:rx2 + 1]

        # 2) Check if SAM mask is broken
        broken = is_broken_mask(sam_mask_roi, min_component_area=40)

        # 3) If broken, use classical backup
        if broken:
            final_mask = segment_chromosome_classical(roi)
            method_used = "classical_fallback"
        else:
            final_mask = sam_mask_roi
            method_used = "sam"

        if final_mask is None or np.count_nonzero(final_mask) == 0:
            print(f"[skip] empty final mask for {base} box {i}")
            continue

        crop = tight_crop_with_mask(roi, final_mask)
        if crop is None or crop.size == 0:
            print(f"[skip] empty crop for {base} box {i}")
            continue

        chrom_counters[cls] = chrom_counters.get(cls, 0) + 1
        crop_idx = chrom_counters[cls]

        out_fname = f"{base}_class{cls}_crop{crop_idx}.jpg"
        out_path = os.path.join(out_dir, out_fname)
        cv2.imwrite(out_path, crop)

        print(f"[saved] {out_fname} using {method_used}")

    print(f"[done] {base}  ({len(boxes)} chromosomes)")

print(f"\nAll crops saved to: {out_dir}")