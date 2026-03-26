import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM


# -------------------------
# Defaults (can be overridden by CLI)
# -------------------------
CONF_THRESH = 0.50
BOX_MARGIN = 0.05
MIN_BOX_AREA = 400

KEEP_LARGEST_COMPONENT = True
MASK_DILATE_PX = 0

WHITE_BACKGROUND = True
SAVE_MASKS = True
CROP_PAD = 2


def read_yolo_txt_flexible(txt_path: str):
    dets = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            vals = list(map(float, parts))
            cls = int(vals[0])
            cx, cy, bw, bh = vals[1], vals[2], vals[3], vals[4]
            conf = vals[5] if len(vals) >= 6 else None
            dets.append({"cls": cls, "cx": cx, "cy": cy, "bw": bw, "bh": bh, "conf": conf})
    return dets


def yolo_norm_to_xyxy_px(det, W, H, margin_frac=0.0):
    cx, cy, bw, bh = det["cx"], det["cy"], det["bw"], det["bh"]
    x1 = (cx - bw / 2) * W
    y1 = (cy - bh / 2) * H
    x2 = (cx + bw / 2) * W
    y2 = (cy + bh / 2) * H

    mx = (x2 - x1) * margin_frac
    my = (y2 - y1) * margin_frac
    x1 -= mx; y1 -= my; x2 += mx; y2 += my

    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def keep_largest_component(mask_bool: np.ndarray) -> np.ndarray:
    mask = (mask_bool.astype(np.uint8) * 255)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask_bool
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest)


def dilate_mask(mask_bool: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask_bool
    k = 2 * px + 1
    kernel = np.ones((k, k), np.uint8)
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.dilate(m, kernel, iterations=1)
    return (m > 0)


def crop_to_mask(img_bgr: np.ndarray, mask_bool: np.ndarray, pad: int = 0):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None, None
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    y1 = max(0, y1 - pad); x1 = max(0, x1 - pad)
    y2 = min(img_bgr.shape[0] - 1, y2 + pad)
    x2 = min(img_bgr.shape[1] - 1, x2 + pad)
    return (x1, y1, x2, y2), img_bgr[y1:y2+1, x1:x2+1].copy()


def run_sam_cutouts(
    img_path: str,
    yolo_txt: str,
    out_dir: str,
    sam_ckpt: str,
    conf_thresh: float = CONF_THRESH,
    box_margin: float = BOX_MARGIN,
    min_box_area: int = MIN_BOX_AREA,
    keep_largest: bool = KEEP_LARGEST_COMPONENT,
    mask_dilate_px: int = MASK_DILATE_PX,
    white_bg: bool = WHITE_BACKGROUND,
    save_masks: bool = SAVE_MASKS,
    crop_pad: int = CROP_PAD,
):
    img_path = str(img_path)
    yolo_txt = str(yolo_txt)
    out_dir = str(out_dir)
    sam_ckpt = str(sam_ckpt)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    H, W = img_bgr.shape[:2]

    dets = read_yolo_txt_flexible(yolo_txt)
    if not dets:
        raise RuntimeError("No detections read from YOLO txt file.")

    has_conf = any(d["conf"] is not None for d in dets)

    prompts = []
    det_meta = []
    for d in dets:
        if has_conf and d["conf"] is not None and d["conf"] < conf_thresh:
            continue
        box = yolo_norm_to_xyxy_px(d, W, H, margin_frac=box_margin)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) < min_box_area:
            continue
        prompts.append(box)
        det_meta.append({"cls": d["cls"], "conf": d["conf"]})

    if not prompts:
        raise RuntimeError("All detections were filtered out. Lower CONF_THRESH/MIN_BOX_AREA.")

    sam = SAM(sam_ckpt)
    results = sam(img_path, bboxes=prompts)
    r = results[0]
    if r.masks is None or r.masks.data is None:
        raise RuntimeError("SAM returned no masks. Check model file and prompt format.")

    masks_np = r.masks.data.cpu().numpy().astype(bool)

    if masks_np.shape[0] != len(prompts):
        print(f"Warning: prompts={len(prompts)} masks={masks_np.shape[0]}")

    per_class_count = {}

    for i in range(min(len(prompts), masks_np.shape[0])):
        cls = det_meta[i]["cls"]
        conf = det_meta[i]["conf"]
        mask = masks_np[i]

        if keep_largest:
            mask = keep_largest_component(mask)
        mask = dilate_mask(mask, mask_dilate_px)

        cutout_full = img_bgr.copy()
        if white_bg:
            cutout_full[~mask] = 255

        bbox, cropped = crop_to_mask(cutout_full, mask, pad=crop_pad)
        if cropped is None or bbox is None:
            continue

        cls_dir = Path(out_dir) / f"class_{cls:02d}"
        cls_dir.mkdir(parents=True, exist_ok=True)

        per_class_count.setdefault(cls, 0)
        per_class_count[cls] += 1
        k = per_class_count[cls]

        conf_str = f"{conf:.3f}" if conf is not None else "na"
        out_img = cls_dir / f"det_{k:02d}_conf_{conf_str}.png"
        cv2.imwrite(str(out_img), cropped)

        if save_masks:
            x1, y1, x2, y2 = bbox
            mask_crop = (mask[y1:y2+1, x1:x2+1].astype(np.uint8) * 255)
            out_m = cls_dir / f"det_{k:02d}_mask.png"
            cv2.imwrite(str(out_m), mask_crop)

            meta_record = {
                "cls": int(cls),
                "det_idx_in_class": int(k),
                "prompt_idx": int(i),
                "confidence": float(conf) if conf is not None else None,
                "orig_image": os.path.basename(img_path),
                "bbox_xyxy_fullimg": [int(x1), int(y1), int(x2), int(y2)],
                "crop_pad": int(crop_pad),
                "cutout_file": out_img.name,
                "mask_file": out_m.name,
            }
            meta_path = cls_dir / f"det_{k:02d}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta_record, f, indent=4)

    print(f"Done. Saved SAM cutouts to: {out_dir}")
    print("Counts by class:", dict(sorted(per_class_count.items())))


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to original chromosome image (jpg/png)")
    ap.add_argument("--yolo", required=True, help="Path to YOLO txt file (cls cx cy w h [conf])")
    ap.add_argument("--out", required=True, help="Output directory for sam_cutouts")
    ap.add_argument("--sam", required=True, help="Path to SAM checkpoint (e.g., sam2.1_b.pt)")

    ap.add_argument("--conf", type=float, default=CONF_THRESH)
    ap.add_argument("--margin", type=float, default=BOX_MARGIN)
    ap.add_argument("--min_area", type=int, default=MIN_BOX_AREA)
    ap.add_argument("--dilate", type=int, default=MASK_DILATE_PX)
    ap.add_argument("--crop_pad", type=int, default=CROP_PAD)
    ap.add_argument("--keep_largest", action="store_true", default=KEEP_LARGEST_COMPONENT)
    ap.add_argument("--no_white_bg", action="store_true", help="Keep original background (not white)")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    run_sam_cutouts(
        img_path=args.img,
        yolo_txt=args.yolo,
        out_dir=args.out,
        sam_ckpt=args.sam,
        conf_thresh=args.conf,
        box_margin=args.margin,
        min_box_area=args.min_area,
        mask_dilate_px=args.dilate,
        crop_pad=args.crop_pad,
        keep_largest=args.keep_largest,
        white_bg=(not args.no_white_bg),
        save_masks=True,
    )


if __name__ == "__main__":
    main()