import cv2
import numpy as np
import os
import glob
from segment_anything import sam_model_registry, SamPredictor

# 1) Load SAM weights
sam = sam_model_registry["vit_b"](checkpoint="sam_b.pt")
predictor = SamPredictor(sam)

imgs_dir   = "norm/images/train"
labels_dir = "norm/labels/train"
out_dir    = "crops/train"

os.makedirs(out_dir, exist_ok=True)

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
    classes    = []

    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
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

    boxes   = np.array(boxes_list, dtype=np.float32)
    classes = np.array(classes, dtype=int)

    # 2) Set image for SAM
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    # 3) For each chromosome: predict mask, apply to white bg, crop tight, save
    chrom_counters = {}  # track how many of each class we've seen per image

    for i, (box, cls) in enumerate(zip(boxes, classes)):
        mask, _, _ = predictor.predict(
            box=box[None, :],
            multimask_output=False
        )
        mask_bool = mask[0].astype(bool)  # (H, W)

        # Apply mask onto white background
        white  = np.full_like(img_bgr, 255)
        result = white.copy()
        result[mask_bool] = img_bgr[mask_bool]

        # Tight crop around the mask
        ys, xs = np.where(mask_bool)
        if ys.size == 0:
            print(f"[skip] empty mask for {base} box {i}")
            continue

        y0, y1_crop = int(ys.min()), int(ys.max())
        x0, x1_crop = int(xs.min()), int(xs.max())
        crop = result[y0:y1_crop+1, x0:x1_crop+1]

        # Unique filename: <original_base>_class<cls>_crop<N>.png
        chrom_counters[cls] = chrom_counters.get(cls, 0) + 1
        crop_idx  = chrom_counters[cls]
        out_fname = f"{base}_class{cls}_crop{crop_idx}.png"
        cv2.imwrite(os.path.join(out_dir, out_fname), crop)

    print(f"[done] {base}  ({len(boxes)} chromosomes)")

print(f"\nAll crops saved to: {out_dir}")