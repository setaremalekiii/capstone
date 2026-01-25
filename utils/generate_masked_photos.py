import cv2
import numpy as np
import os
import glob
# you need to pip install segment_anything package for this to work
from segment_anything import sam_model_registry, SamPredictor

chromosome_classes = {
    0: "A1",
    1: "A2",
    2: "A3",
    3: "B4",
    4: "B5",
    5: "C6",
    6: "C7",
    7: "C8",
    8: "C9",
    9: "C10",
    10: "C11",
    11: "C12",
    12: "D13",
    13: "D14",
    14: "D15",
    15: "E16",
    16: "E17",
    17: "E18",
    18: "F19",
    19: "F20",
    20: "G21",
    21: "G22",
    23: "X",
}

# 1) Load SAM weights 
#  These weights were generated in th generate_mask_weights.py script based on the first 364 images in the normal training folder
sam = sam_model_registry["vit_b"](checkpoint="sam_b.pt")

# Meta predictor that converts the .pt weights to a form usable for prediction
predictor = SamPredictor(sam)

# change these accordingly
imgs_dir   = "norm/images/train"
labels_dir = "norm/labels/train"
out_dir    = "masked_results/train" # we dont technically need this so comment it out if you want to save a bit of time
mask_dir   = "masked_results/train_np" 

os.makedirs(out_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

for img_path in glob.glob(os.path.join(imgs_dir, "*.*")):
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_dir, base + ".txt")
    
    if not os.path.exists(label_path):
        print(f"[skip] no label for {base}")
        continue

    # Load image as pixels  
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[skip] could not read {img_path}")
        continue

    H, W = img_bgr.shape[:2]

    # build boxes + classes from YOLO txt file
    boxes_list = []
    classes = []

    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            cls = int(parts[0])
            xc, yc, w_norm, h_norm = parts[1:]  # YOLO normalized -> this will make values between 0 and 1

            bw = w_norm * W
            bh = h_norm * H
            x_c = xc * W
            y_c = yc * H

            x1 = x_c - bw / 2
            y1 = y_c - bh / 2
            x2 = x_c + bw / 2
            y2 = y_c + bh / 2

            boxes_list.append([x1, y1, x2, y2])
            classes.append(cls)

    if not boxes_list:
        print(f"[skip] no boxes in {label_path}")
        continue

    boxes = np.array(boxes_list, dtype=np.float32)   # shape (N, 4)
    classes = np.array(classes, dtype=int)

    # 2) Tell SAM which image to work on
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    # 3) Get a mask for each box (loop)
    all_masks = []
    all_scores = []

    for box in boxes:
        m, s, _ = predictor.predict(
            box=box[None, :],      # shape (1, 4)
            multimask_output=False
        )
        all_masks.append(m[0])     # (H, W)
        all_scores.append(s[0])

    masks = np.stack(all_masks, axis=0)   # (N, H, W)

    # 4) Visualize masks + YOLO boxes + labels
    overlay = img_bgr.copy()

    for i, (box, cls) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = box
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        # black box
        cv2.rectangle(overlay, p1, p2, (0, 0, 0), 2)

        # map class → chromosome name, e.g. chromosome_classes = {0:"A1", 1:"A2", ...} I still need to add proper classes
        name = str(cls)

    #    label = f"{chromosome_classes[name]}:{i}"
        label = f"{cls}:{i}"

        cv2.putText(
            overlay,
            label,
            (p1[0], p1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),   # black text
            1,
            cv2.LINE_AA
        )

    # then: paint SAM masks in red
    for m in masks:
        overlay[m.astype(bool)] = (0, 0, 255)   # red where mask==1
        
    
    # blend with original for nicer look 
    alpha = 0.5  # transparency factor
    vis = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    best_i = int(np.argmax(all_scores))        # all_scores corresponds to each box
    keep = masks[best_i].astype(bool)          # (H, W)

    white = np.full_like(img_bgr, 255)
    result = white.copy()
    result[keep] = img_bgr[keep]

    cv2.imwrite(os.path.join(out_dir, f"{base}_bestmask_white.png"), result)
    out_path = os.path.join(out_dir, f"{base}_sam_overlay_boxes.png")
    cv2.imwrite(out_path, vis)
    np.save(os.path.join(mask_dir, f"{base}_masks.npy"),   masks)
    np.save(os.path.join(mask_dir, f"{base}_boxes.npy"),   boxes)
    np.save(os.path.join(mask_dir, f"{base}_classes.npy"), classes)

