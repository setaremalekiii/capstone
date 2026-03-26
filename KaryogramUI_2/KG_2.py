import json
import re
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

CUTOUTS_DIR = "/Users/saatvik_11/Desktop/KaryogramUI_2/preprocessed_train"
OUT_PATH    = "/Users/saatvik_11/Desktop/KaryogramUI_2/karyogram.png"
LAYOUT_PATH = "/Users/saatvik_11/Desktop/KaryogramUI_2/karyogram_layout.json"

USE_XY = True
X_CLASS = 22
Y_CLASS = 23

COLS = 6
CELL_W = 240
CELL_H = 280
MARGIN = 25
PAD_BETWEEN = 12
LABEL_H = 30

SHOW_PER_CLASS = 2
PICK_METHOD = "largest_area"   
MAX_CANDIDATES_PER_CLASS = 10  

MAX_IMG_H = CELL_H - LABEL_H - 12



FNAME_RE = re.compile(r"^.*_class(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def class_label(cls: int) -> str:
    if USE_XY and cls == X_CLASS:
        return "X"
    if USE_XY and cls == Y_CLASS:
        return "Y"
    return str(cls + 1)


def class_sort_key(cls: int):
    if USE_XY:
        if cls == X_CLASS:
            return 10_000
        if cls == Y_CLASS:
            return 10_001
    return cls


def resize_keep_aspect(im: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = im.shape[:2]
    scale = min(max_w / w, max_h / h, 1.75)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)


def safe_paste(dst: np.ndarray, src: np.ndarray, x: int, y: int):
    H, W = dst.shape[:2]
    h, w = src.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    dst[y1:y2, x1:x2] = src[sy1:sy2, sx1:sx2]


def load_images_by_class(flat_dir: str):
    """
    NEW FORMAT:
    - One folder (no class subfolders)
    - Files like: 100_class0_0.jpg, 100_class0_1.jpg, ..., 100_class23_1.jpg
    - cls is 0..23
    - homolog index i is 0 or 1
    """
    base = Path(flat_dir)
    if not base.exists():
        raise FileNotFoundError(f"Folder not found: {base}")

    grouped = defaultdict(list)

    for img_path in sorted(base.iterdir()):
        if not img_path.is_file():
            continue

        m = FNAME_RE.match(img_path.name)
        if not m:
            continue

        cls = int(m.group(1))
        pair_i = int(m.group(2))

        if not (0 <= cls <= 23):
            continue
        if pair_i not in (0, 1):
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue

        grouped[cls].append({
            "image": im,
            "cutout_path": str(img_path),
            "mask_path": None,   
            "meta_path": None,   
            "pair_i": pair_i,
            "area": im.shape[0] * im.shape[1],
        })

    if MAX_CANDIDATES_PER_CLASS is not None:
        for cls, items in grouped.items():
            
            items_sorted = sorted(items, key=lambda x: (x["pair_i"], -x["area"]))
            grouped[cls] = items_sorted[:MAX_CANDIDATES_PER_CLASS]

    return grouped


def pick_images(items, k):
    """
    Prefer homolog order: i=0 then i=1.
    If duplicates exist for same homolog, pick largest area first (optional).
    """
    if not items:
        return []

    if PICK_METHOD == "largest_area":
        items = sorted(items, key=lambda x: (x.get("pair_i", 0), -x["area"]))
    else:
        items = sorted(items, key=lambda x: x.get("pair_i", 0))

    return items[:k]


def standard_class_order():
    if USE_XY:
        return list(range(0, 22)) + [X_CLASS, Y_CLASS]
    return None


def build_karyogram(grouped):
    std = standard_class_order()
    if std is None:
        classes = sorted(grouped.keys(), key=class_sort_key)
    else:
        classes = [c for c in std if c in grouped]

    rows = int(np.ceil(len(classes) / COLS))
    canvas_h = rows * (CELL_H + MARGIN) + MARGIN
    canvas_w = COLS * (CELL_W + MARGIN) + MARGIN
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    layout = []

    for idx, cls in enumerate(classes):
        r = idx // COLS
        c = idx % COLS
        x0 = MARGIN + c * (CELL_W + MARGIN)
        y0 = MARGIN + r * (CELL_H + MARGIN)

        cv2.putText(
            canvas, class_label(cls),
            (x0 + 6, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (0, 0, 0), 2, cv2.LINE_AA
        )

        items = grouped.get(cls, [])
        chosen = pick_images(items, SHOW_PER_CLASS)
        if not chosen:
            continue

        if len(chosen) == 1:
            max_w_each = CELL_W - 10
        else:
            max_w_each = (CELL_W - PAD_BETWEEN - 10) // 2

        resized = []
        for item in chosen:
            im = resize_keep_aspect(item["image"], max_w_each, MAX_IMG_H)
            resized.append((im, item))

        total_w = sum(im.shape[1] for im, _ in resized) + PAD_BETWEEN * (len(resized) - 1)
        start_x = x0 + max(0, (CELL_W - total_w) // 2)

        avail_h = CELL_H - LABEL_H
        max_disp_h = max(im.shape[0] for im, _ in resized)
        start_y = y0 + LABEL_H + max(0, (avail_h - max_disp_h) // 2)

        cur_x = start_x
        for im, item in resized:
            safe_paste(canvas, im, cur_x, start_y)

            x1 = cur_x
            y1 = start_y
            x2 = cur_x + im.shape[1]
            y2 = start_y + im.shape[0]

            layout.append({
                "cls": cls,
                "cutout_path": item["cutout_path"],
                "mask_path": item["mask_path"],
                "meta_path": item["meta_path"],
                "placed_xyxy": [int(x1), int(y1), int(x2), int(y2)]
            })

            cur_x += im.shape[1] + PAD_BETWEEN

    return canvas, layout


def main():
    grouped = load_images_by_class(CUTOUTS_DIR)
    if not grouped:
        raise RuntimeError(
            "No images found. Expected files like '100_class0_0.jpg' in: "
            + str(CUTOUTS_DIR)
        )

    karyo, layout = build_karyogram(grouped)

    ok = cv2.imwrite(OUT_PATH, karyo)
    if not ok:
        raise RuntimeError(f"Failed to write image to: {OUT_PATH}")

    with open(LAYOUT_PATH, "w") as f:
        json.dump(layout, f, indent=2)

    print("Saved karyogram:", OUT_PATH)
    print("Saved layout:", LAYOUT_PATH)


if __name__ == "__main__":
    main()