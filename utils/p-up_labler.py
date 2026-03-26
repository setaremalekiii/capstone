#!/usr/bin/env python3
"""
label_flip_ui_json_fullscreen.py

Full-screen OpenCV labeling UI for flip labels saved as JSON dict:
  { "relative/path.png": 0 or 1 }

Keys:
  f = flip (1)
  k = keep (0)
  s = skip
  b = back (undo last label)
  q / ESC = quit
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
WIN_NAME = "Label Flip UI (JSON)"


def get_screen_size_fallback() -> Tuple[int, int]:
    """Try to get screen size; fallback if unavailable."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1920, 1080


def list_images(in_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    else:
        files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    return sorted(files, key=lambda p: p.as_posix())


def load_labels_json(labels_path: Path) -> Dict[str, int]:
    if not labels_path.exists():
        return {}
    with labels_path.open("r") as f:
        data = json.load(f)
    out: Dict[str, int] = {}
    for k, v in data.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def save_labels_json(labels_path: Path, labels: Dict[str, int]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = labels_path.with_suffix(labels_path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    tmp.replace(labels_path)


def rel_key(in_dir: Path, p: Path) -> str:
    try:
        return p.relative_to(in_dir).as_posix()
    except Exception:
        return p.as_posix()


def draw_text_top_right(canvas: np.ndarray, lines: List[str], pad: int = 18) -> None:
    """Draw right-aligned text in the top-right corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thick_bg = 3
    thick_fg = 1

    y = pad + 22
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thick_fg)
        x = canvas.shape[1] - pad - tw

        # shadow/back stroke
        cv2.putText(canvas, line, (x, y), font, scale, (0, 0, 0), thick_bg, cv2.LINE_AA)
        cv2.putText(canvas, line, (x, y), font, scale, (255, 255, 255), thick_fg, cv2.LINE_AA)

        y += th + 14


def make_centered_canvas(
    img_bgr: np.ndarray,
    screen_w: int,
    screen_h: int,
    bg: int = 40,
    max_frac: float = 0.80,
) -> np.ndarray:
    """
    Create full-screen canvas and place the image centered.
    Image is scaled to fit within max_frac of screen (but can be scaled up too).
    """
    canvas = np.full((screen_h, screen_w, 3), bg, dtype=np.uint8)

    h, w = img_bgr.shape[:2]
    max_w = int(screen_w * max_frac)
    max_h = int(screen_h * max_frac)

    # scale factor to fit within max_w/max_h
    scale = min(max_w / w, max_h / h)
    scale = max(scale, 1e-6)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)

    x0 = (screen_w - new_w) // 2
    y0 = (screen_h - new_h) // 2

    # paste
    canvas[y0:y0 + new_h, x0:x0 + new_w] = img_resized
    return canvas


def wait_for_key() -> int:
    """
    Use a small wait loop so the window stays responsive.
    Returns key code (lower 8 bits) or -1.
    """
    while True:
        k = cv2.waitKeyEx(30)
        if k != -1:
            return k & 0xFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_json", default="labels.json")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--screen_w", type=int, default=0)
    ap.add_argument("--screen_h", type=int, default=0)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_json = Path(args.out_json)

    if not in_dir.exists():
        print(f"IN_DIR does not exist: {in_dir}", file=sys.stderr)
        sys.exit(1)

    labels = load_labels_json(out_json)
    files = list_images(in_dir, recursive=args.recursive)
    if not files:
        print(f"No images found in: {in_dir}")
        sys.exit(0)

    # screen size
    sw, sh = (args.screen_w, args.screen_h)
    if sw <= 0 or sh <= 0:
        sw, sh = get_screen_size_fallback()

    # Prepare window — always fullscreen
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WIN_NAME, 0, 0)
    cv2.resizeWindow(WIN_NAME, sw, sh)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # index map for undo
    key_to_index = {rel_key(in_dir, p): i for i, p in enumerate(files)}
    history: List[Tuple[str, int]] = []

    print(f"Found {len(files)} images. Already labeled: {len(labels)}.")
    print("Keys: f=flip(1), k=keep(0), s=skip, b=back, q/ESC=quit")
    print(f"Saving to: {out_json}")

    idx = 0
    while idx < len(files):
        p = files[idx]
        key = rel_key(in_dir, p)

        # skip already labeled (resume)
        if key in labels:
            idx += 1
            continue

        img = cv2.imread(p.as_posix())
        if img is None:
            print(f"[SKIP unreadable] {key}")
            idx += 1
            continue

        canvas = make_centered_canvas(img, sw, sh, bg=40, max_frac=0.80)

        overlay = [
            f"{idx+1}/{len(files)}  remaining (unlabeled) shown",
            f"file: {key}",
            f"labeled_total={len(labels)}   time={datetime.now().strftime('%H:%M:%S')}",
            "f=flip(1)  k=keep(0)  s=skip  b=back  q=quit",
        ]
        draw_text_top_right(canvas, overlay)

        cv2.imshow(WIN_NAME, canvas)
        k = wait_for_key()

        if k in (27, ord("q"), ord("Q")):  # ESC or q
            break

        if k in (ord("s"), ord("S")):
            idx += 1
            continue

        if k in (ord("f"), ord("F")):
            labels[key] = 1
            save_labels_json(out_json, labels)
            history.append((key, 1))
            idx += 1
            continue

        if k in (ord("k"), ord("K")):
            labels[key] = 0
            save_labels_json(out_json, labels)
            history.append((key, 0))
            idx += 1
            continue

        if k in (ord("b"), ord("B")):
            if not history:
                continue
            last_key, _ = history.pop()
            if last_key in labels:
                labels.pop(last_key, None)
                save_labels_json(out_json, labels)
            # jump back to that image
            idx = key_to_index.get(last_key, max(idx - 1, 0))
            continue

        # ignore unknown keys

    cv2.destroyAllWindows()
    save_labels_json(out_json, labels)
    print(f"Saved labels to: {out_json}  (count={len(labels)})")


if __name__ == "__main__":
    main()