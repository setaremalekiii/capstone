#!/usr/bin/env python3
"""
Group JPEG images into per-class folders by parsing filenames like:
  1101374_class6_4.jpg  -> class6

Copies files into:
    output file that you indicate 
Usage:
  python split_by_class.py

Notes:
- Only copies .jpg/.jpeg files (case-insensitive).
- Skips files that don't match the expected pattern.
"""

import os
import re
import shutil
from pathlib import Path

# INPUT folder containing your training images
# you will need to  change this to your own path
INPUT_DIR = Path("C:/Users/setim/Desktop/year 5/457/seperate_by_class/train_croppedv3")

# OUTPUT base folder where class subfolders will be created
# Same as this one 
OUTPUT_BASE = Path("C:\Users\Hanna\Documents\GitHub\capstone\CVAE\class_2")

# Matches "..._class<number>_....jpg"
CLASS_RE = re.compile(r"(?:^|_)class(\d+)(?:_|$)", re.IGNORECASE)

VALID_EXTS = {".jpg", ".jpeg"}


def extract_class_label(filename: str) -> str | None:
    """
    Returns something like 'class6' from a filename, or None if no class found.
    """
    m = CLASS_RE.search(filename)
    if not m:
        return None
    return f"class{m.group(1)}"


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    MAX_IMAGES = 10

    for entry in INPUT_DIR.iterdir():

        if copied >= MAX_IMAGES:
            break
        
        if not entry.is_file():
            continue

        ext = entry.suffix.lower()
        if ext not in VALID_EXTS:
            continue

        class_label = extract_class_label(entry.name)
        if class_label is None:
            skipped += 1
            continue

        class_dir = OUTPUT_BASE / class_label
        class_dir.mkdir(parents=True, exist_ok=True)

        dst = class_dir / entry.name

        # Copy2 preserves metadata; change to shutil.copy if you don't care.
        shutil.copy2(entry, dst)
        copied += 1

    print(f"Done.")
    print(f"Copied:  {copied}")
    print(f"Skipped (no class in filename): {skipped}")
    print(f"Output: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
