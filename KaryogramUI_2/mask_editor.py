# mask_editor_app.py
#
# Full PyQt6 app:
# 1) Launcher: pick Image + YOLO txt + SAM ckpt + output sam_cutouts + optional karyogram script
# 2) Generate masks (runs GeneratorV2.py with CLI args)
# 3) Editor window:
#    - Pinned TOP nav bar
#    - Scrollable CENTER image (QScrollArea)
#    - Pinned BOTTOM controls
#    - Brush mapping fixed (label size == pixmap size; correct mouse coords)
# 4) Karyogram tab shows karyogram.png after generation
#
# Save to: /Users/saatvik_11/Desktop/KaryogramUI/mask_editor_app.py
# Run:
#   cd /Users/saatvik_11/Desktop/KaryogramUI
#   source .venv/bin/activate
#   python3 mask_editor_app.py

import sys
import os
import re
import glob
import subprocess
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QKeySequence, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QMessageBox,
    QFileDialog, QLineEdit, QTabWidget, QScrollArea, QSizePolicy
)

# -------------------------
# QImage helpers
# -------------------------
def cv_bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

def mask_to_rgba_overlay(mask_u8: np.ndarray, alpha: int = 110) -> QImage:
    h, w = mask_u8.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., 0] = 255  # R
    overlay[..., 1] = 0    # G
    overlay[..., 2] = 0    # B
    overlay[..., 3] = (mask_u8 > 0).astype(np.uint8) * alpha
    return QImage(overlay.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()

# -------------------------
# File discovery + sorting
# -------------------------
_CLASS_RE = re.compile(r"class_(\d+)")
_DET_RE = re.compile(r"det_(\d+)")

def sort_key(path: str):
    p = path.replace("\\", "/")
    m = _CLASS_RE.search(p)
    class_id = int(m.group(1)) if m else 10**9
    d = _DET_RE.search(os.path.basename(p))
    det_id = int(d.group(1)) if d else 10**9
    return (class_id, det_id, os.path.basename(p).lower(), p.lower())

def find_cutouts(root_dir: str):
    pattern = os.path.join(root_dir, "**", "det_*_conf_*.png")
    files = glob.glob(pattern, recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=sort_key)
    return files

def derive_mask_path(cutout_path: str):
    base = os.path.basename(cutout_path)
    m = re.match(r"(det_\d+)_conf_.*\.png$", base)
    if m:
        return os.path.join(os.path.dirname(cutout_path), f"{m.group(1)}_mask.png")
    return os.path.splitext(cutout_path)[0] + "_mask.png"


# -------------------------
# Paint widget (FIXED brush mapping)
# -------------------------
class PaintLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # CRITICAL: ensure widget does NOT expand larger than pixmap,
        # otherwise mouse coords include empty padding and brush drifts.
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._drawing = False
        self._last = QPoint(0, 0)

        self.mode = "add"
        self.brush_radius = 10  # in IMAGE pixels

        # Zoom display-only
        self.zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 8.0

        self.base_bgr = None
        self.mask_u8 = None
        self._pix = None

        self.dirty = False

    def load(self, img_path: str, mask_path: str):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros(img.shape[:2], dtype=np.uint8)

        if m.shape[:2] != img.shape[:2]:
            raise ValueError(
                f"Mask size does not match image size.\n"
                f"Image: {img.shape[:2]} Mask: {m.shape[:2]}\n\n"
                f"This usually means the mask and cutout weren't cropped identically."
            )

        self.base_bgr = img
        self.mask_u8 = (m > 127).astype(np.uint8) * 255
        self.dirty = False
        self._render()

    def set_mode(self, mode: str):
        self.mode = mode

    def set_brush_radius(self, r: int):
        self.brush_radius = max(1, int(r))

    def set_zoom(self, z: float):
        z = float(z)
        self.zoom = max(self.min_zoom, min(self.max_zoom, z))
        self._render()

    def _render(self):
        if self.base_bgr is None or self.mask_u8 is None:
            return

        base_q = cv_bgr_to_qimage(self.base_bgr)
        overlay_q = mask_to_rgba_overlay(self.mask_u8, alpha=110)

        pix = QPixmap.fromImage(base_q)
        painter = QPainter(pix)
        painter.drawImage(0, 0, overlay_q)
        painter.end()

        w = max(1, int(pix.width() * self.zoom))
        h = max(1, int(pix.height() * self.zoom))
        scaled = pix.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self._pix = scaled
        self.setPixmap(self._pix)

        # CRITICAL: widget size equals pixmap size -> correct coordinate mapping
        self.setFixedSize(self._pix.size())

    def _paint_at_screen(self, sx: int, sy: int):
        if self.mask_u8 is None:
            return

        # screen coords within label map directly (since label size == pixmap size)
        ix = int(sx / self.zoom)
        iy = int(sy / self.zoom)

        h, w = self.mask_u8.shape[:2]
        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))

        val = 255 if self.mode == "add" else 0
        cv2.circle(self.mask_u8, (ix, iy), self.brush_radius, int(val), thickness=-1)
        self.dirty = True
        self._render()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._last = event.position().toPoint()
            self._paint_at_screen(self._last.x(), self._last.y())

    def mouseMoveEvent(self, event):
        if not self._drawing:
            return
        p = event.position().toPoint()

        x0, y0 = self._last.x(), self._last.y()
        x1, y1 = p.x(), p.y()

        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(n + 1):
            sx = int(round(x0 + (x1 - x0) * i / n))
            sy = int(round(y0 + (y1 - y0) * i / n))
            self._paint_at_screen(sx, sy)

        self._last = p

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False


# -------------------------
# Editor Window with Tabs (Pinned top + bottom)
# -------------------------
class EditorWindow(QMainWindow):
    def __init__(self, root_dir: str, karyo_script: str | None):
        super().__init__()
        self.setWindowTitle("SAM Mask Editor + Karyogram Tab")

        self.root_dir = root_dir
        self.karyo_script = karyo_script

        self.cutouts = find_cutouts(root_dir)
        if not self.cutouts:
            raise FileNotFoundError(
                f"No cutout files found under:\n{root_dir}\n\nExpected pattern: **/det_*_conf_*.png"
            )

        self.idx = 0
        self.img_path = ""
        self.mask_path = ""

        # ---- editor widgets ----
        self.view = PaintLabel()
        self.status = QLabel("")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.btn_prev = QPushButton("← Prev")
        self.btn_next = QPushButton("Next →")
        self.btn_add = QPushButton("Add")
        self.btn_erase = QPushButton("Erase")
        self.btn_save = QPushButton("Save (Ctrl+S)")
        self.btn_karyo = QPushButton("Generate Karyogram")

        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(120)
        self.brush_slider.setValue(10)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(25)
        self.zoom_slider.setMaximum(800)
        self.zoom_slider.setValue(150)

        # Wire up
        self.btn_add.clicked.connect(lambda: self.view.set_mode("add"))
        self.btn_erase.clicked.connect(lambda: self.view.set_mode("erase"))
        self.brush_slider.valueChanged.connect(self.view.set_brush_radius)
        self.zoom_slider.valueChanged.connect(lambda v: self.view.set_zoom(v / 100.0))

        self.btn_save.clicked.connect(self.save_current)
        self.btn_prev.clicked.connect(self.prev_item)
        self.btn_next.clicked.connect(self.next_item)

        self.btn_karyo.clicked.connect(self.run_karyogram)
        self.btn_karyo.setEnabled(bool(self.karyo_script))

        # Shortcuts
        act_save = QAction(self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.save_current)
        self.addAction(act_save)

        act_prev = QAction(self)
        act_prev.setShortcut(QKeySequence(Qt.Key.Key_Left))
        act_prev.triggered.connect(self.prev_item)
        self.addAction(act_prev)

        act_next = QAction(self)
        act_next.setShortcut(QKeySequence(Qt.Key.Key_Right))
        act_next.triggered.connect(self.next_item)
        self.addAction(act_next)

        # -------------------------
        # Editor tab layout (Pinned top + bottom)
        # -------------------------
        # TOP nav bar
        nav_bar = QWidget()
        nav = QHBoxLayout(nav_bar)
        nav.setContentsMargins(10, 10, 10, 6)
        nav.setSpacing(10)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        nav.addWidget(self.status)
        nav_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # CENTER scroll area for image
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(self.view)

        # BOTTOM controls bar
        controls_bar = QWidget()
        controls = QHBoxLayout(controls_bar)
        controls.setContentsMargins(10, 6, 10, 10)
        controls.setSpacing(10)
        controls.addWidget(self.btn_add)
        controls.addWidget(self.btn_erase)
        controls.addWidget(QLabel("Brush"))
        controls.addWidget(self.brush_slider)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Zoom"))
        controls.addWidget(self.zoom_slider)
        controls.addStretch(1)
        controls.addWidget(self.btn_save)
        controls.addWidget(self.btn_karyo)
        controls_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        editor_layout = QVBoxLayout()
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)
        editor_layout.addWidget(nav_bar)
        editor_layout.addWidget(scroll, stretch=1)
        editor_layout.addWidget(controls_bar)

        editor_tab = QWidget()
        editor_tab.setLayout(editor_layout)

        # -------------------------
        # Karyogram tab
        # -------------------------
        self.karyo_label = QLabel("No karyogram generated yet.")
        self.karyo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.karyo_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.karyo_zoom_slider.setMinimum(10)
        self.karyo_zoom_slider.setMaximum(300)
        self.karyo_zoom_slider.setValue(100)
        self.karyo_zoom_slider.valueChanged.connect(self.refresh_karyogram_view)

        self.karyo_status = QLabel("")
        self.karyo_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        kscroll = QScrollArea()
        kscroll.setWidgetResizable(True)
        kscroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        kscroll.setWidget(self.karyo_label)

        kary_controls = QHBoxLayout()
        kary_controls.setContentsMargins(10, 10, 10, 6)
        kary_controls.addWidget(QLabel("Zoom"))
        kary_controls.addWidget(self.karyo_zoom_slider)
        kary_controls.addStretch(1)
        kary_controls.addWidget(self.karyo_status)

        kary_layout = QVBoxLayout()
        kary_layout.setContentsMargins(0, 0, 0, 0)
        kary_layout.setSpacing(0)
        kary_layout.addLayout(kary_controls)
        kary_layout.addWidget(kscroll)

        kary_tab = QWidget()
        kary_tab.setLayout(kary_layout)

        # ---- Tabs ----
        self.tabs = QTabWidget()
        self.tabs.addTab(editor_tab, "Mask Editor")
        self.tabs.addTab(kary_tab, "Karyogram")

        root = QVBoxLayout()
        root.addWidget(self.tabs)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.karyogram_path = None
        self.karyogram_pix = None

        self.load_index(0)

    def load_index(self, i: int):
        i = max(0, min(len(self.cutouts) - 1, i))
        self.idx = i

        self.img_path = self.cutouts[self.idx]
        self.mask_path = derive_mask_path(self.img_path)

        self.view.load(self.img_path, self.mask_path)

        rel_img = os.path.relpath(self.img_path, self.root_dir)
        rel_mask = os.path.relpath(self.mask_path, self.root_dir)
        self.status.setText(f"{self.idx + 1}/{len(self.cutouts)}   {rel_img}   |   mask: {rel_mask}")

        self.btn_prev.setEnabled(self.idx > 0)
        self.btn_next.setEnabled(self.idx < len(self.cutouts) - 1)

    def save_current(self):
        if self.view.mask_u8 is None or self.view.base_bgr is None:
            return

        ok = cv2.imwrite(self.mask_path, self.view.mask_u8)
        if not ok:
            QMessageBox.critical(self, "Save failed", f"Could not write mask:\n{self.mask_path}")
            return

        # Update cutout so downstream karyogram reflects edits
        cutout = self.view.base_bgr.copy()
        cutout[self.view.mask_u8 == 0] = 255
        ok2 = cv2.imwrite(self.img_path, cutout)
        if not ok2:
            QMessageBox.warning(self, "Mask saved", "Mask saved, but failed to update cutout image.")
        else:
            self.view.dirty = False

    def _autosave_if_dirty(self):
        if self.view.dirty:
            self.save_current()

    def next_item(self):
        self._autosave_if_dirty()
        if self.idx < len(self.cutouts) - 1:
            self.load_index(self.idx + 1)

    def prev_item(self):
        self._autosave_if_dirty()
        if self.idx > 0:
            self.load_index(self.idx - 1)

    def _find_karyogram_output(self) -> str | None:
        p1 = os.path.join(self.root_dir, "karyogram.png")
        if os.path.isfile(p1):
            return p1
        if self.karyo_script:
            p2 = os.path.join(os.path.dirname(self.karyo_script), "karyogram.png")
            if os.path.isfile(p2):
                return p2
        return None

    def refresh_karyogram_view(self):
        if self.karyogram_pix is None:
            return
        z = self.karyo_zoom_slider.value() / 100.0
        w = max(1, int(self.karyogram_pix.width() * z))
        h = max(1, int(self.karyogram_pix.height() * z))
        scaled = self.karyogram_pix.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.karyo_label.setPixmap(scaled)
        if self.karyogram_path:
            self.karyo_status.setText(f"{os.path.basename(self.karyogram_path)}   ({w}×{h})")

    def run_karyogram(self):
        if not self.karyo_script:
            QMessageBox.information(self, "No script set", "No karyogram script was provided.")
            return

        self._autosave_if_dirty()

        if not os.path.isfile(self.karyo_script):
            QMessageBox.critical(self, "Script not found", f"Cannot find:\n{self.karyo_script}")
            return

        cmd = [sys.executable, self.karyo_script]
        try:
            p = subprocess.run(
                cmd,
                cwd=os.path.dirname(self.karyo_script) or None,
                capture_output=True,
                text=True
            )
        except Exception as e:
            QMessageBox.critical(self, "Run failed", str(e))
            return

        if p.returncode != 0:
            msg = "Karyogram generation failed.\n\n"
            msg += "STDOUT:\n" + (p.stdout[-4000:] if p.stdout else "(empty)") + "\n\n"
            msg += "STDERR:\n" + (p.stderr[-4000:] if p.stderr else "(empty)")
            QMessageBox.critical(self, "Error", msg)
            return

        out_path = self._find_karyogram_output()
        if not out_path:
            QMessageBox.warning(
                self,
                "Generated but not found",
                "Script ran successfully, but could not find karyogram.png.\n\n"
                "Best fix: update your karyogram generator to always write:\n"
                "  <sam_cutouts_root>/karyogram.png\n"
            )
            return

        pix = QPixmap(out_path)
        if pix.isNull():
            QMessageBox.warning(self, "Load failed", f"Could not load image:\n{out_path}")
            return

        self.karyogram_path = out_path
        self.karyogram_pix = pix
        self.refresh_karyogram_view()
        self.tabs.setCurrentIndex(1)

        QMessageBox.information(self, "Done", "Karyogram generated. Check the Karyogram tab.")


# -------------------------
# Launcher Window
# -------------------------
class Launcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Karyogram Pipeline Launcher")

        self.in_img = QLineEdit()
        self.in_yolo = QLineEdit()
        self.in_sam = QLineEdit()
        self.in_out = QLineEdit()
        self.in_karyo = QLineEdit()

        btn_img = QPushButton("Choose Image")
        btn_yolo = QPushButton("Choose YOLO .txt")
        btn_sam = QPushButton("Choose SAM checkpoint")
        btn_out = QPushButton("Choose Output Folder")
        btn_karyo = QPushButton("Choose Karyogram Script (optional)")

        btn_run = QPushButton("Generate Masks + Open Editor")
        btn_run.setStyleSheet("font-weight: bold; padding: 10px;")

        btn_img.clicked.connect(self.pick_img)
        btn_yolo.clicked.connect(self.pick_yolo)
        btn_sam.clicked.connect(self.pick_sam)
        btn_out.clicked.connect(self.pick_out)
        btn_karyo.clicked.connect(self.pick_karyo)
        btn_run.clicked.connect(self.run_pipeline)

        layout = QVBoxLayout()

        def row(label: str, le: QLineEdit, btn: QPushButton):
            r = QHBoxLayout()
            r.addWidget(QLabel(label))
            r.addWidget(le, stretch=1)
            r.addWidget(btn)
            layout.addLayout(r)

        row("Image:", self.in_img, btn_img)
        row("YOLO txt:", self.in_yolo, btn_yolo)
        row("SAM ckpt (.pt):", self.in_sam, btn_sam)
        row("Output dir (sam_cutouts):", self.in_out, btn_out)
        row("Karyogram script:", self.in_karyo, btn_karyo)

        layout.addWidget(btn_run)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        # Defaults for your folder
        cwd = os.getcwd()
        maybe_sam = os.path.join(cwd, "sam2.1_b.pt")
        if os.path.isfile(maybe_sam):
            self.in_sam.setText(maybe_sam)
        self.in_out.setText(os.path.join(cwd, "sam_cutouts"))

    def pick_img(self):
        p, _ = QFileDialog.getOpenFileName(self, "Pick image", "", "Images (*.png *.jpg *.jpeg)")
        if p:
            self.in_img.setText(p)

    def pick_yolo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Pick YOLO txt", "", "Text (*.txt)")
        if p:
            self.in_yolo.setText(p)

    def pick_sam(self):
        p, _ = QFileDialog.getOpenFileName(self, "Pick SAM checkpoint", "", "Model (*.pt)")
        if p:
            self.in_sam.setText(p)

    def pick_out(self):
        p = QFileDialog.getExistingDirectory(self, "Pick output folder", "")
        if p:
            self.in_out.setText(p)

    def pick_karyo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Pick karyogram script", "", "Python (*.py)")
        if p:
            self.in_karyo.setText(p)

    def run_pipeline(self):
        img = self.in_img.text().strip()
        yolo = self.in_yolo.text().strip()
        sam = self.in_sam.text().strip()
        out = self.in_out.text().strip()
        karyo = self.in_karyo.text().strip() or None

        if not os.path.isfile(img):
            QMessageBox.critical(self, "Missing", "Pick a valid image file.")
            return
        if not os.path.isfile(yolo):
            QMessageBox.critical(self, "Missing", "Pick a valid YOLO txt file.")
            return
        if not os.path.isfile(sam):
            QMessageBox.critical(self, "Missing", "Pick a valid SAM checkpoint (.pt).")
            return
        if not out:
            QMessageBox.critical(self, "Missing", "Pick an output folder.")
            return
        os.makedirs(out, exist_ok=True)

        generator_script = os.path.join(os.getcwd(), "GeneratorV2.py")
        if not os.path.isfile(generator_script):
            QMessageBox.critical(self, "Not found", f"GeneratorV2.py not found in:\n{os.getcwd()}")
            return

        cmd = [
            sys.executable, generator_script,
            "--img", img,
            "--yolo", yolo,
            "--out", out,
            "--sam", sam,
        ]

        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            msg = "Mask generation failed.\n\n"
            msg += "STDOUT:\n" + (p.stdout[-4000:] if p.stdout else "(empty)") + "\n\n"
            msg += "STDERR:\n" + (p.stderr[-4000:] if p.stderr else "(empty)")
            QMessageBox.critical(self, "Error", msg)
            return

        # Open editor window
        try:
            self.editor = EditorWindow(root_dir=out, karyo_script=karyo)
            self.editor.show()
        except Exception as e:
            QMessageBox.critical(self, "Editor failed", str(e))


def main():
    app = QApplication(sys.argv)
    win = Launcher()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()