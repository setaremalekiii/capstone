

import os
import re
import cv2
import math
import glob
import shutil
import argparse
import numpy as np


class Straightener:
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size

    def apply_letterbox(self, img, color=(114,)):
        """
        NOTE: Despite the name, this version intentionally DOES NOT pad.
        It only rescales to fit within target_size (to avoid adding gray borders).
        """
        h, w = img.shape[:2]
        target_w, target_h = self.target_size
        scale = min(target_w / max(w, 1), (target_h * 0.9) / max(h, 1))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized_img

    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num <= 1:
            return mask.astype(bool)
        k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == k)

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        # requires scikit-image
        from skimage.morphology import skeletonize
        return skeletonize(mask).astype(np.uint8)

    def _skeleton_endpoints(self, skel: np.ndarray):
        sk = skel.astype(np.uint8)
        kernel = np.array([[1, 1, 1],
                           [1,10, 1],
                           [1, 1, 1]], np.uint8)
        nbr = cv2.filter2D(sk, -1, kernel)
        ys, xs = np.where(nbr == 11)
        return list(zip(ys.tolist(), xs.tolist()))

    def _trace_skeleton_path(self, skel: np.ndarray, start):
        """
        Graph-based skeleton tracing (diameter heuristic) for stability.
        """
        sk = (skel > 0).astype(np.uint8)
        H, W = sk.shape

        moves = [(-1,-1),(-1,0),(-1,1),
                 (0,-1),       (0,1),
                 (1,-1),(1,0),(1,1)]

        ys, xs = np.where(sk)
        if len(ys) == 0:
            return np.array([start], dtype=np.float32)

        coords = list(zip(ys.tolist(), xs.tolist()))
        idx_of = {p: i for i, p in enumerate(coords)}

        adj = [[] for _ in range(len(coords))]
        endpoints = []
        for i, (y, x) in enumerate(coords):
            deg = 0
            for dy, dx in moves:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and sk[ny, nx]:
                    j = idx_of.get((ny, nx))
                    if j is not None:
                        adj[i].append(j)
                        deg += 1
            if deg == 1:
                endpoints.append(i)

        def bfs(src_idx: int):
            n = len(coords)
            dist = np.full(n, -1, dtype=np.int32)
            parent = np.full(n, -1, dtype=np.int32)
            q = [src_idx]
            dist[src_idx] = 0
            head = 0
            far = src_idx

            while head < len(q):
                u = q[head]
                head += 1
                if dist[u] > dist[far]:
                    far = u
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
            return dist, parent, far

        if endpoints:
            s0 = endpoints[0]
        else:
            s0 = idx_of.get(tuple(start), 0)

        _, _, a = bfs(s0)
        _, parent, b = bfs(a)

        path_idx = []
        cur = b
        while cur != -1:
            path_idx.append(cur)
            if cur == a:
                break
            cur = int(parent[cur])

        if not path_idx:
            return np.array([start], dtype=np.float32)

        path_idx.reverse()
        path = np.array([coords[i] for i in path_idx], dtype=np.float32)
        return path

    def _smooth_and_resample_path(self, path_yx: np.ndarray, step: float = 1.0) -> np.ndarray:
        # requires scipy
        from scipy.signal import savgol_filter

        L = len(path_yx)
        if L < 7:
            pts = path_yx.astype(np.float32)
        else:
            max_win = L if L % 2 == 1 else L - 1
            win = min(31, max_win)
            if win < 5:
                win = max_win
            if win % 2 == 0:
                win -= 1
            poly = 3 if win >= 7 else 2
            y = savgol_filter(path_yx[:, 0], win, poly, mode="interp")
            x = savgol_filter(path_yx[:, 1], win, poly, mode="interp")
            pts = np.stack([y, x], axis=1).astype(np.float32)

        if len(pts) < 2:
            return pts
        d = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
        s = np.concatenate([[0.0], np.cumsum(d)])
        total = float(s[-1])
        if total < 2:
            return pts
        new_s = np.arange(0.0, total, step, dtype=np.float32)
        y2 = np.interp(new_s, s, pts[:, 0]).astype(np.float32)
        x2 = np.interp(new_s, s, pts[:, 1]).astype(np.float32)
        return np.stack([y2, x2], axis=1)

    def classical_straighten_dt(self, img: np.ndarray, max_half_width: int = 16, radius_scale: float = 1.15) -> np.ndarray:
        """
        Straighten grayscale chromosome image into a strip of width (2*max_half_width+1).
        """
        g = cv2.GaussianBlur(img, (5, 5), 0)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)

        mask = th > 0
        mask = self._keep_largest_component(mask)
        if mask.sum() < 80:
            return img

        img_f = img.astype(np.float32)
        bg = float(np.median(img_f[~mask])) if (~mask).any() else float(np.median(img_f))

        dist = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)

        valid_r = dist[dist > 0]
        r_ref = float(np.median(valid_r)) if valid_r.size else 0.0
        hw_floor = int(np.clip(radius_scale * r_ref * 0.7, 6, max_half_width))

        skel = self._skeletonize(mask)
        ends = self._skeleton_endpoints(skel)
        if len(ends) < 2:
            return img

        path_yx = self._trace_skeleton_path(skel, ends[0])
        if len(path_yx) < 15:
            return img

        pts = self._smooth_and_resample_path(path_yx, step=1)

        dy = np.gradient(pts[:, 0])
        dx = np.gradient(pts[:, 1])
        nrm = np.sqrt(dy * dy + dx * dx) + 1e-6
        ty, tx = dy / nrm, dx / nrm

        ty = cv2.GaussianBlur(ty.reshape(-1, 1), (9, 1), 0).flatten()
        tx = cv2.GaussianBlur(tx.reshape(-1, 1), (9, 1), 0).flatten()
        nrm = np.sqrt(ty * ty + tx * tx) + 1e-6
        ty, tx = ty / nrm, tx / nrm

        ny, nx = -tx, ty

        L = len(pts)
        width = 2 * max_half_width + 1
        out = np.zeros((L, width), dtype=np.float32)

        offsets_full = np.arange(-max_half_width, max_half_width + 1, dtype=np.float32)

        H, W = img.shape
        for i, (cy, cx) in enumerate(pts):
            iy = int(round(cy))
            ix = int(round(cx))
            iyc = min(max(iy, 0), H - 1)
            ixc = min(max(ix, 0), W - 1)
            r = dist[iyc, ixc]
            local_hw = int(np.clip(radius_scale * r, 3, max_half_width))
            local_hw = max(local_hw, hw_floor)

            offsets = offsets_full.copy()
            outside = (offsets < -local_hw) | (offsets > local_hw)

            ys = cy + offsets * ny[i]
            xs = cx + offsets * nx[i]

            map_x = xs.astype(np.float32)[None, :]
            map_y = ys.astype(np.float32)[None, :]
            samp = cv2.remap(
                img_f, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )[0, :]

            samp[outside] = bg
            out[i, :] = samp

        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def preprocess(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Full pipeline:
          straighten -> CLAHE -> sharpen -> resize (no padding)
        """
        img = self.classical_straighten_dt(img_gray, max_half_width=16, radius_scale=1.15)

        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(7, 7))
        img_clahe = clahe.apply(img)

        gaussian_3 = cv2.GaussianBlur(img_clahe, (3, 3), 0.5)
        img_sharpened = cv2.addWeighted(img_clahe, 2, gaussian_3, -1, 0)

        out = self.apply_letterbox(img_sharpened)

        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)

        return out


# -----------------------
# GeneratorV2-style I/O
# -----------------------

def is_cutout_image(filename: str) -> bool:
    # Keep GeneratorV2 cutout images, exclude masks/meta
    # e.g. det_01_conf_0.923.png
    if filename.endswith("_mask.png"):
        return False
    if filename.endswith("_meta.json"):
        return False
    return bool(re.match(r"^det_\d+_conf_.*\.(png|jpg|jpeg)$", filename, flags=re.IGNORECASE))


def straighten_generatorv2_cutouts(
    in_root: str,
    out_root: str = "straightened_cutouts",
    target_size=(64, 64),
    copy_masks_and_meta: bool = True,
):
    in_root = str(in_root)
    out_root = str(out_root)

    if not os.path.isdir(in_root):
        raise FileNotFoundError(f"Input folder not found: {in_root}")

    os.makedirs(out_root, exist_ok=True)

    straightener = Straightener(target_size=target_size)

    class_dirs = sorted(glob.glob(os.path.join(in_root, "class_*")))
    if not class_dirs:
        raise RuntimeError(f"No class_* folders found under: {in_root}")

    n_written = 0
    n_skipped = 0

    for cls_dir in class_dirs:
        cls_name = os.path.basename(cls_dir)
        out_cls_dir = os.path.join(out_root, cls_name)
        os.makedirs(out_cls_dir, exist_ok=True)

        for fn in sorted(os.listdir(cls_dir)):
            in_path = os.path.join(cls_dir, fn)

            # copy meta/mask if requested
            if copy_masks_and_meta and (fn.endswith("_mask.png") or fn.endswith("_meta.json")):
                shutil.copy2(in_path, os.path.join(out_cls_dir, fn))
                continue

            if not is_cutout_image(fn):
                continue

            img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[skip] failed to read: {in_path}")
                n_skipped += 1
                continue

            try:
                out_img = straightener.preprocess(img)
            except Exception as e:
                print(f"[skip] error straightening {in_path}: {e}")
                n_skipped += 1
                continue

            out_path = os.path.join(out_cls_dir, fn)  
            ok = cv2.imwrite(out_path, out_img)
            if not ok:
                print(f"[skip] failed to write: {out_path}")
                n_skipped += 1
                continue

            n_written += 1

    print(f"Done. Wrote {n_written} straightened images to: {out_root}")
    if copy_masks_and_meta:
        print("Also copied *_mask.png and *_meta.json files as-is.")
    if n_skipped:
        print(f"Skipped: {n_skipped}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Path to GeneratorV2 output folder (e.g., sam_cutouts)")
    ap.add_argument("--out_root", default="straightened_cutouts", help="Output root folder name/path")
    ap.add_argument("--target_w", type=int, default=64)
    ap.add_argument("--target_h", type=int, default=64)
    ap.add_argument("--no_copy_aux", action="store_true", help="Do not copy *_mask.png and *_meta.json")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    straighten_generatorv2_cutouts(
        in_root=args.in_root,
        out_root=args.out_root,
        target_size=(args.target_w, args.target_h),
        copy_masks_and_meta=(not args.no_copy_aux),
    )


if __name__ == "__main__":
    main()