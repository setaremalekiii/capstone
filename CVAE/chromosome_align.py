"""Class for laoding cropped chromosome dataset"""
import os
import re
import cv2
import torch
import math
import yaml
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import numpy as np
import glob
from torchvision.utils import save_image

class ChromosomeDataset(Dataset):
  def __init__(self, img_paths, target_size, transform = False):
    self.img_paths = img_paths
    self.target_size = target_size
    self.transform = transform
    self.num_classes = 24

    self.do_straighten = True
    self.straight_half_width = 16  # output strip width = 2*16+1 = 33 px


    #self.labels = self.extract_labels(img_paths)
    self.to_tensor = ToTensor()

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)

  def apply_letterbox(self, img, color=(114,)):
    h, w = img.shape[:2]
    target_w, target_h = self.target_size
    scale = min(target_w / w, (target_h * 0.9) / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_img

  def classical_straighten_dt(self, img: np.ndarray, max_half_width: int = 16, radius_scale: float = 1.15) -> np.ndarray:
    g = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)

    mask = th > 0
    mask = self._keep_largest_component(mask)
    if mask.sum() < 80:
        return img

    # background estimate from outside the chromosome mask (more stable than median of whole image)
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

    #pts = extend_pts_both_ends(path_yx, step=1, n_steps=3)
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

        # fill outside radius with background (instead of 0)
        samp[outside] = bg

        # ✅ THIS LINE WAS MISSING / INDENTED WRONG IN YOUR VERSION
        out[i, :] = samp

    # ✅ no global normalization (prevents stripe amplification)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

  def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
      return mask
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == k
  
  def rotate_chromosome_to_vertical(self, img: np.ndarray, vertical_tol: float = 10.0) -> np.ndarray:
    import cv2
    import numpy as np

    g = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = th > 0
    mask = self._keep_largest_component(mask)
    if mask.sum() < 80:
        return img

    ys, xs = np.where(mask)
    if len(xs) < 2:
        return img

    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0, keepdims=True)
    X = pts - mean

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]   # [vx, vy]

    vx, vy = float(v[0]), float(v[1])

    # axis angle relative to +x axis
    angle = np.degrees(np.arctan2(vy, vx)) % 180.0

    # rotation needed to make axis vertical
    rot_deg = 90.0 - angle
    if rot_deg > 90.0:
        rot_deg -= 180.0
    elif rot_deg < -90.0:
        rot_deg += 180.0

    # skip if already close enough to vertical
    if abs(rot_deg) <= vertical_tol:
        return img

    print(f"angle={angle:.2f}, rot_deg={rot_deg:.2f}")

    H, W = img.shape
    center = (W / 2.0, H / 2.0)
    M = cv2.getRotationMatrix2D(center, -rot_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_W = int(H * sin + W * cos)
    new_H = int(H * cos + W * sin)

    M[0, 2] += (new_W / 2.0) - center[0]
    M[1, 2] += (new_H / 2.0) - center[1]

    rotated = cv2.warpAffine(
        img,
        M,
        (new_W, new_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    return rotated
  
  def rotate_chromosome_by_endpoints(self, img: np.ndarray) -> np.ndarray:
    import cv2
    import numpy as np

    g = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = th > 0
    mask = self._keep_largest_component(mask)
    if mask.sum() < 80:
        return img

    skel = self._skeletonize(mask)
    ends = self._skeleton_endpoints(skel)
    if len(ends) < 2:
        return img

    ends = np.array(ends, dtype=np.float32)

    # choose farthest pair if more than 2 endpoints are found
    if len(ends) > 2:
        best_i, best_j, best_d = 0, 1, -1
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                d = np.sum((ends[i] - ends[j]) ** 2)
                if d > best_d:
                    best_d = d
                    best_i, best_j = i, j
        p1, p2 = ends[best_i], ends[best_j]
    else:
        p1, p2 = ends[0], ends[1]

    dy = float(p2[0] - p1[0])
    dx = float(p2[1] - p1[1])

    angle = np.degrees(np.arctan2(dy, dx)) % 180.0

    # how far from vertical
    print(angle)
    # how far from vertical
    deviation = abs(angle)

    vertical_tol = 45.0

    if 65 < deviation < 115:
        return img   # don't rotate

    # otherwise rotate
    rot_deg = 90.0 - angle
    if rot_deg > 90:
        rot_deg -= 180
    H, W = img.shape
    center = (W / 2.0, H / 2.0)
    M = cv2.getRotationMatrix2D(center, -rot_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_W = int(H * sin + W * cos)
    new_H = int(H * cos + W * sin)

    M[0, 2] += (new_W / 2.0) - center[0]
    M[1, 2] += (new_H / 2.0) - center[1]

    rotated = cv2.warpAffine(
        img,
        M,
        (new_W, new_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    return rotated

  def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    return skeletonize(mask).astype(np.uint8)

  def _skeleton_endpoints(self, skel: np.ndarray):
    sk = skel.astype(np.uint8)
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], np.uint8)
    nbr = cv2.filter2D(sk, -1, kernel)
    ys, xs = np.where(nbr == 11)
    return list(zip(ys.tolist(), xs.tolist()))
  
  def _trace_skeleton_path(self, skel: np.ndarray, start):
    """
    Non-greedy skeleton tracing:
    - Build 8-neighborhood graph over skeleton pixels
    - Find a long main path using two BFS passes (graph diameter heuristic)
    - Return the shortest path between the two farthest nodes
    """
    sk = (skel > 0).astype(np.uint8)
    H, W = sk.shape

    # 8-neighborhood
    moves = [(-1,-1),(-1,0),(-1,1),
             (0,-1),        (0,1),
             (1,-1),(1,0),(1,1)]

    # List all skeleton pixels
    ys, xs = np.where(sk)
    if len(ys) == 0:
        return np.array([start], dtype=np.float32)

    coords = list(zip(ys.tolist(), xs.tolist()))
    idx_of = {p: i for i, p in enumerate(coords)}

    # Build adjacency list + find endpoints (degree == 1)
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
        """Return (dist, parent, farthest_idx)."""
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

    # Choose a BFS start:
    # Prefer an endpoint (more stable). If none, use provided start if it's on skeleton; else any skeleton pixel.
    if endpoints:
        s0 = endpoints[0]
    else:
        s0 = idx_of.get(tuple(start), 0)

    # BFS twice to get a long path (diameter heuristic)
    _, _, a = bfs(s0)
    _, parent, b = bfs(a)

    # Reconstruct path from b back to a
    path_idx = []
    cur = b
    while cur != -1:
        path_idx.append(cur)
        if cur == a:
            break
        cur = int(parent[cur])

    # If something went wrong, fallback to single point
    if not path_idx:
        return np.array([start], dtype=np.float32)

    path_idx.reverse()
    path = np.array([coords[i] for i in path_idx], dtype=np.float32)  # (y, x)
    return path

  def _smooth_and_resample_path(self, path_yx: np.ndarray, step: float = 1.0) -> np.ndarray:
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

  def extract_labels(self, img_paths):
    labels = []
    for path in img_paths:
      filename = os.path.basename(path)
      # img name is in format: "{image_base_name}_class{int(class_id)}_{i}.jpg"
      match = re.search(r"class(\d+)", filename)
      if match:
        labels.append(int(match.group(1)))
      else:
        raise ValueError(f"Could not extract class_id from {path}")
    return labels

def export_preprocessed_images(
    img_paths,
    target_size=(64, 64),
    out_dir="preprocessed_out",
    do_augment=False
):
    os.makedirs(out_dir, exist_ok=True)

    # make a dataset instance so we can reuse your methods
    ds = ChromosomeDataset(img_paths, target_size= (254, 254), transform=do_augment)
    all_imgs = []
    for p in img_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"skip (failed to read): {p}")
            continue
        
        img = ds.rotate_chromosome_to_vertical(img)

        img_letterbox = ds.apply_letterbox(img)

        # ensure uint8 [0..255]
        if img_letterbox.dtype != np.uint8:
            img_letterbox = np.clip(img_letterbox, 0, 255).astype(np.uint8)

        base = os.path.basename(p)
        name, ext = os.path.splitext(base)

        # optional: include rotation in filename if augmented
        if do_augment:
            out_name = f"{name}_augRot{theta_deg:.1f}.jpg"
        else:
            out_name = f"{name}.jpg"

        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, img_letterbox)
    
    print(f"Done. Wrote {len(img_paths)} images to: {out_dir}")
  
  
if __name__ == "__main__":
  MAIN_DIR = "C:/Users/Owner/Downloads/Capstone images/Images with annotation/CVAE Folder/model/CVAE"
  cropped_box_path = os.path.join(MAIN_DIR, 'data', 'cropped_v2')
  recon_dir = os.path.join(cropped_box_path, 'train')
  paths = glob.glob("../../data/cropped_v3/train_croppedv3/*.jpg")

  #dataset = ChromosomeDataset(paths, (64,64), True)
  recon_path = os.path.join(MAIN_DIR, f"data.png")
  # tensor_img, _ = dataset[3]
  # save_image(tensor_img, recon_path, normalize=True)

  export_preprocessed_images(
        img_paths=paths,
        target_size=(64, 64),
        out_dir=os.path.join(MAIN_DIR, "data", "preprocessed_train"),
        do_augment=False  # set True if you want rotated/flipped/brightness versions saved
    )
