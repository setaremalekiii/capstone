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
  
  # def __getitem__(self, idx):
  #   img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
    
  #   if self.transform:
  #     img = self.random_augment(img)

  #   #img = self.classical_straighten_dt(img, max_half_width=16, radius_scale=1.15)
      
  #   # img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
  #   clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
  #   img_clahe = clahe.apply(img)
    
  #   gaussian_3 = cv2.GaussianBlur(img_clahe, (3, 3), 0.5)
  #   img_sharpened = cv2.addWeighted(img_clahe, 2, gaussian_3, -1, 0)

  #   img_letterbox = self.apply_letterbox(img_sharpened)
  #   img_tensor = self.to_tensor(img_letterbox)

  #   # one hot encoding:
  #   img_label = torch.tensor(self.labels[idx]).long()
  #   one_hot_label = F.one_hot(img_label, num_classes=self.num_classes).float()
    
  #   return img_tensor, one_hot_label

  def __getitem__(self, idx):
    img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)

    theta_deg = 0.0
    if self.transform:
        img, theta_deg = self.random_augment(img)   # <-- now returns angle

    img = self.classical_straighten_dt(img, max_half_width=16, radius_scale=1.15)

    # --- your preprocessing ---
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    gaussian_3 = cv2.GaussianBlur(img_clahe, (3, 3), 0.5)
    img_sharpened = cv2.addWeighted(img_clahe, 2, gaussian_3, -1, 0)

    img_letterbox = self.apply_letterbox(img_sharpened)
    img_tensor = self.to_tensor(img_letterbox)

    # --- chromosome one-hot ---
    img_label = torch.tensor(self.labels[idx]).long()
    chr_onehot = F.one_hot(img_label, num_classes=self.num_classes).float()

    # --- rotation sin/cos ---
    theta = math.radians(float(theta_deg))
    rot_cond = torch.tensor([math.sin(theta), math.cos(theta)], dtype=torch.float32)

    # --- combined condition ---
    cond = torch.cat([chr_onehot, rot_cond], dim=0)  # shape: [num_classes + 2]

    return img_tensor, cond
  
  # def random_augment(self, img):
  #   # horizontal flip
  #   if torch.rand(1) < 0.5:
  #       img = cv2.flip(img, 1)
    
  #   # random brightness change
  #   alpha = 1.0 + (torch.rand(1).item() - 0.5) * 0.4  # 0.8 to 1.2
  #   img = np.clip(img * alpha, 0, 255).astype(np.uint8)
  #   return img

  def random_augment(self, img):
    theta_deg = float(np.random.uniform(0, 360))

    if theta_deg != 0:
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    if torch.rand(1).item() < 0.5:
        img = cv2.flip(img, 1)

    alpha = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
    img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    return img, theta_deg
  
  # def apply_letterbox(self, img, color=(114,)):
  #   h, w = img.shape[:2]
  #   if w > h:
  #     # rotate the image if wider:
  #     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
  #     h, w = img.shape[:2] 
      
  #   target_w, target_h = self.target_size
    
  #   scale = min(target_w / w, target_h / h)
  #   new_w, new_h = int(w*scale), int(h*scale)
    
  #   resized_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
    
  #   # calculate padding
  #   pad_w = target_w - new_w 
  #   pad_h = target_h - new_h 
  #   top = pad_h // 2
  #   bottom = pad_h - top 
  #   left = pad_w // 2 
  #   right = pad_w - left
    
  #   # pad img
  #   padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)
  #   return padded_img

  def apply_letterbox(self, img, color=(114,)):
    h, w = img.shape[:2]
    target_w, target_h = self.target_size
    scale = min(target_w / w, (target_h * 0.9) / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return padded_img

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
    if len(path_yx) < 12:
        return img

    #pts = extend_pts_both_ends(path_yx, step=1.0, n_steps=20)
    pts = self._smooth_and_resample_path(path_yx, step=1.0)

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
    # --- pad along the length so tips don't get trimmed by resizing/letterbox ---
    # pad_rows = 12  # try 8-20
    # out = np.pad(out, ((pad_rows, pad_rows), (0, 0)), mode="edge")

    return out

  def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
      return mask
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == k

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

def extend_pts_both_ends(pts, step=1.0, n_steps=20):
    pts = pts.astype(np.float32)
    if len(pts) < 3:
        return pts

    # pick a more stable direction using a few points
    k = min(5, len(pts) - 1)

    # start direction: from point k to point 0
    v0 = pts[0] - pts[k]
    # end direction: from point -k-1 to last
    v1 = pts[-1] - pts[-k-1]

    def safe_unit(v):
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return None
        return v / n

    u0 = safe_unit(v0)
    u1 = safe_unit(v1)

    # if direction is degenerate, don’t extend that side
    start_ext = []
    end_ext = []

    if u0 is not None:
        for s in range(n_steps, 0, -1):
            start_ext.append(pts[0] + u0 * (-step * s))
        start_ext = np.stack(start_ext, axis=0)

    if u1 is not None:
        for s in range(1, n_steps + 1):
            end_ext.append(pts[-1] + u1 * (step * s))
        end_ext = np.stack(end_ext, axis=0)

    if len(start_ext) and len(end_ext):
        return np.vstack([start_ext, pts, end_ext])
    elif len(start_ext):
        return np.vstack([start_ext, pts])
    elif len(end_ext):
        return np.vstack([pts, end_ext])
    else:
        return pts
    
def export_preprocessed_images(
    img_paths,
    target_size=(500, 500 ),
    out_dir="preprocessed_out",
    do_augment=False
):
    os.makedirs(out_dir, exist_ok=True)

    # make a dataset instance so we can reuse your methods
    ds = ChromosomeDataset(img_paths, target_size= (500, 500), transform=do_augment)

    for p in img_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"skip (failed to read): {p}")
            continue

        # --- SAME PIPELINE AS __getitem__ ---
        theta_deg = 0.0
        if do_augment:
            img, theta_deg = ds.random_augment(img)
        
        img = ds.classical_straighten_dt(img, max_half_width=16, radius_scale=1.15)

        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)

        gaussian_3 = cv2.GaussianBlur(img_clahe, (3, 3), 0.5)
        img_sharpened = cv2.addWeighted(img_clahe, 2, gaussian_3, -1, 0)

        img_letterbox = ds.apply_letterbox(img_sharpened)

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
  MAIN_DIR = "C:/Users/pierc/Downloads/CVAE/capstone/CVAE"
  cropped_box_path = os.path.join(MAIN_DIR, 'data', 'cropped_v2')
  recon_dir = os.path.join(cropped_box_path, 'train')
  paths = glob.glob("../data/cropped_v3/train_croppedv3/*.jpg")

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

