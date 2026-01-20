"""Class for laoding cropped chromosome dataset"""
import os
import re
import cv2
import torch
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

    self.labels = self.extract_labels(img_paths)
    self.to_tensor = ToTensor()

  def __len__(self):
    return len(self.img_paths)
  
  def __getitem__(self, idx):
    img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
    
    if self.transform:
      img = self.random_augment(img)

    # img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    gaussian_3 = cv2.GaussianBlur(img_clahe, (3, 3), 0.5)
    img_sharpened = cv2.addWeighted(img_clahe, 2, gaussian_3, -1, 0)

    img_letterbox = self.apply_letterbox(img_sharpened)
    img_tensor = self.to_tensor(img_letterbox)

    # one hot encoding:
    img_label = torch.tensor(self.labels[idx]).long()
    one_hot_label = F.one_hot(img_label, num_classes=self.num_classes).float()
    
    return img_tensor, one_hot_label
  
  def random_augment(self, img):
    # horizontal flip
    if torch.rand(1) < 0.5:
        img = cv2.flip(img, 1)
    
    # random brightness change
    alpha = 1.0 + (torch.rand(1).item() - 0.5) * 0.4  # 0.8 to 1.2
    img = np.clip(img * alpha, 0, 255).astype(np.uint8)
    return img

  def apply_letterbox(self, img, color=(114,)):
    h, w = img.shape[:2]
    if w > h:
      # rotate the image if wider:
      img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
      h, w = img.shape[:2] 
      
    target_w, target_h = self.target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    resized_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
    
    # calculate padding
    pad_w = target_w - new_w 
    pad_h = target_h - new_h 
    top = pad_h // 2
    bottom = pad_h - top 
    left = pad_w // 2 
    right = pad_w - left
    
    # pad img
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)
    return padded_img
    
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
  
if __name__ == "__main__":
  MAIN_DIR = "/scratch/st-li1210-1/pearl/karyotype-detector/"
  cropped_box_path = os.path.join(MAIN_DIR, 'data', 'cropped_v2')
  recon_dir = os.path.join(cropped_box_path, 'train')
  paths = glob.glob(f"{recon_dir}/*.jpg")

  dataset = ChromosomeDataset(paths, (64,64), True)
  recon_path = os.path.join(MAIN_DIR, f"data.png")
  tensor_img, _ = dataset[3]
  save_image(tensor_img, recon_path, normalize=True)

