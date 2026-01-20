"""Script for training the ConvCVAE"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ConvCVAE import ConvCVAE
import logging
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure
import pickle
import json

def calculate_class_thresholds(anomaly_scores, class_labels, output_dir):
  print("Calculating class threshold...")
  anomaly_scores = np.array(anomaly_scores)
  class_labels = np.array(class_labels)
  unique_classes = np.unique(class_labels)

  global_min = anomaly_scores.min()
  global_max = anomaly_scores.max() if anomaly_scores.max() > anomaly_scores.min() else global_min + 1e-6

  class_stats = {}
  class_thresholds = {}
  class_thresholds_normalized = {}
  
  for cls in unique_classes:
    cls_mask = class_labels == cls
    cls_scores = anomaly_scores[cls_mask]
    
    if len(cls_scores) > 10:
      cls_mean = np.mean(cls_scores)
      cls_var = np.var(cls_scores)
      cls_threshold = cls_mean + 2 * np.sqrt(cls_var)
    else:
      cls_mean = np.mean(anomaly_scores)
      cls_var = np.var(anomaly_scores)
      cls_threshold = cls_mean + 2 * np.sqrt(cls_var)
      
    cls_threshold_normalized = (cls_threshold - global_min) / (global_max - global_min)
    class_thresholds[cls] = cls_threshold
    class_thresholds_normalized[cls] = cls_threshold_normalized
    
  data = {
      'class_mean': cls_mean,
      'class_var': cls_var,
      'class_thresholds': class_thresholds,
      'class_thresholds_normalized': class_thresholds_normalized,
      'global_min': float(global_min),
      'global_max': float(global_max)
    }

  os.makedirs(output_dir, exist_ok=True)
  with open(os.path.join(output_dir, 'class_distributions.json'), 'w') as f:
    json.dump(data, f, indent=2)
      
  print(f"Saved class distributions to: {os.path.join(output_dir, 'class_distributions.json')}")
  return class_stats, class_thresholds, class_thresholds_normalized, global_min, global_max

def loss_function(x_recon, x, mu, logvar, ssim_fn, beta, gamma, alpha):
  # BCE = F.binary_cross_entropy(x_recon, x, reduction="sum") / bsize
  # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / bsize
  x = x.clamp(0, 1)
  x_recon = x_recon.clamp(0, 1)
  
  BCE = F.binary_cross_entropy(x_recon, x, reduction="none").mean(dim=(1, 2, 3))
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
  
  SSIM = ssim_fn(x_recon, x)
  SSIM_loss = 1 - SSIM

  total_loss = gamma * BCE + beta * KLD + alpha * SSIM_loss
  return total_loss, BCE, KLD, SSIM_loss


def train(batch, model, optimizer, device, ssim_fn, beta, gamma, alpha):
    model.train()
    x, y = batch
    x, y = x.to(device), y.to(device)

    x_recon, mu, logvar = model(x, y)
    optimizer.zero_grad()
    total_loss, recon_loss, kld_loss, ssim_loss = loss_function(x_recon, x, mu, logvar, ssim_fn, beta, gamma, alpha)
    loss = total_loss.mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"Grad Norm: {grad_norm:.4f}")
    optimizer.step()
    return loss.item(), recon_loss.mean().item(), kld_loss.mean().item(), ssim_loss.mean().item(), total_loss, y
  
def validate(batch, model, device, ssim_fn, beta, gamma, alpha):
    model.eval()
    x, y = batch
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        x_recon, mu, logvar = model(x, y)
        total_loss, bce, kld, ssim_loss = loss_function(x_recon, x, mu, logvar, ssim_fn, beta, gamma, alpha)
        return total_loss.mean().item(), bce.mean().item(), kld.mean().item(), ssim_loss.mean().item()
  
def plot_loss(a_data, b_data, a_label, b_label, title):
  fig = plt.figure()
  plt.plot(a_data, label=a_label)
  plt.plot(b_data, label=b_label)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.title(title)
  return fig

def plot_training_loss(total_bce_losses, total_kld_losses, total_ssim_losses, total_val_loss, n_epochs):
  fig = plt.figure()
  plt.plot(range(1, n_epochs + 1), total_bce_losses, label='BCE Loss')
  plt.plot(range(1, n_epochs + 1), total_kld_losses, label='KLD Loss')
  plt.plot(range(1, n_epochs + 1), total_ssim_losses, label='SSIM Loss')
  plt.plot(range(1, n_epochs + 1), total_val_loss, label='Validation Loss', linestyle='--')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.title('Training and Validation Loss Components')  
  plt.grid(True)
  return fig

def train_model(model, optimizer, scheduler, train_dataloader, val_dataloader, device, n_epochs, output_dir, patience, min_beta_value = 0.1):
  if not train_dataloader or not val_dataloader:
    raise ValueError("Dataloaders cannot be empty")
  if not isinstance(model, nn.Module):
    raise ValueError("Model must be a PyTorch nn.Module")
  if device.type == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError("CUDA device requested but not available")
      
  model.to(device)
  
  weights_dir = os.path.join(output_dir, 'weights')
  os.makedirs(weights_dir, exist_ok=True)
  
  recon_dir = os.path.join(output_dir, 'reconstructions')
  os.makedirs(recon_dir, exist_ok=True)
  
  patience_counter = 0
  stopped_epoch = 0
  
  ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device)

  all_anomaly_scores = []
  all_class_labels = []
  
  total_training_loss, total_val_loss = [], []
  total_bce_losses, total_kld_losses, total_ssim_losses = [], [], []
  total_val_bce_losses, total_val_kld_losses, total_val_ssim_losses = [], [], []
  best_val_loss = float("inf")
  
  for epoch in range(n_epochs):
    beta = min(min_beta_value + 0.9 * epoch / n_epochs, 1.0)
    alpha = min(min_beta_value + 0.9 * epoch / n_epochs, 1.0)
    gamma = max(1.0 - 0.99 * epoch / n_epochs, 0.01)
    print(f"Epoch: {epoch+1}/{n_epochs} | Beta: {beta:.4f} | Gamma: {gamma:.4f} | Alpha: {alpha:.4f}")
    train_epoch_losses, train_bce_losses, train_kld_losses, train_ssim_losses = [], [], [], []
    val_bce_losses, val_kld_losses, val_ssim_losses, val_epoch_losses= [], [], [], []
    for batch_idx, train_batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")):
      train_batch_loss, total_bce_loss, train_kld_loss, train_ssim_loss, losses, labels = train(train_batch, model, optimizer, device, ssim_fn, beta, gamma, alpha)

      train_epoch_losses.append(train_batch_loss)
      train_bce_losses.append(total_bce_loss)
      train_kld_losses.append(train_kld_loss)
      train_ssim_losses.append(train_ssim_loss)
      
      all_anomaly_scores.extend(losses.cpu().detach().numpy().tolist())
      all_class_labels.extend(labels.cpu().numpy().tolist())
      
      current_lr = optimizer.param_groups[0]['lr']
      print(f"Epoch {epoch+1} Batch {batch_idx} - Loss: {train_batch_loss:.4f}, BCE: {total_bce_loss:.4f}, gamma:{gamma}, KLD: {train_kld_loss:.4f}, beta:{beta}, SSIM: {train_ssim_loss:.4f}, alpha:{alpha}, LR: {current_lr:.6f}")
      
    scheduler.step()

    for val_batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]"):
      val_loss, val_bce_loss, val_kld_loss, val_ssim_loss = validate(val_batch, model, device, ssim_fn, beta, gamma, alpha)
      val_epoch_losses.append(val_loss)
      val_bce_losses.append(val_bce_loss)
      val_kld_losses.append(val_kld_loss)
      val_ssim_losses.append(val_ssim_loss)
    
    training_per_epoch_loss = np.mean(train_epoch_losses)
    val_per_epoch_loss = np.mean(val_epoch_losses)
    bce_per_epoch_loss = np.mean(train_bce_losses)
    kld_per_epoch_loss = np.mean(train_kld_losses)
    ssim_per_epoch_loss = np.mean(train_ssim_losses)
    val_bce_per_epoch_loss = np.mean(val_bce_losses)
    val_kld_per_epoch_loss = np.mean(val_kld_losses)
    val_ssim_per_epoch_loss = np.mean(val_ssim_losses)

    total_training_loss.append(training_per_epoch_loss)
    total_val_loss.append(val_per_epoch_loss)
    total_bce_losses.append(bce_per_epoch_loss)
    total_kld_losses.append(kld_per_epoch_loss)
    total_ssim_losses.append(ssim_per_epoch_loss)
    total_val_bce_losses.append(val_bce_per_epoch_loss)
    total_val_kld_losses.append(val_kld_per_epoch_loss)
    total_val_ssim_losses.append(val_ssim_per_epoch_loss)
    
    print(f"\nEpoch: {epoch+1}/{n_epochs} | "
            f"Training Loss: {training_per_epoch_loss:.4f}, "
            f"Val Loss: {val_per_epoch_loss:.4f}, "
            f"BCE: {bce_per_epoch_loss:.4f}/{val_bce_per_epoch_loss:.4f}, "
            f"KLD: {kld_per_epoch_loss:.4f}/{val_kld_per_epoch_loss:.4f}, "
            f"SSIM: {ssim_per_epoch_loss:.4f}/{val_ssim_per_epoch_loss:.4f}")
                  
    # save best model
    if val_per_epoch_loss < best_val_loss:
      best_val_loss = val_per_epoch_loss
      patience_counter = 0
      best_model_path = os.path.join(weights_dir, "best.pth")
      torch.save(model.state_dict(), best_model_path)
      print(f"Best model updated (Val Loss: {best_val_loss:.4f})", flush = True)
    else: 
      patience_counter += 1
      print(f"No improvement for {patience_counter} epoch(s)")
    if patience_counter >= patience:
      print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
      break
    
    if (epoch + 1) % 10 == 0:
      model.eval()
      with torch.no_grad():
        recon_batch = next(iter(val_dataloader))
        x_val, y_val = recon_batch
        x_val, y_val = x_val.to(device), y_val.to(device)
        x_recon, _, _ = model(x_val, y_val)

        x_val = x_val.clamp(0, 1)

        # Concatenate original and reconstruction vertically
        recon_grid = torch.cat([x_val, x_recon], dim=0)
        recon_path = os.path.join(recon_dir, f"epoch_{epoch+1}_recon.png")
        save_image(recon_grid.cpu(), recon_path, nrow=x_val.size(0), normalize=True)
        print(f"[Epoch {epoch+1}] Saved reconstruction to {recon_path}")
    
    stopped_epoch = epoch + 1
  
  class_stats, class_thresholds, class_thresholds_normalized, global_min, global_max = calculate_class_thresholds(all_anomaly_scores, all_class_labels, output_dir)

  last_weight_path = os.path.join(weights_dir, "last.pth")
  torch.save(model.state_dict(), last_weight_path)
  
  rk_loss_plot = plot_training_loss(total_bce_losses, total_kld_losses, total_ssim_losses, total_val_loss, stopped_epoch)
  rk_loss_plot.savefig(os.path.join(output_dir, 'training_loss_curve.png'))
  plt.close(rk_loss_plot)
  
  val_fig = plot_training_loss(total_val_bce_losses, total_val_kld_losses, total_val_ssim_losses, total_val_loss, stopped_epoch)
  val_fig.savefig(os.path.join(output_dir, 'val_loss_curve.png'))
  plt.close(val_fig)
  
  train_loss_plot = plot_loss(total_training_loss, total_val_loss, 'Train Loss', 'Val Loss', 'Training and Validation Losses over Epochs')
  train_loss_plot.savefig(os.path.join(output_dir, 'train_val_loss_curve.png'))
  plt.close(train_loss_plot)

  return (total_training_loss, total_val_loss, total_bce_losses, total_kld_losses, total_ssim_losses,
            total_val_bce_losses, total_val_kld_losses, total_val_ssim_losses)

