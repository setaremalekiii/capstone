"""Script for training the ConvCVAE"""
#Basically the same as Train 4 with a few modifications

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from ConvCVAE import ConvCVAE
import logging
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pathlib import Path
from torchvision.transforms.functional import rotate, InterpolationMode


def loss_function(x_recon, x, mu, logvar, ssim_fn, beta, gamma, alpha, bsize):
  BCE = F.binary_cross_entropy(x_recon, x, reduction="sum")/bsize
  KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())/bsize
  #x = x.clamp(0, 1)
  #x_recon = x_recon.clamp(0, 1)
  
  #BCE = F.binary_cross_entropy(x_recon, x, reduction="none").mean(dim=(1, 2, 3))
  #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
  #BCE = BCE.mean()
  #KLD = KLD.mean()
  SSIM = ssim_fn(x_recon, x)
  SSIM_loss = (1 - SSIM).mean()
  SSIM_loss = SSIM_loss.mean()
  total_loss = gamma * BCE + beta * KLD + alpha * SSIM_loss
  return total_loss, BCE, KLD, SSIM_loss

def train(batch, model, optimizer, device, ssim_fn, beta, gamma, alpha):
  model.train()
  x, y = batch
  x, y = x.to(device), y.to(device)
  bsize = x.size(0)
  
  # print(f"x min: {x.min().item()}, max: {x.max().item()}")
  x_recon, mu, logvar = model(x, y)
  # print(f"x_recon min: {x_recon.min().item()}, max: {x_recon.max().item()}")  

  optimizer.zero_grad()
  loss, recon_loss, kld_loss, ssim_loss = loss_function(x_recon, x, mu, logvar, ssim_fn, beta, gamma, alpha, bsize)
  loss.backward()
  grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  print(f"Grad Norm: {grad_norm:.4f}")
  optimizer.step()
  return loss.item(), recon_loss.item(), kld_loss.item(), ssim_loss.item()

def validate(batch, model, device, ssim_fn, beta, gamma, alpha):
  model.eval()
  x, y = batch
  x, y = x.to(device), y.to(device)
  bsize = x.size(0)

  with torch.no_grad():
    x_recon, mu, logvar = model(x, y)
    loss, recon_loss, kld_loss, ssim_loss = loss_function(x_recon, x, mu, logvar, ssim_fn, beta, gamma, alpha, bsize)
  return loss.item(), recon_loss.item(), kld_loss.item(), ssim_loss.item()

def plot_loss(a_data, b_data, a_label, b_label, title):
  if len(a_data) != len(b_data):
    raise ValueError("Data lists must have equal length")
  fig = plt.figure()
  plt.plot(range(1, len(a_data) + 1), a_data, label=a_label)
  plt.plot(range(1, len(b_data) + 1), b_data, label=b_label)
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

def extract_latents(model, dataloader, device):
    model.eval()

    latents = []

    with torch.no_grad():
        for batch in dataloader:
            # adjust this depending on how your dataset returns data
            # most of your code uses (x, c) for CVAE
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, c = batch
                x = x.to(device)
                c = c.to(device)

                # ---- IMPORTANT ----
                # adapt this line to your model API
                mu, logvar = model.encode(x, c)

            else:
                x = batch.to(device)
                mu, logvar = model.encode(x)

            latents.append(mu.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    return latents

def plot_latent_space(model, data_loader, device, n_samples=100):
  model.eval()
  mus = []
  labels = []

  with torch.no_grad():
    n = 0
    for x, y in data_loader:
         x = x.to(device)
         y = y.to(device)

         mu, logvar = model.encode(x, y)   # CVAE needs y
         mus.append(mu.cpu().numpy())
         labels.append(y.cpu().numpy())

         n += x.size(0)
         if n >= n_samples:
           break

  mus = np.concatenate(mus, axis=0)[:n_samples]
  labels = np.concatenate(labels, axis=0)[:n_samples]

    # If y is one-hot [B,24], convert to class id for coloring
  if labels.ndim == 2:
     labels = labels.argmax(axis=1)

  fig = plt.figure(figsize=(10, 8))
  plt.scatter(mus[:, 0], mus[:, 1], c=labels, cmap="viridis", s=8, alpha=0.7)
  plt.colorbar(label="Condition / class")
  plt.xlabel("mu[0]")
  plt.ylabel("mu[1]")
  plt.title("CVAE latent means (first 2 dims)")
  plt.grid(True)
  return fig

def plot_mu_vs_sigma_all_dims(
    model,
    data_loader,
    device,
    n_samples=500,
    max_dims=None,
    save_dir=None,
):
    model.eval()
    mus, sigmas = [], []

    with torch.no_grad():
        n = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            mu, logvar = model.encode(x, y)

            sigma = torch.exp(0.5 * logvar)

            mus.append(mu.cpu())
            sigmas.append(sigma.cpu())

            n += x.size(0)
            if n >= n_samples:
                break

    mu_vals = torch.cat(mus, dim=0)[:n_samples]
    sigma_vals = torch.cat(sigmas, dim=0)[:n_samples]

    latent_dim = mu_vals.shape[1]
    if max_dims is not None:
        latent_dim = min(latent_dim, max_dims)

    figs = []

    for i in range(latent_dim):
        fig = plt.figure(figsize=(7, 6))
        plt.scatter(
            mu_vals[:, i].numpy(),
            mu_vals[:, i+1].numpy(),
            s=8,
            alpha=0.6,
        )
        plt.xlabel(f"mu[{i}]")
        plt.ylabel(f"mu[{i+1}]")
        plt.title(f"Mean vs Uncertainty (latent dim {i})")
        plt.grid(True)

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / f"mu_vs_sigma_dim_{i}.png", dpi=150)
            plt.close(fig)

        figs.append(fig)

    return figs

def rotate_batch(x, angle_degrees: float):
    """
    x: [B, C, H, W] torch tensor
    returns rotated x with same shape
    """
    return torch.stack([
        rotate(img, angle_degrees, interpolation=InterpolationMode.BILINEAR)
        for img in x
    ], dim=0)

@torch.no_grad()
def collect_mu_by_angle(model, dataloader, device, angles, label_getter=None, id_getter=None):
    model.eval()
    mu_dict = {}

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch["image"]
        else:
            raise ValueError("Unknown batch format")

        x = x.to(device)

        y = label_getter(batch).to(device) if label_getter is not None else None
        ids = id_getter(batch) if id_getter is not None else None

        if ids is None:
            ids = list(range(len(mu_dict), len(mu_dict) + x.shape[0]))

        # If your model is conditional, enforce y exists
        if y is None:
            raise ValueError("This model is conditional: encode() needs y. Provide label_getter.")

        mu_angles = []  # list of [B, latent_dim]
        for a in angles:
            x_rot = rotate_batch(x, a)
            mu, logvar = model.encode(x_rot, y)
            mu_angles.append(mu.detach().cpu().numpy())

        mu_angles = np.stack(mu_angles, axis=0)  # [num_angles, B, latent_dim]

        for b_idx, sample_id in enumerate(ids):
            mu_dict[sample_id] = mu_angles[:, b_idx, :]

    return mu_dict


def rotation_variance_scores(mu_dict, agg="mean"):
    """
    mu_dict[id] = [num_angles, latent_dim]
    Returns:
      scores: [latent_dim] rotation sensitivity score
    """
    per_image_var = []
    for sample_id, mu in mu_dict.items():
        # variance across angles: [latent_dim]
        per_image_var.append(mu.var(axis=0))

    per_image_var = np.stack(per_image_var, axis=0)  # [num_images, latent_dim]

    if agg == "mean":
        return per_image_var.mean(axis=0)
    elif agg == "median":
        return np.median(per_image_var, axis=0)
    else:
        raise ValueError("agg must be 'mean' or 'median'")



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


  total_training_loss, total_val_loss = [], []
  total_bce_losses, total_kld_losses, total_ssim_losses = [], [], []
  total_val_bce_losses, total_val_kld_losses, total_val_ssim_losses = [], [], []
  best_val_loss = float("inf")
  


  for epoch in range(n_epochs):
    beta = min(min_beta_value + 0.9 * epoch / n_epochs, 1)
    alpha = 1.0
    gamma = max(1.0 - 0.99 * epoch / n_epochs, 0.1)
    print(f"Epoch: {epoch+1}/{n_epochs} | Beta: {beta:.4f} | Gamma: {gamma:.4f} | Alpha: {alpha:.4f}")
    train_epoch_losses, train_bce_losses, train_kld_losses, train_ssim_losses = [], [], [], []
    val_bce_losses, val_kld_losses, val_ssim_losses, val_epoch_losses= [], [], [], []


    for batch_idx, train_batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")):
      train_batch_loss, total_bce_loss, train_kld_loss, train_ssim_loss = train(train_batch, model, optimizer, device, ssim_fn, beta, gamma, alpha)

      train_epoch_losses.append(train_batch_loss)
      train_bce_losses.append(total_bce_loss)
      train_kld_losses.append(train_kld_loss)
      train_ssim_losses.append(train_ssim_loss)
      
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

        # x_val = x_val.clamp(0, 1)

        # Concatenate original and reconstruction vertically
        recon_grid = torch.cat([x_val, x_recon], dim=0)
        recon_path = os.path.join(recon_dir, f"epoch_{epoch+1}_recon.png")
        save_image(recon_grid.cpu(), recon_path, nrow=x_val.size(0), normalize=True)
        print(f"[Epoch {epoch+1}] Saved reconstruction to {recon_path}")
        
    stopped_epoch = epoch + 1

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

  latent_space = plot_latent_space(model, train_dataloader, device)
  latent_space.savefig(os.path.join(output_dir, 'Latent_space.png'))
  plt.close(latent_space)

  #mu_vs_sigma = plot_mu_vs_sigma(model, train_dataloader, device)
  #mu_vs_sigma.savefig(os.path.join(output_dir, 'mu_vs_sigma.png'))
  #plt.close(mu_vs_sigma)
  #latents = extract_latents(model, train_dataloader, device)

  #np.save(os.path.join(output_dir, "latents.npy"), latents)

  angles = list(range(0, 360, 10))  # e.g. every 10 degrees

  # If your dataloader yields (x, y, ids):
  label_getter = lambda batch: batch[1]      # adjust if needed
  mu_dict = collect_mu_by_angle(
    model,
    train_dataloader,
    device,
    angles=angles,
    label_getter=label_getter,  # or set correctly
    id_getter=None      # <-- important
  )

  scores = rotation_variance_scores(mu_dict, agg="mean")  # [latent_dim]

  top = np.argsort(scores)[::-1]
  

  df = pd.DataFrame({
    "latent_dim": np.arange(len(scores)),
    "rotation_variance_score": scores
  })

  # sort by score (highest first)
  df = df.sort_values("rotation_variance_score", ascending=False)

  out_path = os.path.join(output_dir, "rotation_latent_scores.csv")
  df.to_csv(out_path, index=False)

  d = top[0]

  plt.figure()

  for sample_id in list(mu_dict.keys())[:5]:   # first 5 chromosomes
    mu = mu_dict[sample_id]
    plt.plot(angles, mu[:, d], alpha=0.7)

  plt.xlabel("Rotation angle (degrees)")
  plt.ylabel(f"mu[{d}]")
  plt.title(f"Latent dim {d} across chromosomes")
  plt.grid(True)
  plt.show()
  out = os.path.join(output_dir, f"latent_dim_{d}_vs_angle.png")
  plt.savefig(out, dpi=150, bbox_inches="tight")
  plt.close()

#   d1, d2, d3 = top[:3]

#   # flatten all samples + angles into one big set of points
#   points = []
#   angle_vals = []

#   for sample_id, mu in mu_dict.items():  # mu: [num_angles, latent_dim]
#     pts = mu[:, [d1, d2, d3]]          # [num_angles, 3]
#     points.append(pts)
#     angle_vals.append(np.array(angles))  # [num_angles]

#   points = np.vstack(points)        # [N, 3]
#   angle_vals = np.concatenate(angle_vals)  # [N]

#   fig = plt.figure()
#   ax = fig.add_subplot(111, projection="3d")

#   sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=angle_vals)
#   fig.colorbar(sc, ax=ax, label="angle (deg)")

#   ax.set_xlabel(f"mu[{d1}]")
#   ax.set_ylabel(f"mu[{d2}]")
#   ax.set_zlabel(f"mu[{d3}]")
#   ax.set_title("Top-3 rotation dims: all samples × angles")

#   plt.show()

  plot_mu_vs_sigma_all_dims(
    model,
    train_dataloader,
    device,
    n_samples=500,
    save_dir= os.path.join(output_dir)
)
  
  

  return total_training_loss, total_val_loss
