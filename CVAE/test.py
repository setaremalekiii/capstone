"""Script for performing inference/detection with the ConvCVAE"""
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

#def calc_anomaly_score():

def loss_function(x_recon, x, mu, logvar, bsize, ssim_fn, beta, gamma, alpha):
  x = x.clamp(0, 1)
  x_recon = x_recon.clamp(0, 1)
  
  BCE = F.binary_cross_entropy(x_recon, x, reduction="none").mean(dim=(1, 2, 3))

  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
  
  SSIM = ssim_fn(x_recon, x, reduction='none')
  SSIM = SSIM.mean(dim=(1, 2, 3))
  SSIM_loss = 1 - SSIM

  total_loss = gamma * BCE + beta * KLD + alpha * SSIM_loss
  return total_loss, BCE, KLD, SSIM_loss


def test_model(model, test_dataloader, device, output_dir, beta, gamma, alpha, save_recon, model_path):
  recon_dir = os.path.join(output_dir, 'reconstructions')
  os.makedirs(recon_dir, exist_ok=True)
  
  ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device)

  model.load_state_dict(torch.load(model_path, map_location=device))
  model.to(device)
  model.eval()
  
  total_loss, total_bce, total_kld, total_ssim = 0.0, 0.0, 0.0, 0.0
  count = 0
  
  anomaly_scores = []
  true_labels = []
  
  with torch.no_grad()
    for batch in test_dataloader:
      x, y = x.to(device), y.to(device)
      bsize = x.size(0)
      
      x_recon,mu, logvar = model(x, y)
      
      loss, bce, kld, ssim = loss_function(x_recon, x, mu, logvar, bsize, beta, gamma, alpha)
      total_loss += loss.item() * bsize
      total_bce += bce.item() * bsize
      total_kld += kld.item() * bsize
      total_ssim += ssim.item() * bsize
      count += bsize
      
      sample_anomaly_scores = (gamma * bce + beta * kld + alpha * ssim_loss).cpu().numpy()
      anomaly_scores.extend(sample_anomaly_scores.tolist())
      true_labels.extend(y.cpu().numpy().tolist())
      
      if save_recon and batch_idx == 0:
        comparison = torch.cat([x, x_recon], dim=0)
        recon_path = os.path.join(recon_dir, 'test_recon.png')
        save_image(comparison.cpu(), recon_path, nrow=x.size(0), normalize=True)
        print(f"Saved test reconstructions to: {os.path.join(output_dir, 'test_recon.png')}")
    
    
    avg_loss = total_loss / count
    avg_bce = total_bce / count
    avg_kld = total_kld / count
    avg_ssim = total_ssim / count

    print(f"Test Loss: {avg_loss:.4f} | BCE: {avg_bce:.4f} | KLD: {avg_kld:.4f} | SSIM: {avg_ssim:.4f}")
    return avg_loss, avg_bce, avg_kld, avg_ssim

