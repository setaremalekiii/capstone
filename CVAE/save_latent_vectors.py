
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

def encode_decode(model, data_loader, device, n_samples=5000):
  model.eval()
  mus = []
  labels = []

  with torch.no_grad():
    n = 0
    for x, y in data_loader:
         x = x.to(device)
         y = y.to(device)

         mu, logvar = model.encode(x, y)   # CVAE needs y

         with open('360_embeddings.txt', 'w') as file:
           file.write(f"{mu.cpu().numpy()}\n")
        
         reconstruct_image(model, mu, y, 'outputs/temp_recon', device)

def reconstruct_image(model, latent_vector, condition_vector, output_dir, device):
  latent_vector = latent_vector.to(device)
  condition_vector = condition_vector.to(device)

  x_recon = model.decode(latent_vector, condition_vector)

  recon_dir = os.path.join(output_dir, 'reconstructions_from_latent')

  recon_path = os.path.join(recon_dir, f"recon.png")
  save_image(x_recon.cpu(), recon_path, nrow=latent_vector.size(0), normalize=True)
  print(f"Saved reconstruction to {recon_path}")


# for i in range(360):
#     file_name = f"rotss_{i:03d}_class0_0.jpg"

#     # Encode the input to get the latent vector
#     mu, logvar = model.encode(x, y)
    
#     # Save the latent vector to a text file
#     with open('360_embeddings.txt', 'a') as file:
#         file.write(f"{mu.cpu().numpy()}\n")


