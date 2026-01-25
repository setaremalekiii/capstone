"""Convolutional Conditional Variational AutoEncoder model"""

import torch
import torch.nn as nn

class ConvCVAE(nn.Module):
  
  def __init__(self, img_size, img_channels=1, label_dim=24, latent_dim=64, deeper = False):
      super(ConvCVAE, self).__init__()
      self.img_channels = img_channels
      self.label_dim = label_dim
      self.latent_dim = latent_dim
      self.img_size = img_size

      # Encoder
      if deeper:
        print("Training deeper model...")
        self.enc_conv = nn.Sequential(
              nn.Conv2d(img_channels + label_dim, 32, kernel_size=3, stride=2, padding=1), # (1, 32, 32 , 32 if 64x64
              nn.ReLU(),
              nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
              nn.ReLU(),
              nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
              nn.ReLU(),
              nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
              nn.ReLU()
          )
      else:
        self.enc_conv = nn.Sequential(
            nn.Conv2d(img_channels + label_dim, 32, kernel_size=3, stride=2, padding=1), # (1, 32, 32 , 32 if 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
      
      with torch.no_grad():
        dummy = torch.zeros(1, img_channels + label_dim, *img_size)
        out = self.enc_conv(dummy)
        _, c, h, w = out.size()
        
      self.conv_out_channels = c
      self.conv_out_h = h
      self.conv_out_w = w
      self.conv_out_size = c * h * w
      
      # separate u and logvar calcualtion for each dimension in the latent_dim
      self.fc_mu = nn.Linear(self.conv_out_size, latent_dim)
      self.fc_logvar = nn.Linear(self.conv_out_size, latent_dim)
      self.fc_dec = nn.Linear(latent_dim + label_dim, self.conv_out_size)

      # Decoder
      if deeper:
        self.dec_conv = nn.Sequential(
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, img_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid()
        )
      else:
        self.dec_conv = nn.Sequential(
          nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(32, img_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
          nn.Sigmoid()
        )
    
  def encode(self, x, y):
    y_map = y.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.img_size[0], self.img_size[1])
    # concatenating image channels to labels
    print(f"y_map shape: {y_map.shape}, x shape: {x.shape}")
    x_cond = torch.cat([x, y_map], dim=1) 
    print(f"x_cond shape: {x_cond.shape}")
    h = self.enc_conv(x_cond)
      
    out = h.view(h.size(0), -1)  # flatten to [B, C*H*W]
    mu = self.fc_mu(out)
    logvar = self.fc_logvar(out)
    logvar = torch.clamp(logvar, min=-10, max=10)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    # Reparameterization: required since normal distribution is not differentiable
    # returns latent vector z 
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z, y):
    z_cond = torch.cat([z, y], dim=1)
    h = self.fc_dec(z_cond)
    h = h.view(-1, self.conv_out_channels, self.conv_out_h, self.conv_out_w)
    x_recon = self.dec_conv(h)
      
    return x_recon

  def forward(self, x, y):
    mu, logvar = self.encode(x, y)
    print(f"mu mean: {mu.mean().item():.4f}, logvar mean: {logvar.mean().item():.4f}")

    z = self.reparameterize(mu, logvar)
    print(f"z mean: {z.mean().item():.4f}, std: {z.std().item():.4f}")

    x_recon = self.decode(z, y)
    print(f"x_recon min: {x_recon.min().item():.4f}, max: {x_recon.max().item():.4f}")
    
    return x_recon, mu, logvar
  


    
