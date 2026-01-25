"""Convolutional Conditional Variational AutoEncoder model"""
# bringing in pytorth neural netowork modules
import torch
import torch.nn as nn

# this is an example of inheritance where ConvCVAE class inherits from nn.Module class
# this allows ConvCAE to be trainable  as it  stores parameters and can 
class ConvCVAE(nn.Module):
  # constructor: images (64 x 64), 1 channel so grayscale, label are 24 of them, latent dim is 64, more or less layers
  # Z = n x latent_dim
  def __init__(self, img_size, img_channels=1, label_dim=24, latent_dim=64, deeper = False):
      super(ConvCVAE, self).__init__()
      self.img_channels = img_channels
      self.label_dim = label_dim
      self.latent_dim = latent_dim
      self.img_size = img_size

      # Encode s
      if deeper:
        print("Training deeper model...")
        # Sequential is a function that allows us to stack layers 
        self.enc_conv = nn.Sequential(
              # 2d is just ax + b for the convolutional layer and we plug that  into our 
              #RelU function which is just a max function of (0,x)
              # input channels = img_channels + label_dim because we are concatenating the labels to the image channels
              nn.Conv2d(img_channels + label_dim, 32, kernel_size=3, stride=2, padding=1), # (1, 32, 32 , 32 if 64x64
             
              # we could also try nn.sigmoid or nn.tanh later? maybe each person trains a diff one?
              nn.ReLU(),
              # the output dim is passed in as an input into the next layer there are more parameters here:
              # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
              # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
              nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
              nn.ReLU(),
              # padding adds zeros to keep the size the same but stride indicates the step 
              # size for the kernel so rn its checking every other pixel
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
      # determines the size of the output from the conv layers 128 or 256 
      #  NO_GRAD means dont calculate gradients for this operation
      # [1, 25, H, W] shuold be a sample output 
      # then we can get smth like if deeper=False: [1, 128, 8, 8] if deeper=True: [1, 256, 4, 4]
      #  this is the tesnor shape after applying the conv layers
      with torch.no_grad():
        dummy = torch.zeros(1, img_channels + label_dim, *img_size)
        out = self.enc_conv(dummy)
        _, c, h, w = out.size()
        
      self.conv_out_channels = c
      self.conv_out_h = h
      self.conv_out_w = w
      # if deeper=False: conv_out_size = 128 * 8 * 8 = 8192 
      # if deeper=True: conv_out_size = 256 * 4 * 4 = 4096
      self.conv_out_size = c * h * w
      print(f"conv output size: {self.conv_out_size}")
      
      # separate u and logvar calcualtion for each dimension in the latent_dim
      # this is the bottle neck ur going from 4096 → 64 or 8192 → 64 
      # now this is the latent space with the mean and logvar for each dimension 
      # latent space is a vector of size latent_dim
      #  mu is center of the gaussian distribution IN the latent space with latent dim output vector size
      self.fc_mu = nn.Linear(self.conv_out_size, latent_dim)
      # nn.Linear  -> y=xA^T+b
      #tells the model how “uncertain/spread out” each latent dimension should be.
      self.fc_logvar = nn.Linear(self.conv_out_size, latent_dim)
      # expands that back into the size needed to reshape into a feature map to decode 
      self.fc_dec = nn.Linear(latent_dim + label_dim, self.conv_out_size)

      # Decoder
      if deeper:
        # class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, 
        # padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', 
        # device=None, dtype=None)
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
      
  # encodes input image x  and its label y into mu and logvar parameters
  # x → [B, 1, H, W]  B images 1 channel (grayscale) height H width W
  # y → [B, 24] B labels each of size 24 (one hot encoded)
  def encode(self, x, y):
    # y_map → [B, 24, H, W]
    y_map = y.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.img_size[0], self.img_size[1])
    # concatenating (linking together) image channels to labels
    print(f"y_map shape: {y_map.shape}, x shape: {x.shape}")
    x_cond = torch.cat([x, y_map], dim=1) 
    print(f"x_cond shape: {x_cond.shape}")
    h = self.enc_conv(x_cond)
    # CNN encoder compresses spatially into features.
    out = h.view(h.size(0), -1)  # flatten to [B, C*H*W]
    mu = self.fc_mu(out)
    logvar = self.fc_logvar(out)
    # another value we can change if needed
    logvar = torch.clamp(logvar, min=-10, max=10)
    return mu, logvar

  # this is applied per sample?
  def reparameterize(self, mu, logvar):
    # Reparameterization: required since normal distribution is not differentiable
    # returns latent vector z 
    # log variance to standard deviation
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z, y):
    z_cond = torch.cat([z, y], dim=1)
    h = self.fc_dec(z_cond)
    h = h.view(-1, self.conv_out_channels, self.conv_out_h, self.conv_out_w)
    x_recon = self.dec_conv(h)
      
    return x_recon

  # runs forward pass and returns a tensor of reconstructed images, mu, logvar
  def forward(self, x, y):
    mu, logvar = self.encode(x, y)
    # print(f"mu mean: {mu.mean().item():.4f}, logvar mean: {logvar.mean().item():.4f}")

    z = self.reparameterize(mu, logvar)
    # print(f"z mean: {z.mean().item():.4f}, std: {z.std().item():.4f}")

    x_recon = self.decode(z, y)
    # print(f"x_recon min: {x_recon.min().item():.4f}, max: {x_recon.max().item():.4f}")
    
    return x_recon, mu, logvar

    
