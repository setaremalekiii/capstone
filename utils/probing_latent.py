import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def probe_latent_dimension(model, x, y, device, dim_to_probe, sweep_range=(-3, 3), steps=9):
    """
    model: trained ConvCVAE
    x: single image tensor [1, C, H, W]
    y: corresponding condition tensor [1, num_classes]
    dim_to_probe: which latent dimension to modify
    """

    model.eval()
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        mu, logvar = model.encode(x, y)

        # Use deterministic latent (no sampling noise)
        base_z = mu.clone()

        sweep_values = torch.linspace(
            sweep_range[0],
            sweep_range[1],
            steps
        ).to(device)

        decoded_images = []

        for val in sweep_values:
            z = base_z.clone()
            z[:, dim_to_probe] = val  # modify only one dimension

            recon = model.decode(z, y)
            decoded_images.append(recon.cpu())

    decoded_images = torch.cat(decoded_images, dim=0)

    grid = make_grid(decoded_images, nrow=steps, normalize=True)

    plt.figure(figsize=(15, 3))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Latent dim {dim_to_probe} sweep from {sweep_range[0]} to {sweep_range[1]}")
    plt.show()

# Here's an example of how to use this function to probe :) 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.load_state_dict(torch.load("best.pth"))
# model.to(device)

# # get one batch
# x_batch, y_batch = next(iter(train_dataloader))

# # take first sample
# x_sample = x_batch[0:1]
# y_sample = y_batch[0:1]

# # probe dimension 3 for example
# probe_latent_dimension(model, x_sample, y_sample, device, dim_to_probe=3)
