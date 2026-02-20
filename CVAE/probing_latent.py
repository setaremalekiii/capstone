import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

@torch.no_grad()
def probe_latent_dimension_save(
    model,
    x, y,
    device,
    dim_to_probe,
    out_dir,
    sweep_range=(-3, 3),
    steps=9,
    sample_id="0",
    include_original=True,
    centered=True,          # sweep around mu instead of overwriting it
):
    """
    Saves a grid image showing reconstructions as you sweep one latent dim.
    x: [1, C, H, W]
    y: [1, label_dim]
    """

    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    x = x.to(device)
    y = y.to(device)

    mu, logvar = model.encode(x, y)
    base_z = mu.clone()

    vals = torch.linspace(sweep_range[0], sweep_range[1], steps, device=device)

    recons = []
    for v in vals:
        z = base_z.clone()
        if centered:
            z[:, dim_to_probe] = base_z[:, dim_to_probe] + v
        else:
            z[:, dim_to_probe] = v
        x_recon = model.decode(z, y)
        recons.append(x_recon.cpu())

    recon_stack = torch.cat(recons, dim=0)  # [steps, C, H, W]
    recon_grid = make_grid(recon_stack, nrow=steps, normalize=True)

    if include_original:
        orig_grid = make_grid(x.cpu(), nrow=1, normalize=True)
        # stack original above recon grid
        # pad original to same width visually by putting it in a figure
        plt.figure(figsize=(max(steps * 1.5, 10), 4))
        plt.subplot(2, 1, 1)
        plt.imshow(orig_grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"Original (sample {sample_id})")

        plt.subplot(2, 1, 2)
        plt.imshow(recon_grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"Dim {dim_to_probe} sweep ({sweep_range[0]} → {sweep_range[1]}) {'around mu' if centered else ''}")

    else:
        plt.figure(figsize=(max(steps * 1.5, 10), 2.5))
        plt.imshow(recon_grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"Sample {sample_id} | Dim {dim_to_probe} sweep ({sweep_range[0]} → {sweep_range[1]})")

    fname = f"sample_{sample_id}_dim_{dim_to_probe}_sweep.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path