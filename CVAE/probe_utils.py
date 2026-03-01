# probe_utils.py
import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from typing import Sequence, Union, Literal

@torch.no_grad()
def probe_latent_dimension_save(
    model,
    x, y,
    device,
    dims_to_probe: Union[int, Sequence[int]],
    out_dir,
    sweep_range=(-2, 2),
    steps=9,
    sample_id="0",
    include_original=True,
    centered=True, 
    probe_mode: Literal["single", "together", "one_by_one"] = "single",
):
    """
    Saves a grid image showing reconstructions as you sweep one latent dimension.

    model: trained ConvCVAE
    x: [1, C, H, W]
    y: [1, label_dim]
    device: torch device
    dims_to_probe: which latent axis to change
    out_dir: folder to save images
    sweep_range: (lo, hi) range for the sweep
    steps: number of sweep points
    sample_id: string used in filename
    include_original: include original image above reconstructions
    centered: if True use z_d = mu_d + delta, else z_d = value 
    probe_mode: "single", "together", or "one_by_one"
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    x = x.to(device)
    y = y.to(device)

    # Encode to get a stable latent anchor
    mu, logvar = model.encode(x, y)
    base_z = mu.clone()

    # Normalize dimension inputs in case they are not already lists or 
    # the together single etc is not indicated
    if isinstance(dims_to_probe, int):
        dims = [dims_to_probe]
    else:
        dims = list(dims_to_probe)

    if probe_mode == "single":
        if len(dims) != 1:
            raise ValueError('probe_mode="single" requires exactly one dim.')
    elif probe_mode not in ("together", "one_by_one"):
        raise ValueError('probe_mode must be one of: "single", "together", "one_by_one".')

    latent_dim = base_z.shape[1]
    for d in dims:
        if not (0 <= d < latent_dim):
            raise ValueError(f"Invalid dim {d}. Latent dim is {latent_dim}.")

    vals = torch.linspace(sweep_range[0], sweep_range[1], steps, device=device)

    recons = []
    if probe_mode in ("single", "together"):
        for v in vals:
            z = base_z.clone()
            if centered:
                z[:, dims_to_probe] = base_z[:, dims_to_probe] + v
            else:
                z[:, dims_to_probe] = v

            x_recon = model.decode(z, y)
            recons.append(x_recon.cpu())

        recon_stack = torch.cat(recons, dim=0)  # [steps, C, H, W]
        recon_grid = make_grid(recon_stack, nrow=steps, normalize=True)
        title_dims = dims[0] if len(dims) == 1 else dims
        title = (
            f"Dims {title_dims} sweep ({sweep_range[0]} → {sweep_range[1]})"
            + (" around mu" if centered else "")
        )
    else:
        # one_by_one: multiple rows, one per dim
        for d in dims:
            for v in vals:
                z = base_z.clone()
                if centered:
                    z[:, d] = base_z[:, d] + v
                else:
                    z[:, d] = v
                x_recon = model.decode(z, y)
                recons.append(x_recon.cpu())

        recon_stack = torch.cat(recons, dim=0)  # [len(dims)*steps, C, H, W]
        recon_grid = make_grid(recon_stack, nrow=steps, normalize=True)

        title = (
            f"Dims {dims} swept one-by-one ({sweep_range[0]} → {sweep_range[1]})"
            + (" around mu" if centered else "")
            + f" | rows follow dims order"
        )


    # Plot (optionally include original)
    if include_original:
        orig_grid = make_grid(x.cpu(), nrow=1, normalize=True)

        # Heuristic figsize
        height = 4 if probe_mode != "one_by_one" else max(4, 2.5 + 1.2 * len(dims))
        plt.figure(figsize=(max(steps * 1.5, 10), height))

        plt.subplot(2, 1, 1)
        plt.imshow(orig_grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"Original (sample {sample_id})")

        plt.subplot(2, 1, 2)
        plt.imshow(recon_grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title(title)
    else:
        height = 2.5 if probe_mode != "one_by_one" else max(2.5, 1.2 * len(dims))
        plt.figure(figsize=(max(steps * 1.5, 10), height))
        plt.imshow(recon_grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"Sample {sample_id} | {title}")

    # Save
    dims_str = "-".join(map(str, dims))
    fname = f"sample_{sample_id}_dims_{dims_str}_{probe_mode}_sweep.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path