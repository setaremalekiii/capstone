# Usage script to load a trained ConvCVAE, build the val dataloader, and save latent-dim probe grids.

import os
import glob
import yaml
import torch
from torch.utils.data import DataLoader

from ConvCVAE import ConvCVAE
from data import ChromosomeDataset
from probe_utils import probe_latent_dimension_save  # <-- put your save-probe function in probe_utils.py


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # ---- CONFIG (edit these) ----
    yaml_path = "data.yaml"  # same YAML you used for training
    weights_path = "best_32.pth"  # <-- update
    out_dir = "probe_outputs/exp1_best"  # where probe images will be saved

    imgsize = 64
    latent_dim = 32
    deeper = False
    batch_size = 32

    # Which latent dims to probe
    dims_to_probe = list(range(32))  # edit as needed

    # How many images to probe (from validation set)
    num_images_to_probe = 50

    # Probe sweep settings
    sweep_range = (-2, 2)  # recommended when centered=True
    steps = 9
    centered = True        # sweep around mu
    include_original = True
    # -----------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Build model (MUST match training)
    model = ConvCVAE(
        img_size=(imgsize, imgsize),
        latent_dim=latent_dim,
        deeper=deeper
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 2) Build validation dataloader (same as your main.py)
    cfg = load_yaml(yaml_path)
    val_img_paths = glob.glob(f"{cfg['val']}/*.jpg")
    val_data = ChromosomeDataset(val_img_paths, target_size=(imgsize, imgsize), transform=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # 3) Probe and save
    saved = 0
    global_idx = 0

    for x_batch, y_batch in val_loader:
        b = x_batch.size(0)

        for j in range(b):
            if saved >= num_images_to_probe:
                break

            x = x_batch[j:j+1]
            y = y_batch[j:j+1]
            sample_id = str(global_idx)

            for d in dims_to_probe:
                # Optional: organize by dim subfolders
                dim_dir = os.path.join(out_dir, f"dim_{d:02d}")
                os.makedirs(dim_dir, exist_ok=True)

                probe_latent_dimension_save(
                    model=model,
                    x=x,
                    y=y,
                    device=device,
                    dim_to_probe=d,
                    out_dir=dim_dir,
                    sweep_range=sweep_range,
                    steps=steps,
                    sample_id=sample_id,
                    include_original=include_original,
                    centered=centered,
                )

            saved += 1
            global_idx += 1

        if saved >= num_images_to_probe:
            break

    print(f"Saved probe grids for {saved} images into: {out_dir}")


if __name__ == "__main__":
    main()