# Make sure to cd into the CVAE folfer before running this script or else it will not work!
import os
import glob
import yaml
import torch
from torch.utils.data import DataLoader

from ConvCVAE import ConvCVAE
from data import ChromosomeDataset
from probe_utils import probe_latent_dimension_save, save_latent_vector
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # CONFIG (edit these) as needed make maybe give a new path for each run
    yaml_path = "CVAE/data.yaml"  # same YAML you used for training
    weights_path = "best_32.pth"  # <-- update later with new weights
    out_dir = "probe_outputs/new_test"  # where probe images will be saved

    imgsize = 64
    latent_dim = 32
    deeper = False
    batch_size = 32

    # Which latent dims to probe
    #dims_to_probe = list(range(latent_dim+1))  # edit as needed you can either do all dims for uncomment the line below to customize which dim u wanna probe
    dims_to_probe = [9,25,16]
    probe_mode = "together"
    

    # Probe sweep settings
    sweep_range = (-2, 2)  # recommended when centered=True
    steps = 9
    centered = True # sweep around mu
    include_original = True


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

    # How many images to probe (from validation set)
    num_images_to_probe = len(val_data)

    # 3) Probe and save
    saved = 0
    global_idx = 0

    group_dir = os.path.join(out_dir, f"group_{'-'.join(map(str, dims_to_probe))}")
    os.makedirs(group_dir, exist_ok=True)

    lat_dir = "latent_vector_file"
    os.makedirs(lat_dir, exist_ok=True)

    for x_batch, y_batch in val_loader:
        b = x_batch.size(0)

        for j in range(b):
            if saved >= num_images_to_probe:
                break

            x = x_batch[j:j+1]
            y = y_batch[j:j+1]
            sample_id = str(global_idx)

            # save_latent_vector(
            # model=model,
            # x=x,
            # y=y,
            # device=device,
            # out_dir=lat_dir,
            # sample_id=sample_id,
            # angles=range(1, 360),
            # txt_name="latents_by_angle.txt"
            # )

            probe_latent_dimension_save(
            model=model,
            x=x,
            y=y,
            device=device,
            dims_to_probe=dims_to_probe,   # <-- pass full list
            out_dir=group_dir,
            sweep_range=sweep_range,
            steps=steps,
            sample_id=sample_id,
            include_original=include_original,
            centered=centered,
            probe_mode=probe_mode,         # "together" or "one_by_one"
            )

            saved += 1
            global_idx += 1

        if saved >= num_images_to_probe:
            break

    print(f"Saved probe grids for {saved} images into: {out_dir}")

if __name__ == "__main__":
    main()