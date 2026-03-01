import os
import torch
from torch.utils.data import Dataset

class LatentAngleDataset(Dataset):
    def __init__(self, root_dir="latent_vector_file", filename="latents_by_angle.txt"):
        path = os.path.join(root_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        X, y = [], []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # sample_id = parts[0]  # keep if you want
                angle = float(parts[1])
                z = [float(v) for v in parts[2:]]
                y.append(angle)
                X.append(z)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.latent_dim = self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]