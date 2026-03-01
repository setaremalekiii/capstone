import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from latent_data import LatentAngleDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = LatentAngleDataset("latent_vector_file", "latents_by_angle.txt")
    D = ds.latent_dim

    n_val = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    # standardize X using train stats (makes coefficients comparable)
    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))], dim=0)
    mean = X_train.mean(0).to(device)
    std = X_train.std(0).clamp_min(1e-6).to(device)

    model = nn.Linear(D, 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    def eval_mse(loader):
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                pred = model((X - mean) / std)
                loss = loss_fn(pred, y)
                total += loss.item() * X.size(0)
                n += X.size(0)
        return total / max(1, n)

    for ep in range(1, 31):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            pred = model((X - mean) / std)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {ep:02d} | train MSE {eval_mse(train_loader):.3f} | val MSE {eval_mse(val_loader):.3f}")

    # Inspect coefficients
    w = model.weight.detach().cpu().squeeze(0)
    idx = torch.argsort(torch.abs(w), descending=True)
    print("\nTop dims by |weight|:")
    for k in range(min(10, D)):
        i = idx[k].item()
        print(f"dim {i:02d}: w={w[i].item(): .6f}")

if __name__ == "__main__":
    main()