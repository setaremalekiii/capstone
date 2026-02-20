from probing_latent import probe_latent_dimension
from torch import model, torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("best.pth"))
model.to(device)

# get one batch
x_batch, y_batch = next(iter(train_dataloader))

# take first sample
x_sample = x_batch[0:1]
y_sample = y_batch[0:1]

# probe dimension 3 for example
probe_latent_dimension(model, x_sample, y_sample, device, dim_to_probe=3)
