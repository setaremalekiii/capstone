import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------
# CONFIG
# -----------------------------
LATENTS_PATH = "outputs/train/exp186/latents.npy"   # expects shape (N, 64)
OUT_DIR = "pca_out"
N_COMPONENTS = 64              # keep all PCs to assess full variation
# -----------------------------

import os
os.makedirs(OUT_DIR, exist_ok=True)

# Load latents
Z = np.load(LATENTS_PATH)  # (N, 64)
if Z.ndim != 2 or Z.shape[1] != 64:
    raise ValueError(f"Expected latents of shape (N, 64). Got {Z.shape}.")

# Standardize (important for PCA unless you *want* raw-scale PCA)
scaler = StandardScaler(with_mean=True, with_std=True)
Zs = scaler.fit_transform(Z)

# PCA
pca = PCA(n_components=N_COMPONENTS, svd_solver="full", random_state=0)
pcs = pca.fit_transform(Zs)

# Explained variance
evr = pca.explained_variance_ratio_          # (64,)
ev = pca.explained_variance_                 # (64,)
components = pca.components_                 # (64, 64) rows=PCs, cols=latent dims

# Save explained variance per PC
df_pc = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(N_COMPONENTS)],
    "explained_variance_ratio": evr,
    "explained_variance": ev
})
df_pc.to_csv(os.path.join(OUT_DIR, "pc_explained_variance.csv"), index=False)

# ------------------------------------------
# Rank latent dimensions by contribution
# Weighted squared loadings
# ------------------------------------------
# components[k, j] is loading of latent dim j on PC k
# Score per latent dim j = sum_k (loading^2 * EVR_k)
scores = (components**2 * evr[:, None]).sum(axis=0)  # (64,)

df_dims = pd.DataFrame({
    "latent_dim": np.arange(64),
    "score_weighted_loading": scores
}).sort_values("score_weighted_loading", ascending=False)

df_dims["rank"] = np.arange(1, 65)
df_dims.to_csv(os.path.join(OUT_DIR, "latent_dim_rank.csv"), index=False)

# Also save the loadings matrix for inspection
df_loadings = pd.DataFrame(
    components,
    index=[f"PC{i+1}" for i in range(N_COMPONENTS)],
    columns=[f"z{i}" for i in range(64)]
)
df_loadings.to_csv(os.path.join(OUT_DIR, "pc_loadings_matrix.csv"))

# Save transformed PCs (optional)
df_pcs = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(N_COMPONENTS)])
df_pcs.to_csv(os.path.join(OUT_DIR, "samples_in_pc_space.csv"), index=False)

print("Saved:")
print(f"- {OUT_DIR}/pc_explained_variance.csv")
print(f"- {OUT_DIR}/latent_dim_rank.csv")
print(f"- {OUT_DIR}/pc_loadings_matrix.csv")
print(f"- {OUT_DIR}/samples_in_pc_space.csv")

# Print top 10 latent dims
print("\nTop 10 latent dims by weighted-loading score:")
print(df_dims.head(10).to_string(index=False))