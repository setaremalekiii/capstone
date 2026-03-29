#!/usr/bin/env python3
"""
train_linear.py
===============
Shared training script — works with features from ANY of the 3 options:
    pixel_features.npz  (Option 1)
    pca_features.npz    (Option 2)
    cnn_features.npz    (Option 3)

All three save the same format: X=(N, D), labels=(N,), keys=(N,)

Usage
-----
python train_linear.py \
    --features  pca_features.npz \
    --out_model orientation_model.npz \
    [--val_split 0.2] \
    [--C 1.0]

Output
------
orientation_model.npz:
    W, b, val_acc, val_auc, feature_dim
"""

import argparse
from pathlib import Path

import numpy as np


# ── minimal implementations (no sklearn needed) ───────────────────────────── #

def standardize(X_tr, X_va):
    mean = X_tr.mean(axis=0)
    std  = X_tr.std(axis=0) + 1e-8
    return (X_tr - mean) / std, (X_va - mean) / std, mean, std


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def logistic_regression_train(X, y, C=1.0, max_iter=1000, tol=1e-5, lr=0.1):
    """
    L2-regularized logistic regression via gradient descent.
    class_weight='balanced' equivalent applied via sample weights.
    """
    N, D   = X.shape
    W      = np.zeros(D, dtype=np.float64)
    b      = 0.0
    lam    = 1.0 / (C * N)   # L2 regularization strength

    # balanced class weights
    n_pos  = y.sum()
    n_neg  = N - n_pos
    w_pos  = N / (2.0 * n_pos) if n_pos > 0 else 1.0
    w_neg  = N / (2.0 * n_neg) if n_neg > 0 else 1.0
    sample_w = np.where(y == 1, w_pos, w_neg)

    prev_loss = float("inf")
    for it in range(max_iter):
        logits = X @ W + b
        probs  = sigmoid(logits)
        errors = (probs - y) * sample_w

        grad_W = X.T @ errors / N + lam * W
        grad_b = errors.mean()

        W -= lr * grad_W
        b -= lr * grad_b

        loss = -(sample_w * (y * np.log(probs + 1e-12) +
                             (1 - y) * np.log(1 - probs + 1e-12))).mean()
        loss += 0.5 * lam * (W ** 2).sum()

        if abs(prev_loss - loss) < tol:
            print(f"  Converged at iteration {it+1}")
            break
        prev_loss = loss

        if it % 100 == 0:
            print(f"  iter {it:4d}  loss={loss:.5f}")

    return W, b


def predict(X, W, b, threshold=0.5):
    probs = sigmoid(X @ W + b)
    return (probs >= threshold).astype(int), probs


def roc_auc(y_true, y_prob):
    pairs  = sorted(zip(y_prob, y_true), reverse=True)
    n_pos  = sum(y_true)
    n_neg  = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = auc = prev_tp = 0
    for prob, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += (tp + prev_tp) / 2.0
            prev_tp = tp
    auc += (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


def stratified_split(X, y, keys, val_frac, seed=42):
    rng    = np.random.default_rng(seed)
    idx0   = np.where(y == 0)[0]
    idx1   = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n_val0 = max(1, int(len(idx0) * val_frac))
    n_val1 = max(1, int(len(idx1) * val_frac))
    val_idx   = np.concatenate([idx0[:n_val0],  idx1[:n_val1]])
    train_idx = np.concatenate([idx0[n_val0:],  idx1[n_val1:]])
    return train_idx, val_idx


def classification_report(y_true, y_pred):
    for cls, name in [(0, "keep(0)"), (1, "flip(1)")]:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        sup  = sum(t == cls for t in y_true)
        print(f"  {name:<10}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  support={sup}")


# ─────────────────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",  required=True, help="*.npz from any extract_features_*.py")
    ap.add_argument("--out_model", default="orientation_model.npz")
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--C",         type=float, default=1.0,
                    help="Inverse regularization (higher = less regularization)")
    ap.add_argument("--lr",        type=float, default=0.1)
    ap.add_argument("--max_iter",  type=int,   default=2000)
    ap.add_argument("--seed",      type=int,   default=42)
    args = ap.parse_args()

    data   = np.load(args.features, allow_pickle=True)
    X      = data["X"].astype(np.float64)
    labels = data["labels"].astype(int)
    keys   = data["keys"]

    N, D = X.shape
    print(f"[INFO] Loaded {N} samples, feature_dim={D}")
    print(f"       keep(0)={int((labels==0).sum())}  flip(1)={int((labels==1).sum())}")

    if N < 10:
        raise RuntimeError("Too few samples. Label more images first.")

    train_idx, val_idx = stratified_split(X, labels, keys, args.val_split, args.seed)
    X_tr, y_tr = X[train_idx], labels[train_idx]
    X_va, y_va = X[val_idx],   labels[val_idx]
    print(f"[SPLIT] train={len(train_idx)}  val={len(val_idx)}")

    # standardize
    X_tr_s, X_va_s, sc_mean, sc_std = standardize(X_tr, X_va)

    print(f"\n[TRAIN] C={args.C}  lr={args.lr}  max_iter={args.max_iter}")
    W, b = logistic_regression_train(X_tr_s, y_tr.astype(np.float64),
                                      C=args.C, max_iter=args.max_iter, lr=args.lr)

    y_pred, y_prob = predict(X_va_s, W, b)
    val_acc = float((y_pred == y_va).mean())
    val_auc = roc_auc(y_va.tolist(), y_prob.tolist())

    print(f"\n── Validation results ──────────────────────────────────")
    print(f"   Accuracy : {val_acc:.4f}")
    print(f"   ROC-AUC  : {val_auc:.4f}")
    classification_report(y_va.tolist(), y_pred.tolist())

    tp = sum(t == 1 and p == 1 for t, p in zip(y_va, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_va, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_va, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_va, y_pred))
    print(f"\n── Confusion Matrix ────────────────────────────────────")
    print(f"                  Predicted")
    print(f"               keep(0)  flip(1)")
    print(f"  Actual keep(0)  {tn:5d}   {fp:5d}")
    print(f"  Actual flip(1)  {fn:5d}   {tp:5d}")

    # absorb scaler into W and b so inference needs no scaler
    W_raw = W / sc_std
    b_raw = float(b - np.dot(W / sc_std, sc_mean))

    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        W=W_raw.astype(np.float32),
        b=np.array([b_raw], dtype=np.float32),
        val_acc=np.array([val_acc], dtype=np.float32),
        val_auc=np.array([val_auc], dtype=np.float32),
        feature_dim=np.array([D]),
    )
    print(f"\n[SAVED] Model → {out_path}")


if __name__ == "__main__":
    main()