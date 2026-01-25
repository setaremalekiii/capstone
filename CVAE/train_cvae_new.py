""""new script for training CVAE on OBB  boxed images to find p or q band"""

import torch
import torch.nn.functional as Func
import numpy as np
from ConvCVAE import ConvCVAE
import logging
from tqdm import tqdm # progress bar
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure
import json

Logger = logging.getLogger(__name__)

def k_fold_split(n, k_folds):
    """Splits the data indices into k folds for cross-validation."""
    fold_size = len(n) // k_folds
    folds = []
    for i in range(k_folds):
        start = i * fold_size
        end = start + fold_size if i != k_folds - 1 else len(n)
        folds.append(n[start:end])
    return folds

def calculate_class_thresholds(anomaly_scores, true_labels):
    """Calculates optimal thresholds for each class based on anomaly scores."""
    class_thresholds = {}
    for class_label in set(true_labels):
        class_scores = [score for score, label in zip(anomaly_scores, true_labels) if label == class_label]
        threshold = np.percentile(class_scores, 95)  # Example: 95th percentile as threshold
        class_thresholds[class_label] = threshold
    return class_thresholds
    