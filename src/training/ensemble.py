"""Validation-only weighted ensemble search across CNN-MIL, Foundation-MIL, Fusion-MIL."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def get_val_probs(model, dataset, mode, device):
    """Returns (probs, labels) arrays for every bag in the dataset."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for item in dataset:
            if mode == "fusion":
                cnn_bag, found_bag, label, _ = item
                cnn_bag, found_bag = cnn_bag.to(device), found_bag.to(device)
                logit, _ = model(cnn_bag, found_bag)
            else:
                bag, label, _ = item
                bag = bag.to(device)
                logit, _ = model(bag)
            probs.append(torch.sigmoid(logit).item())
            labels.append(label.item())
    return np.array(probs), np.array(labels)


def search_ensemble_weights(val_probs_dict, val_labels, step=0.05):
    """val_probs_dict: {"cnn": array, "foundation": array, "fusion": array}. Grid search on validation only."""
    keys = list(val_probs_dict.keys())
    assert len(keys) == 3
    k1, k2, k3 = keys
    best_weights, best_auroc = None, 0.0
    grid = np.arange(0, 1.001, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1 + 1e-9:
                continue
            w3 = max(0.0, w3)
            ensemble_prob = w1 * val_probs_dict[k1] + w2 * val_probs_dict[k2] + w3 * val_probs_dict[k3]
            auroc = roc_auc_score(val_labels, ensemble_prob)
            if auroc > best_auroc:
                best_auroc = auroc
                best_weights = {k1: round(float(w1), 3), k2: round(float(w2), 3), k3: round(float(w3), 3)}
    return best_weights, best_auroc
