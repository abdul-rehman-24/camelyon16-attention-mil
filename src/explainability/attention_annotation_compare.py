"""Compares model attention weights against expert tumor annotations (point-in-polygon test)."""
import numpy as np
from matplotlib.path import Path

def points_in_any_polygon(points_xy, polygons):
    inside = np.zeros(len(points_xy), dtype=bool)
    for poly in polygons:
        path = Path(poly)
        inside |= path.contains_points(points_xy)
    return inside

def get_slide_attention(model, cache, slide_id, device):
    import torch
    idx = [i for i, s in enumerate(cache["slide_id"]) if s == slide_id]
    cnn_bag = torch.stack([cache["cnn_feature"][i] for i in idx]).to(device)
    found_bag = torch.stack([cache["foundation_feature"][i] for i in idx]).to(device)
    xs = np.array([cache["x"][i] for i in idx])
    ys = np.array([cache["y"][i] for i in idx])
    model.eval()
    with torch.no_grad():
        logit, attn = model(cnn_bag, found_bag)
    return xs, ys, attn.cpu().numpy(), torch.sigmoid(logit).item()

def compute_tumor_attention_stats(xs, ys, attn_weights, polygons, patch_size=256):
    centers = np.stack([xs + patch_size/2, ys + patch_size/2], axis=1)
    inside_tumor = points_in_any_polygon(centers, polygons)
    stats = {
        "n_patches": len(xs),
        "n_patches_in_tumor_region": int(inside_tumor.sum()),
        "mean_attention_in_tumor": float(attn_weights[inside_tumor].mean()) if inside_tumor.any() else None,
        "mean_attention_outside_tumor": float(attn_weights[~inside_tumor].mean()) if (~inside_tumor).any() else None,
        "attention_ratio": None,
    }
    if stats["mean_attention_in_tumor"] and stats["mean_attention_outside_tumor"]:
        stats["attention_ratio"] = stats["mean_attention_in_tumor"] / stats["mean_attention_outside_tumor"]
    return stats, inside_tumor
