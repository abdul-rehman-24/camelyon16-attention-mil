"""Generates patch coordinates (not saved images) at a target level, filtered by
tissue percentage using the low-res tissue mask."""
import numpy as np

def extract_patch_coords(slide, tissue_mask, mask_level, patch_size=256,
                          target_level=0, tissue_thresh=0.5, stride=None):
    """Returns list of dicts: {x, y, level} — (x,y) are level-0 coordinates,
    suitable for slide.read_region((x, y), target_level, (patch_size, patch_size))."""
    stride = stride or patch_size
    level_downsample = slide.level_downsamples[target_level]
    mask_downsample = slide.level_downsamples[mask_level]

    level_w, level_h = slide.level_dimensions[target_level]
    mask_h, mask_w = tissue_mask.shape

    # scale factor from target_level pixel coords -> mask pixel coords
    scale = level_downsample / mask_downsample

    coords = []
    for y in range(0, level_h - patch_size + 1, stride):
        for x in range(0, level_w - patch_size + 1, stride):
            mx0, my0 = int(x * scale), int(y * scale)
            mx1, my1 = int((x + patch_size) * scale), int((y + patch_size) * scale)
            mx1, my1 = min(mx1, mask_w), min(my1, mask_h)
            if mx1 <= mx0 or my1 <= my0:
                continue
            region = tissue_mask[my0:my1, mx0:mx1]
            tissue_frac = region.mean() if region.size > 0 else 0.0
            if tissue_frac >= tissue_thresh:
                level0_x = int(x * level_downsample)
                level0_y = int(y * level_downsample)
                coords.append({"x": level0_x, "y": level0_y, "level": target_level, "tissue_frac": float(tissue_frac)})
    return coords

def apply_patch_cap(coords, cap, seed=42):
    """Randomly subsamples coords to at most `cap` patches, deterministic via seed."""
    import random
    if len(coords) <= cap:
        return coords
    rng = random.Random(seed)
    return rng.sample(coords, cap)
