"""Tissue/background segmentation via Otsu thresholding on the saturation channel,
computed at a low-resolution pyramid level for speed."""
import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from scipy import ndimage

def get_tissue_mask(slide, mask_level=None):
    """Returns (mask, level_used, level_dims). mask is bool array, True=tissue."""
    if mask_level is None:
        # pick a level with downsample close to 32x for a fast, stable mask
        target = 32.0
        downsamples = slide.level_downsamples
        mask_level = min(range(len(downsamples)), key=lambda i: abs(downsamples[i] - target))

    dims = slide.level_dimensions[mask_level]
    img = slide.read_region((0, 0), mask_level, dims).convert("RGB")
    img_np = np.array(img)

    hsv = rgb2hsv(img_np)
    saturation = hsv[:, :, 1]

    thresh = threshold_otsu(saturation)
    mask = saturation > thresh

    # clean up: remove tiny specks, fill small holes
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_fill_holes(mask)

    return mask, mask_level, dims
