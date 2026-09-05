"""Save/load embedding caches with a version-tagged meta.json (Section 16)."""
import torch
import json
import os

CACHE_VERSION = "v1_cnn1280_phikon1024_patch256_level0_cap4000"

def save_cache(cache_dict, path):
    torch.save(cache_dict, path)
    meta = {"version": CACHE_VERSION, "n_patches": len(cache_dict["slide_id"])}
    with open(path.replace(".pt", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def load_cache(path):
    cache = torch.load(path, weights_only=False)
    meta_path = path.replace(".pt", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["version"] == CACHE_VERSION, f"Cache version mismatch! expected {CACHE_VERSION}, got {meta['version']}"
    return cache
