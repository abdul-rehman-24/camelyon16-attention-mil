"""Groups cached patch embeddings into per-slide bags for MIL. One bag = one slide (variable N)."""
import torch
import numpy as np
from torch.utils.data import Dataset

LABEL_MAP = {"normal": 0, "tumor": 1}

class MILBagDataset(Dataset):
    """
    mode: "cnn" -> returns (cnn_bag, label)
          "foundation" -> returns (found_bag, label)
          "fusion" -> returns (cnn_bag, found_bag, label)
    """
    def __init__(self, cache, slide_ids, mode="fusion"):
        self.mode = mode
        slide_arr = np.array(cache["slide_id"])
        self.bags = []
        for sid in slide_ids:
            idx = np.where(slide_arr == sid)[0]
            label = LABEL_MAP[cache["label"][idx[0]]]
            cnn_bag = torch.stack([cache["cnn_feature"][i] for i in idx])
            found_bag = torch.stack([cache["foundation_feature"][i] for i in idx])
            self.bags.append({"slide_id": sid, "label": label, "cnn": cnn_bag, "found": found_bag})

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        b = self.bags[idx]
        label_t = torch.tensor([b["label"]], dtype=torch.float32)
        if self.mode == "cnn":
            return b["cnn"], label_t, b["slide_id"]
        elif self.mode == "foundation":
            return b["found"], label_t, b["slide_id"]
        else:
            return b["cnn"], b["found"], label_t, b["slide_id"]


def get_all_slide_ids(cache):
    seen = []
    for sid in cache["slide_id"]:
        if sid not in seen:
            seen.append(sid)
    return seen
