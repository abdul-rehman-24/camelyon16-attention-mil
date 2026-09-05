"""Frozen Phikon-v2 patch encoder (1024-d CLS token)."""
import torch
from transformers import AutoImageProcessor, AutoModel

def build_foundation_encoder(device="cuda"):
    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    model.eval().to(device)
    return model, processor

@torch.no_grad()
def extract_foundation_batch(model, processor, pil_images, device="cuda"):
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
    return cls_embedding.cpu()
