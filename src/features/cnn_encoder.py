"""Frozen EfficientNet-B0 patch encoder (1280-d pooled features)."""
import timm
import torch
import torchvision.transforms as T

def build_cnn_encoder(device="cuda"):
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)  # num_classes=0 -> pooled features
    model.eval().to(device)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform

@torch.no_grad()
def extract_cnn_batch(model, transform, pil_images, device="cuda"):
    batch = torch.stack([transform(img) for img in pil_images]).to(device)
    feats = model(batch)  # (B, 1280)
    return feats.cpu()
