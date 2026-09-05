"""Gated Attention MIL (Ilse et al. 2018) — shared core for CNN-MIL, Foundation-MIL, Fusion-MIL."""
import torch
import torch.nn as nn


class GatedAttention(nn.Module):
    """Produces one scalar attention weight per instance in the bag."""
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, instances):
        # instances: (N, in_dim) -> weights: (N, 1)
        A_V = self.attention_V(instances)
        A_U = self.attention_U(instances)
        A = self.attention_w(A_V * A_U)          # (N, 1) gated attention logits
        A = torch.softmax(A, dim=0)               # normalize across the bag
        return A


class AttentionMIL(nn.Module):
    """
    Single-branch Attention MIL (used for both CNN-MIL and Foundation-MIL).
    Input: bag of N instance embeddings (N variable per slide).
    """
    def __init__(self, in_dim, proj_dim=512, attn_hidden=128, num_classes=1):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(in_dim, proj_dim), nn.ReLU())
        self.attention = GatedAttention(proj_dim, attn_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, bag):
        # bag: (N, in_dim) — one slide's instances
        h = self.projection(bag)          # (N, proj_dim)
        A = self.attention(h)             # (N, 1)
        slide_embedding = (A * h).sum(dim=0, keepdim=True)  # (1, proj_dim), weighted sum
        logit = self.classifier(slide_embedding)            # (1, num_classes)
        return logit, A.squeeze(-1)       # logit + per-patch attention weights


class FusionAttentionMIL(nn.Module):
    """
    Two-branch Fusion Attention MIL (CNN + Foundation), per-branch projection then
    concatenation before attention (V1's fusion philosophy, reused per Section 1.2/9).
    """
    def __init__(self, cnn_dim=1280, found_dim=1024, proj_dim=512, attn_hidden=128, num_classes=1):
        super().__init__()
        self.cnn_projection = nn.Sequential(nn.Linear(cnn_dim, proj_dim), nn.ReLU())
        self.found_projection = nn.Sequential(nn.Linear(found_dim, proj_dim), nn.ReLU())
        fused_dim = proj_dim * 2
        self.attention = GatedAttention(fused_dim, attn_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, cnn_bag, found_bag):
        # cnn_bag: (N, cnn_dim), found_bag: (N, found_dim) — same N, same slide
        h_cnn = self.cnn_projection(cnn_bag)        # (N, proj_dim)
        h_found = self.found_projection(found_bag)  # (N, proj_dim)
        h_fused = torch.cat([h_cnn, h_found], dim=1)  # (N, fused_dim)
        A = self.attention(h_fused)                 # (N, 1)
        slide_embedding = (A * h_fused).sum(dim=0, keepdim=True)
        logit = self.classifier(slide_embedding)
        return logit, A.squeeze(-1)
