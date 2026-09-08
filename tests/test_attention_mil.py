"""Shape and forward-pass smoke tests for the Attention MIL models.
Run with: pytest tests/test_attention_mil.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from models.attention_mil import GatedAttention, AttentionMIL, FusionAttentionMIL


def test_gated_attention_output_shape():
    N, in_dim = 50, 256
    attn = GatedAttention(in_dim=in_dim, hidden_dim=128)
    x = torch.randn(N, in_dim)
    weights = attn(x)
    assert weights.shape == (N, 1)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5), "attention weights must sum to 1 (softmax)"


def test_cnn_mil_forward_shape():
    N = 100
    model = AttentionMIL(in_dim=1280, proj_dim=512)
    bag = torch.randn(N, 1280)
    logit, attn = model(bag)
    assert logit.shape == (1, 1)
    assert attn.shape == (N,)
    assert torch.allclose(attn.sum(), torch.tensor(1.0), atol=1e-5)


def test_foundation_mil_forward_shape():
    N = 100
    model = AttentionMIL(in_dim=1024, proj_dim=512)
    bag = torch.randn(N, 1024)
    logit, attn = model(bag)
    assert logit.shape == (1, 1)
    assert attn.shape == (N,)


def test_fusion_mil_forward_shape():
    N = 100
    model = FusionAttentionMIL(cnn_dim=1280, found_dim=1024, proj_dim=512)
    cnn_bag = torch.randn(N, 1280)
    found_bag = torch.randn(N, 1024)
    logit, attn = model(cnn_bag, found_bag)
    assert logit.shape == (1, 1)
    assert attn.shape == (N,)
    assert torch.allclose(attn.sum(), torch.tensor(1.0), atol=1e-5)


def test_variable_bag_size():
    """MIL bags have variable N per slide -- models must handle this without shape errors."""
    model = AttentionMIL(in_dim=1280)
    for N in [1, 5, 500, 4000]:
        bag = torch.randn(N, 1280)
        logit, attn = model(bag)
        assert logit.shape == (1, 1)
        assert attn.shape == (N,)


def test_fusion_mil_gradient_flow():
    """Backward pass should not error and should produce non-zero gradients."""
    model = FusionAttentionMIL(cnn_dim=1280, found_dim=1024)
    cnn_bag = torch.randn(10, 1280)
    found_bag = torch.randn(10, 1024)
    label = torch.tensor([[1.0]])

    logit, _ = model(cnn_bag, found_bag)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "no gradients flowed through the fusion model"


if __name__ == "__main__":
    test_gated_attention_output_shape()
    test_cnn_mil_forward_shape()
    test_foundation_mil_forward_shape()
    test_fusion_mil_forward_shape()
    test_variable_bag_size()
    test_fusion_mil_gradient_flow()
    print("✅ All tests passed")
