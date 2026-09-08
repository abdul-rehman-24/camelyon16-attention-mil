"""
Thin CLI entrypoint for training a single MIL model on a cached embedding set.

Usage:
    python scripts/run_training.py --cache_path embeddings_cache/phase3_final150_embeddings.pt \\
        --model fusion --epochs 40 --ckpt_dir checkpoints/fusion_mil_run

Requires: an embedding cache already built via src/features/ (see README "Reproducing this work").
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.optim as optim

from features.cache_io import load_cache
from data.phase3_split import make_phase3_split
from data.bag_dataset import MILBagDataset
from models.attention_mil import AttentionMIL, FusionAttentionMIL
from training.train_mil import train_mil_model


MODEL_BUILDERS = {
    "cnn": lambda: (AttentionMIL(in_dim=1280), "cnn"),
    "foundation": lambda: (AttentionMIL(in_dim=1024), "foundation"),
    "fusion": lambda: (FusionAttentionMIL(cnn_dim=1280, found_dim=1024), "fusion"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_path", required=True, help="Path to a cached embeddings .pt file")
    parser.add_argument("--model", choices=["cnn", "foundation", "fusion"], required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ckpt_dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cache = load_cache(args.cache_path)
    print(f"Loaded cache: {len(cache['slide_id'])} patches, {len(set(cache['slide_id']))} slides")

    train_ids, val_ids = make_phase3_split(cache, val_frac=args.val_frac, seed=args.seed)
    train_labels = [cache["label"][cache["slide_id"].index(s)] for s in train_ids]
    print(f"Train: {len(train_ids)} slides, Val: {len(val_ids)} slides")

    n_normal = train_labels.count("normal")
    n_tumor = train_labels.count("tumor")
    pos_weight = torch.tensor([n_normal / n_tumor]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model, mode = MODEL_BUILDERS[args.model]()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_ds = MILBagDataset(cache, train_ids, mode=mode)
    val_ds = MILBagDataset(cache, val_ids, mode=mode)

    best_auroc, epochs_run, _ = train_mil_model(
        model, train_ds, val_ds, criterion, optimizer, args.epochs, device,
        ckpt_dir=args.ckpt_dir, mode=mode
    )
    print(f"\n✅ Done. Best val AUROC: {best_auroc:.4f} (stopped at epoch {epochs_run})")
    print(f"Checkpoint saved to: {args.ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
