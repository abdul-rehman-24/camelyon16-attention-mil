"""Training/eval loop for Attention MIL models. Batch size = 1 bag (variable N per slide) — standard for MIL."""
import time
import torch
from sklearn.metrics import roc_auc_score


def run_mil_epoch(model, dataset, criterion, optimizer, device, mode="fusion", train_mode=True):
    model.train() if train_mode else model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.set_grad_enabled(train_mode):
        for item in dataset:
            if mode == "fusion":
                cnn_bag, found_bag, label, _ = item
                cnn_bag, found_bag, label = cnn_bag.to(device), found_bag.to(device), label.to(device)
                if train_mode: optimizer.zero_grad()
                logit, _ = model(cnn_bag, found_bag)
            else:
                bag, label, _ = item
                bag, label = bag.to(device), label.to(device)
                if train_mode: optimizer.zero_grad()
                logit, _ = model(bag)

            loss = criterion(logit, label.unsqueeze(0) if label.dim() == 1 else label)
            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            prob = torch.sigmoid(logit).item()
            all_labels.append(label.item())
            all_probs.append(prob)

    avg_loss = total_loss / len(dataset)
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else None
    return avg_loss, auroc, all_labels, all_probs


def train_mil_model(model, train_ds, val_ds, criterion, optimizer, epochs, device,
                     ckpt_dir, mode="fusion", verbose=True):
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_auroc = 0.0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_auroc, _, _ = run_mil_epoch(model, train_ds, criterion, optimizer, device, mode, True)
        val_loss, val_auroc, _, _ = run_mil_epoch(model, val_ds, criterion, optimizer, device, mode, False)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} train_auroc={train_auroc} "
                  f"| val_loss={val_loss:.4f} val_auroc={val_auroc} | {time.time()-t0:.1f}s")

        if val_auroc is not None and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_auroc": val_auroc},
                       f"{ckpt_dir}/best.pt")
            if verbose:
                print(f"  💾 New best (val_auroc={val_auroc:.4f})")

    return best_val_auroc
