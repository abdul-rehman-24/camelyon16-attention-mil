"""Training/eval loop for Attention MIL models. Batch size = 1 bag (variable N per slide) — standard for MIL.
Now with: LR scheduler (ReduceLROnPlateau on val_loss) + early stopping (patience on val_auroc)."""
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
                     ckpt_dir, mode="fusion", verbose=True,
                     use_scheduler=True, early_stop_patience=8, min_delta=1e-4):
    """
    early_stop_patience: epochs to wait for val_auroc improvement before stopping.
    Returns (best_val_auroc, epochs_actually_run, history dict).
    """
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_auroc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_auroc": [], "val_auroc": []}

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_auroc, _, _ = run_mil_epoch(model, train_ds, criterion, optimizer, device, mode, True)
        val_loss, val_auroc, _, _ = run_mil_epoch(model, val_ds, criterion, optimizer, device, mode, False)

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auroc"].append(train_auroc)
        history["val_auroc"].append(val_auroc)

        current_lr = optimizer.param_groups[0]["lr"]
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} train_auroc={train_auroc} "
                  f"| val_loss={val_loss:.4f} val_auroc={val_auroc} | lr={current_lr:.2e} | {time.time()-t0:.1f}s")

        if val_auroc is not None and val_auroc > best_val_auroc + min_delta:
            best_val_auroc = val_auroc
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_auroc": val_auroc},
                       f"{ckpt_dir}/best.pt")
            if verbose:
                print(f"  💾 New best (val_auroc={val_auroc:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            if verbose:
                print(f"  ⏹️  Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
            break

    return best_val_auroc, epoch + 1, history
