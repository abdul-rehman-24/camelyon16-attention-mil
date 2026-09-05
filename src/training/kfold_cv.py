"""Slide-level stratified k-fold CV for MIL models — checks if a result is
consistent across splits, not just a lucky single train/val partition."""
import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_kfold_splits(cache, n_splits=4, seed=42):
    """Returns list of (train_ids, val_ids) tuples, one per fold. Slide-level, stratified by label."""
    seen, labels = [], []
    for sid, lbl in zip(cache["slide_id"], cache["label"]):
        if sid not in seen:
            seen.append(sid)
            labels.append(lbl)
    seen = np.array(seen)
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(seen, labels):
        folds.append((list(seen[train_idx]), list(seen[val_idx])))
    return folds


def run_kfold_cv(cache, model_builder, mode, criterion_builder, epochs, device,
                  ckpt_dir_prefix, n_splits=4, seed=42, verbose=True):
    """
    model_builder: () -> fresh model instance (must reset each fold!)
    criterion_builder: (train_labels_list) -> criterion (rebuilds pos_weight per fold)
    Returns dict: {fold_num: {"best_val_auroc":.., "epochs_run":..}} + summary stats.
    """
    import torch
    from data.bag_dataset import MILBagDataset
    from training.train_mil import train_mil_model

    folds = make_kfold_splits(cache, n_splits=n_splits, seed=seed)
    fold_results = {}

    for fold_num, (train_ids, val_ids) in enumerate(folds):
        if verbose:
            print(f"\n{'='*50}\nFOLD {fold_num+1}/{n_splits}\n{'='*50}")
            print(f"Train: {train_ids}\nVal: {val_ids}")

        train_labels = [cache["label"][cache["slide_id"].index(s)] for s in train_ids]

        train_ds = MILBagDataset(cache, train_ids, mode=mode)
        val_ds = MILBagDataset(cache, val_ids, mode=mode)

        model = model_builder().to(device)
        criterion = criterion_builder(train_labels, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_val_auroc, epochs_run, history = train_mil_model(
            model, train_ds, val_ds, criterion, optimizer, epochs, device,
            ckpt_dir=f"{ckpt_dir_prefix}_fold{fold_num}", mode=mode, verbose=verbose
        )
        fold_results[fold_num] = {
            "best_val_auroc": best_val_auroc, "epochs_run": epochs_run,
            "train_ids": train_ids, "val_ids": val_ids
        }

    aurocs = [v["best_val_auroc"] for v in fold_results.values() if v["best_val_auroc"] is not None]
    summary = {
        "n_folds": n_splits,
        "mean_val_auroc": float(np.mean(aurocs)) if aurocs else None,
        "std_val_auroc": float(np.std(aurocs)) if aurocs else None,
        "per_fold_auroc": aurocs,
    }
    return fold_results, summary
