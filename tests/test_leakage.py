"""Verifies that stratified train/val/test splits never leak slides across sets.
Run with: pytest tests/test_leakage.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_fake_cache(n_tumor=20, n_normal=30):
    slide_id, label = [], []
    for i in range(n_tumor):
        for _ in range(10):
            slide_id.append(f"tumor_{i:03d}")
            label.append("tumor")
    for i in range(n_normal):
        for _ in range(10):
            slide_id.append(f"normal_{i:03d}")
            label.append("normal")
    return {"slide_id": slide_id, "label": label}


def test_phase3_split_no_leakage():
    from data.phase3_split import make_phase3_split, check_slide_leakage
    cache = _make_fake_cache()
    train_ids, val_ids = make_phase3_split(cache, val_frac=0.15, seed=42)
    passed, overlap = check_slide_leakage(train_ids, val_ids)
    assert passed, f"leakage detected: {overlap}"
    assert len(overlap) == 0


def test_phase3_split_covers_all_slides():
    from data.phase3_split import make_phase3_split
    cache = _make_fake_cache()
    train_ids, val_ids = make_phase3_split(cache, val_frac=0.15, seed=42)
    all_slides = set(cache["slide_id"])
    assert set(train_ids) | set(val_ids) == all_slides, "some slides missing from split"


def test_kfold_splits_no_leakage():
    from training.kfold_cv import make_kfold_splits
    cache = _make_fake_cache()
    folds = make_kfold_splits(cache, n_splits=4, seed=42)
    for train_ids, val_ids in folds:
        overlap = set(train_ids) & set(val_ids)
        assert len(overlap) == 0, f"fold leakage: {overlap}"


if __name__ == "__main__":
    test_phase3_split_no_leakage()
    test_phase3_split_covers_all_slides()
    test_kfold_splits_no_leakage()
    print("✅ All leakage tests passed")
