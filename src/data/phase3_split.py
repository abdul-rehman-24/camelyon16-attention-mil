"""Final stratified slide-level split for the 150-slide Phase 3 dataset."""
from sklearn.model_selection import train_test_split

def make_phase3_split(cache, val_frac=0.15, seed=42):
    seen, labels = [], []
    for sid, lbl in zip(cache["slide_id"], cache["label"]):
        if sid not in seen:
            seen.append(sid)
            labels.append(lbl)
    train_ids, val_ids = train_test_split(seen, test_size=val_frac, stratify=labels, random_state=seed)
    return train_ids, val_ids

def check_slide_leakage(train_ids, val_ids):
    overlap = set(train_ids) & set(val_ids)
    return len(overlap) == 0, overlap
