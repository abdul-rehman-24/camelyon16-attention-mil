# V2 Session Resume Guide (updated after Pilot phase)

## RUN AGAIN (new Kaggle session)
1. Environment: `pip install openslide-python timm transformers`, `apt-get install openslide-tools`
2. Clone repo: `git clone https://github.com/abdul-rehman-24/camelyon16-attention-mil.git`
3. Attach Kaggle Dataset `camelyon16-v2-artifacts` via "+ Add Data"
4. Load cached pilot embeddings:
```python
   from features.cache_io import load_cache
   cache = load_cache("/kaggle/input/camelyon16-v2-artifacts/embeddings/pilot_embeddings.pt")
```

## DO NOT RERUN
- 5-WSI download (raw WSIs deleted; embeddings cached)
- Tissue detection + patch extraction on pilot
- CNN + Phikon-v2 feature extraction on pilot (20,000 patches)
- Phikon-v2 feasibility gate (PASS confirmed)

## STATUS
Pilot phase (Section 5) fully complete: 5 WSIs (3 tumor + 2 normal), tissue detection,
patch extraction with 4,000/slide cap, CNN+Phikon-v2 embeddings cached (20,000 patches,
187MB), verified round-trip, backed up to Kaggle Dataset `camelyon16-v2-artifacts`.

## NEXT
- Section 9: Implement Attention MIL models (CNN-MIL, Foundation-MIL, Fusion-MIL)
- Smoke-test all three on pilot's 20,000-patch cache (5 bags)
- Then decide: Phase 2 (20-WSI) scale-up vs. finishing MIL training loop first
