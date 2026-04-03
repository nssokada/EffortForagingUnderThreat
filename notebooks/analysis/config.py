"""
Shared configuration for analysis notebooks.
Auto-detects repo root and latest preprocessing outputs.

Usage (in any notebook):
    from config import *
    # Gives you: SAMPLE, REPO_ROOT, DATA_DIR, VIGOR_DIR, MODEL_DIR, EXCLUDE, BKW

Toggle sample:
    Set SAMPLE = "confirmatory" below to switch datasets.
"""

import os
from pathlib import Path

# ============================================================
# TOGGLE: "exploratory" or "confirmatory"
# ============================================================
SAMPLE = "exploratory"
# ============================================================

# Find repo root
REPO_ROOT = Path(os.getcwd())
for _ in range(5):
    if (REPO_ROOT / '.git').exists():
        break
    REPO_ROOT = REPO_ROOT.parent
os.chdir(REPO_ROOT)

# Sample-specific paths
_sample_dir = REPO_ROOT / "data" / f"{SAMPLE}_350"
_processed = _sample_dir / "processed"
_stage5_candidates = sorted(_processed.glob("stage5_*"))
if _stage5_candidates:
    DATA_DIR = _stage5_candidates[-1]
else:
    raise FileNotFoundError(f"No stage5 output found in {_processed}")

# Results go to sample-specific subdirectories
RESULTS_BASE = REPO_ROOT / "results"
VIGOR_DIR = RESULTS_BASE / "stats" / "vigor_analysis"
MODEL_DIR = RESULTS_BASE / "stats" / "joint_optimal"

# Standard exclusions (calibration outliers — recomputed per sample)
if SAMPLE == "exploratory":
    EXCLUDE = [154, 197, 208]
else:
    EXCLUDE = []  # Will be determined after preprocessing confirmatory data

# Default bambi MCMC kwargs
BKW = dict(draws=2000, tune=1000, chains=4, progressbar=False, random_seed=42)

print(f"Sample: {SAMPLE}")
print(f"Data:   {DATA_DIR.name}")
