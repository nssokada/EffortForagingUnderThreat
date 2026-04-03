"""
Shared configuration for analysis notebooks.
Auto-detects repo root and latest preprocessing outputs.

Usage (in any notebook):
    from config import *
    # Gives you: REPO_ROOT, DATA_DIR, VIGOR_DIR, MODEL_DIR, EXCLUDE, BKW
"""

import os
from pathlib import Path

# Find repo root
REPO_ROOT = Path(os.getcwd())
for _ in range(5):
    if (REPO_ROOT / '.git').exists():
        break
    REPO_ROOT = REPO_ROOT.parent
os.chdir(REPO_ROOT)

# Auto-detect latest stage5 output
_processed = REPO_ROOT / "data" / "exploratory_350" / "processed"
_stage5_candidates = sorted(_processed.glob("stage5_*"))
if _stage5_candidates:
    DATA_DIR = _stage5_candidates[-1]
else:
    raise FileNotFoundError(f"No stage5 output found in {_processed}")

# Standard paths
VIGOR_DIR = REPO_ROOT / "results" / "stats" / "vigor_analysis"
MODEL_DIR = REPO_ROOT / "results" / "stats" / "joint_optimal"

# Standard exclusions (calibration outliers)
EXCLUDE = [154, 197, 208]

# Default bambi MCMC kwargs
BKW = dict(draws=2000, tune=1000, chains=4, progressbar=False, random_seed=42)

print(f"Repo: {REPO_ROOT}")
print(f"Data: {DATA_DIR.name}")
