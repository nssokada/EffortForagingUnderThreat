"""
Shared configuration for analysis notebooks.

Loads BOTH exploratory and confirmatory samples for side-by-side analysis.
All paths auto-detected from repo structure.

Usage (in any notebook):
    from config import *

Provides:
    REPO_ROOT           — repository root
    SAMPLES             — dict with 'exploratory' and 'confirmatory' data
    EXP, CONF           — shorthand for SAMPLES['exploratory'], SAMPLES['confirmatory']
    BKW                 — default bambi MCMC kwargs
    USE_GPU             — whether GPU is available

Each sample dict contains:
    data_dir, model_dir, vigor_dir, params_path, exclude, label, color
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# ============================================================
# Repo root
# ============================================================
REPO_ROOT = Path(os.getcwd())
for _ in range(5):
    if (REPO_ROOT / '.git').exists():
        break
    REPO_ROOT = REPO_ROOT.parent
os.chdir(REPO_ROOT)

# ============================================================
# GPU detection
# ============================================================
try:
    import jax
    USE_GPU = any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in jax.devices())
except ImportError:
    USE_GPU = False

# ============================================================
# Sample configuration
# ============================================================
@dataclass
class SampleConfig:
    name: str
    label: str
    color: str
    data_dir: Path
    model_dir: Path
    vigor_dir: Path
    params_path: Path
    model_input_dir: Path
    exclude: List[int] = field(default_factory=list)
    n_subjects: int = 0

def _find_stage5(sample_name):
    processed = REPO_ROOT / "data" / f"{sample_name}_350" / "processed"
    candidates = sorted(processed.glob("stage5_*"))
    if candidates:
        return candidates[-1]
    return None

def _build_sample(name, label, color, exclude):
    s5 = _find_stage5(name)
    if s5 is None:
        return None
    model_dir = REPO_ROOT / "results" / "stats" / "joint_optimal" / name
    vigor_dir = REPO_ROOT / "results" / "stats" / "vigor_analysis"
    params_path = model_dir / "mcmc_m4_params.csv"
    model_input_dir = REPO_ROOT / "data" / f"model_input_{name}"
    if not model_input_dir.exists():
        model_input_dir = REPO_ROOT / "data" / "model_input"

    import pandas as pd
    n = 0
    if params_path.exists():
        n = len(pd.read_csv(params_path))

    return SampleConfig(
        name=name, label=label, color=color,
        data_dir=s5, model_dir=model_dir, vigor_dir=vigor_dir,
        params_path=params_path, model_input_dir=model_input_dir,
        exclude=exclude, n_subjects=n,
    )

# Build both samples
EXP = _build_sample("exploratory", "Exploratory (N=290)", "#1f77b4", [154, 197, 208])
CONF = _build_sample("confirmatory", "Confirmatory (N=281)", "#d62728", [])

SAMPLES = {}
if EXP: SAMPLES['exploratory'] = EXP
if CONF: SAMPLES['confirmatory'] = CONF

# ============================================================
# Shared constants
# ============================================================
# Default bambi MCMC kwargs
BKW = dict(draws=2000, tune=1000, chains=4, progressbar=False, random_seed=42)

# Model constants (from fitted population params)
MODEL_CONSTANTS = dict(gamma=0.76, hazard=0.481, sp=0.25, C_pen=5.0)

# MCMC settings
MCMC_SETTINGS = dict(
    num_warmup=2000, num_samples=4000, num_chains=4,
    target_accept=0.95, max_tree_depth=10,
)

# ============================================================
# Print summary
# ============================================================
for name, s in SAMPLES.items():
    print(f"{s.label}: {s.data_dir.name} | params: {s.n_subjects} subjects")
print(f"GPU: {'Yes' if USE_GPU else 'No'}")
