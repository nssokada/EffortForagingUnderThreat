"""
Fit FETExponentialBias (best model) on the full 293-subject exploratory dataset.
Full MCMC settings: 2000 warmup / 4000 samples / 4 chains.
"""

import sys
from pathlib import Path
import glob
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from modeling import configure_device, FETExponentialBias, ModelFitter

# --- Device ---
configure_device(use_gpu=False)

# --- Data ---
stage5_dirs = sorted(glob.glob(str(
    ROOT / 'data' / 'exploratory_350' / 'processed' / 'stage5_filtered_data_*'
)))
if not stage5_dirs:
    raise FileNotFoundError("No stage5 output found.")
STAGE5_DIR = Path(stage5_dirs[-1])
data = pd.read_csv(STAGE5_DIR / 'behavior.csv')
print(f"Data: {len(data)} trials, {data['subj'].nunique()} subjects")
print(f"Loaded from: {STAGE5_DIR}")

# --- Output ---
SAVE_DIR = ROOT / 'results' / 'model_fits' / 'exploratory'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Fit ---
MCMC_CONFIG = dict(
    num_warmup=2000,
    num_samples=4000,
    num_chains=4,
    target_accept_prob=0.90,
    R_H=5.0,
    R_L=1.0,
    C=5.0,
)

print("\nFitting FETExponentialBias ...")
print(f"Config: {MCMC_CONFIG}")
print(f"Saving to: {SAVE_DIR}\n")

fitter = ModelFitter(FETExponentialBias())
fitter.fit(data, **MCMC_CONFIG)
fitter.save(SAVE_DIR / 'FET_Exp_Bias_fit.pkl')

# Also save data alongside the fit
data.to_pickle(SAVE_DIR / 'original_data.pkl')

print(f"\nDone. Fit saved to: {SAVE_DIR / 'FET_Exp_Bias_fit.pkl'}")
