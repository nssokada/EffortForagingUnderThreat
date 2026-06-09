"""
Does affect reshape behavior? Within-subject test on probe trials.

On each probe trial we have a forced-choice + a single self-report rating
(anxiety OR confidence — they alternate, not both). On the same trial we
also have the pressing rate (vigor). This lets us ask: does within-subject
affect variation predict within-subject vigor variation, beyond what task
conditions (T, D) and parameters (ω, κ) explain?

Fit separately for anxiety-probe and confidence-probe trials:

  M_base:   vigor_z ~ T_z + D_z + omega_z + kappa_z         + (1 | subj)
  M_affect: vigor_z ~ T_z + D_z + omega_z + kappa_z + aff_z + (1 | subj)
  M_int:    vigor_z ~ T_z*aff_z + D_z + omega_z + kappa_z   + (1 | subj)

Compare M_affect vs M_base: does affect add predictive value above task + params?
Compare M_int vs M_affect:  does affect modulate the slope of vigor on T?

Outputs:
  results/stats/affect_analysis/affect_reshapes_behavior.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)
NB_DIR = REPO_ROOT / "notebooks" / "analysis"
sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import zscore

from load_data import load_both  # type: ignore


SAMPLES = {
    "exploratory": {
        "stage5": "data/exploratory_350/processed/stage5_filtered_data_20260403_133425",
        "m4_params": "results/stats/joint_optimal/exploratory/mcmc_m4_params.csv",
    },
    "confirmatory": {
        "stage5": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413",
        "m4_params": "results/stats/joint_optimal/confirmatory/mcmc_m4_params.csv",
    },
}


def build_probe_trial_data():
    """Build pooled long-form probe-trial table: one row per probe trial,
    with the question label, the rating, vigor, T, D, and per-subject (ω, κ).
    """
    exp_data, conf_data = load_both()
    all_data = {"exploratory": exp_data, "confirmatory": conf_data}

    rows = []
    for sample_name, paths in SAMPLES.items():
        feelings = pd.read_csv(Path(paths["stage5"]) / "feelings.csv", low_memory=False)
        m4 = pd.read_csv(paths["m4_params"])

        # Each row in feelings.csv is already one probe trial with one question.
        # Keep the rating (response) and the question label.
        f = feelings[["subj", "trialNumber", "threat", "distance", "questionLabel", "response"]].copy()
        f = f.dropna(subset=["response"])

        d = all_data[sample_name]
        if "vigor" not in d:
            print(f"  WARNING: no trial-level vigor data for {sample_name}")
            continue
        vigor_df = d["vigor"][["subj", "trial", "norm_rate"]].copy()
        vigor_df = vigor_df.rename(columns={"trial": "trialNumber", "norm_rate": "vigor"})

        probe_data = f.merge(vigor_df, on=["subj", "trialNumber"], how="inner")
        probe_data = probe_data.merge(m4[["subj", "omega", "kappa"]], on="subj", how="inner")
        probe_data["sample"] = sample_name
        rows.append(probe_data)

    pooled = pd.concat(rows, ignore_index=True)
    pooled = pooled.dropna(subset=["response", "vigor"])

    pooled["log_omega"] = np.log(pooled["omega"])
    pooled["log_kappa"] = np.log(pooled["kappa"])
    pooled["T_z"] = zscore(pooled["threat"].astype(float).values, nan_policy="omit")
    pooled["D_z"] = zscore(pooled["distance"].astype(float).values, nan_policy="omit")
    pooled["omega_z"] = zscore(pooled["log_omega"].values, nan_policy="omit")
    pooled["kappa_z"] = zscore(pooled["log_kappa"].values, nan_policy="omit")
    pooled["vigor_z"] = zscore(pooled["vigor"].values, nan_policy="omit")
    # Within-question z-score so anxiety and confidence each go in on the same scale.
    pooled["aff_z"] = np.nan
    for q in pooled["questionLabel"].unique():
        m = pooled["questionLabel"] == q
        pooled.loc[m, "aff_z"] = zscore(pooled.loc[m, "response"].values, nan_policy="omit")
    return pooled


def fit_and_compare(pooled, formulas):
    """Fit a series of nested models and print fixed effects + log-lik comparisons."""
    fits = {}
    for label, formula in formulas.items():
        print(f"\n--- {label} ---")
        print(f"  formula: {formula}")
        try:
            # Random intercept by subject for simplicity (random slope on T_z adds complexity)
            model = smf.mixedlm(formula, data=pooled, groups=pooled["subj"]).fit(reml=False, method="lbfgs")
            print(f"  converged: {model.converged}")
            print(f"  logL: {model.llf:.2f}, AIC: {model.aic:.2f}")
            print("  fixed effects (β [SE, z, p]):")
            for term in model.fe_params.index:
                if term == "Intercept": continue
                b = model.fe_params[term]
                se = model.bse_fe[term]
                z = model.tvalues[term]
                p = model.pvalues[term]
                sig = " ★" if p < 0.05 else ""
                sig = " ★★" if p < 0.01 else sig
                sig = " ★★★" if p < 0.001 else sig
                print(f"    {term:30s} {b:+.4f}  [SE {se:.4f}, z = {z:+.2f}, p = {p:.4g}]{sig}")
            fits[label] = model
        except Exception as e:
            print(f"  ERROR: {e}")
            fits[label] = None
    return fits


def main():
    print("=" * 70)
    print("DOES AFFECT RESHAPE BEHAVIOR? Within-subject probe-trial test")
    print("=" * 70)
    pooled = build_probe_trial_data()
    print(f"\nProbe trials available: {len(pooled)}")
    print(f"  anxiety probes: {(pooled['questionLabel'] == 'anxiety').sum()}")
    print(f"  confidence probes: {(pooled['questionLabel'] == 'confidence').sum()}")
    print(f"Unique subjects: {pooled['subj'].nunique()}")

    formulas = {
        "M_base":   "vigor_z ~ T_z + D_z + omega_z + kappa_z",
        "M_affect": "vigor_z ~ T_z + D_z + omega_z + kappa_z + aff_z",
        "M_int":    "vigor_z ~ T_z + D_z + omega_z + kappa_z + aff_z + T_z:aff_z + D_z:aff_z",
    }

    all_rows = []
    fits_by_q = {}
    for q in ["anxiety", "confidence"]:
        print("\n" + "#" * 70)
        print(f"# QUESTION = {q.upper()}")
        print("#" * 70)
        sub = pooled[pooled["questionLabel"] == q].copy()
        print(f"  trials: {len(sub)}, subjects: {sub['subj'].nunique()}")
        fits = fit_and_compare(sub, formulas)
        fits_by_q[q] = fits

        print(f"\n  MODEL COMPARISON [{q}]")
        for label, model in fits.items():
            if model is not None:
                print(f"    {label:10s}  AIC = {model.aic:.2f}   logL = {model.llf:.2f}")
        if fits.get("M_base") and fits.get("M_affect"):
            d_aic = fits["M_affect"].aic - fits["M_base"].aic
            d_ll = fits["M_affect"].llf - fits["M_base"].llf
            print(f"    Δ(M_affect − M_base): ΔAIC = {d_aic:+.2f}, ΔlogL = {d_ll:+.2f}  "
                  f"→ {q} {'IMPROVES' if d_aic < 0 else 'HURTS'} fit by {abs(d_aic):.1f} AIC")
        if fits.get("M_affect") and fits.get("M_int"):
            d_aic = fits["M_int"].aic - fits["M_affect"].aic
            d_ll = fits["M_int"].llf - fits["M_affect"].llf
            print(f"    Δ(M_int − M_affect):  ΔAIC = {d_aic:+.2f}, ΔlogL = {d_ll:+.2f}  "
                  f"→ {q} interactions {'help' if d_aic < 0 else 'do not help'}")

        for label, model in fits.items():
            if model is None: continue
            for term in model.fe_params.index:
                all_rows.append({
                    "question": q, "model": label, "term": term,
                    "beta": float(model.fe_params[term]),
                    "se": float(model.bse_fe[term]),
                    "z": float(model.tvalues[term]),
                    "p": float(model.pvalues[term]),
                    "logL": float(model.llf), "AIC": float(model.aic),
                })

    out_dir = REPO_ROOT / "results" / "stats" / "affect_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_dir / "affect_reshapes_behavior.csv", index=False)
    print(f"\nSaved: {out_dir / 'affect_reshapes_behavior.csv'}")


if __name__ == "__main__":
    main()
