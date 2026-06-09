"""
Does (ω, κ) modulate the slope of anxiety/confidence on threat and distance?

Tests whether per-subject parameters MODERATE the within-subject affective
reactivity to task dimensions. The natural intuition: subjects with high ω
should show *steeper* anxiety-to-threat slopes (they're more threat-sensitive),
not just higher mean anxiety. Same logic for κ on confidence.

This is the reactivity test that result_503 / 507 didn't run. Those tested
main effects of (ω, κ) on average affect. This tests parameter × task
interactions on trial-level affect.

Model (per channel: anxiety, confidence):
  response ~ T_z + D_z + omega_z + kappa_z
           + T_z:omega_z + T_z:kappa_z
           + D_z:omega_z + D_z:kappa_z
           + (1 + T_z | subj)

Trial-level probe data, pooled across samples (N ≈ 10,288 probe trials,
N ≈ 571 subjects).
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


def build_pooled():
    rows = []
    for sample, paths in SAMPLES.items():
        feelings = pd.read_csv(Path(paths["stage5"]) / "feelings.csv", low_memory=False)
        m4 = pd.read_csv(paths["m4_params"])
        df = feelings.merge(m4, on="subj", how="inner")
        df["sample"] = sample
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)
    pooled = pooled.dropna(subset=["response"]).copy()
    pooled["T"] = pooled["threat"].astype(float)
    pooled["D"] = pooled["distance"].astype(int).astype(float)
    pooled["log_omega"] = np.log(pooled["omega"])
    pooled["log_kappa"] = np.log(pooled["kappa"])
    # POOLED z-scoring on trial-level data (each subject appears multiple times so subject-level params are repeated)
    pooled["omega_z"] = zscore(pooled["log_omega"].values, nan_policy="omit")
    pooled["kappa_z"] = zscore(pooled["log_kappa"].values, nan_policy="omit")
    pooled["T_z"] = zscore(pooled["T"].values, nan_policy="omit")
    pooled["D_z"] = zscore(pooled["D"].values, nan_policy="omit")
    return pooled


def fit_channel(pooled: pd.DataFrame, channel: str):
    sub = pooled[pooled["questionLabel"] == channel].copy()
    n_obs = len(sub)
    n_subj = sub["subj"].nunique()
    print(f"\n--- {channel} | N obs = {n_obs}, N subj = {n_subj} ---")

    formula = (
        "response ~ T_z + D_z + omega_z + kappa_z + "
        "T_z:omega_z + T_z:kappa_z + D_z:omega_z + D_z:kappa_z"
    )
    print(f"  formula: {formula}")
    print(f"  random: (1 + T_z | subj)")

    # statsmodels MixedLM with random intercept and random slope on T_z
    model = smf.mixedlm(
        formula,
        data=sub,
        groups=sub["subj"],
        re_formula="~T_z",
    ).fit(reml=False, method="lbfgs")

    print(f"\n  Convergence: {model.converged}")
    print(f"  LogLik: {model.llf:.2f}, AIC: {model.aic:.2f}")
    print("\n  Fixed effects:")
    rows = []
    for term in ["Intercept", "T_z", "D_z", "omega_z", "kappa_z",
                 "T_z:omega_z", "T_z:kappa_z", "D_z:omega_z", "D_z:kappa_z"]:
        if term not in model.fe_params.index:
            continue
        beta = float(model.fe_params[term])
        se = float(model.bse_fe[term])
        z = float(model.tvalues[term])
        p = float(model.pvalues[term])
        sig = "★" if p < 0.05 else " "
        sigstrong = "★★" if p < 0.01 else sig
        sigvstrong = "★★★" if p < 0.001 else sigstrong
        print(f"    {term:18s} β = {beta:+.4f} (SE {se:.4f}, z = {z:+.2f}, p = {p:.4g}) {sigvstrong}")
        rows.append({
            "channel": channel, "term": term, "n_obs": n_obs, "n_subj": n_subj,
            "beta": beta, "se": se, "z": z, "p": p,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("AFFECT REACTIVITY × (ω, κ) interactions")
    print("=" * 70)
    pooled = build_pooled()
    print(f"\nPooled probe trials: {len(pooled)}")
    print(f"Subjects: {pooled['subj'].nunique()}")
    print(f"Channels: {pooled['questionLabel'].unique()}")

    all_rows = []
    for channel in ("anxiety", "confidence"):
        df = fit_channel(pooled, channel)
        all_rows.append(df)

    out = pd.concat(all_rows, ignore_index=True)
    out_path = REPO_ROOT / "results" / "stats" / "affect_analysis" / "affect_reactivity_interactions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Headline: the four interaction terms per channel
    print("\n" + "=" * 70)
    print("HEADLINE — does (ω, κ) modulate affective REACTIVITY to T and D?")
    print("=" * 70)
    interaction_terms = ["T_z:omega_z", "T_z:kappa_z", "D_z:omega_z", "D_z:kappa_z"]
    pivot = out[out["term"].isin(interaction_terms)].pivot_table(
        index="term", columns="channel", values="beta"
    )
    pivot_p = out[out["term"].isin(interaction_terms)].pivot_table(
        index="term", columns="channel", values="p"
    )
    print("\nβ values:")
    print(pivot.round(4).to_string())
    print("\np values:")
    print(pivot_p.round(4).to_string())

    sig = out[(out["term"].isin(interaction_terms)) & (out["p"] < 0.05)]
    print(f"\nSignificant interactions: {len(sig)} of {len(interaction_terms) * 2}")
    if len(sig) > 0:
        print(sig[["channel", "term", "beta", "se", "z", "p"]].to_string(index=False))


if __name__ == "__main__":
    main()
