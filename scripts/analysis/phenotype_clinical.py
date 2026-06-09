"""
Phenotype × mental health analysis.

Earlier analyses tested whether (ω, κ) or (angle, magnitude) predict clinical
scales (results 604, 508 Analysis 1). Both mostly null. But the phenotype
categorization (HH/HL/LH/LL via median splits on P_heavy and mean_vigor) is
*behavioral*, not parameter-based, and could in principle reveal clinical
signal that parameter analyses missed — phenotypes are what a clinician
would actually observe.

For each of 17 clinical scales / subscales:
  1. One-way ANOVA across phenotypes
  2. Bayesian regression: scale_z ~ phenotype (HH = reference)

Outputs:
  results/stats/clinical/phenotype_clinical_anova.csv
  results/stats/clinical/phenotype_clinical_regression.csv
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
import bambi as bmb
import arviz as az
from scipy.stats import zscore, f_oneway

from config import BKW  # type: ignore
from load_data import load_both  # type: ignore


CLINICAL_SCALES = [
    "DASS21_Anxiety", "DASS21_Depression", "DASS21_Stress", "DASS21_Total",
    "PHQ9_Total", "OASIS_Total", "STAI_Trait", "STAI_State", "STICSA_Total",
    "AMI_Behavioural", "AMI_Social", "AMI_Emotional", "AMI_Total",
    "MFIS_Physical", "MFIS_Cognitive", "MFIS_Psychosocial", "MFIS_Total",
]


def build_pooled():
    exp_data, conf_data = load_both()
    exp_master = exp_data["master"].reset_index().rename(columns={"index": "subj"}).copy()
    conf_master = conf_data["master"].reset_index().rename(columns={"index": "subj"}).copy()
    exp_master["sample"] = "exploratory"
    conf_master["sample"] = "confirmatory"
    pooled = pd.concat([exp_master, conf_master], ignore_index=True)
    pooled["omega_z"] = zscore(np.log(pooled["omega"]).values, nan_policy="omit")
    pooled["kappa_z"] = zscore(np.log(pooled["kappa"]).values, nan_policy="omit")
    # Phenotype via median splits
    med_choice = pooled["p_heavy"].median()
    med_vigor = pooled["mean_vigor"].median()
    pooled["high_choice"] = (pooled["p_heavy"] > med_choice).astype(int)
    pooled["high_vigor"] = (pooled["mean_vigor"] > med_vigor).astype(int)
    def lbl(r):
        if r["high_choice"] == 1 and r["high_vigor"] == 1: return "HH"
        if r["high_choice"] == 1 and r["high_vigor"] == 0: return "HL"
        if r["high_choice"] == 0 and r["high_vigor"] == 1: return "LH"
        return "LL"
    pooled["phenotype"] = pooled.apply(lbl, axis=1)
    return pooled


def main():
    print("=" * 70)
    print("PHENOTYPE × MENTAL HEALTH analysis")
    print("=" * 70)
    pooled = build_pooled()
    print(f"\nPooled N: {len(pooled)}")
    print("Phenotype counts:")
    print(pooled["phenotype"].value_counts())

    # ANOVAs
    print("\n" + "=" * 70)
    print("STEP 1 — One-way ANOVA per clinical scale")
    print("=" * 70)
    anova_rows = []
    for scale in CLINICAL_SCALES:
        if scale not in pooled.columns:
            continue
        groups = [pooled[pooled["phenotype"] == p][scale].dropna().values for p in ["HH", "HL", "LH", "LL"]]
        if any(len(g) < 5 for g in groups):
            continue
        F, p = f_oneway(*groups)
        sig = "★" if p < 0.05 else " "
        sig_strong = "★★" if p < 0.01 else sig
        sig_vstrong = "★★★" if p < 0.001 else sig_strong
        print(f"  {scale:22s} F = {F:.2f}, p = {p:.4g}  HH={groups[0].mean():+.2f}, "
              f"HL={groups[1].mean():+.2f}, LH={groups[2].mean():+.2f}, LL={groups[3].mean():+.2f} {sig_vstrong}")
        anova_rows.append({
            "scale": scale, "F": F, "p": p,
            "HH_mean": groups[0].mean(), "HL_mean": groups[1].mean(),
            "LH_mean": groups[2].mean(), "LL_mean": groups[3].mean(),
            "n_HH": len(groups[0]), "n_HL": len(groups[1]),
            "n_LH": len(groups[2]), "n_LL": len(groups[3]),
        })
    anova_df = pd.DataFrame(anova_rows)
    anova_df.to_csv(REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_clinical_anova.csv", index=False)

    # Bayesian regressions — only run on scales that passed ANOVA significance
    print("\n" + "=" * 70)
    print("STEP 2 — Bayesian regressions (HH = reference) for ANOVA-significant scales")
    print("=" * 70)
    sig_scales = anova_df[anova_df["p"] < 0.05]["scale"].tolist()
    print(f"\nScales with ANOVA p < 0.05: {sig_scales}")
    print(f"(of {len(anova_df)} tested)")

    reg_rows = []
    for scale in sig_scales:
        df = pooled[[scale, "phenotype"]].dropna().copy()
        df[f"{scale}_z"] = zscore(df[scale].values, nan_policy="omit")
        df["phenotype_factor"] = pd.Categorical(df["phenotype"], categories=["HH", "HL", "LH", "LL"])
        print(f"\n  --- {scale} ---")
        m = bmb.Model(f"{scale}_z ~ phenotype_factor", data=df)
        s = az.summary(m.fit(**BKW), hdi_prob=0.95)
        for term in s.index:
            if term in ["sigma", "Intercept"]:
                continue
            beta = s.loc[term, "mean"]
            lo = s.loc[term, "hdi_2.5%"]
            hi = s.loc[term, "hdi_97.5%"]
            sig = "★" if (lo > 0) or (hi < 0) else " "
            print(f"    {term:30s} β = {beta:+.3f} [{lo:+.3f}, {hi:+.3f}] {sig}")
            reg_rows.append({
                "scale": scale, "contrast": term, "beta": beta,
                "hdi_lo": lo, "hdi_hi": hi, "sig": (lo > 0) or (hi < 0),
            })
    pd.DataFrame(reg_rows).to_csv(REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_clinical_regression.csv", index=False)

    # Summary
    print("\n" + "=" * 70)
    print("HEADLINE — phenotype × clinical scales")
    print("=" * 70)
    print(f"\nANOVA significant (p < 0.05): {len(sig_scales)} of {len(anova_df)}")
    print(f"With Bonferroni correction (α = 0.05/{len(anova_df)} = {0.05/len(anova_df):.4f}):")
    bonf_sig = anova_df[anova_df["p"] < 0.05/len(anova_df)]
    print(f"  {len(bonf_sig)} of {len(anova_df)} survive Bonferroni")
    if len(bonf_sig) > 0:
        print(bonf_sig[["scale", "F", "p", "HH_mean", "HL_mean", "LH_mean", "LL_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
