"""
Phenotype × metacognition profile analysis.

For each behavioral phenotype (defined by median splits on P(heavy) and
mean_vigor), characterize the full parametric + metacognitive + outcome
profile. Then test whether metacognitive variability WITHIN phenotype
predicts adaptive performance beyond phenotype membership.

This directly addresses the question: do the energy-safety tradeoff
phenotypes have distinct metacognitive signatures, AND does variability
in metacognition within a phenotype affect outcomes?

Outputs:
  results/stats/clinical/phenotype_metacog_profile.csv      — group means + SDs
  results/stats/clinical/phenotype_metacog_outcomes.csv    — phenotype × metacog → outcomes
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


def build_pooled():
    exp_data, conf_data = load_both()
    exp_master = exp_data["master"].reset_index().rename(columns={"index": "subj"}).copy()
    conf_master = conf_data["master"].reset_index().rename(columns={"index": "subj"}).copy()
    exp_master["sample"] = "exploratory"
    conf_master["sample"] = "confirmatory"
    pooled = pd.concat([exp_master, conf_master], ignore_index=True)
    pooled["omega_z"] = zscore(np.log(pooled["omega"]).values, nan_policy="omit")
    pooled["kappa_z"] = zscore(np.log(pooled["kappa"]).values, nan_policy="omit")
    return pooled


def define_phenotypes(pooled: pd.DataFrame) -> pd.DataFrame:
    """Median split on P(heavy) and mean_vigor — defines HH/HL/LH/LL."""
    med_choice = pooled["p_heavy"].median()
    med_vigor = pooled["mean_vigor"].median()
    pooled["high_choice"] = (pooled["p_heavy"] > med_choice).astype(int)
    pooled["high_vigor"] = (pooled["mean_vigor"] > med_vigor).astype(int)
    def lbl(r):
        if r["high_choice"] == 1 and r["high_vigor"] == 1:
            return "HH"
        if r["high_choice"] == 1 and r["high_vigor"] == 0:
            return "HL"
        if r["high_choice"] == 0 and r["high_vigor"] == 1:
            return "LH"
        return "LL"
    pooled["phenotype"] = pooled.apply(lbl, axis=1)
    print(f"Phenotype N:")
    print(pooled["phenotype"].value_counts())
    print(f"Median P(heavy) = {med_choice:.3f}; median mean_vigor = {med_vigor:.3f}")
    return pooled


def phenotype_profile(pooled: pd.DataFrame) -> pd.DataFrame:
    """Mean + SD per phenotype across parametric, metacognitive, outcome variables."""
    vars_to_profile = [
        "p_heavy", "mean_vigor", "omega", "kappa", "omega_z", "kappa_z",
        "mean_confidence", "mean_anxiety", "anx_calibration", "anx_slope",
        "earnings", "escape_rate", "pct_opt",
    ]
    rows = []
    for phen in ["HH", "HL", "LH", "LL"]:
        sub = pooled[pooled["phenotype"] == phen]
        rec = {"phenotype": phen, "N": len(sub)}
        for v in vars_to_profile:
            if v in sub.columns:
                rec[f"{v}_mean"] = sub[v].mean()
                rec[f"{v}_sd"] = sub[v].std()
        rows.append(rec)
    return pd.DataFrame(rows)


def anova_across_phenotypes(pooled: pd.DataFrame, var: str):
    """One-way ANOVA on var across the four phenotypes."""
    if var not in pooled.columns:
        return None
    groups = [pooled[pooled["phenotype"] == p][var].dropna().values for p in ["HH", "HL", "LH", "LL"]]
    if any(len(g) < 5 for g in groups):
        return None
    f, p = f_oneway(*groups)
    return {"var": var, "F": f, "p": p,
            "HH_mean": groups[0].mean(), "HL_mean": groups[1].mean(),
            "LH_mean": groups[2].mean(), "LL_mean": groups[3].mean()}


def fit_phenotype_metacog_outcome(pooled: pd.DataFrame, outcome: str) -> dict:
    """Test: outcome ~ phenotype + confidence_z + anx_calibration_z + phenotype:metacog interactions"""
    df = pooled[[outcome, "phenotype", "mean_confidence", "anx_calibration"]].dropna().copy()
    if len(df) < 50:
        return None
    df[f"{outcome}_z"] = zscore(df[outcome].values, nan_policy="omit")
    df["confidence_z"] = zscore(df["mean_confidence"].values, nan_policy="omit")
    df["anx_cal_z"] = zscore(df["anx_calibration"].values, nan_policy="omit")
    df["phenotype_factor"] = pd.Categorical(df["phenotype"], categories=["HH", "HL", "LH", "LL"])

    formula = f"{outcome}_z ~ phenotype_factor + confidence_z + anx_cal_z"
    print(f"\nMain-effects: {formula}")
    m_main = bmb.Model(formula, data=df)
    s_main = az.summary(m_main.fit(**BKW), hdi_prob=0.95)
    print(s_main[["mean", "hdi_2.5%", "hdi_97.5%"]].round(3).to_string())

    formula_int = f"{outcome}_z ~ phenotype_factor * (confidence_z + anx_cal_z)"
    print(f"\nWith interactions: {formula_int}")
    m_int = bmb.Model(formula_int, data=df)
    s_int = az.summary(m_int.fit(**BKW), hdi_prob=0.95)
    print(s_int[["mean", "hdi_2.5%", "hdi_97.5%"]].round(3).to_string())

    return {"main": s_main, "int": s_int}


def main():
    print("=" * 70)
    print("PHENOTYPE × METACOGNITION PROFILE ANALYSIS")
    print("=" * 70)
    pooled = build_pooled()
    print(f"\nPooled N: {len(pooled)}")
    pooled = define_phenotypes(pooled)

    # Step 1: phenotype profile
    print("\n" + "=" * 70)
    print("STEP 1 — Phenotype profile (mean ± sd per group)")
    print("=" * 70)
    profile = phenotype_profile(pooled)
    print(profile.round(3).to_string(index=False))
    profile.to_csv(REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_profile.csv", index=False)

    # Step 2: ANOVA across phenotypes
    print("\n" + "=" * 70)
    print("STEP 2 — One-way ANOVA across phenotypes")
    print("=" * 70)
    anova_rows = []
    for v in ["omega", "kappa", "mean_confidence", "mean_anxiety", "anx_calibration",
              "earnings", "escape_rate", "pct_opt"]:
        rec = anova_across_phenotypes(pooled, v)
        if rec:
            print(f"  {v:20s} F = {rec['F']:.2f}, p = {rec['p']:.3g}  "
                  f"(HH={rec['HH_mean']:+.3f}, HL={rec['HL_mean']:+.3f}, "
                  f"LH={rec['LH_mean']:+.3f}, LL={rec['LL_mean']:+.3f})")
            anova_rows.append(rec)
    pd.DataFrame(anova_rows).to_csv(REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_anova.csv", index=False)

    # Step 3: variability within phenotype
    print("\n" + "=" * 70)
    print("STEP 3 — Metacognitive variability WITHIN phenotypes (SDs)")
    print("=" * 70)
    sd_table = pooled.groupby("phenotype")[["mean_confidence", "mean_anxiety", "anx_calibration"]].std()
    print(sd_table.round(3).to_string())

    # Step 4: phenotype × metacog → outcomes
    print("\n" + "=" * 70)
    print("STEP 4 — Does metacognitive variability within phenotype predict outcomes?")
    print("=" * 70)
    for outcome in ["earnings", "pct_opt", "escape_rate"]:
        print(f"\n*** Outcome: {outcome} ***")
        fit_phenotype_metacog_outcome(pooled, outcome)

    print("\nAll done. CSVs saved to results/stats/clinical/")


if __name__ == "__main__":
    main()
