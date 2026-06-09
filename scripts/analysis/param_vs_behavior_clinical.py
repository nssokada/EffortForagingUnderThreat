"""
Does the model do useful work? Param vs. raw behavior for clinical prediction.

The Fung-style analysis found that p_heavy_shift (high-T minus low-T choice shift)
and affect intercepts predict apathy scales. But (ω, κ) alone mostly don't predict
those same scales. Question: is p_heavy_shift just a noisy proxy for the
parameters (in which case the model is doing the work, just nonlinearly), or
does behavior carry clinical signal BEYOND what the parameters can explain?

Tests:
  1. Does (ω, κ) predict p_heavy_shift? (the model parameters should drive shift)
  2. Multiple regression: AMI_Behavioural ~ ω_z + κ_z + p_heavy_shift_z + sample
     - If shift β shrinks to null when params controlled → params do the work
     - If shift β survives → behavior carries info params miss
  3. Same logic for confidence_intercept, anxiety_intercept

Tells us whether the model is the right level of description or whether direct
behavioral readouts are doing all the work.

Outputs: results/stats/clinical/param_vs_behavior_clinical.csv
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
import statsmodels.api as sm
from scipy.stats import zscore, pearsonr

from load_data import load_both  # type: ignore


def build_data():
    exp, conf = load_both()
    rows_beh = []
    rows_master = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        beh = d["vigor"][["subj", "T_round", "is_heavy", "norm_rate"]].copy()
        beh["sample"] = sample
        rows_beh.append(beh)
        m = d["master"].reset_index().rename(columns={"index": "subj"}).copy()
        m["sample"] = sample
        rows_master.append(m)
    beh = pd.concat(rows_beh, ignore_index=True)
    master = pd.concat(rows_master, ignore_index=True)

    # IMPORTANT: groupby must include sample because subj numbering overlaps across samples
    grp = beh.groupby(["subj", "sample", "T_round"]).agg(p_heavy=("is_heavy", "mean"),
                                                          vigor=("norm_rate", "mean")).reset_index()
    p_wide = grp.pivot_table(index=["subj", "sample"], columns="T_round", values="p_heavy").reset_index()
    v_wide = grp.pivot_table(index=["subj", "sample"], columns="T_round", values="vigor").reset_index()
    p_wide.columns = ["subj", "sample"] + [f"p_heavy_T{c}" for c in p_wide.columns[2:]]
    v_wide.columns = ["subj", "sample"] + [f"vigor_T{c}" for c in v_wide.columns[2:]]
    out = master.merge(p_wide, on=["subj", "sample"]).merge(v_wide, on=["subj", "sample"])
    out["p_heavy_shift"] = out["p_heavy_T0.9"] - out["p_heavy_T0.1"]
    out["vigor_shift"] = out["vigor_T0.9"] - out["vigor_T0.1"]
    out["sample_dummy"] = (out["sample"] == "confirmatory").astype(int)
    out["omega_z"] = zscore(np.log(out["omega"]).values)
    out["kappa_z"] = zscore(np.log(out["kappa"]).values)

    # Merge structure-based affect on BOTH subj AND sample (subj numbering overlaps)
    slopes = pd.read_csv(REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_slopes_subjects.csv")
    out = out.merge(slopes[["subj", "sample", "confidence_intercept", "anxiety_intercept",
                             "confidence_slope_T", "anxiety_slope_T",
                             "confidence_slope_D", "anxiety_slope_D"]], on=["subj", "sample"], how="left")
    return out


def fit(df, lhs, rhs):
    sub = df[[lhs] + rhs + ["sample_dummy"]].dropna().copy()
    if len(sub) < 30:
        return None
    sub[lhs] = zscore(sub[lhs].values)
    for r in rhs:
        sub[r] = zscore(sub[r].values)
    X = sm.add_constant(sub[rhs + ["sample_dummy"]].values)
    res = sm.OLS(sub[lhs].values, X).fit()
    return res


def main():
    df = build_data()
    print(f"N: {len(df)}")
    rows = []

    print("\n" + "=" * 78)
    print("STEP 1 — Do (ω, κ) predict p_heavy_shift and vigor_shift?")
    print("=" * 78)
    for shift in ["p_heavy_shift", "vigor_shift"]:
        res = fit(df, shift, ["omega_z", "kappa_z"])
        print(f"\n  {shift} ~ ω_z + κ_z + sample")
        print(f"     R² = {res.rsquared:.4f}")
        for i, name in enumerate(["ω_z", "κ_z"]):
            sig = "★" if res.pvalues[i+1] < 0.05 else " "
            print(f"     {name:5s}: β={res.params[i+1]:+.3f}  SE={res.bse[i+1]:.3f}  t={res.tvalues[i+1]:+.2f}  p={res.pvalues[i+1]:.4g} {sig}")
        # Also pearson r
        sub = df[[shift, "omega_z", "kappa_z"]].dropna()
        for p, name in [("omega_z", "ω"), ("kappa_z", "κ")]:
            r, pv = pearsonr(sub[shift], sub[p])
            print(f"     pearson r({name}, {shift}) = {r:+.3f} (p = {pv:.4g})")

    print("\n" + "=" * 78)
    print("STEP 2 — When (ω, κ) AND behavior are both predictors of AMI_Behavioural,")
    print("         which carries the signal?")
    print("=" * 78)
    targets = ["AMI_Behavioural", "AMI_Total", "DASS21_Anxiety", "DASS21_Depression"]
    for target in targets:
        if target not in df.columns:
            continue
        print(f"\n  → {target}")
        # Model 1: just params
        res1 = fit(df, target, ["omega_z", "kappa_z"])
        # Model 2: just behavior shift
        res2 = fit(df, target, ["p_heavy_shift"])
        # Model 3: just affect intercept
        res3 = fit(df, target, ["confidence_intercept", "anxiety_intercept"])
        # Model 4: all together
        res4 = fit(df, target, ["omega_z", "kappa_z", "p_heavy_shift",
                                 "confidence_intercept", "anxiety_intercept"])
        # Print clean comparison
        for label, res, terms in [
            ("params only", res1, ["omega_z", "kappa_z"]),
            ("shift only", res2, ["p_heavy_shift"]),
            ("affect intercepts only", res3, ["confidence_intercept", "anxiety_intercept"]),
            ("all together", res4, ["omega_z", "kappa_z", "p_heavy_shift",
                                     "confidence_intercept", "anxiety_intercept"]),
        ]:
            if res is None: continue
            print(f"     [{label:24s}]  R² = {res.rsquared:.4f}")
            for i, name in enumerate(terms):
                sig = "★" if res.pvalues[i+1] < 0.05 else " "
                print(f"        {name:22s}: β={res.params[i+1]:+.3f}  p={res.pvalues[i+1]:.4g} {sig}")
            rows.append(dict(model=label, outcome=target, rsq=res.rsquared,
                             betas={n: float(res.params[i+1]) for i, n in enumerate(terms)},
                             ps={n: float(res.pvalues[i+1]) for i, n in enumerate(terms)}))

    pd.DataFrame(rows).to_csv(REPO_ROOT / "results" / "stats" / "clinical" / "param_vs_behavior_clinical.csv", index=False)


if __name__ == "__main__":
    main()
