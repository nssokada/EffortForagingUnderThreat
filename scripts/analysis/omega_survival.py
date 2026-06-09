"""
Does high ω predict better survival? Normative validation.

ω is the model parameter that weights capture cost in the value function.
If it has a normative interpretation as "internal price of capture", then
subjects with higher ω should:
  1. Be captured less often (lower per-trial capture rate)
  2. Escape more often when actually attacked (higher P(escape | attack))
  3. The relationship should hold beyond κ

Tests:
  1. escape_rate ~ ω + κ + sample, pooled
  2. Same in each sample separately
  3. Per-threat-level escape rate: does ω matter more at high T?
  4. Total capture rate (P(captured) across all trials)

Outputs: results/stats/joint_optimal/omega_survival.csv
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
    rows_master = []
    rows_beh = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d["master"].reset_index().rename(columns={"index": "subj"}).copy()
        m["sample"] = sample
        rows_master.append(m)
        b = d["beh"][["subj", "isAttackTrial", "trialEndState", "threat"]].copy()
        b["sample"] = sample
        rows_beh.append(b)
    master = pd.concat(rows_master, ignore_index=True)
    beh = pd.concat(rows_beh, ignore_index=True)
    master["omega_z"] = zscore(np.log(master["omega"]).values)
    master["kappa_z"] = zscore(np.log(master["kappa"]).values)
    master["sample_dummy"] = (master["sample"] == "confirmatory").astype(int)
    return master, beh


def fit(df, lhs, rhs, sample_dummy=True):
    sub = df[[lhs] + rhs + (["sample_dummy"] if sample_dummy else [])].dropna().copy()
    sub[lhs + "_z"] = zscore(sub[lhs].values, nan_policy="omit")
    for r in rhs:
        sub[r] = zscore(sub[r].values, nan_policy="omit")
    cols = rhs + (["sample_dummy"] if sample_dummy else [])
    X = sm.add_constant(sub[cols].values)
    res = sm.OLS(sub[lhs + "_z"].values, X).fit()
    return res


def main():
    master, beh = build_data()
    print(f"N = {len(master)}    Subjects: {master['subj'].nunique()}")
    print(f"  exp: {(master['sample']=='exploratory').sum()}    conf: {(master['sample']=='confirmatory').sum()}")
    print(f"  Mean escape_rate: {master['escape_rate'].mean():.3f}    SD: {master['escape_rate'].std():.3f}")
    print(f"  Mean earnings: {master['earnings'].mean():.2f}    SD: {master['earnings'].std():.2f}")

    print("\n" + "=" * 78)
    print("1. POOLED: escape_rate ~ ω + κ + sample")
    print("=" * 78)
    res = fit(master, "escape_rate", ["omega_z", "kappa_z"])
    print(f"   R² = {res.rsquared:.4f}")
    print(f"   ω_z   β = {res.params[1]:+.3f}  SE {res.bse[1]:.3f}  t = {res.tvalues[1]:+.2f}  p = {res.pvalues[1]:.4g}")
    print(f"   κ_z   β = {res.params[2]:+.3f}  SE {res.bse[2]:.3f}  t = {res.tvalues[2]:+.2f}  p = {res.pvalues[2]:.4g}")

    # Marginal pearson
    r_ω, p_ω = pearsonr(master["omega_z"], master["escape_rate"])
    r_κ, p_κ = pearsonr(master["kappa_z"], master["escape_rate"])
    print(f"\n   marginal r(ω, escape) = {r_ω:+.3f} (p = {p_ω:.4g})")
    print(f"   marginal r(κ, escape) = {r_κ:+.3f} (p = {p_κ:.4g})")

    print("\n" + "=" * 78)
    print("2. WITHIN-SAMPLE replication: escape_rate ~ ω + κ")
    print("=" * 78)
    for sample in ["exploratory", "confirmatory"]:
        sub = master[master["sample"] == sample].copy()
        res = fit(sub, "escape_rate", ["omega_z", "kappa_z"], sample_dummy=False)
        print(f"\n   --- {sample} (N={len(sub)}) ---")
        print(f"   R² = {res.rsquared:.4f}")
        print(f"   ω_z   β = {res.params[1]:+.3f}  SE {res.bse[1]:.3f}  t = {res.tvalues[1]:+.2f}  p = {res.pvalues[1]:.4g}")
        print(f"   κ_z   β = {res.params[2]:+.3f}  SE {res.bse[2]:.3f}  t = {res.tvalues[2]:+.2f}  p = {res.pvalues[2]:.4g}")

    print("\n" + "=" * 78)
    print("3. PER-THREAT-LEVEL escape rate × ω")
    print("=" * 78)
    att = beh[beh["isAttackTrial"] == 1].copy()
    att["escaped"] = (att["trialEndState"] == "escaped").astype(int)
    per_subj_T = att.groupby(["subj", "sample", "threat"])["escaped"].mean().reset_index()
    wide = per_subj_T.pivot_table(index=["subj", "sample"], columns="threat", values="escaped").reset_index()
    wide.columns = ["subj", "sample"] + [f"escape_T{c}" for c in wide.columns[2:]]
    m2 = master.merge(wide, on=["subj", "sample"], how="left")
    for col in [c for c in m2.columns if c.startswith("escape_T")]:
        sub = m2.dropna(subset=[col])
        if len(sub) < 30: continue
        res = fit(sub, col, ["omega_z", "kappa_z"])
        sig = "★" if res.pvalues[1] < 0.05 else " "
        print(f"   {col:14s}  (N={len(sub)})   ω_z β = {res.params[1]:+.3f}, p = {res.pvalues[1]:.4g} {sig}   κ_z β = {res.params[2]:+.3f}, p = {res.pvalues[2]:.4g}")

    print("\n" + "=" * 78)
    print("4. TOTAL CAPTURE RATE (per-trial, attack OR not) ~ ω + κ")
    print("=" * 78)
    cap = beh.copy()
    cap["captured"] = (cap["trialEndState"] == "captured").astype(int)
    per_subj_cap = cap.groupby(["subj", "sample"])["captured"].mean().reset_index()
    per_subj_cap.columns = ["subj", "sample", "captures_per_trial"]
    m3 = master.merge(per_subj_cap, on=["subj", "sample"])
    res = fit(m3, "captures_per_trial", ["omega_z", "kappa_z"])
    print(f"   R² = {res.rsquared:.4f}")
    print(f"   ω_z β = {res.params[1]:+.3f}  p = {res.pvalues[1]:.4g}")
    print(f"   κ_z β = {res.params[2]:+.3f}  p = {res.pvalues[2]:.4g}")
    r, p = pearsonr(m3["omega_z"], m3["captures_per_trial"])
    print(f"   marginal r(ω, captures_per_trial) = {r:+.3f}, p = {p:.4g}")
    print(f"   Mean captures/trial: {m3['captures_per_trial'].mean():.3f}    SD: {m3['captures_per_trial'].std():.3f}")

    # Save
    out_path = REPO_ROOT / "results" / "stats" / "joint_optimal" / "omega_survival.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m3[["subj", "sample", "omega", "kappa", "omega_z", "kappa_z", "escape_rate",
        "captures_per_trial", "earnings"]].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
