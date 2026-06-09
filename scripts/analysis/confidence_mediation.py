"""
Does confidence (or anxiety) mediate (ω, κ) → behavioral outcomes?

Multivariate mediation. Both ω and κ entered simultaneously as exposures so
each parameter's mediation is partialed against the other. Pooled across
samples with a sample dummy. Bootstrap CI on the indirect effect (5000 iter,
resampling subjects with replacement).

For each behavioral outcome Y in {earnings, pct_opt, p_heavy, mean_vigor,
escape_rate} and each mediator M in {mean_confidence, mean_anxiety}:

  a-path:  M  = α + a_ω·ω_z + a_κ·κ_z + δ·sample
  b-path:  Y  = α + b·M + c'_ω·ω_z + c'_κ·κ_z + δ·sample
  total:   Y  = α + c_ω·ω_z + c_κ·κ_z + δ·sample

  indirect_ω = a_ω · b   (CI via bootstrap)
  indirect_κ = a_κ · b   (CI via bootstrap)

Outputs:
  results/stats/affect_analysis/confidence_mediation.csv
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
from scipy.stats import zscore

from load_data import load_both  # type: ignore


OUTCOMES = ["earnings", "pct_opt", "p_heavy", "mean_vigor", "escape_rate"]
MEDIATORS = ["mean_confidence", "mean_anxiety"]
N_BOOT = 5000
RNG = np.random.default_rng(20260605)


def build_pooled():
    exp, conf = load_both()
    em = exp["master"].reset_index().rename(columns={"index": "subj"}).copy()
    cm = conf["master"].reset_index().rename(columns={"index": "subj"}).copy()
    em["sample"] = 0  # exploratory
    cm["sample"] = 1  # confirmatory
    df = pd.concat([em, cm], ignore_index=True)
    df["log_omega"] = np.log(df["omega"])
    df["log_kappa"] = np.log(df["kappa"])
    # Pooled z-scoring (we want one common scale across both samples)
    df["omega_z"] = zscore(df["log_omega"].values, nan_policy="omit")
    df["kappa_z"] = zscore(df["log_kappa"].values, nan_policy="omit")
    return df


def fit_paths(df, mediator, outcome):
    """Return (a_om, a_kp, b, c_prime_om, c_prime_kp, c_om, c_kp) point estimates."""
    Xa = sm.add_constant(df[["omega_z", "kappa_z", "sample"]].values)
    Ma = sm.OLS(df[mediator].values, Xa).fit()
    a_om, a_kp = Ma.params[1], Ma.params[2]

    Xb = sm.add_constant(df[[mediator, "omega_z", "kappa_z", "sample"]].values)
    Mb = sm.OLS(df[outcome].values, Xb).fit()
    b = Mb.params[1]
    cp_om, cp_kp = Mb.params[2], Mb.params[3]

    Xc = sm.add_constant(df[["omega_z", "kappa_z", "sample"]].values)
    Mc = sm.OLS(df[outcome].values, Xc).fit()
    c_om, c_kp = Mc.params[1], Mc.params[2]

    return dict(a_om=a_om, a_kp=a_kp, b=b, cp_om=cp_om, cp_kp=cp_kp, c_om=c_om, c_kp=c_kp)


def bootstrap_indirect(df, mediator, outcome, n_boot=N_BOOT):
    """Bootstrap subject-resamples; return arrays of indirect_om and indirect_kp."""
    n = len(df)
    ind_om = np.empty(n_boot)
    ind_kp = np.empty(n_boot)
    idx_all = np.arange(n)
    for i in range(n_boot):
        idx = RNG.choice(idx_all, size=n, replace=True)
        sub = df.iloc[idx]
        try:
            p = fit_paths(sub, mediator, outcome)
            ind_om[i] = p["a_om"] * p["b"]
            ind_kp[i] = p["a_kp"] * p["b"]
        except Exception:
            ind_om[i] = np.nan
            ind_kp[i] = np.nan
    return ind_om, ind_kp


def main():
    print("=" * 78)
    print("MEDIATION: (ω, κ) → behavioral outcomes  via  {confidence, anxiety}")
    print(f"Bootstrap iterations: {N_BOOT}")
    print("=" * 78)
    df = build_pooled()
    df = df.dropna(subset=["omega_z", "kappa_z"] + MEDIATORS + OUTCOMES).copy()
    # Z-score outcomes + mediators (within pooled) so β's are on a common scale
    for c in MEDIATORS + OUTCOMES:
        df[c] = zscore(df[c].values, nan_policy="omit")
    print(f"\nPooled N (complete cases): {len(df)}")
    print(f"Exploratory: {(df['sample']==0).sum()}  Confirmatory: {(df['sample']==1).sum()}")

    rows = []
    for mediator in MEDIATORS:
        print(f"\n{'#' * 78}\n# Mediator: {mediator}\n{'#' * 78}")
        for outcome in OUTCOMES:
            pt = fit_paths(df, mediator, outcome)
            ind_om_boot, ind_kp_boot = bootstrap_indirect(df, mediator, outcome)
            ci_om = np.nanpercentile(ind_om_boot, [2.5, 97.5])
            ci_kp = np.nanpercentile(ind_kp_boot, [2.5, 97.5])
            # Bootstrap p (two-tailed, proportion of resamples crossing zero)
            p_om = 2 * min((ind_om_boot < 0).mean(), (ind_om_boot > 0).mean())
            p_kp = 2 * min((ind_kp_boot < 0).mean(), (ind_kp_boot > 0).mean())
            ind_om = pt["a_om"] * pt["b"]
            ind_kp = pt["a_kp"] * pt["b"]
            prop_om = ind_om / pt["c_om"] if pt["c_om"] != 0 else np.nan
            prop_kp = ind_kp / pt["c_kp"] if pt["c_kp"] != 0 else np.nan
            sig_om = " ★" if (ci_om[0] > 0) or (ci_om[1] < 0) else ""
            sig_kp = " ★" if (ci_kp[0] > 0) or (ci_kp[1] < 0) else ""

            print(f"\n→ outcome = {outcome}")
            print(f"   a_ω = {pt['a_om']:+.3f}   a_κ = {pt['a_kp']:+.3f}   b = {pt['b']:+.3f}")
            print(f"   c (total): ω {pt['c_om']:+.3f}, κ {pt['c_kp']:+.3f}   "
                  f"c' (direct): ω {pt['cp_om']:+.3f}, κ {pt['cp_kp']:+.3f}")
            print(f"   indirect_ω = {ind_om:+.4f} [95% CI {ci_om[0]:+.4f}, {ci_om[1]:+.4f}]  "
                  f"p_boot = {p_om:.4f}  prop_med = {prop_om:+.2%}{sig_om}")
            print(f"   indirect_κ = {ind_kp:+.4f} [95% CI {ci_kp[0]:+.4f}, {ci_kp[1]:+.4f}]  "
                  f"p_boot = {p_kp:.4f}  prop_med = {prop_kp:+.2%}{sig_kp}")

            rows.append({
                "mediator": mediator, "outcome": outcome,
                "a_omega": pt["a_om"], "a_kappa": pt["a_kp"], "b": pt["b"],
                "c_omega": pt["c_om"], "c_kappa": pt["c_kp"],
                "cprime_omega": pt["cp_om"], "cprime_kappa": pt["cp_kp"],
                "indirect_omega": ind_om, "indirect_omega_lo": ci_om[0], "indirect_omega_hi": ci_om[1], "p_omega": p_om,
                "indirect_kappa": ind_kp, "indirect_kappa_lo": ci_kp[0], "indirect_kappa_hi": ci_kp[1], "p_kappa": p_kp,
                "prop_mediated_omega": prop_om, "prop_mediated_kappa": prop_kp,
            })

    out = REPO_ROOT / "results" / "stats" / "affect_analysis" / "confidence_mediation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
