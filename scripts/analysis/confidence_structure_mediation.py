"""
Mediation v2: confidence/anxiety STRUCTURE, not just mean.

The mean-confidence mediation test (confidence_mediation.py) was null. But mean
collapses across the way a subject's confidence is *structured* in response
to contingencies. Here we test the structure directly using the per-subject
regression decomposition already produced for result_510:

  intercept  — baseline level of confidence/anxiety after partialing T, D
  slope_T    — within-subject reactivity to threat
  slope_D    — within-subject reactivity to distance
  cal_T      — Pearson r(threat, response) per subject

For each (mediator, outcome) and each parameter (ω, κ):
  Run multivariate mediation with both ω_z and κ_z as exposures, sample-
  controlled. Bootstrap 5000 iter on the indirect effect.

Additionally: parallel multi-mediator model that enters all three structure
measures (intercept, slope_T, slope_D) at once, so the indirect through each
is partialed against the others.

Outputs:
  results/stats/affect_analysis/confidence_structure_mediation.csv  (single-mediator)
  results/stats/affect_analysis/confidence_structure_mediation_multi.csv  (joint)
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


SUBJECTS_CSV = REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_slopes_subjects.csv"
OUT_DIR = REPO_ROOT / "results" / "stats" / "affect_analysis"

OUTCOMES = ["earnings", "pct_opt", "p_heavy", "mean_vigor", "escape_rate"]
SINGLE_MEDIATORS = [
    "confidence_intercept", "confidence_slope_T", "confidence_slope_D", "confidence_cal_T",
    "anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D", "anxiety_cal_T",
]
CONF_STRUCTURE = ["confidence_intercept", "confidence_slope_T", "confidence_slope_D"]
ANX_STRUCTURE = ["anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D"]

N_BOOT = 5000
RNG = np.random.default_rng(20260605)


def load_data():
    df = pd.read_csv(SUBJECTS_CSV)
    df["sample_dummy"] = (df["sample"] == "confirmatory").astype(int)
    df["omega_z"] = df["omega_z_pool"]
    df["kappa_z"] = df["kappa_z_pool"]
    return df


def zscore_cols(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = zscore(out[c].values, nan_policy="omit")
    return out


def fit_single(df, mediator, outcome):
    """Single-mediator mediation. Returns dict of paths."""
    Xa = sm.add_constant(df[["omega_z", "kappa_z", "sample_dummy"]].values)
    Ma = sm.OLS(df[mediator].values, Xa).fit()
    a_om, a_kp = Ma.params[1], Ma.params[2]

    Xb = sm.add_constant(df[[mediator, "omega_z", "kappa_z", "sample_dummy"]].values)
    Mb = sm.OLS(df[outcome].values, Xb).fit()
    b = Mb.params[1]
    cp_om, cp_kp = Mb.params[2], Mb.params[3]

    Xc = sm.add_constant(df[["omega_z", "kappa_z", "sample_dummy"]].values)
    Mc = sm.OLS(df[outcome].values, Xc).fit()
    c_om, c_kp = Mc.params[1], Mc.params[2]

    return dict(a_om=a_om, a_kp=a_kp, b=b, cp_om=cp_om, cp_kp=cp_kp, c_om=c_om, c_kp=c_kp)


def fit_multi(df, mediators, outcome):
    """Parallel multi-mediator model. mediators is a list of column names.

    Returns: per-mediator (a_om, a_kp, b) and overall (c, c')."""
    n_med = len(mediators)
    # a-paths: each mediator regressed on (ω, κ, sample)
    a_oms, a_kps = np.zeros(n_med), np.zeros(n_med)
    for i, m in enumerate(mediators):
        Xa = sm.add_constant(df[["omega_z", "kappa_z", "sample_dummy"]].values)
        Ma = sm.OLS(df[m].values, Xa).fit()
        a_oms[i] = Ma.params[1]
        a_kps[i] = Ma.params[2]
    # b-paths: outcome on all mediators jointly + ω, κ, sample
    Xb = sm.add_constant(df[mediators + ["omega_z", "kappa_z", "sample_dummy"]].values)
    Mb = sm.OLS(df[outcome].values, Xb).fit()
    bs = Mb.params[1:1 + n_med]
    cp_om = Mb.params[1 + n_med]
    cp_kp = Mb.params[1 + n_med + 1]
    # total
    Xc = sm.add_constant(df[["omega_z", "kappa_z", "sample_dummy"]].values)
    Mc = sm.OLS(df[outcome].values, Xc).fit()
    c_om, c_kp = Mc.params[1], Mc.params[2]
    return dict(a_oms=a_oms, a_kps=a_kps, bs=bs, cp_om=cp_om, cp_kp=cp_kp, c_om=c_om, c_kp=c_kp)


def bootstrap_single(df, mediator, outcome, n_boot=N_BOOT):
    n = len(df)
    ind_om = np.empty(n_boot)
    ind_kp = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.choice(np.arange(n), size=n, replace=True)
        sub = df.iloc[idx]
        try:
            p = fit_single(sub, mediator, outcome)
            ind_om[i] = p["a_om"] * p["b"]
            ind_kp[i] = p["a_kp"] * p["b"]
        except Exception:
            ind_om[i] = np.nan
            ind_kp[i] = np.nan
    return ind_om, ind_kp


def bootstrap_multi(df, mediators, outcome, n_boot=N_BOOT):
    n = len(df)
    n_med = len(mediators)
    ind_om = np.empty((n_boot, n_med))
    ind_kp = np.empty((n_boot, n_med))
    for i in range(n_boot):
        idx = RNG.choice(np.arange(n), size=n, replace=True)
        sub = df.iloc[idx]
        try:
            p = fit_multi(sub, mediators, outcome)
            ind_om[i, :] = p["a_oms"] * p["bs"]
            ind_kp[i, :] = p["a_kps"] * p["bs"]
        except Exception:
            ind_om[i, :] = np.nan
            ind_kp[i, :] = np.nan
    return ind_om, ind_kp


def main():
    print("=" * 78)
    print("MEDIATION v2 — confidence/anxiety STRUCTURE as mediator")
    print("=" * 78)
    df = load_data()
    df = df.dropna(subset=["omega_z", "kappa_z"] + SINGLE_MEDIATORS + OUTCOMES).copy()
    # Z-score mediators + outcomes for unit comparison
    df = zscore_cols(df, SINGLE_MEDIATORS + OUTCOMES)
    print(f"\nN: {len(df)} (exp {(df['sample']=='exploratory').sum()}, "
          f"conf {(df['sample']=='confirmatory').sum()})")

    # ── Single-mediator analysis ──────────────────────────────────────────
    print("\n" + "#" * 78)
    print("# Single mediator (one structure measure at a time)")
    print("#" * 78)
    single_rows = []
    for mediator in SINGLE_MEDIATORS:
        print(f"\n--- mediator: {mediator} ---")
        for outcome in OUTCOMES:
            pt = fit_single(df, mediator, outcome)
            ind_om_b, ind_kp_b = bootstrap_single(df, mediator, outcome)
            ci_om = np.nanpercentile(ind_om_b, [2.5, 97.5])
            ci_kp = np.nanpercentile(ind_kp_b, [2.5, 97.5])
            p_om = 2 * min((ind_om_b < 0).mean(), (ind_om_b > 0).mean())
            p_kp = 2 * min((ind_kp_b < 0).mean(), (ind_kp_b > 0).mean())
            ind_om = pt["a_om"] * pt["b"]
            ind_kp = pt["a_kp"] * pt["b"]
            sig_om = "★" if (ci_om[0] > 0) or (ci_om[1] < 0) else " "
            sig_kp = "★" if (ci_kp[0] > 0) or (ci_kp[1] < 0) else " "
            print(f"  {outcome:14s}  indirect_ω={ind_om:+.4f} [{ci_om[0]:+.4f},{ci_om[1]:+.4f}] p={p_om:.3f} {sig_om}   "
                  f"indirect_κ={ind_kp:+.4f} [{ci_kp[0]:+.4f},{ci_kp[1]:+.4f}] p={p_kp:.3f} {sig_kp}")
            single_rows.append({
                "mediator": mediator, "outcome": outcome,
                "a_omega": pt["a_om"], "a_kappa": pt["a_kp"], "b": pt["b"],
                "c_omega": pt["c_om"], "c_kappa": pt["c_kp"],
                "cprime_omega": pt["cp_om"], "cprime_kappa": pt["cp_kp"],
                "indirect_omega": ind_om, "ind_om_lo": ci_om[0], "ind_om_hi": ci_om[1], "p_omega": p_om,
                "indirect_kappa": ind_kp, "ind_kp_lo": ci_kp[0], "ind_kp_hi": ci_kp[1], "p_kappa": p_kp,
            })
    pd.DataFrame(single_rows).to_csv(OUT_DIR / "confidence_structure_mediation.csv", index=False)
    print(f"\nSaved single-mediator results: {OUT_DIR / 'confidence_structure_mediation.csv'}")

    # ── Multi-mediator analysis ───────────────────────────────────────────
    print("\n" + "#" * 78)
    print("# Parallel multi-mediator (intercept + slope_T + slope_D jointly)")
    print("#" * 78)
    multi_rows = []
    for label, mediators in [("confidence_structure", CONF_STRUCTURE), ("anxiety_structure", ANX_STRUCTURE)]:
        print(f"\n--- mediator set: {label} ({mediators}) ---")
        for outcome in OUTCOMES:
            pt = fit_multi(df, mediators, outcome)
            ind_om_b, ind_kp_b = bootstrap_multi(df, mediators, outcome)
            # Total indirect = sum across parallel mediators
            total_ind_om = pt["a_oms"] @ pt["bs"]
            total_ind_kp = pt["a_kps"] @ pt["bs"]
            total_ind_om_boot = np.nansum(ind_om_b, axis=1)
            total_ind_kp_boot = np.nansum(ind_kp_b, axis=1)
            ci_om = np.nanpercentile(total_ind_om_boot, [2.5, 97.5])
            ci_kp = np.nanpercentile(total_ind_kp_boot, [2.5, 97.5])
            p_om = 2 * min((total_ind_om_boot < 0).mean(), (total_ind_om_boot > 0).mean())
            p_kp = 2 * min((total_ind_kp_boot < 0).mean(), (total_ind_kp_boot > 0).mean())
            sig_om = "★" if (ci_om[0] > 0) or (ci_om[1] < 0) else " "
            sig_kp = "★" if (ci_kp[0] > 0) or (ci_kp[1] < 0) else " "
            print(f"  {outcome:14s}  TOTAL_ind_ω={total_ind_om:+.4f} [{ci_om[0]:+.4f},{ci_om[1]:+.4f}] p={p_om:.3f} {sig_om}   "
                  f"TOTAL_ind_κ={total_ind_kp:+.4f} [{ci_kp[0]:+.4f},{ci_kp[1]:+.4f}] p={p_kp:.3f} {sig_kp}")
            # Per-mediator breakdown
            for i, m in enumerate(mediators):
                ci_m_om = np.nanpercentile(ind_om_b[:, i], [2.5, 97.5])
                ci_m_kp = np.nanpercentile(ind_kp_b[:, i], [2.5, 97.5])
                p_m_om = 2 * min((ind_om_b[:, i] < 0).mean(), (ind_om_b[:, i] > 0).mean())
                p_m_kp = 2 * min((ind_kp_b[:, i] < 0).mean(), (ind_kp_b[:, i] > 0).mean())
                ind_om = pt["a_oms"][i] * pt["bs"][i]
                ind_kp = pt["a_kps"][i] * pt["bs"][i]
                sm_om = "★" if (ci_m_om[0] > 0) or (ci_m_om[1] < 0) else " "
                sm_kp = "★" if (ci_m_kp[0] > 0) or (ci_m_kp[1] < 0) else " "
                print(f"      via {m:24s}  ω={ind_om:+.4f} [{ci_m_om[0]:+.4f},{ci_m_om[1]:+.4f}] p={p_m_om:.3f} {sm_om}   "
                      f"κ={ind_kp:+.4f} [{ci_m_kp[0]:+.4f},{ci_m_kp[1]:+.4f}] p={p_m_kp:.3f} {sm_kp}")
                multi_rows.append({
                    "mediator_set": label, "mediator": m, "outcome": outcome,
                    "a_omega": pt["a_oms"][i], "a_kappa": pt["a_kps"][i], "b": pt["bs"][i],
                    "indirect_omega": ind_om, "ind_om_lo": ci_m_om[0], "ind_om_hi": ci_m_om[1], "p_omega": p_m_om,
                    "indirect_kappa": ind_kp, "ind_kp_lo": ci_m_kp[0], "ind_kp_hi": ci_m_kp[1], "p_kappa": p_m_kp,
                    "total_indirect_omega": total_ind_om, "total_indirect_kappa": total_ind_kp,
                    "cprime_omega": pt["cp_om"], "cprime_kappa": pt["cp_kp"],
                    "c_omega": pt["c_om"], "c_kappa": pt["c_kp"],
                })

    pd.DataFrame(multi_rows).to_csv(OUT_DIR / "confidence_structure_mediation_multi.csv", index=False)
    print(f"\nSaved multi-mediator results: {OUT_DIR / 'confidence_structure_mediation_multi.csv'}")


if __name__ == "__main__":
    main()
