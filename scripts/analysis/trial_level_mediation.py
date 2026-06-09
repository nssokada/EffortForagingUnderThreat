"""
Trial-level mediation v3: per-trial affect as the mediator.

Mean confidence (v1) and confidence STRUCTURE (v2) both produced essentially
null mediations. The legitimate criticism: those are between-subject
summaries — they can't catch the work confidence does *moment-to-moment* as
the subject responds to the actual trial they're on.

This version operates at the trial level. On each probe trial we have:
  - T_z, D_z (the contingencies the subject is responding to)
  - the actual rating (anxiety OR confidence — one per probe trial)
  - the actual vigor on that trial
  - the subject's (ω_z, κ_z) — between-subject

Models (probe trials only, separately for anxiety and confidence questions):
  c-total:  vigor_z ~ T_z + D_z + ω_z + κ_z                   + (1|subj)
  a-path:   aff_z   ~ T_z + D_z + ω_z + κ_z                   + (1|subj)
  b/c'-path:vigor_z ~ T_z + D_z + ω_z + κ_z + aff_z           + (1|subj)

Indirect_ω = a_ω · b   (Monte Carlo CI from joint sampling distribution)
Indirect_κ = a_κ · b

We also decompose aff_z into:
  - aff_between (per-subject mean rating)
  - aff_within  (trial rating minus subject mean)
and test each as a separate mediator. This separates *trait-like* mediation
(subjects with high mean affect have systematically different vigor) from
*state-like* mediation (the moment-to-moment regulation we showed in
affect_reshapes_behavior.py).

Outputs:
  results/stats/affect_analysis/trial_level_mediation.csv
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

N_MC = 20000
RNG = np.random.default_rng(20260605)


def build_pooled():
    exp, conf = load_both()
    rows = []
    for sample_name, paths in SAMPLES.items():
        f = pd.read_csv(Path(paths["stage5"]) / "feelings.csv", low_memory=False)
        m4 = pd.read_csv(paths["m4_params"])
        f = f[["subj", "trialNumber", "threat", "distance", "questionLabel", "response"]].dropna(subset=["response"]).copy()
        d = exp if sample_name == "exploratory" else conf
        v = d["vigor"][["subj", "trial", "norm_rate"]].rename(columns={"trial": "trialNumber", "norm_rate": "vigor"})
        x = f.merge(v, on=["subj", "trialNumber"], how="inner")
        x = x.merge(m4[["subj", "omega", "kappa"]], on="subj", how="inner")
        x["sample"] = sample_name
        rows.append(x)
    df = pd.concat(rows, ignore_index=True).dropna(subset=["response", "vigor"]).copy()
    df["log_omega"] = np.log(df["omega"])
    df["log_kappa"] = np.log(df["kappa"])
    df["T_z"] = zscore(df["threat"].astype(float).values, nan_policy="omit")
    df["D_z"] = zscore(df["distance"].astype(float).values, nan_policy="omit")
    df["omega_z"] = zscore(df["log_omega"].values, nan_policy="omit")
    df["kappa_z"] = zscore(df["log_kappa"].values, nan_policy="omit")
    df["vigor_z"] = zscore(df["vigor"].values, nan_policy="omit")
    # within-question z-score of affect
    df["aff_z"] = np.nan
    for q in df["questionLabel"].unique():
        m = df["questionLabel"] == q
        df.loc[m, "aff_z"] = zscore(df.loc[m, "response"].values, nan_policy="omit")
    # between-within decomposition (within question, within subject)
    df["aff_between"] = np.nan
    df["aff_within"] = np.nan
    for q in df["questionLabel"].unique():
        m = df["questionLabel"] == q
        sub = df.loc[m]
        means = sub.groupby("subj")["aff_z"].transform("mean")
        df.loc[m, "aff_between"] = means
        df.loc[m, "aff_within"] = sub["aff_z"].values - means.values
    return df


def fit_mm(formula, data):
    return smf.mixedlm(formula, data=data, groups=data["subj"]).fit(reml=False, method="lbfgs")


def extract(model, terms):
    """Return dict of {term: (beta, se)}."""
    out = {}
    for t in terms:
        if t in model.fe_params.index:
            out[t] = (float(model.fe_params[t]), float(model.bse_fe[t]))
        else:
            out[t] = (np.nan, np.nan)
    return out


def mc_indirect_ci(a_mean, a_se, b_mean, b_se, n=N_MC):
    """Monte Carlo CI for indirect effect (assuming independence of a, b)."""
    a = RNG.normal(a_mean, a_se, size=n)
    b = RNG.normal(b_mean, b_se, size=n)
    prod = a * b
    ci = np.percentile(prod, [2.5, 97.5])
    p = 2 * min((prod < 0).mean(), (prod > 0).mean())
    return float(np.mean(prod)), float(ci[0]), float(ci[1]), float(p)


def run_one(df_q, q_label, mediator_col, mediator_label):
    """Fit total, a-path, b-path. Compute indirect for ω, κ via Monte Carlo CI."""
    fm_total = "vigor_z ~ T_z + D_z + omega_z + kappa_z"
    fm_a = f"{mediator_col} ~ T_z + D_z + omega_z + kappa_z"
    fm_b = f"vigor_z ~ T_z + D_z + omega_z + kappa_z + {mediator_col}"
    Mt = fit_mm(fm_total, df_q)
    Ma = fit_mm(fm_a, df_q)
    Mb = fit_mm(fm_b, df_q)
    tot = extract(Mt, ["omega_z", "kappa_z"])
    a = extract(Ma, ["omega_z", "kappa_z"])
    bf = extract(Mb, [mediator_col, "omega_z", "kappa_z"])
    b_mean, b_se = bf[mediator_col]
    # Indirect via MC
    ind_om = mc_indirect_ci(a["omega_z"][0], a["omega_z"][1], b_mean, b_se)
    ind_kp = mc_indirect_ci(a["kappa_z"][0], a["kappa_z"][1], b_mean, b_se)
    # Direct effects from b-path
    cp_om = bf["omega_z"]
    cp_kp = bf["kappa_z"]
    # Total
    c_om = tot["omega_z"]
    c_kp = tot["kappa_z"]

    print(f"\n   ── mediator: {mediator_label} ──")
    print(f"   total c (ω): {c_om[0]:+.4f} (SE {c_om[1]:.4f})    c (κ): {c_kp[0]:+.4f} (SE {c_kp[1]:.4f})")
    print(f"   direct c' (ω): {cp_om[0]:+.4f} (SE {cp_om[1]:.4f})    c' (κ): {cp_kp[0]:+.4f} (SE {cp_kp[1]:.4f})")
    print(f"   a (ω→M): {a['omega_z'][0]:+.4f} (SE {a['omega_z'][1]:.4f})    a (κ→M): {a['kappa_z'][0]:+.4f} (SE {a['kappa_z'][1]:.4f})")
    print(f"   b (M→vigor): {b_mean:+.4f} (SE {b_se:.4f})")
    sig_om = "★" if (ind_om[1] > 0) or (ind_om[2] < 0) else " "
    sig_kp = "★" if (ind_kp[1] > 0) or (ind_kp[2] < 0) else " "
    print(f"   indirect_ω: {ind_om[0]:+.5f}  [95% MC CI {ind_om[1]:+.5f}, {ind_om[2]:+.5f}]  p = {ind_om[3]:.4f} {sig_om}")
    print(f"   indirect_κ: {ind_kp[0]:+.5f}  [95% MC CI {ind_kp[1]:+.5f}, {ind_kp[2]:+.5f}]  p = {ind_kp[3]:.4f} {sig_kp}")
    prop_om = ind_om[0] / c_om[0] if c_om[0] != 0 else np.nan
    prop_kp = ind_kp[0] / c_kp[0] if c_kp[0] != 0 else np.nan
    print(f"   prop_mediated_ω: {prop_om:+.2%}    prop_mediated_κ: {prop_kp:+.2%}")
    return {
        "question": q_label, "mediator": mediator_label,
        "c_omega": c_om[0], "c_kappa": c_kp[0],
        "cprime_omega": cp_om[0], "cprime_kappa": cp_kp[0],
        "a_omega": a["omega_z"][0], "a_kappa": a["kappa_z"][0], "b": b_mean,
        "indirect_omega": ind_om[0], "ind_om_lo": ind_om[1], "ind_om_hi": ind_om[2], "p_omega": ind_om[3],
        "indirect_kappa": ind_kp[0], "ind_kp_lo": ind_kp[1], "ind_kp_hi": ind_kp[2], "p_kappa": ind_kp[3],
        "prop_mediated_omega": prop_om, "prop_mediated_kappa": prop_kp,
    }


def main():
    print("=" * 78)
    print("MEDIATION v3 — per-trial affect as mediator (mixed-effects, Monte Carlo CI)")
    print("=" * 78)
    df = build_pooled()
    print(f"\nTotal probe trials: {len(df)}    Subjects: {df['subj'].nunique()}")
    print(f"  anxiety probes: {(df['questionLabel']=='anxiety').sum()}")
    print(f"  confidence probes: {(df['questionLabel']=='confidence').sum()}")

    rows = []
    for q_label in ["anxiety", "confidence"]:
        print("\n" + "#" * 78)
        print(f"# question = {q_label}")
        print("#" * 78)
        df_q = df[df["questionLabel"] == q_label].copy()
        # Re-z aff_z within this question so it's centered for this analysis
        df_q["aff_z"] = zscore(df_q["aff_z"].values, nan_policy="omit")
        df_q["aff_between"] = zscore(df_q["aff_between"].values, nan_policy="omit")
        df_q["aff_within"] = zscore(df_q["aff_within"].values, nan_policy="omit")

        # 1) Total per-trial affect
        rows.append(run_one(df_q, q_label, "aff_z", "trial-level aff_z (total)"))
        # 2) Between-subject component (per-subject mean affect)
        rows.append(run_one(df_q, q_label, "aff_between", "between-subject mean (trait)"))
        # 3) Within-subject component (trial deviation from subject mean)
        rows.append(run_one(df_q, q_label, "aff_within", "within-subject deviation (state)"))

    out = REPO_ROOT / "results" / "stats" / "affect_analysis" / "trial_level_mediation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
