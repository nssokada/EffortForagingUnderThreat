"""
Foraging-theoretic optimum analysis with κ_opt calibrated to typical human.

Plan summary:
  1. Define foraging objective:
       W_opt(u, choice, T, D, κ_opt) = S(u, T, D)·R − (1−S(u, T, D))·(R+C) − κ_opt·(u−req)²·D
     with S(u, T, D) = exp(−hazard·T^γ·D / u). ω is implicitly = 1 (face-value capture).

  2. Compute group-median observed vigor per (T, D_heavy, cookie_weight) cell.

  3. Grid-search κ_opt to find κ_opt* that minimizes SSE between optimum-predicted
     vigor pattern and group-median observed vigor pattern. This is the
     "species-typical" foraging-optimum anchor.

  4. Solve foraging optimum at κ_opt* and at sensitivity bounds (κ_opt*/2 and 2·κ_opt*).

  5. Per-subject signed deviation in choice and vigor at each κ_opt.

  6. Test within each sample (exploratory, confirmatory):
       Δ ~ ω_z + κ_z (parameter directions)
       Δ ~ ω_z + κ_z + affect_slopes (residual variance from affect)

  7. Report robustness across the κ_opt range.

Outputs:
  results/stats/joint_optimal/foraging_optimum_grid.csv
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


# ── Task and model constants ──────────────────────────────────────────────
GAMMA = 0.86      # threat exponent (population posterior)
HAZARD = 0.832    # hazard rate (population posterior)
C_PENALTY = 5.0   # fixed capture penalty
R_HEAVY = 5.0
R_LIGHT = 1.0
REQ_HEAVY = 0.9
REQ_LIGHT = 0.4
D_LIGHT = 1.0     # light cookie always at close distance
D_HEAVY_LEVELS = [1.0, 2.0, 3.0]
T_LEVELS = [0.1, 0.5, 0.9]

U_GRID = np.linspace(0.1, 2.0, 200)
KAPPA_OPT_SEARCH = np.logspace(-3, 1, 50)


# ── Foraging-optimum solver ───────────────────────────────────────────────
def survival(u, T, D):
    return np.exp(-HAZARD * (T ** GAMMA) * D / np.clip(u, 0.1, None))


def value(u, T, D, R, req, kappa_opt):
    """Foraging-optimum value: ω = 1, κ = κ_opt."""
    S = survival(u, T, D)
    return S * R - (1 - S) * (R + C_PENALTY) - kappa_opt * (u - req) ** 2 * D


def solve_optimum(T, D, R, req, kappa_opt):
    """Return (u_star, V_star) for the foraging optimum at this condition."""
    W = value(U_GRID, T, D, R, req, kappa_opt)
    idx = int(np.argmax(W))
    return float(U_GRID[idx]), float(W[idx])


def optimal_pattern(kappa_opt):
    """Return optimal P(heavy), u_heavy, u_light at each (T, D_heavy) cell.

    Returns a DataFrame with columns: T, D_heavy, P_heavy_opt, u_heavy_opt, u_light_opt.
    """
    rows = []
    for T in T_LEVELS:
        u_L, V_L = solve_optimum(T, D_LIGHT, R_LIGHT, REQ_LIGHT, kappa_opt)
        for D_H in D_HEAVY_LEVELS:
            u_H, V_H = solve_optimum(T, D_H, R_HEAVY, REQ_HEAVY, kappa_opt)
            rows.append({
                "T": T, "D_heavy": D_H,
                "P_heavy_opt": int(V_H > V_L),
                "u_heavy_opt": u_H,
                "u_light_opt": u_L,
            })
    return pd.DataFrame(rows)


# ── Observed group-typical pattern ────────────────────────────────────────
def build_observed():
    """Per-subject × condition observed P(heavy) and vigor by cookie."""
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        v = d["vigor"][["subj", "T_round", "distance", "is_heavy", "norm_rate"]].copy()
        v["sample"] = sample
        rows.append(v)
    beh = pd.concat(rows, ignore_index=True)
    # Per (subj, sample, T, D_heavy) — D_heavy = distance on trial
    # P(heavy): mean of is_heavy
    p_heavy = beh.groupby(["subj", "sample", "T_round", "distance"])["is_heavy"].mean().reset_index()
    p_heavy.columns = ["subj", "sample", "T", "D_heavy", "p_heavy_obs"]

    # Vigor on heavy trials (when chose heavy)
    h_trials = beh[beh["is_heavy"] == 1]
    u_heavy = h_trials.groupby(["subj", "sample", "T_round", "distance"])["norm_rate"].mean().reset_index()
    u_heavy.columns = ["subj", "sample", "T", "D_heavy", "u_heavy_obs"]

    # Vigor on light trials (D doesn't matter on light, but record per (T, D_heavy) for join)
    l_trials = beh[beh["is_heavy"] == 0]
    u_light = l_trials.groupby(["subj", "sample", "T_round", "distance"])["norm_rate"].mean().reset_index()
    u_light.columns = ["subj", "sample", "T", "D_heavy", "u_light_obs"]

    obs = p_heavy.merge(u_heavy, on=["subj", "sample", "T", "D_heavy"], how="left")
    obs = obs.merge(u_light, on=["subj", "sample", "T", "D_heavy"], how="left")
    return obs


def group_median_pattern(obs):
    """Median across subjects of observed vigor at each (T, D_heavy)."""
    grp = obs.groupby(["T", "D_heavy"]).agg(
        u_heavy_med=("u_heavy_obs", "median"),
        u_light_med=("u_light_obs", "median"),
        p_heavy_med=("p_heavy_obs", "median"),
    ).reset_index()
    return grp


# ── Calibrate κ_opt to typical human ──────────────────────────────────────
def calibrate_kappa_opt(group_med, kappa_grid=KAPPA_OPT_SEARCH):
    """Find κ_opt* minimizing SSE between optimum vigor and group-median vigor.

    Both heavy and light vigor contribute. Sum over all (T, D) cells for heavy
    and (T) cells for light (since light is at D=1).
    """
    sse = []
    for kop in kappa_grid:
        opt = optimal_pattern(kop)
        merged = group_med.merge(opt, on=["T", "D_heavy"])
        # Heavy vigor SSE — across 9 cells
        err_h = (merged["u_heavy_opt"] - merged["u_heavy_med"]) ** 2
        # Light vigor SSE — light is the same at fixed T (across D_heavy)
        # So just take 3 unique T values' average light_opt and light_med
        light_med_by_T = merged.groupby("T")["u_light_med"].mean()
        light_opt_by_T = merged.groupby("T")["u_light_opt"].mean()
        err_l = (light_opt_by_T - light_med_by_T) ** 2
        sse.append(float(err_h.sum() + err_l.sum()))
    sse = np.array(sse)
    idx = int(np.argmin(sse))
    return kappa_grid[idx], sse


# ── Per-subject deviations ────────────────────────────────────────────────
def per_subject_deviation(obs, kappa_opt):
    opt = optimal_pattern(kappa_opt)
    df = obs.merge(opt, on=["T", "D_heavy"], how="left")
    df["delta_choice"] = df["p_heavy_obs"] - df["P_heavy_opt"]
    df["delta_uH"] = df["u_heavy_obs"] - df["u_heavy_opt"]
    df["delta_uL"] = df["u_light_obs"] - df["u_light_opt"]
    # Aggregate to per-subject summary (signed sum across cells)
    agg = df.groupby(["subj", "sample"]).agg(
        delta_choice_sum=("delta_choice", lambda x: x.dropna().sum()),
        delta_uH_sum=("delta_uH", lambda x: x.dropna().sum()),
        delta_uL_sum=("delta_uL", lambda x: x.dropna().sum()),
        n_cells_choice=("delta_choice", lambda x: x.dropna().shape[0]),
        n_cells_uH=("delta_uH", lambda x: x.dropna().shape[0]),
        n_cells_uL=("delta_uL", lambda x: x.dropna().shape[0]),
    ).reset_index()
    return agg


def merge_params_affect(dev_df, exp, conf):
    em = exp["master"].reset_index().rename(columns={"index": "subj"}).copy()
    cm = conf["master"].reset_index().rename(columns={"index": "subj"}).copy()
    em["sample"] = "exploratory"; cm["sample"] = "confirmatory"
    master = pd.concat([em, cm], ignore_index=True)
    master["omega_z"] = zscore(np.log(master["omega"]).values)
    master["kappa_z"] = zscore(np.log(master["kappa"]).values)
    dev_df = dev_df.merge(master[["subj", "sample", "omega_z", "kappa_z"]], on=["subj", "sample"])

    # Affect slopes per subject
    slopes = pd.read_csv(REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_slopes_subjects.csv")
    keep = ["subj", "sample", "anxiety_intercept", "confidence_intercept",
            "anxiety_slope_T", "confidence_slope_T",
            "anxiety_slope_D", "confidence_slope_D"]
    slopes = slopes[keep]
    dev_df = dev_df.merge(slopes, on=["subj", "sample"], how="left")
    return dev_df


def fit_z(df, lhs, rhs):
    sub = df[[lhs] + rhs].dropna().copy()
    if len(sub) < 30:
        return None
    sub[lhs + "_z"] = zscore(sub[lhs].values, nan_policy="omit")
    for r in rhs:
        sub[r] = zscore(sub[r].values, nan_policy="omit")
    X = sm.add_constant(sub[rhs].values)
    return sm.OLS(sub[lhs + "_z"].values, X).fit()


def test_predictions(dev_df, label, kappa_opt, results_rows):
    print(f"\n{'=' * 78}\n## {label} (κ_opt = {kappa_opt:.4f})\n{'=' * 78}")
    affect_preds = ["anxiety_slope_T", "confidence_slope_T",
                    "anxiety_slope_D", "confidence_slope_D",
                    "anxiety_intercept", "confidence_intercept"]
    for sample in ["exploratory", "confirmatory"]:
        sub = dev_df[dev_df["sample"] == sample].copy()
        print(f"\n--- {sample} (N = {len(sub)}) ---")
        for outcome in ["delta_choice_sum", "delta_uH_sum", "delta_uL_sum"]:
            res_base = fit_z(sub, outcome, ["omega_z", "kappa_z"])
            if res_base is None:
                continue
            print(f"\n  {outcome} ~ ω + κ    R² = {res_base.rsquared:.4f}")
            for i, n in enumerate(["omega_z", "kappa_z"]):
                sig = "★" if res_base.pvalues[i+1] < 0.05 else " "
                print(f"    {n:14s} β={res_base.params[i+1]:+.3f} p={res_base.pvalues[i+1]:.4g} {sig}")
            results_rows.append({
                "label": label, "kappa_opt": kappa_opt, "sample": sample,
                "outcome": outcome, "model": "params_only", "R2": res_base.rsquared,
                "beta_omega": res_base.params[1], "p_omega": res_base.pvalues[1],
                "beta_kappa": res_base.params[2], "p_kappa": res_base.pvalues[2],
            })

            res_full = fit_z(sub, outcome, ["omega_z", "kappa_z"] + affect_preds)
            if res_full is None:
                continue
            d_r2 = res_full.rsquared - res_base.rsquared
            print(f"  {outcome} ~ ω + κ + affect_slopes   R² = {res_full.rsquared:.4f}   ΔR² = {d_r2:+.4f}")
            for i, n in enumerate(["omega_z", "kappa_z"] + affect_preds):
                sig = "★" if res_full.pvalues[i+1] < 0.05 else " "
                print(f"    {n:24s} β={res_full.params[i+1]:+.3f} p={res_full.pvalues[i+1]:.4g} {sig}")
            row = {
                "label": label, "kappa_opt": kappa_opt, "sample": sample,
                "outcome": outcome, "model": "with_affect", "R2": res_full.rsquared,
                "delta_R2": d_r2,
                "beta_omega": res_full.params[1], "p_omega": res_full.pvalues[1],
                "beta_kappa": res_full.params[2], "p_kappa": res_full.pvalues[2],
            }
            for i, n in enumerate(affect_preds):
                row[f"beta_{n}"] = res_full.params[3 + i]
                row[f"p_{n}"] = res_full.pvalues[3 + i]
            results_rows.append(row)


def main():
    print("=" * 78)
    print("FORAGING OPTIMUM ANALYSIS — κ_opt calibrated to median human")
    print("=" * 78)

    obs = build_observed()
    print(f"\nObserved table: {len(obs)} (subj, sample, T, D) rows")

    group_med = group_median_pattern(obs)
    print("\nGroup-median observed pattern:")
    print(group_med.to_string(index=False))

    # Sanity check: are observed vigor patterns adaptive (rise with T)?
    print("\n[Sanity] Mean of group-median vigor by T:")
    print(group_med.groupby("T")[["u_heavy_med", "u_light_med"]].mean())

    # Calibrate κ_opt
    print("\n" + "=" * 78)
    print("CALIBRATION: find κ_opt* matching group-median vigor")
    print("=" * 78)
    kopt_star, sse_curve = calibrate_kappa_opt(group_med)
    print(f"\nκ_opt* = {kopt_star:.4f}   (min SSE = {sse_curve.min():.4f})")
    print(f"SSE range across grid: {sse_curve.min():.3f} to {sse_curve.max():.3f}")

    # Show the optimum at κ_opt*
    opt_star = optimal_pattern(kopt_star)
    print(f"\nForaging optimum pattern at κ_opt* = {kopt_star:.4f}:")
    print(opt_star.to_string(index=False))

    # Sensitivity bounds
    kopt_half = kopt_star / 2.0
    kopt_double = kopt_star * 2.0
    print(f"\nSensitivity bounds: half = {kopt_half:.4f}, double = {kopt_double:.4f}")

    # Compute per-subject deviations at each κ_opt
    print("\n" + "=" * 78)
    print("PER-SUBJECT DEVIATION TESTS")
    print("=" * 78)
    exp, conf = load_both()
    results_rows = []

    for label, kop in [("κ_opt*", kopt_star), ("κ_opt*/2", kopt_half), ("2·κ_opt*", kopt_double)]:
        dev = per_subject_deviation(obs, kop)
        dev = merge_params_affect(dev, exp, conf)
        test_predictions(dev, label, kop, results_rows)

    out = REPO_ROOT / "results" / "stats" / "joint_optimal" / "foraging_optimum_grid.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
