"""
Optimal execution analysis: do (ω, κ) preferences shift subjects across the
W(u)-predicted optimum, and what does that cost them in earnings?

Three analyses:

  Part 1 — Subjective optimum and deviation
    - Compute u*_subj per (subject, condition) under W(u; ω_i, κ_i, T, D, R, req)
    - Compute observed u from cell-means
    - Aggregate signed deviation (u_obs − u*_subj) per subject
    - Regress deviation on (ω_z, κ_z) — predicted: β_ω > 0, β_κ < 0

  Part 2 — Survival vs earnings trade-off
    - escape_rate ~ mean_vigor (predicted positive, monotonic — the "more pressing = more survival" link)
    - earnings ~ mean_vigor (predicted positive — model has no objective effort cost, so more vigor → more earnings via more survival)
    - earnings ~ deviation_from_u*_subj (signed and |·|) — does subjective optimum match earnings-optimal?

  Part 3 — Choice T-specificity (re-introduce in clean form what 402's stimulus tuning did)
    - Trial-level multilevel Bayesian logistic: choice ~ T·ω + T·κ + D·ω + D·κ + main effects + (1|subj)
    - WITH random intercepts (the 402 version did not)
    - Tests:
      * ω × T > 0 in magnitude → ω is T-specific (steeper avoidance under threat)
      * κ × T ≈ 0 → κ is NOT T-specific (uniform suppression across T)
      * κ × D nonzero → κ is D-specific via demand cost
      * ω × D smaller → ω is less D-specific

Outputs:
  results/stats/individual_diffs/optimal_execution_deviation.csv
  results/stats/individual_diffs/optimal_execution_earnings.csv
  results/stats/individual_diffs/optimal_execution_t_specificity.csv
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
from scipy.stats import zscore, pearsonr

from config import BKW  # type: ignore
from load_data import load_both  # type: ignore


# ---------- Model constants (from scripts/modeling/joint_optimal/param_recovery_v8c.py) ----------
C_PENALTY = 5.0
U_GRID = np.linspace(0.1, 2.5, 300)  # extended above 1.5 to capture the model's optimum when it's high

# Population params per sample (M4 posterior means from mcmc_convergence_diagnostics.csv)
POP_PARAMS = {
    "exploratory": {"gamma": 0.847, "hazard": 0.551},
    "confirmatory": {"gamma": 0.827, "hazard": 0.382},
}


def exp_survival(u: np.ndarray, T: float, D: float, gamma: float, hazard: float) -> np.ndarray:
    return np.exp(-hazard * (T ** gamma) * D / np.clip(u, 0.1, None))


def compute_W(u: np.ndarray, omega: float, kappa: float, T: float, D: float, R: float, req: float, gamma: float, hazard: float) -> np.ndarray:
    S = exp_survival(u, T, D, gamma, hazard)
    return S * R - (1.0 - S) * omega * (R + C_PENALTY) - kappa * (u - req) ** 2 * D


def find_ustar_subjective(omega: float, kappa: float, T: float, D: float, R: float, req: float, gamma: float, hazard: float) -> float:
    """Grid argmax of W(u) using subject's own (ω, κ)."""
    W_vals = compute_W(U_GRID, omega, kappa, T, D, R, req, gamma, hazard)
    return float(U_GRID[np.argmax(W_vals)])


def find_ustar_population(kappa_pop: float, omega_pop: float, T: float, D: float, R: float, req: float, gamma: float, hazard: float) -> float:
    """u* under population-mean (ω, κ) — the 'rational forager' benchmark."""
    return find_ustar_subjective(omega_pop, kappa_pop, T, D, R, req, gamma, hazard)


def D_traveled(is_heavy: int, exp_dist: float) -> float:
    """Distance the subject actually traversed.
    Light cookie is always at D=1 regardless of trial's heavy-distance condition."""
    return float(exp_dist) if is_heavy == 1 else 1.0


def load_per_subj_params(sample: str) -> pd.DataFrame:
    """Per-subject ω, κ from M4 fit. Returns DataFrame with subj, omega, kappa, log_omega, log_kappa."""
    path = REPO_ROOT / "results" / "stats" / "joint_optimal" / sample / "mcmc_m4_params.csv"
    df = pd.read_csv(path)
    df["log_omega"] = np.log(df["omega"])
    df["log_kappa"] = np.log(df["kappa"])
    return df


def load_cell_means(sample: str) -> pd.DataFrame:
    """Per-subject, per-condition observed pressing rates."""
    path = REPO_ROOT / "data" / f"model_input_{sample}" / "vigor_cell_means.csv"
    return pd.read_csv(path)


def load_choice_trials(sample: str) -> pd.DataFrame:
    """Per-trial choice data."""
    path = REPO_ROOT / "data" / f"model_input_{sample}" / "choice_trials.csv"
    return pd.read_csv(path)


_MASTER_CACHE: dict = {}


def load_master(sample: str) -> pd.DataFrame:
    """Per-subject master table from load_both — includes escape_rate, earnings, mean_vigor."""
    if not _MASTER_CACHE:
        exp_data, conf_data = load_both()
        if exp_data is not None:
            _MASTER_CACHE["exploratory"] = exp_data["master"].copy()
        if conf_data is not None:
            _MASTER_CACHE["confirmatory"] = conf_data["master"].copy()
    return _MASTER_CACHE[sample].copy()


# ---------- PART 1 + 2: Per-subject deviation from u* and earnings analysis ----------

def part1_deviation_analysis(sample: str) -> dict:
    print(f"\n{'='*60}\nPART 1+2: {sample.upper()} sample\n{'='*60}")

    params = load_per_subj_params(sample)
    cells = load_cell_means(sample)
    gamma = POP_PARAMS[sample]["gamma"]
    hazard = POP_PARAMS[sample]["hazard"]
    print(f"Population: γ = {gamma}, hazard = {hazard}")
    print(f"N subjects: {len(params)}, N cells: {len(cells)}")

    # Population-mean (ω, κ) for the "rational forager" benchmark
    omega_pop = params["omega"].mean()
    kappa_pop = params["kappa"].mean()
    print(f"Population mean: ω̄ = {omega_pop:.3f}, κ̄ = {kappa_pop:.3f}")

    # Merge per-subject params with cell-means
    cells_with_params = cells.merge(params[["subj", "omega", "kappa"]],
                                     left_on="subj_idx", right_on="subj", how="inner")

    # Compute u*_subj and u*_pop per row; D used in W is the traversed distance
    u_star_subj = []
    u_star_pop = []
    D_used = []
    for _, row in cells_with_params.iterrows():
        D_w = D_traveled(int(row["is_heavy"]), row["actual_dist"])
        u_s = find_ustar_subjective(row["omega"], row["kappa"], row["T_round"],
                                     D_w, row["actual_R"], row["actual_req"], gamma, hazard)
        u_p = find_ustar_subjective(omega_pop, kappa_pop, row["T_round"],
                                     D_w, row["actual_R"], row["actual_req"], gamma, hazard)
        u_star_subj.append(u_s)
        u_star_pop.append(u_p)
        D_used.append(D_w)
    cells_with_params["u_star_subj"] = u_star_subj
    cells_with_params["u_star_pop"] = u_star_pop
    cells_with_params["D_traveled"] = D_used
    cells_with_params["dev_subj"] = cells_with_params["mean_rate"] - cells_with_params["u_star_subj"]
    cells_with_params["dev_pop"] = cells_with_params["mean_rate"] - cells_with_params["u_star_pop"]
    cells_with_params["abs_dev_subj"] = np.abs(cells_with_params["dev_subj"])
    cells_with_params["abs_dev_pop"] = np.abs(cells_with_params["dev_pop"])

    # Aggregate to subject level
    subj_dev = cells_with_params.groupby("subj_idx").agg(
        mean_dev_subj=("dev_subj", "mean"),
        mean_dev_pop=("dev_pop", "mean"),
        mean_abs_dev_subj=("abs_dev_subj", "mean"),
        mean_abs_dev_pop=("abs_dev_pop", "mean"),
        mean_observed=("mean_rate", "mean"),
        mean_ustar_subj=("u_star_subj", "mean"),
        mean_ustar_pop=("u_star_pop", "mean"),
    ).reset_index()
    subj_dev = subj_dev.merge(params[["subj", "omega", "kappa", "log_omega", "log_kappa"]],
                               left_on="subj_idx", right_on="subj", how="inner")
    subj_dev["omega_z"] = zscore(subj_dev["log_omega"].values, nan_policy="omit")
    subj_dev["kappa_z"] = zscore(subj_dev["log_kappa"].values, nan_policy="omit")

    # Bring in behavioral outcomes from master (load_both)
    master = load_master(sample).reset_index().rename(columns={"index": "subj"})
    keep_cols = [c for c in ["subj", "escape_rate", "earnings", "mean_vigor", "p_heavy", "oc_ratio", "pct_opt"] if c in master.columns]
    print(f"  master columns kept: {keep_cols}, n_rows: {len(master)}")
    subj_dev = subj_dev.merge(master[keep_cols], on="subj", how="inner")

    print(f"\nN subjects with full data: {len(subj_dev)}")
    print(f"Subject-mean u*_subj across conditions: {subj_dev['mean_ustar_subj'].mean():.3f} ± {subj_dev['mean_ustar_subj'].std():.3f}")
    print(f"Subject-mean u*_pop across conditions:  {subj_dev['mean_ustar_pop'].mean():.3f} ± {subj_dev['mean_ustar_pop'].std():.3f}")
    print(f"Subject-mean observed u:                {subj_dev['mean_observed'].mean():.3f} ± {subj_dev['mean_observed'].std():.3f}")
    print(f"Mean signed deviation (u_obs - u*_subj):  {subj_dev['mean_dev_subj'].mean():+.4f}")
    print(f"Mean signed deviation (u_obs - u*_pop):   {subj_dev['mean_dev_pop'].mean():+.4f}")

    # ---- Regress signed deviation on (ω_z, κ_z) ----
    out = {"sample": sample, "n": len(subj_dev), "omega_pop": omega_pop, "kappa_pop": kappa_pop}
    if "escape_rate" in subj_dev.columns:
        print("\n--- Regression: signed deviation (u_obs − u*_subj) ~ ω_z + κ_z ---")
        m = bmb.Model("mean_dev_subj ~ omega_z + kappa_z", data=subj_dev)
        s = az.summary(m.fit(**BKW), hdi_prob=0.95)
        for t in ["omega_z", "kappa_z"]:
            print(f"  {t}: β = {s.loc[t,'mean']:+.4f} [{s.loc[t,'hdi_2.5%']:+.4f}, {s.loc[t,'hdi_97.5%']:+.4f}]")
            out[f"dev_subj_{t}_mean"] = s.loc[t, "mean"]
            out[f"dev_subj_{t}_hdi_lo"] = s.loc[t, "hdi_2.5%"]
            out[f"dev_subj_{t}_hdi_hi"] = s.loc[t, "hdi_97.5%"]

        # ---- Deviation from population optimum ----
        print("\n--- Regression: signed deviation (u_obs − u*_pop) ~ ω_z + κ_z ---")
        m2 = bmb.Model("mean_dev_pop ~ omega_z + kappa_z", data=subj_dev)
        s2 = az.summary(m2.fit(**BKW), hdi_prob=0.95)
        for t in ["omega_z", "kappa_z"]:
            print(f"  {t}: β = {s2.loc[t,'mean']:+.4f} [{s2.loc[t,'hdi_2.5%']:+.4f}, {s2.loc[t,'hdi_97.5%']:+.4f}]")
            out[f"dev_pop_{t}_mean"] = s2.loc[t, "mean"]
            out[f"dev_pop_{t}_hdi_lo"] = s2.loc[t, "hdi_2.5%"]
            out[f"dev_pop_{t}_hdi_hi"] = s2.loc[t, "hdi_97.5%"]

        # ---- Survival vs earnings linearity in vigor ----
        print("\n--- Earnings vs survival vs vigor (raw Pearson) ---")
        r_e_v, p_e_v = pearsonr(subj_dev["mean_vigor"], subj_dev["earnings"])
        r_s_v, p_s_v = pearsonr(subj_dev["mean_vigor"], subj_dev["escape_rate"])
        print(f"  r(earnings, mean_vigor)   = {r_e_v:+.3f} (p={p_e_v:.3g})")
        print(f"  r(escape, mean_vigor)     = {r_s_v:+.3f} (p={p_s_v:.3g})")
        out["r_earnings_vigor"] = r_e_v
        out["r_escape_vigor"] = r_s_v

        # ---- Earnings ~ |deviation from u*_subj| and |deviation from u*_pop| ----
        print("\n--- Regression: earnings ~ |u_obs - u*_subj| + |u_obs - u*_pop| + ω_z + κ_z ---")
        m3 = bmb.Model("earnings ~ mean_abs_dev_subj + mean_abs_dev_pop + omega_z + kappa_z", data=subj_dev)
        s3 = az.summary(m3.fit(**BKW), hdi_prob=0.95)
        for t in ["mean_abs_dev_subj", "mean_abs_dev_pop", "omega_z", "kappa_z"]:
            print(f"  {t}: β = {s3.loc[t,'mean']:+.3f} [{s3.loc[t,'hdi_2.5%']:+.3f}, {s3.loc[t,'hdi_97.5%']:+.3f}]")
            out[f"earnings_{t}_mean"] = s3.loc[t, "mean"]
            out[f"earnings_{t}_hdi_lo"] = s3.loc[t, "hdi_2.5%"]
            out[f"earnings_{t}_hdi_hi"] = s3.loc[t, "hdi_97.5%"]

    # Save the per-subject deviation table
    out_dir = REPO_ROOT / "results" / "stats" / "individual_diffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    subj_dev.to_csv(out_dir / f"optimal_execution_subjects_{sample}.csv", index=False)
    print(f"\nSaved: {out_dir / f'optimal_execution_subjects_{sample}.csv'}")

    return out


# ---------- PART 3: T-specificity in choice ----------

def part3_t_specificity(sample: str) -> dict:
    print(f"\n{'='*60}\nPART 3 (T-specificity in choice): {sample.upper()}\n{'='*60}")

    trials = load_choice_trials(sample)
    params = load_per_subj_params(sample)
    params["omega_z"] = zscore(params["log_omega"].values, nan_policy="omit")
    params["kappa_z"] = zscore(params["log_kappa"].values, nan_policy="omit")
    print(f"N trials: {len(trials)}, columns: {list(trials.columns)[:8]}")

    trials = trials.merge(params[["subj", "omega_z", "kappa_z"]],
                           left_on="subj_idx", right_on="subj", how="inner")
    # Find threat column (may be 'threat' or 'T_round')
    t_col = "threat" if "threat" in trials.columns else "T_round"
    trials["T_z"] = zscore(trials[t_col].astype(float).values, nan_policy="omit")
    # Use heavy-cookie distance D_H (column may vary; choose the right one)
    d_col = None
    for cand in ["actual_dist", "D_H", "distance_H", "dist_H"]:
        if cand in trials.columns:
            d_col = cand
            break
    if d_col is None:
        print(f"  Warning: no distance column found in {list(trials.columns)}")
        return {"sample": sample, "skipped": True}
    trials["D_z"] = zscore(trials[d_col].astype(float).values, nan_policy="omit")

    if "choice" in trials.columns:
        trials["choice_int"] = trials["choice"].astype(int)
    elif "choice_int" not in trials.columns:
        print(f"  Warning: no choice column found")
        return {"sample": sample, "skipped": True}

    print(f"  Using distance column: {d_col}")
    print(f"  N subjects: {trials['subj_idx'].nunique()}")
    print(f"  N choice rows: {len(trials)}")

    # ---- Multilevel logistic with proper random intercepts ----
    # choice ~ T + D + ω + κ + T·ω + T·κ + D·ω + D·κ + (1|subj)
    print("\nFitting multilevel logistic with random intercepts...")
    mod = bmb.Model(
        "choice_int ~ T_z + D_z + omega_z + kappa_z + T_z:omega_z + D_z:omega_z + T_z:kappa_z + D_z:kappa_z + (1|subj_idx)",
        data=trials, family="bernoulli",
    )
    res = mod.fit(draws=1500, tune=500, chains=4, progressbar=False, random_seed=42)
    s = az.summary(res, hdi_prob=0.95)

    out = {"sample": sample}
    print("\nResults (β [95% HDI]):")
    for t in ["T_z", "D_z", "omega_z", "kappa_z",
              "T_z:omega_z", "T_z:kappa_z", "D_z:omega_z", "D_z:kappa_z"]:
        if t not in s.index:
            continue
        b = s.loc[t, "mean"]
        lo = s.loc[t, "hdi_2.5%"]
        hi = s.loc[t, "hdi_97.5%"]
        sig = "★" if (lo > 0 or hi < 0) else " "
        kind = "INT" if ":" in t else "main"
        print(f"  {kind:4s} {t:<20}: β = {b:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")
        out[f"{t}_mean"] = b
        out[f"{t}_hdi_lo"] = lo
        out[f"{t}_hdi_hi"] = hi
        out[f"{t}_sig"] = (lo > 0 or hi < 0)

    return out


def main():
    deviation_rows = []
    tspec_rows = []
    for sample in ["exploratory", "confirmatory"]:
        deviation_rows.append(part1_deviation_analysis(sample))
        tspec_rows.append(part3_t_specificity(sample))

    out_dir = REPO_ROOT / "results" / "stats" / "individual_diffs"
    pd.DataFrame(deviation_rows).to_csv(out_dir / "optimal_execution_deviation.csv", index=False)
    pd.DataFrame(tspec_rows).to_csv(out_dir / "optimal_execution_t_specificity.csv", index=False)
    print(f"\nSaved: {out_dir / 'optimal_execution_deviation.csv'}")
    print(f"Saved: {out_dir / 'optimal_execution_t_specificity.csv'}")


if __name__ == "__main__":
    main()
