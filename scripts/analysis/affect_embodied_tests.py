"""
Embodied affect tests — Tests A, B, C for whether anxiety/confidence track
model-derived S(u*) BEYOND what raw threat (T) and distance (D) explain.

The result_501 finding (affect ~ S_probe) is necessary but not sufficient
for the embodied affect claim. S(u*) is a nonlinear transform of (T, D)
weighted by per-subject (ω, κ), so the bare correlation with affect could
just reflect that affect tracks raw threat. To establish that affect tracks
embodied survival prospects (the Frame C claim), we need to show S(u*)
carries predictive content BEYOND T + D, or that subject-specific (ω, κ)
shapes affect even when (T, D) are held fixed.

Three nested tests:

  TEST A — Incremental variance
    Fit: anxiety ~ T_z + D_z + S(u*)_z + (1|subj)
    Question: does S(u*) remain significant after controlling for T, D?
    Verdict: positive if β(S(u*)) HDI excludes zero with |β| meaningfully > 0

  TEST B — Model comparison
    Fit two competing single-predictor models:
      M_TD:   anxiety ~ T_z + D_z + (1|subj)
      M_S:    anxiety ~ S(u*)_z + (1|subj)
    Compare on AIC and log-likelihood. Which one is the better predictor?

  TEST C — Between-subject embodied content
    Fit: anxiety ~ T_z + D_z + omega_z + kappa_z + (1|subj)
    Question: do subject-specific (ω, κ) predict affect AFTER controlling
    for trial-level (T, D)? The embodied claim predicts:
      - High κ → more anxiety (high κ → lower predicted u* → lower S → more anxiety)
      - High ω → less anxiety (ω mobilises execution → higher S → less anxiety)
      - BUT result_503 says ω → confidence not anxiety, so ω → anxiety may be null
    Run for both anxiety AND confidence on both samples.

Outputs:
  results/stats/affect_analysis/embodied_tests_<sample>.csv
  results/stats/affect_analysis/embodied_tests_summary.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import expit
from scipy.stats import zscore


C_PENALTY = 5.0
UGRID = np.linspace(0.1, 1.5, 40)

SAMPLES = {
    "exploratory": {
        "stage5": "data/exploratory_350/processed/stage5_filtered_data_20260403_133425",
        "m4_params": "results/stats/joint_optimal/exploratory/mcmc_m4_params.csv",
        "m4_conv": "results/stats/joint_optimal/exploratory/mcmc_convergence_diagnostics.csv",
    },
    "confirmatory": {
        "stage5": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413",
        "m4_params": "results/stats/joint_optimal/confirmatory/mcmc_m4_params.csv",
        "m4_conv": "results/stats/joint_optimal/confirmatory/mcmc_convergence_diagnostics.csv",
    },
}


def m4_population_params(conv_path: Path):
    conv = pd.read_csv(conv_path)
    m4 = conv[conv["model"] == "M4"].set_index("parameter")["mean"]
    gamma_v = float(np.clip(np.exp(m4["gr"]), 0.1, 3.0))
    hazard_v = float(np.exp(m4["hr"]))
    sp_v = float(np.clip(np.exp(m4["spr"]), 0.01, 1.0))
    return gamma_v, hazard_v, sp_v


def compute_s_probe(df: pd.DataFrame, gamma_v: float, hazard_v: float, sp_v: float) -> np.ndarray:
    T = df["T"].to_numpy()
    D = df["D_model"].to_numpy()
    R = df["R"].to_numpy()
    req = df["req"].to_numpy()
    om = df["omega"].to_numpy()
    ka = df["kappa"].to_numpy()
    n = len(df)
    ug = UGRID[None, :]
    req_b = req[:, None]
    speed = expit((ug - 0.25 * req_b) / sp_v)
    Tg = (T ** gamma_v)[:, None]
    D_b = D[:, None].astype(float)
    S = np.exp(-hazard_v * Tg * D_b / np.clip(speed, 0.01, None))
    W = (S * R[:, None]
         - (1.0 - S) * om[:, None] * (R[:, None] + C_PENALTY)
         - ka[:, None] * (ug - req_b) ** 2 * D_b)
    idx_star = W.argmax(axis=1)
    return S[np.arange(n), idx_star]


def fit_lmm(formula: str, data: pd.DataFrame):
    """Fit MixedLM and return summary tuple."""
    model = smf.mixedlm(formula, data=data, groups=data["subj"]).fit(reml=False, method="lbfgs")
    return model


def extract_coef(model, term: str) -> dict:
    if term not in model.fe_params.index:
        return {f"{term}_beta": np.nan, f"{term}_se": np.nan,
                f"{term}_z": np.nan, f"{term}_p": np.nan}
    return {
        f"{term}_beta": float(model.fe_params[term]),
        f"{term}_se": float(model.bse_fe[term]),
        f"{term}_z": float(model.tvalues[term]),
        f"{term}_p": float(model.pvalues[term]),
    }


def fit_sample(sample_name: str, paths: dict) -> pd.DataFrame:
    print("=" * 70)
    print(f"Sample: {sample_name}")
    print("=" * 70)
    feelings = pd.read_csv(Path(paths["stage5"]) / "feelings.csv", low_memory=False)
    m4 = pd.read_csv(paths["m4_params"])
    gamma_v, hazard_v, sp_v = m4_population_params(paths["m4_conv"])
    print(f"  Population params: gamma={gamma_v:.4f}, hazard={hazard_v:.4f}, sp={sp_v:.4f}")

    df = feelings.merge(m4, on="subj", how="inner")
    df["T"] = df["threat"].astype(float)
    df["D_model"] = (df["distance"].astype(int) + 1)
    df["is_heavy"] = (df["trialCookie_rewardValue"] == 5.0).astype(int)
    df["R"] = df["trialCookie_rewardValue"].astype(float)
    df["req"] = np.where(df["is_heavy"] == 1, 0.9, 0.4)
    df = df.dropna(subset=["response"]).copy()

    df["S_probe"] = compute_s_probe(df, gamma_v, hazard_v, sp_v)
    df["S_probe_z"] = zscore(df["S_probe"])
    df["T_z"] = zscore(df["T"])
    df["D_z"] = zscore(df["D_model"].astype(float))
    df["log_omega"] = np.log(df["omega"])
    df["log_kappa"] = np.log(df["kappa"])
    df["omega_z"] = zscore(df["log_omega"])
    df["kappa_z"] = zscore(df["log_kappa"])

    rows = []
    for channel in ("anxiety", "confidence"):
        sub = df[df["questionLabel"] == channel].copy()
        if len(sub) == 0:
            continue
        n_obs = len(sub)
        n_subj = sub["subj"].nunique()
        print(f"\n  --- {channel} | N obs = {n_obs}, N subj = {n_subj} ---")

        row: dict = {"sample": sample_name, "channel": channel,
                     "n_obs": n_obs, "n_subj": n_subj}

        # Test A: incremental S(u*) above T + D
        print("\n  TEST A — anxiety ~ T_z + D_z + S(u*)_z + (1|subj)")
        m_A = fit_lmm("response ~ T_z + D_z + S_probe_z", sub)
        for term in ["T_z", "D_z", "S_probe_z", "Intercept"]:
            row.update({f"A_{k}": v for k, v in extract_coef(m_A, term).items()})
        row["A_loglik"] = float(m_A.llf)
        row["A_aic"] = float(m_A.aic)
        for term in ["T_z", "D_z", "S_probe_z"]:
            coef = extract_coef(m_A, term)
            print(f"    β({term}) = {coef[f'{term}_beta']:+.4f}  z = {coef[f'{term}_z']:+.2f}  p = {coef[f'{term}_p']:.3g}")

        # Test B: model comparison — single-predictor models
        print("\n  TEST B — model comparison: S(u*) vs (T + D)")
        m_S = fit_lmm("response ~ S_probe_z", sub)
        m_TD = fit_lmm("response ~ T_z + D_z", sub)
        row["B_S_loglik"] = float(m_S.llf)
        row["B_S_aic"] = float(m_S.aic)
        row["B_TD_loglik"] = float(m_TD.llf)
        row["B_TD_aic"] = float(m_TD.aic)
        row["B_delta_aic_S_minus_TD"] = row["B_S_aic"] - row["B_TD_aic"]
        row["B_delta_loglik_S_minus_TD"] = row["B_S_loglik"] - row["B_TD_loglik"]
        print(f"    M_S  : logL = {row['B_S_loglik']:.2f},  AIC = {row['B_S_aic']:.2f}")
        print(f"    M_TD : logL = {row['B_TD_loglik']:.2f},  AIC = {row['B_TD_aic']:.2f}")
        print(f"    ΔAIC (S − TD) = {row['B_delta_aic_S_minus_TD']:+.2f}  ({'S wins' if row['B_delta_aic_S_minus_TD'] < 0 else 'TD wins'})")
        print(f"    ΔlogL (S − TD) = {row['B_delta_loglik_S_minus_TD']:+.2f}")

        # Test C: between-subject (ω, κ) at fixed (T, D)
        print("\n  TEST C — anxiety ~ T_z + D_z + omega_z + kappa_z + (1|subj)")
        m_C = fit_lmm("response ~ T_z + D_z + omega_z + kappa_z", sub)
        for term in ["T_z", "D_z", "omega_z", "kappa_z", "Intercept"]:
            row.update({f"C_{k}": v for k, v in extract_coef(m_C, term).items()})
        row["C_loglik"] = float(m_C.llf)
        row["C_aic"] = float(m_C.aic)
        for term in ["T_z", "D_z", "omega_z", "kappa_z"]:
            coef = extract_coef(m_C, term)
            print(f"    β({term}) = {coef[f'{term}_beta']:+.4f}  z = {coef[f'{term}_z']:+.2f}  p = {coef[f'{term}_p']:.3g}")

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    out_dir = REPO_ROOT / "results" / "stats" / "affect_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name, paths in SAMPLES.items():
        res = fit_sample(name, paths)
        path = out_dir / f"embodied_tests_{name}.csv"
        res.to_csv(path, index=False)
        print(f"\nSaved: {path}")
        all_rows.append(res)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_dir / "embodied_tests_summary.csv", index=False)
    print(f"\nSaved: {out_dir / 'embodied_tests_summary.csv'}")

    # Print headline summary
    print("\n" + "=" * 70)
    print("HEADLINE SUMMARY")
    print("=" * 70)
    cols = ["sample", "channel",
            "A_T_z_beta", "A_D_z_beta", "A_S_probe_z_beta",
            "B_delta_aic_S_minus_TD",
            "C_omega_z_beta", "C_kappa_z_beta"]
    print(combined[cols].to_string(index=False))


if __name__ == "__main__":
    main()
