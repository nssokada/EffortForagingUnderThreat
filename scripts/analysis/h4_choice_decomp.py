"""
H4 follow-up: add P(heavy) to the polar/Cartesian decomposition that 208
already runs on escape_rate, mean_vigor, and pct_opt. Computes the
parameter-mediated prediction for r(choice, vigor) and compares to the
observed marginal correlation — the test that anchors the embodied W(u)
framing for [[result_401]].

Mirrors the conventions in notebooks/analysis/H4_profiles_optimality.ipynb
(cell at lines ~230-275): bambi with BKW (4 chains × 2,000 draws + 1,000
tuning), log-transform-then-z-score on ω and κ, polar = (angle, magnitude),
Cartesian = (ω, κ), both with the interaction term.

Outputs:
  results/stats/individual_diffs/h4_choice_decomp.csv      — full coefficient table
  results/stats/individual_diffs/h4_predicted_r_cv.csv     — predicted vs observed r(choice, vigor)
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Make notebook helpers importable and chdir to repo root (config.py sets REPO_ROOT = cwd)
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


OUTCOMES_CHOICE = ["p_heavy"]
OUTCOMES_VIGOR = ["mean_vigor"]


def fit_decomp(master: pd.DataFrame, outcome: str) -> dict:
    """Fit polar + Cartesian decomposition (with interaction) AND main-effects-only
    Cartesian regression, mirroring notebook conventions."""
    df = master[[outcome, "omega_z", "kappa_z", "angle_z"]].dropna().copy()
    df["mag"] = np.sqrt(df["omega_z"] ** 2 + df["kappa_z"] ** 2)
    df["mag_z"] = zscore(df["mag"].values, nan_policy="omit")
    df["ang_x_mag"] = df["angle_z"] * df["mag_z"]
    df["om_x_kap"] = df["omega_z"] * df["kappa_z"]

    out: dict = {"n": len(df)}

    # Polar (with interaction) — for 208 table style
    m_pol = bmb.Model(f"{outcome} ~ angle_z + mag_z + ang_x_mag", data=df)
    s_pol = az.summary(m_pol.fit(**BKW), hdi_prob=0.95)
    out["polar"] = s_pol

    # Cartesian (with interaction) — for 208 table style
    m_cart = bmb.Model(f"{outcome} ~ omega_z + kappa_z + om_x_kap", data=df)
    s_cart = az.summary(m_cart.fit(**BKW), hdi_prob=0.95)
    out["cartesian"] = s_cart

    # Cartesian main-effects-only — clean partial slopes for the predicted-r calculation
    m_main = bmb.Model(f"{outcome} ~ omega_z + kappa_z", data=df)
    s_main = az.summary(m_main.fit(**BKW), hdi_prob=0.95)
    out["main_effects"] = s_main

    return out


def fmt(s: pd.DataFrame, term: str) -> str:
    return (
        f"b={s.loc[term, 'mean']:+.4f} "
        f"[{s.loc[term, 'hdi_2.5%']:+.4f}, {s.loc[term, 'hdi_97.5%']:+.4f}]  "
        f"Rhat={s.loc[term, 'r_hat']:.3f}  ESS={int(s.loc[term, 'ess_bulk'])}"
    )


def record_rows(records: list, sample: str, outcome: str, parameterization: str, summary: pd.DataFrame, terms: list[str]):
    for t in terms:
        records.append(
            dict(
                sample=sample,
                outcome=outcome,
                parameterization=parameterization,
                term=t,
                mean=summary.loc[t, "mean"],
                hdi_lo=summary.loc[t, "hdi_2.5%"],
                hdi_hi=summary.loc[t, "hdi_97.5%"],
                r_hat=summary.loc[t, "r_hat"],
                ess_bulk=int(summary.loc[t, "ess_bulk"]),
            )
        )


def predicted_r_choice_vigor(
    beta_oc: float, beta_kc: float,
    beta_ov: float, beta_kv: float,
    r_omega_kappa: float,
    sd_choice: float, sd_vigor: float,
) -> tuple[float, float]:
    """Compute parameter-mediated r(choice, vigor) from the four partial coefficients.

    Under choice = β_ωc·ω + β_κc·κ + ε_c and vigor = β_ωv·ω + β_κv·κ + ε_v with
    z-scored ω, κ (Var=1, Cov=r_omega_kappa):

        Cov_param(c, v) = β_ωc·β_ωv + β_κc·β_κv + r(ω,κ)·(β_ωc·β_κv + β_κc·β_ωv)
        r_param        = Cov_param / (SD(c) · SD(v))

    Returns (Cov_param, r_param).
    """
    cov_param = (
        beta_oc * beta_ov
        + beta_kc * beta_kv
        + r_omega_kappa * (beta_oc * beta_kv + beta_kc * beta_ov)
    )
    r_param = cov_param / (sd_choice * sd_vigor)
    return cov_param, r_param


def main():
    exp_data, conf_data = load_both()
    all_data = {"exploratory": exp_data, "confirmatory": conf_data}
    all_data = {k: v for k, v in all_data.items() if v is not None}

    records: list = []
    pred_records: list = []

    for sample_name, d in all_data.items():
        label = d["config"].label
        master = d["master"].copy()
        master["mag"] = np.sqrt(master["omega_z"] ** 2 + master["kappa_z"] ** 2)
        master["mag_z"] = zscore(master["mag"].values, nan_policy="omit")

        r_ok = master[["omega_z", "kappa_z"]].corr().iloc[0, 1]
        print(f"\n===== {label} =====")
        print(f"r(omega_z, kappa_z) = {r_ok:+.3f}")

        # Fit decomps for choice AND vigor (we need both for the predicted-r calculation)
        fits = {}
        for outcome in OUTCOMES_CHOICE + OUTCOMES_VIGOR:
            if outcome not in master.columns:
                print(f"  SKIP {outcome}: not in master")
                continue
            print(f"\n--- {outcome} ---")
            fits[outcome] = fit_decomp(master, outcome)
            s_pol = fits[outcome]["polar"]
            s_cart = fits[outcome]["cartesian"]
            s_main = fits[outcome]["main_effects"]

            print("  Polar (w/ interaction):")
            for t in ["angle_z", "mag_z", "ang_x_mag"]:
                print(f"    {t:14s} {fmt(s_pol, t)}")
            record_rows(records, sample_name, outcome, "polar", s_pol, ["angle_z", "mag_z", "ang_x_mag"])

            print("  Cartesian (w/ interaction):")
            for t in ["omega_z", "kappa_z", "om_x_kap"]:
                print(f"    {t:14s} {fmt(s_cart, t)}")
            record_rows(records, sample_name, outcome, "cartesian", s_cart, ["omega_z", "kappa_z", "om_x_kap"])

            print("  Cartesian main-effects-only (used for predicted-r):")
            for t in ["omega_z", "kappa_z"]:
                print(f"    {t:14s} {fmt(s_main, t)}")
            record_rows(records, sample_name, outcome, "cartesian_main", s_main, ["omega_z", "kappa_z"])

        # Predicted vs observed r(choice, vigor)
        if "p_heavy" in fits and "mean_vigor" in fits:
            s_c = fits["p_heavy"]["main_effects"]
            s_v = fits["mean_vigor"]["main_effects"]
            beta_oc = s_c.loc["omega_z", "mean"]
            beta_kc = s_c.loc["kappa_z", "mean"]
            beta_ov = s_v.loc["omega_z", "mean"]
            beta_kv = s_v.loc["kappa_z", "mean"]
            df_cv = master[["p_heavy", "mean_vigor"]].dropna()
            sd_c = df_cv["p_heavy"].std()
            sd_v = df_cv["mean_vigor"].std()
            r_obs, p_obs = pearsonr(df_cv["p_heavy"], df_cv["mean_vigor"])

            cov_p, r_p = predicted_r_choice_vigor(beta_oc, beta_kc, beta_ov, beta_kv, r_ok, sd_c, sd_v)

            # Decompose into pathway contributions
            cov_omega = beta_oc * beta_ov                           # ω-pathway contribution (z=1 variance)
            cov_kappa = beta_kc * beta_kv                           # κ-pathway contribution
            cov_cross = r_ok * (beta_oc * beta_kv + beta_kc * beta_ov)  # cross term

            print(f"\n--- Predicted r(choice, vigor) ---")
            print(f"  β_ωc = {beta_oc:+.4f},  β_κc = {beta_kc:+.4f}")
            print(f"  β_ωv = {beta_ov:+.4f},  β_κv = {beta_kv:+.4f}")
            print(f"  r(ω, κ) = {r_ok:+.3f}")
            print(f"  SD(choice) = {sd_c:.4f},  SD(vigor) = {sd_v:.4f}")
            print(f"  Cov_ω-pathway   = {cov_omega:+.5f}")
            print(f"  Cov_κ-pathway   = {cov_kappa:+.5f}")
            print(f"  Cov_cross       = {cov_cross:+.5f}")
            print(f"  Cov_param total = {cov_p:+.5f}")
            print(f"  r_predicted (parameter-mediated) = {r_p:+.4f}")
            print(f"  r_observed   (sample correlation) = {r_obs:+.4f}  (p={p_obs:.3g})")

            pred_records.append(dict(
                sample=sample_name,
                n=len(df_cv),
                r_omega_kappa=r_ok,
                beta_omega_choice=beta_oc,
                beta_kappa_choice=beta_kc,
                beta_omega_vigor=beta_ov,
                beta_kappa_vigor=beta_kv,
                sd_choice=sd_c,
                sd_vigor=sd_v,
                cov_omega_pathway=cov_omega,
                cov_kappa_pathway=cov_kappa,
                cov_cross_term=cov_cross,
                cov_param_total=cov_p,
                r_predicted=r_p,
                r_observed=r_obs,
                p_observed=p_obs,
            ))

    out_dir = REPO_ROOT / "results" / "stats" / "individual_diffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    decomp_csv = out_dir / "h4_choice_decomp.csv"
    pred_csv = out_dir / "h4_predicted_r_cv.csv"
    pd.DataFrame(records).to_csv(decomp_csv, index=False)
    pd.DataFrame(pred_records).to_csv(pred_csv, index=False)
    print(f"\nSaved: {decomp_csv}")
    print(f"Saved: {pred_csv}")


if __name__ == "__main__":
    main()
