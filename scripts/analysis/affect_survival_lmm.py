"""
Trial-level affect ~ S_probe LMM — result_501.

For each probe trial, compute the M4 model-derived survival probability
S_probe under the subject's optimal pressing rate u*, then fit two
mixed-effects models:

    anxiety_response    ~ S_probe_z + (1 | subj)
    confidence_response ~ S_probe_z + (1 | subj)

S_probe and u* are computed from each subject's fitted (ω, κ) and the
population (γ, h, σ_sp) from M4's posterior means, exactly as in
scripts/mcmc/run_model_comparison_mcmc.py's evaluate_fit for M4.

Outputs (per sample):
    results/stats/affect_analysis/s_probe_affect_lmm_<sample>.csv
        — coefficients, z, p, n_obs, n_subj for each (channel, sample)

Run on both samples sequentially; print a unified summary.
"""

import argparse
from pathlib import Path

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


def m4_population_params(conv_path):
    """Extract M4 population params (gamma, hazard, sp) from convergence CSV."""
    conv = pd.read_csv(conv_path)
    m4 = conv[conv["model"] == "M4"].set_index("parameter")["mean"]
    gamma_v = float(np.clip(np.exp(m4["gr"]), 0.1, 3.0))
    hazard_v = float(np.exp(m4["hr"]))
    sp_v = float(np.clip(np.exp(m4["spr"]), 0.01, 1.0))
    return gamma_v, hazard_v, sp_v


def compute_s_probe(df, gamma_v, hazard_v, sp_v):
    """Vectorized: argmax_u W(u) per probe, then S_probe = S(u*, T, D)."""
    T = df["T"].to_numpy()
    D = df["D_model"].to_numpy()
    R = df["R"].to_numpy()
    req = df["req"].to_numpy()
    om = df["omega"].to_numpy()
    ka = df["kappa"].to_numpy()

    n = len(df)
    ug = UGRID[None, :]                                 # (1, 40)
    req_b = req[:, None]                                # (n, 1)
    speed = expit((ug - 0.25 * req_b) / sp_v)           # (n, 40)
    Tg = (T ** gamma_v)[:, None]                        # (n, 1)
    D_b = D[:, None].astype(float)
    S = np.exp(-hazard_v * Tg * D_b / np.clip(speed, 0.01, None))   # (n, 40)
    W = (S * R[:, None]
         - (1.0 - S) * om[:, None] * (R[:, None] + C_PENALTY)
         - ka[:, None] * (ug - req_b) ** 2 * D_b)        # (n, 40)
    idx_star = W.argmax(axis=1)                         # (n,)
    return S[np.arange(n), idx_star]                    # (n,)


def fit_sample(sample_name, paths):
    print("=" * 70)
    print(f"Sample: {sample_name}")
    print("=" * 70)
    feelings = pd.read_csv(Path(paths["stage5"]) / "feelings.csv", low_memory=False)
    m4 = pd.read_csv(paths["m4_params"])
    gamma_v, hazard_v, sp_v = m4_population_params(paths["m4_conv"])
    print(f"  Population params: gamma={gamma_v:.4f}, hazard={hazard_v:.4f}, sp={sp_v:.4f}")
    print(f"  Feelings: {len(feelings)} rows, {feelings['subj'].nunique()} subjects")
    print(f"  M4 per-subj: {len(m4)} subjects")

    df = feelings.merge(m4, on="subj", how="inner")
    # Map task conditions to model conditions
    df["T"] = df["threat"].astype(float)
    df["D_model"] = (df["distance"].astype(int) + 1)  # {0,1,2} → {1,2,3}
    df["is_heavy"] = (df["trialCookie_rewardValue"] == 5.0).astype(int)
    df["R"] = df["trialCookie_rewardValue"].astype(float)
    df["req"] = np.where(df["is_heavy"] == 1, 0.9, 0.4)

    # Drop rows missing the response (shouldn't happen but safe)
    df = df.dropna(subset=["response"]).copy()
    print(f"  Merged rows after dropna: {len(df)}, {df['subj'].nunique()} subjects")

    df["S_probe"] = compute_s_probe(df, gamma_v, hazard_v, sp_v)
    df["S_probe_z"] = zscore(df["S_probe"])

    # Fit two LMMs
    rows = []
    for channel in ("anxiety", "confidence"):
        sub = df[df["questionLabel"] == channel].copy()
        n_obs = len(sub)
        n_subj = sub["subj"].nunique()
        print(f"\n  --- {channel} ---")
        print(f"  N obs = {n_obs}, N subj = {n_subj}")
        print(f"  S_probe summary: mean={sub['S_probe'].mean():.3f}, "
              f"sd={sub['S_probe'].std():.3f}, "
              f"min={sub['S_probe'].min():.3f}, max={sub['S_probe'].max():.3f}")

        model = smf.mixedlm(
            "response ~ S_probe_z",
            data=sub,
            groups=sub["subj"],
        ).fit(reml=False)

        beta = float(model.fe_params["S_probe_z"])
        se = float(model.bse_fe["S_probe_z"])
        z = float(model.tvalues["S_probe_z"])
        p = float(model.pvalues["S_probe_z"])
        intercept = float(model.fe_params["Intercept"])

        print(f"  Intercept = {intercept:.4f}")
        print(f"  β(S_probe_z) = {beta:.4f} (SE {se:.4f}, z = {z:.3f}, p = {p:.4e})")

        rows.append({
            "sample": sample_name,
            "channel": channel,
            "n_obs": n_obs,
            "n_subj": n_subj,
            "intercept": intercept,
            "beta": beta,
            "se": se,
            "z": z,
            "p": p,
            "s_probe_mean": float(sub["S_probe"].mean()),
            "s_probe_sd": float(sub["S_probe"].std()),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="exploratory,confirmatory")
    parser.add_argument("--out-dir", default="results/stats/affect_analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for name in args.samples.split(","):
        name = name.strip()
        if name not in SAMPLES:
            print(f"Unknown sample: {name}")
            continue
        res = fit_sample(name, SAMPLES[name])
        out_path = out_dir / f"s_probe_affect_lmm_{name}.csv"
        res.to_csv(out_path, index=False)
        print(f"\nSaved {out_path}")
        all_rows.append(res)

    combined = pd.concat(all_rows, ignore_index=True)
    print("\n" + "=" * 70)
    print("SUMMARY — affect ~ S_probe (z-scored) + (1|subj), per channel × sample")
    print("=" * 70)
    print(combined[["sample", "channel", "n_obs", "n_subj", "intercept",
                    "beta", "se", "z", "p"]].to_string(index=False))


if __name__ == "__main__":
    main()
