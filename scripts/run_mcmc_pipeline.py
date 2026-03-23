"""
run_mcmc_pipeline.py
====================
Full MCMC pipeline for the effort-foraging-under-threat paper.

Three stages:
  1. Choice MCMC → λ, k_i, β_i (proper posteriors)
  2. Vigor MCMC  → α_i, δ_i (λ fixed from choice)
  3. Joint MCMC  → correlated [log(k), log(β), α, δ] with LKJ prior (robustness)

Usage:
  # GPU (recommended):
  python3 scripts/run_mcmc_pipeline.py --platform gpu

  # CPU (slow but works):
  python3 scripts/run_mcmc_pipeline.py --platform cpu --num-warmup 500 --num-samples 500

  # Skip stages:
  python3 scripts/run_mcmc_pipeline.py --platform gpu --stages 1,2    # choice + vigor only
  python3 scripts/run_mcmc_pipeline.py --platform gpu --stages 3      # joint only (uses saved λ)
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random

# ============================================================
# Parse arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--platform", default="cuda", choices=["cpu", "cuda", "gpu"])
parser.add_argument("--num-chains", type=int, default=4)
parser.add_argument("--num-warmup", type=int, default=1000)
parser.add_argument("--num-samples", type=int, default=1000)
parser.add_argument("--max-tree-depth", type=int, default=10)
parser.add_argument("--target-accept", type=float, default=0.90)
parser.add_argument("--stages", default="1,2,3", help="Comma-separated stages to run")
args = parser.parse_args()

STAGES = [int(s) for s in args.stages.split(",")]

platform = "cuda" if args.platform == "gpu" else args.platform
numpyro.set_platform(platform)
if platform == "cuda":
    numpyro.enable_x64(False)  # float32 on GPU is fine

print("=" * 60)
print("MCMC Pipeline — Effort Foraging Under Threat")
print("=" * 60)
print(f"Platform:      {args.platform}")
print(f"Devices:       {jax.devices()}")
print(f"Chains:        {args.num_chains}")
print(f"Warmup:        {args.num_warmup}")
print(f"Samples:       {args.num_samples}")
print(f"Tree depth:    {args.max_tree_depth}")
print(f"Target accept: {args.target_accept}")
print(f"Stages:        {STAGES}")
print()

# ============================================================
# Paths
# ============================================================
ROOT = Path("/workspace")
STAGE5 = ROOT / "data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
VIGOR_PROC = ROOT / "data/exploratory_350/processed/vigor_processed"
STAT_DIR = ROOT / "results/stats"
MCMC_DIR = ROOT / "results/model_fits/exploratory"
STAT_DIR.mkdir(parents=True, exist_ok=True)
MCMC_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load and prep data
# ============================================================
print("Loading data...", flush=True)
beh = pd.read_csv(STAGE5 / "behavior.csv")
feelings = pd.read_csv(STAGE5 / "feelings.csv")

# Trial-level vigor
ts = pd.read_parquet(VIGOR_PROC / "smoothed_vigor_ts.parquet")
vigor_trial = ts.groupby(["subj", "trial"])["vigor_norm"].mean().reset_index()
vigor_trial.columns = ["subj", "global_trial", "vigor_norm"]

# Remove probes
probe_per_subj = feelings.groupby("subj")["trialNumber"].apply(set).reset_index()
probe_per_subj.columns = ["subj", "probe_set"]
vigor_trial = vigor_trial.merge(probe_per_subj, on="subj", how="left")
vigor_trial["is_probe"] = vigor_trial.apply(
    lambda r: r["global_trial"] in r["probe_set"], axis=1
)
vb = vigor_trial[~vigor_trial["is_probe"]].copy().drop(columns=["probe_set", "is_probe"])
vb = vb.sort_values(["subj", "global_trial"])
vb["trial"] = vb.groupby("subj").cumcount() + 1

merged = beh.merge(vb[["subj", "trial", "vigor_norm"]], on=["subj", "trial"], how="inner")
merged["effort_chosen"] = merged["choice"] * merged["effort_H"] + (1 - merged["choice"]) * merged["effort_L"]
merged["excess"] = merged["vigor_norm"] - merged["effort_chosen"]

subj_ids = sorted(merged["subj"].unique())
subj_map = {s: i for i, s in enumerate(subj_ids)}
merged["subj_idx"] = merged["subj"].map(subj_map)
N_subj = len(subj_ids)

# JAX arrays
si = jnp.array(merged["subj_idx"].values, dtype=jnp.int32)
T_arr = jnp.array(merged["threat"].values, dtype=jnp.float32)
dH = jnp.array(merged["distance_H"].values, dtype=jnp.float32)
dL = jnp.array(merged["distance_L"].values, dtype=jnp.float32)
eH = jnp.array(merged["effort_H"].values, dtype=jnp.float32)
eL = jnp.array(merged["effort_L"].values, dtype=jnp.float32)
ch = jnp.array(merged["choice"].values, dtype=jnp.float32)
dc = jnp.array(
    (merged["choice"] * merged["distance_H"] + (1 - merged["choice"]) * merged["distance_L"]).values,
    dtype=jnp.float32,
)
ex = jnp.array(merged["excess"].values, dtype=jnp.float32)

print(f"  N = {N_subj} subjects, {len(merged):,} trials")


# ============================================================
# Model definitions
# ============================================================
def choice_model(si, T, dH, dL, eH, eL, ch=None):
    """L3_add: additive effort, hyperbolic survival, option-specific S."""
    tau = numpyro.sample("tau", dist.LogNormal(0, 1))
    lam = numpyro.sample("lam", dist.LogNormal(0, 1))
    mu_logk = numpyro.sample("mu_logk", dist.Normal(0, 1))
    sd_logk = numpyro.sample("sd_logk", dist.HalfNormal(1))
    mu_logb = numpyro.sample("mu_logb", dist.Normal(0, 1))
    sd_logb = numpyro.sample("sd_logb", dist.HalfNormal(1))
    with numpyro.plate("subj", N_subj):
        logk = numpyro.sample("logk", dist.Normal(mu_logk, sd_logk))
        logb = numpyro.sample("logb", dist.Normal(mu_logb, sd_logb))
    k = jnp.exp(logk[si])
    beta = jnp.exp(logb[si])
    SH = (1 - T) + T / (1.0 + lam * dH)
    SL = (1 - T) + T / (1.0 + lam * dL)
    SVH = 5.0 * SH - k * eH - beta * (1 - SH)
    SVL = 1.0 * SL - k * eL - beta * (1 - SL)
    numpyro.sample("ch", dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def vigor_model(si, danger, excess=None):
    """Hierarchical vigor: excess = α_i + δ_i · danger + ε."""
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0, 1))
    sd_alpha = numpyro.sample("sd_alpha", dist.HalfNormal(1))
    mu_delta = numpyro.sample("mu_delta", dist.Normal(0, 1))
    sd_delta = numpyro.sample("sd_delta", dist.HalfNormal(1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    with numpyro.plate("subj", N_subj):
        alpha = numpyro.sample("alpha", dist.Normal(mu_alpha, sd_alpha))
        delta = numpyro.sample("delta", dist.Normal(mu_delta, sd_delta))
    mu = alpha[si] + delta[si] * danger
    numpyro.sample("excess", dist.Normal(mu, sigma), obs=excess)


N_PARAMS = 4


def joint_model(si, T, dH, dL, dc, eH, eL, ex, ch, lam_fixed=None):
    """Joint choice + vigor with correlated random effects."""
    if lam_fixed is not None:
        lam = lam_fixed
    else:
        lam = numpyro.sample("lam", dist.LogNormal(0, 1))
    tau = numpyro.sample("tau", dist.LogNormal(0, 1))
    sigma_excess = numpyro.sample("sigma_excess", dist.HalfNormal(1))

    mu_logk = numpyro.sample("mu_logk", dist.Normal(0, 1))
    mu_logbeta = numpyro.sample("mu_logbeta", dist.Normal(0, 1))
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0, 1))
    mu_delta = numpyro.sample("mu_delta", dist.Normal(0, 1))
    mu = jnp.stack([mu_logk, mu_logbeta, mu_alpha, mu_delta])

    sigma_logk = numpyro.sample("sigma_logk", dist.HalfNormal(1))
    sigma_logbeta = numpyro.sample("sigma_logbeta", dist.HalfNormal(1))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1))
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(1))
    sigma_vec = jnp.stack([sigma_logk, sigma_logbeta, sigma_alpha, sigma_delta])

    L_omega = numpyro.sample("L_omega", dist.LKJCholesky(N_PARAMS, concentration=2.0))
    L_sigma = jnp.diag(sigma_vec) @ L_omega

    with numpyro.plate("subjects", N_subj):
        z_raw = numpyro.sample(
            "z_raw",
            dist.MultivariateNormal(jnp.zeros(N_PARAMS), scale_tril=jnp.eye(N_PARAMS)),
        )
    theta = mu[None, :] + z_raw @ L_sigma.T
    k_i = jnp.exp(theta[:, 0])
    beta_i = jnp.exp(theta[:, 1])
    alpha_i = theta[:, 2]
    delta_i = theta[:, 3]

    k_t = k_i[si]; beta_t = beta_i[si]
    alpha_t = alpha_i[si]; delta_t = delta_i[si]

    S_H = (1.0 - T) + T / (1.0 + lam * dH)
    S_L = (1.0 - T) + T / (1.0 + lam * dL)
    SV_H = 5.0 * S_H - k_t * eH - beta_t * (1.0 - S_H)
    SV_L = 1.0 * S_L - k_t * eL - beta_t * (1.0 - S_L)
    numpyro.sample("choice", dist.Bernoulli(logits=tau * (SV_H - SV_L)), obs=ch)

    S_chosen = (1.0 - T) + T / (1.0 + lam * dc)
    mu_ex = alpha_t + delta_t * (1.0 - S_chosen)
    numpyro.sample("excess", dist.Normal(mu_ex, sigma_excess), obs=ex)


def run_mcmc(model, model_args, model_kwargs, name, rng_seed=0):
    """Run NUTS MCMC and return the MCMC object."""
    print(f"\n{'=' * 60}")
    print(f"Running MCMC: {name}")
    print(f"{'=' * 60}", flush=True)

    kernel = NUTS(
        model,
        max_tree_depth=args.max_tree_depth,
        target_accept_prob=args.target_accept,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method="parallel" if platform == "cuda" else "sequential",
        progress_bar=True,
    )

    rng_key = random.PRNGKey(rng_seed)
    t0 = time.time()
    mcmc.run(rng_key, *model_args, **model_kwargs)
    elapsed = time.time() - t0

    print(f"\n  Elapsed: {elapsed / 60:.1f} min")
    mcmc.print_summary(exclude_deterministic=True)
    return mcmc, elapsed


def print_diagnostics(mcmc, param_names=None):
    """Print convergence diagnostics."""
    samples = mcmc.get_samples()
    summary = {}
    for name, vals in samples.items():
        if param_names and name not in param_names:
            continue
        if vals.ndim == 1:
            summary[name] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "n_eff": float(numpyro.diagnostics.effective_sample_size(vals[None, :])),
            }
    if summary:
        print("\n  Key parameters:")
        for name, s in summary.items():
            print(f"    {name:<16s}: {s['mean']:+8.4f} (±{s['std']:.4f}), ESS={s['n_eff']:.0f}")


# ============================================================
# STAGE 1: Choice MCMC
# ============================================================
if 1 in STAGES:
    choice_data = (si, T_arr, dH, dL, eH, eL)
    choice_kwargs = {"ch": ch}

    mcmc_choice, t_choice = run_mcmc(
        choice_model, choice_data, choice_kwargs, "Choice Model (L3_add)", rng_seed=42
    )

    samples_c = mcmc_choice.get_samples()
    lam_mcmc = np.array(samples_c["lam"])
    lam_mean = float(lam_mcmc.mean())
    lam_sd = float(lam_mcmc.std())
    print(f"\n  ★ λ = {lam_mean:.2f} (±{lam_sd:.2f})")

    # Save
    logk_post = np.array(samples_c["logk"])
    logb_post = np.array(samples_c["logb"])
    k_mcmc = np.exp(logk_post).mean(axis=0)
    beta_mcmc = np.exp(logb_post).mean(axis=0)

    choice_params = pd.DataFrame({
        "subj": subj_ids,
        "k_mcmc": k_mcmc,
        "k_mcmc_sd": np.exp(logk_post).std(axis=0),
        "beta_mcmc": beta_mcmc,
        "beta_mcmc_sd": np.exp(logb_post).std(axis=0),
        "logk_mcmc": logk_post.mean(axis=0),
        "logb_mcmc": logb_post.mean(axis=0),
    })
    choice_params.to_csv(STAT_DIR / "mcmc_choice_params.csv", index=False)

    pop_choice = pd.DataFrame([{
        "lam": lam_mean, "lam_sd": lam_sd,
        "lam_2.5": float(np.percentile(lam_mcmc, 2.5)),
        "lam_97.5": float(np.percentile(lam_mcmc, 97.5)),
        "tau": float(np.array(samples_c["tau"]).mean()),
        "mu_logk": float(np.array(samples_c["mu_logk"]).mean()),
        "mu_logb": float(np.array(samples_c["mu_logb"]).mean()),
        "elapsed_min": t_choice / 60,
    }])
    pop_choice.to_csv(STAT_DIR / "mcmc_choice_population.csv", index=False)
    print(f"  Saved: mcmc_choice_params.csv, mcmc_choice_population.csv")

else:
    # Load saved λ
    pop_choice = pd.read_csv(STAT_DIR / "mcmc_choice_population.csv")
    lam_mean = float(pop_choice["lam"].values[0])
    choice_params = pd.read_csv(STAT_DIR / "mcmc_choice_params.csv")
    print(f"  Loaded saved λ = {lam_mean:.2f}")


# ============================================================
# STAGE 2: Vigor MCMC (λ fixed from choice)
# ============================================================
if 2 in STAGES:
    S_chosen = (1.0 - T_arr) + T_arr / (1.0 + lam_mean * dc)
    danger = 1.0 - S_chosen

    vigor_data = (si, danger)
    vigor_kwargs = {"excess": ex}

    mcmc_vigor, t_vigor = run_mcmc(
        vigor_model, vigor_data, vigor_kwargs,
        f"Vigor Model (λ={lam_mean:.1f} from choice)", rng_seed=43
    )

    samples_v = mcmc_vigor.get_samples()
    alpha_post = np.array(samples_v["alpha"])
    delta_post = np.array(samples_v["delta"])

    alpha_mcmc = alpha_post.mean(axis=0)
    delta_mcmc = delta_post.mean(axis=0)

    mu_delta = np.array(samples_v["mu_delta"])
    sd_delta = np.array(samples_v["sd_delta"])
    print(f"\n  ★ μ_δ = {mu_delta.mean():.4f} (±{mu_delta.std():.4f})")
    print(f"  ★ σ_δ = {sd_delta.mean():.4f}")
    print(f"  ★ P(μ_δ > 0) = {(mu_delta > 0).mean():.4f}")
    print(f"  ★ % subjects δ > 0: {(delta_mcmc > 0).mean() * 100:.1f}%")

    vigor_params = pd.DataFrame({
        "subj": subj_ids,
        "alpha_mcmc": alpha_mcmc,
        "alpha_mcmc_sd": alpha_post.std(axis=0),
        "delta_mcmc": delta_mcmc,
        "delta_mcmc_sd": delta_post.std(axis=0),
    })
    vigor_params.to_csv(STAT_DIR / "mcmc_vigor_params.csv", index=False)

    pop_vigor = pd.DataFrame([{
        "mu_delta": float(mu_delta.mean()),
        "mu_delta_sd": float(mu_delta.std()),
        "mu_delta_2.5": float(np.percentile(mu_delta, 2.5)),
        "mu_delta_97.5": float(np.percentile(mu_delta, 97.5)),
        "sd_delta": float(sd_delta.mean()),
        "mu_alpha": float(np.array(samples_v["mu_alpha"]).mean()),
        "pct_delta_pos": float((delta_mcmc > 0).mean()),
        "elapsed_min": t_vigor / 60,
    }])
    pop_vigor.to_csv(STAT_DIR / "mcmc_vigor_population.csv", index=False)
    print(f"  Saved: mcmc_vigor_params.csv, mcmc_vigor_population.csv")

    # Cross-domain correlations
    print(f"\n  Cross-domain correlations (independent MCMC):")
    pairs = [
        ("log(β)", choice_params["logb_mcmc"].values, "δ", delta_mcmc),
        ("log(k)", choice_params["logk_mcmc"].values, "δ", delta_mcmc),
        ("log(k)", choice_params["logk_mcmc"].values, "α", alpha_mcmc),
        ("log(β)", choice_params["logb_mcmc"].values, "α", alpha_mcmc),
        ("α", alpha_mcmc, "δ", delta_mcmc),
    ]
    cross_results = []
    for n1, v1, n2, v2 in pairs:
        r, p = stats.pearsonr(v1, v2)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {n1 + ' × ' + n2:<16s}: r={r:+.3f}, p={p:.2e} {sig}")
        cross_results.append({"param_1": n1, "param_2": n2, "r": round(r, 4), "p": p})
    pd.DataFrame(cross_results).to_csv(STAT_DIR / "mcmc_cross_correlations.csv", index=False)

else:
    vigor_params = pd.read_csv(STAT_DIR / "mcmc_vigor_params.csv")
    print(f"  Loaded saved vigor params")


# ============================================================
# STAGE 3: Joint MCMC (correlated random effects)
# ============================================================
if 3 in STAGES:
    from functools import partial

    joint_fn = partial(joint_model, lam_fixed=lam_mean)
    joint_data = (si, T_arr, dH, dL, dc, eH, eL, ex, ch)

    mcmc_joint, t_joint = run_mcmc(
        joint_fn, joint_data, {},
        f"Joint Model (λ={lam_mean:.1f} fixed, LKJ correlated)",
        rng_seed=44,
    )

    samples_j = mcmc_joint.get_samples()

    # Extract correlation matrix
    L_omega = np.array(samples_j["L_omega"])
    omega = np.einsum("...ij,...kj->...ik", L_omega, L_omega)

    NAMES = ["log_k", "log_beta", "alpha", "delta"]
    print(f"\n  LKJ Posterior Correlations:")
    print(f"  {'Pair':<22s} {'ρ':>7s} {'95% CI':>16s} {'P(>0)':>7s} {'ESS':>7s}")
    print("  " + "-" * 62)

    joint_corr_rows = []
    for i in range(N_PARAMS):
        for j in range(i + 1, N_PARAMS):
            rho = omega[:, i, j]
            m = float(rho.mean())
            lo = float(np.percentile(rho, 2.5))
            hi = float(np.percentile(rho, 97.5))
            p_pos = float((rho > 0).mean())
            ess = float(numpyro.diagnostics.effective_sample_size(rho[None, :]))
            print(f"  {NAMES[i]:>8s} × {NAMES[j]:<8s}   {m:+7.3f} [{lo:+6.3f}, {hi:+6.3f}] {p_pos:7.3f} {ess:7.0f}")
            joint_corr_rows.append({
                "param_1": NAMES[i], "param_2": NAMES[j],
                "rho_mean": round(m, 4), "rho_sd": round(float(rho.std()), 4),
                "rho_2.5": round(lo, 4), "rho_97.5": round(hi, 4),
                "P_positive": round(p_pos, 4), "ESS": round(ess, 0),
            })

    pd.DataFrame(joint_corr_rows).to_csv(STAT_DIR / "mcmc_joint_correlations.csv", index=False)

    # Population params
    mu_delta_j = np.array(samples_j["mu_delta"])
    sd_delta_j = np.array(samples_j["sigma_delta"])
    print(f"\n  μ_δ = {mu_delta_j.mean():.4f} (±{mu_delta_j.std():.4f})")
    print(f"  σ_δ = {sd_delta_j.mean():.4f}")

    # Rhat check
    print(f"\n  Convergence (max Rhat across all params):")
    all_rhat = []
    for name, vals in samples_j.items():
        if vals.ndim >= 1:
            rhat = numpyro.diagnostics.split_gelman_rubin(
                vals.reshape(args.num_chains, -1, *vals.shape[1:])
            )
            if np.isscalar(rhat):
                all_rhat.append(rhat)
            else:
                all_rhat.append(float(np.max(rhat)))
    max_rhat = max(all_rhat) if all_rhat else float("nan")
    print(f"    Max Rhat = {max_rhat:.4f} ({'GOOD' if max_rhat < 1.05 else 'WARNING'})")

    pop_joint = pd.DataFrame([{
        "mu_delta": float(mu_delta_j.mean()),
        "sigma_delta": float(sd_delta_j.mean()),
        "max_rhat": max_rhat,
        "elapsed_min": t_joint / 60,
        "lam_fixed": lam_mean,
    }])
    pop_joint.to_csv(STAT_DIR / "mcmc_joint_population.csv", index=False)
    print(f"\n  Saved: mcmc_joint_correlations.csv, mcmc_joint_population.csv")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)

if 1 in STAGES:
    print(f"  Choice: λ = {lam_mean:.2f} (±{lam_sd:.2f}), {t_choice/60:.1f} min")
if 2 in STAGES:
    print(f"  Vigor:  μ_δ = {mu_delta.mean():.4f}, {t_vigor/60:.1f} min")
    bd = [r for r in cross_results if r["param_1"] == "log(β)" and r["param_2"] == "δ"]
    if bd:
        print(f"  ★ Independent MCMC: log(β) × δ = {bd[0]['r']:+.3f}")
if 3 in STAGES:
    bd_j = [r for r in joint_corr_rows if r["param_1"] == "log_beta" and r["param_2"] == "delta"]
    if bd_j:
        print(f"  ★ Joint MCMC: ρ(β, δ) = {bd_j[0]['rho_mean']:+.3f} [{bd_j[0]['rho_2.5']:+.3f}, {bd_j[0]['rho_97.5']:+.3f}]")
    print(f"  Joint: {t_joint/60:.1f} min, max Rhat = {max_rhat:.4f}")

print("\nOutputs in results/stats/mcmc_*.csv")
