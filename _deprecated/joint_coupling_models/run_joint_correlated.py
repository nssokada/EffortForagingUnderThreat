"""
run_joint_correlated.py
========================
Joint choice-vigor model with CORRELATED random effects.

The key difference from run_joint_choice_vigor_model.py:
  Subject parameters [log(k), log(β), α, δ] are drawn from a
  multivariate normal with a full covariance matrix. The off-diagonal
  elements directly estimate cross-domain coupling (e.g., β-δ correlation).

Model specification:
  Survival:  S = (1-T) + T / (1 + λ·D)
  Choice:    SV_H = R_H·S - k_i·E_H - β_i·(1-S)
             SV_L = R_L·S - k_i·E_L - β_i·(1-S)
             choice ~ Bernoulli(sigmoid(τ·(SV_H - SV_L)))
  Vigor:     excess_ij = α_i + δ_i·(1-S_ij) + ε_ij
             ε_ij ~ Normal(0, σ_excess)
  where excess_ij = vigor_norm_ij - effort_chosen_ij

Subject-level (correlated):
  θ_i = [log(k_i), log(β_i), α_i, δ_i] ~ MVN(μ, Σ)
  Σ = diag(σ) · Ω · diag(σ)
  Ω ~ LKJCholesky(η=2)   (correlation matrix)
  σ_j ~ HalfNormal(1.0)  (marginal SDs)

Population parameters: λ, τ, σ_excess

Fit: SVI with AutoMultivariateNormal guide, 30000 steps, Adam lr=0.002
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
import numpyro.optim as optim

numpyro.set_platform("cpu")

print("=" * 60)
print("Joint Choice-Vigor Model — Correlated Random Effects")
print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"NumPyro version: {numpyro.__version__}")
print(f"Devices: {jax.devices()}")
print()

# ============================================================
# Paths
# ============================================================
ROOT = Path("/workspace")
STAGE5 = ROOT / "data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
VIGOR_PROC = ROOT / "data/exploratory_350/processed/vigor_processed"
STAT_DIR = ROOT / "results/stats"
RESULTS_DIR = ROOT / "results"
STAT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Load data (same pipeline as run_joint_choice_vigor_model.py)
# ============================================================
print("Loading behavior.csv ...", flush=True)
beh = pd.read_csv(STAGE5 / "behavior.csv")
print(f"  {len(beh):,} trials, {beh['subj'].nunique()} subjects")

print("Loading smoothed_vigor_ts.parquet ...", flush=True)
ts = pd.read_parquet(VIGOR_PROC / "smoothed_vigor_ts.parquet")
print(f"  {len(ts):,} rows, {ts['subj'].nunique()} subjects")

# ============================================================
# 2. Compute trial-level mean vigor_norm
# ============================================================
print("\nComputing trial-level mean vigor_norm ...", flush=True)
vigor_trial = ts.groupby(["subj", "trial"])["vigor_norm"].mean().reset_index()
vigor_trial.columns = ["subj", "global_trial", "vigor_norm"]

# ============================================================
# 3. Remove probe trials, align to behavioral trial index
# ============================================================
feelings = pd.read_csv(STAGE5 / "feelings.csv")
probe_per_subj = feelings.groupby("subj")["trialNumber"].apply(set).reset_index()
probe_per_subj.columns = ["subj", "probe_set"]

vigor_trial = vigor_trial.merge(probe_per_subj, on="subj", how="left")
vigor_trial["is_probe"] = vigor_trial.apply(
    lambda r: r["global_trial"] in r["probe_set"], axis=1
)
vigor_beh = vigor_trial[~vigor_trial["is_probe"]].copy()
vigor_beh = vigor_beh.drop(columns=["probe_set", "is_probe"])
vigor_beh = vigor_beh.sort_values(["subj", "global_trial"])
vigor_beh["trial"] = vigor_beh.groupby("subj").cumcount() + 1
print(f"  Behavioral vigor: {len(vigor_beh):,} rows")

# ============================================================
# 4. Merge behavior + vigor
# ============================================================
print("\nMerging behavior and vigor ...", flush=True)
merged = beh.merge(
    vigor_beh[["subj", "trial", "vigor_norm"]],
    on=["subj", "trial"],
    how="inner"
)
print(f"  Merged: {len(merged):,} trials (of {len(beh):,} behavioral)")
print(f"  Subjects: {merged['subj'].nunique()}")

merged["effort_chosen"] = (
    merged["choice"] * merged["effort_H"] +
    (1 - merged["choice"]) * merged["effort_L"]
)
merged["excess"] = merged["vigor_norm"] - merged["effort_chosen"]
print(f"  Excess vigor: mean={merged['excess'].mean():.3f}, SD={merged['excess'].std():.3f}")

# ============================================================
# 5. Build arrays for JAX
# ============================================================
print("\nBuilding JAX arrays ...", flush=True)
subj_ids = sorted(merged["subj"].unique())
subj_map = {s: i for i, s in enumerate(subj_ids)}
merged["subj_idx"] = merged["subj"].map(subj_map)
N_subj = len(subj_ids)

subj_idx = jnp.array(merged["subj_idx"].values, dtype=jnp.int32)
threat = jnp.array(merged["threat"].values, dtype=jnp.float32)
dist_H = jnp.array(merged["distance_H"].values, dtype=jnp.float32)
dist_L = jnp.array(merged["distance_L"].values, dtype=jnp.float32)  # always 1
effort_H = jnp.array(merged["effort_H"].values, dtype=jnp.float32)
effort_L = jnp.array(merged["effort_L"].values, dtype=jnp.float32)
choice = jnp.array(merged["choice"].values, dtype=jnp.int32)
choice_f = jnp.array(merged["choice"].values, dtype=jnp.float32)
excess_obs = jnp.array(merged["excess"].values, dtype=jnp.float32)

# Distance of chosen option (for vigor S)
dist_chosen = jnp.array(
    (merged["choice"] * merged["distance_H"] + (1 - merged["choice"]) * merged["distance_L"]).values,
    dtype=jnp.float32
)

R_H = 5.0
R_L = 1.0

print(f"  N subjects: {N_subj}")
print(f"  N trials:   {len(merged):,}")
print(f"  distance_L unique: {sorted(merged['distance_L'].unique())}")
print(f"  distance_H unique: {sorted(merged['distance_H'].unique())}")

# ============================================================
# 6. Model definition — Correlated random effects
# ============================================================
N_PARAMS = 4  # [log(k), log(β), α, δ]
PARAM_NAMES = ["log_k", "log_beta", "alpha", "delta"]


def joint_correlated_model(subj_idx, threat, dist_H, dist_L, dist_chosen,
                           effort_H, effort_L, excess_obs, choice_obs, N_subj,
                           R_H=5.0, R_L=1.0, lam_fixed=None):
    """
    Joint choice-vigor with correlated subject-level parameters.
    θ_i = [log(k), log(β), α, δ] ~ MVN(μ, Σ)

    CRITICAL: S is option-specific in choice (S_H ≠ S_L when D_H ≠ D_L)
    so that β is identified. For vigor, S uses the chosen option's distance.

    If lam_fixed is provided, λ is fixed (from choice-only fit) to prevent
    the vigor likelihood from inflating λ. This preserves the "shared S"
    computation as defined by choice, which is where S is well-identified.
    """
    # ---- Population parameters ----
    if lam_fixed is not None:
        lam = lam_fixed
    else:
        lam = numpyro.sample("lambda", dist.LogNormal(0.0, 1.0))
    tau = numpyro.sample("tau", dist.LogNormal(0.0, 1.0))
    sigma_excess = numpyro.sample("sigma_excess", dist.HalfNormal(1.0))

    # ---- MVN hyperparameters ----
    # Means for [log(k), log(β), α, δ]
    mu_logk = numpyro.sample("mu_logk", dist.Normal(0.0, 1.0))
    mu_logbeta = numpyro.sample("mu_logbeta", dist.Normal(0.0, 1.0))
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0.0, 1.0))
    mu_delta = numpyro.sample("mu_delta", dist.Normal(0.0, 1.0))
    mu = jnp.stack([mu_logk, mu_logbeta, mu_alpha, mu_delta])

    # Marginal SDs
    sigma_logk = numpyro.sample("sigma_logk", dist.HalfNormal(1.0))
    sigma_logbeta = numpyro.sample("sigma_logbeta", dist.HalfNormal(1.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(1.0))
    sigma_vec = jnp.stack([sigma_logk, sigma_logbeta, sigma_alpha, sigma_delta])

    # Correlation matrix via LKJ Cholesky
    L_omega = numpyro.sample(
        "L_omega",
        dist.LKJCholesky(N_PARAMS, concentration=2.0)
    )
    # Scale to covariance Cholesky: L_Sigma = diag(sigma) @ L_omega
    L_sigma = jnp.diag(sigma_vec) @ L_omega

    # ---- Per-subject parameters (non-centered) ----
    with numpyro.plate("subjects", N_subj):
        z_raw = numpyro.sample(
            "z_raw",
            dist.MultivariateNormal(jnp.zeros(N_PARAMS), scale_tril=jnp.eye(N_PARAMS))
        )
    # θ_i = μ + L_Σ · z_raw_i
    theta = mu[None, :] + z_raw @ L_sigma.T  # (N_subj, 4)

    # Unpack and transform
    logk_i = theta[:, 0]
    logbeta_i = theta[:, 1]
    alpha_i = theta[:, 2]
    delta_i = theta[:, 3]

    k_i = jnp.exp(logk_i)      # k > 0
    beta_i = jnp.exp(logbeta_i)  # β > 0

    # ---- Index to trial level ----
    k_t = k_i[subj_idx]
    beta_t = beta_i[subj_idx]
    alpha_t = alpha_i[subj_idx]
    delta_t = delta_i[subj_idx]

    # ---- Option-specific survival (CRITICAL for β identification) ----
    S_H = (1.0 - threat) + threat / (1.0 + lam * dist_H)
    S_L = (1.0 - threat) + threat / (1.0 + lam * dist_L)

    # ---- Choice likelihood ----
    SV_H = R_H * S_H - k_t * effort_H - beta_t * (1.0 - S_H)
    SV_L = R_L * S_L - k_t * effort_L - beta_t * (1.0 - S_L)
    delta_SV = tau * (SV_H - SV_L)

    with numpyro.plate("trials_choice", len(choice_obs)):
        numpyro.sample("choice", dist.Bernoulli(logits=delta_SV), obs=choice_obs)

    # ---- Vigor likelihood (S uses chosen option's distance) ----
    S_chosen = (1.0 - threat) + threat / (1.0 + lam * dist_chosen)
    mu_excess = alpha_t + delta_t * (1.0 - S_chosen)

    with numpyro.plate("trials_vigor", len(excess_obs)):
        numpyro.sample("excess", dist.Normal(mu_excess, sigma_excess), obs=excess_obs)


# ============================================================
# 7. Choice-only model (for comparison)
# ============================================================
def choice_only_model(subj_idx, threat, dist_H, dist_L,
                      effort_H, effort_L, choice_obs, N_subj,
                      R_H=5.0, R_L=1.0):
    """Choice-only with correlated k, β. Option-specific S."""
    lam = numpyro.sample("lambda", dist.LogNormal(0.0, 1.0))
    tau = numpyro.sample("tau", dist.LogNormal(0.0, 1.0))

    mu_logk = numpyro.sample("mu_logk", dist.Normal(0.0, 1.0))
    mu_logbeta = numpyro.sample("mu_logbeta", dist.Normal(0.0, 1.0))
    mu = jnp.stack([mu_logk, mu_logbeta])

    sigma_logk = numpyro.sample("sigma_logk", dist.HalfNormal(1.0))
    sigma_logbeta = numpyro.sample("sigma_logbeta", dist.HalfNormal(1.0))
    sigma_vec = jnp.stack([sigma_logk, sigma_logbeta])

    L_omega = numpyro.sample("L_omega", dist.LKJCholesky(2, concentration=2.0))
    L_sigma = jnp.diag(sigma_vec) @ L_omega

    with numpyro.plate("subjects", N_subj):
        z_raw = numpyro.sample(
            "z_raw",
            dist.MultivariateNormal(jnp.zeros(2), scale_tril=jnp.eye(2))
        )
    theta = mu[None, :] + z_raw @ L_sigma.T
    k_i = jnp.exp(theta[:, 0])
    beta_i = jnp.exp(theta[:, 1])

    k_t = k_i[subj_idx]
    beta_t = beta_i[subj_idx]

    S_H = (1.0 - threat) + threat / (1.0 + lam * dist_H)
    S_L = (1.0 - threat) + threat / (1.0 + lam * dist_L)
    SV_H = R_H * S_H - k_t * effort_H - beta_t * (1.0 - S_H)
    SV_L = R_L * S_L - k_t * effort_L - beta_t * (1.0 - S_L)
    delta_SV = tau * (SV_H - SV_L)

    with numpyro.plate("trials", len(choice_obs)):
        numpyro.sample("choice", dist.Bernoulli(logits=delta_SV), obs=choice_obs)


# ============================================================
# 8. STEP 1: Fit choice-only model to get λ
# ============================================================
N_STEPS = 30000
LR = 0.002
PRINT_EVERY = 3000

print("\n" + "=" * 60)
print(f"STEP 1: Fitting Choice-Only Model (SVI, {N_STEPS} steps, lr={LR})")
print("  Purpose: estimate λ from choice data (where S is well-identified)")
print("=" * 60, flush=True)

rng_key = jax.random.PRNGKey(42)
rng_key, rng_init_c, rng_run_c = jax.random.split(rng_key, 3)

guide_choice = AutoMultivariateNormal(choice_only_model)
optimizer_choice = optim.Adam(step_size=LR)
svi_choice = SVI(
    choice_only_model, guide_choice, optimizer_choice,
    loss=Trace_ELBO()
)

choice_args = (subj_idx, threat, dist_H, dist_L, effort_H, effort_L, choice, N_subj)
choice_kwargs = {"R_H": R_H, "R_L": R_L}

svi_state_choice = svi_choice.init(rng_init_c, *choice_args, **choice_kwargs)

t0 = time.time()
elbo_history_choice = []
for step in range(N_STEPS):
    svi_state_choice, loss = svi_choice.update(svi_state_choice, *choice_args, **choice_kwargs)
    elbo_history_choice.append(-float(loss))
    if (step + 1) % PRINT_EVERY == 0:
        elapsed = time.time() - t0
        recent = np.mean(elbo_history_choice[-500:])
        print(f"  Step {step+1:5d}/{N_STEPS}  ELBO={-float(loss):.1f}  "
              f"(avg500={recent:.1f}, {elapsed:.0f}s)", flush=True)

final_elbo_choice = np.mean(elbo_history_choice[-500:])
print(f"\n  Choice-only final ELBO (avg last 500): {final_elbo_choice:.1f}", flush=True)

params_choice = svi_choice.get_params(svi_state_choice)

# Extract λ from choice-only posterior
n_post = 4000
rng_key, rng_post_c = jax.random.split(rng_key)
post_choice = guide_choice.sample_posterior(rng_post_c, params_choice, sample_shape=(n_post,))
lam_choice_samples = np.array(post_choice["lambda"])
lam_choice_mean = float(lam_choice_samples.mean())
lam_choice_sd = float(lam_choice_samples.std())
print(f"\n  λ (choice-only): {lam_choice_mean:.3f} (±{lam_choice_sd:.3f})")

# ============================================================
# 9. STEP 2: Fit joint model with λ FIXED from choice
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 2: Fitting Joint Correlated Model (SVI, {N_STEPS} steps, lr={LR})")
print(f"  λ FIXED at {lam_choice_mean:.3f} (from choice-only fit)")
print("  Guide: AutoMultivariateNormal")
print("=" * 60, flush=True)

rng_key, rng_init_j, rng_run_j = jax.random.split(rng_key, 3)

# Use functools.partial to fix lam_fixed
from functools import partial
joint_model_fixed_lam = partial(
    joint_correlated_model,
    lam_fixed=lam_choice_mean
)

guide_joint = AutoMultivariateNormal(joint_model_fixed_lam)
optimizer_joint = optim.Adam(step_size=LR)
svi_joint = SVI(
    joint_model_fixed_lam, guide_joint, optimizer_joint,
    loss=Trace_ELBO()
)

joint_args = (
    subj_idx, threat, dist_H, dist_L, dist_chosen,
    effort_H, effort_L, excess_obs, choice, N_subj
)
joint_kwargs = {"R_H": R_H, "R_L": R_L}

print("  Initializing ...", flush=True)
svi_state_joint = svi_joint.init(rng_init_j, *joint_args, **joint_kwargs)

print(f"  Running {N_STEPS} steps ...", flush=True)
t0 = time.time()
elbo_history = []
for step in range(N_STEPS):
    svi_state_joint, loss = svi_joint.update(svi_state_joint, *joint_args, **joint_kwargs)
    elbo_history.append(-float(loss))
    if (step + 1) % PRINT_EVERY == 0:
        elapsed = time.time() - t0
        recent = np.mean(elbo_history[-500:])
        print(f"  Step {step+1:5d}/{N_STEPS}  ELBO={-float(loss):.1f}  "
              f"(avg500={recent:.1f}, {elapsed:.0f}s)", flush=True)

final_elbo_joint = np.mean(elbo_history[-500:])
print(f"\n  Joint model final ELBO (avg last 500): {final_elbo_joint:.1f}", flush=True)

params_joint = svi_joint.get_params(svi_state_joint)

# ============================================================
# 10. Extract posterior samples
# ============================================================
print("\nSampling from posteriors ...", flush=True)
rng_key, rng_post_j = jax.random.split(rng_key)

post_joint = guide_joint.sample_posterior(rng_post_j, params_joint, sample_shape=(n_post,))


def post_summary(samples, name):
    vals = np.array(samples[name])
    return float(np.mean(vals)), float(np.std(vals))


# ============================================================
# 11. Extract correlation matrix from joint model
# ============================================================
print("\n" + "=" * 60)
print("CORRELATION MATRIX (joint model posterior)")
print("=" * 60)

L_omega_samples = np.array(post_joint["L_omega"])  # (n_post, 4, 4)

# Reconstruct correlation matrices: Ω = L_ω @ L_ω^T
omega_samples = np.einsum("...ij,...kj->...ik", L_omega_samples, L_omega_samples)  # (n_post, 4, 4)

omega_mean = omega_samples.mean(axis=0)
omega_sd = omega_samples.std(axis=0)

# Credible intervals for each off-diagonal
print(f"\n  {'Pair':<20s} {'mean':>7s} {'SD':>7s} {'2.5%':>7s} {'97.5%':>7s} {'P(>0)':>7s}")
print("  " + "-" * 55)
pairs = []
for i in range(N_PARAMS):
    for j in range(i + 1, N_PARAMS):
        rho_ij = omega_samples[:, i, j]
        m = float(rho_ij.mean())
        s = float(rho_ij.std())
        lo = float(np.percentile(rho_ij, 2.5))
        hi = float(np.percentile(rho_ij, 97.5))
        p_pos = float(np.mean(rho_ij > 0))
        name = f"{PARAM_NAMES[i]} × {PARAM_NAMES[j]}"
        print(f"  {name:<20s} {m:+7.3f} {s:7.3f} [{lo:+6.3f}, {hi:+6.3f}] {p_pos:7.3f}")
        pairs.append({
            "param_1": PARAM_NAMES[i],
            "param_2": PARAM_NAMES[j],
            "rho_mean": round(m, 4),
            "rho_sd": round(s, 4),
            "rho_2.5": round(lo, 4),
            "rho_97.5": round(hi, 4),
            "P_positive": round(p_pos, 4),
        })

corr_df = pd.DataFrame(pairs)

# ============================================================
# 12. Extract population parameters
# ============================================================
print("\n" + "=" * 60)
print("POPULATION PARAMETERS")
print("=" * 60)

pop = {}
# λ is fixed from choice-only
pop["lambda"] = lam_choice_mean
pop["lambda_sd"] = lam_choice_sd
print(f"  {'lambda (fixed)':<16s}: {lam_choice_mean:+8.4f} (±{lam_choice_sd:.4f}) [from choice-only]")

for name in ["tau", "sigma_excess",
             "mu_logk", "mu_logbeta", "mu_alpha", "mu_delta",
             "sigma_logk", "sigma_logbeta", "sigma_alpha", "sigma_delta"]:
    m, s = post_summary(post_joint, name)
    pop[name] = m
    pop[name + "_sd"] = s
    print(f"  {name:<16s}: {m:+8.4f} (±{s:.4f})")

# ============================================================
# 13. Extract per-subject parameters
# ============================================================
print("\nExtracting per-subject parameters ...", flush=True)

# z_raw: (n_post, N_subj, 4)
z_raw_post = np.array(post_joint["z_raw"])

# Hyperparameters per sample
mu_logk_post = np.array(post_joint["mu_logk"])[:, None]
mu_logbeta_post = np.array(post_joint["mu_logbeta"])[:, None]
mu_alpha_post = np.array(post_joint["mu_alpha"])[:, None]
mu_delta_post = np.array(post_joint["mu_delta"])[:, None]

sigma_logk_post = np.array(post_joint["sigma_logk"])[:, None]
sigma_logbeta_post = np.array(post_joint["sigma_logbeta"])[:, None]
sigma_alpha_post = np.array(post_joint["sigma_alpha"])[:, None]
sigma_delta_post = np.array(post_joint["sigma_delta"])[:, None]

# Reconstruct L_sigma for each sample
# L_sigma = diag(sigma) @ L_omega
# theta = mu + z_raw @ L_sigma.T
sigma_vecs = np.stack([
    np.array(post_joint["sigma_logk"]),
    np.array(post_joint["sigma_logbeta"]),
    np.array(post_joint["sigma_alpha"]),
    np.array(post_joint["sigma_delta"]),
], axis=-1)  # (n_post, 4)

# diag(sigma) @ L_omega: (n_post, 4, 4)
L_sigma_post = np.einsum("...i,ij,...jk->...ik",
                          sigma_vecs,
                          np.eye(N_PARAMS),
                          L_omega_samples)
# Actually: diag(sigma_s) @ L_omega_s for each s
L_sigma_post = sigma_vecs[:, :, None] * L_omega_samples  # broadcast: (n_post, 4, 4)

mu_vec = np.stack([
    np.array(post_joint["mu_logk"]),
    np.array(post_joint["mu_logbeta"]),
    np.array(post_joint["mu_alpha"]),
    np.array(post_joint["mu_delta"]),
], axis=-1)  # (n_post, 4)

# theta_i = mu + z_raw_i @ L_sigma.T
# (n_post, N_subj, 4) = (n_post, 1, 4) + (n_post, N_subj, 4) @ (n_post, 4, 4)
theta_post = mu_vec[:, None, :] + np.einsum("...si,...ji->...sj", z_raw_post, L_sigma_post)

logk_post = theta_post[:, :, 0]
logbeta_post = theta_post[:, :, 1]
alpha_post = theta_post[:, :, 2]
delta_post = theta_post[:, :, 3]

k_post = np.exp(logk_post)
beta_post = np.exp(logbeta_post)

# Posterior means per subject
k_mean = k_post.mean(axis=0)
beta_mean = beta_post.mean(axis=0)
alpha_mean = alpha_post.mean(axis=0)
delta_mean = delta_post.mean(axis=0)

k_sd = k_post.std(axis=0)
beta_sd = beta_post.std(axis=0)
alpha_sd = alpha_post.std(axis=0)
delta_sd = delta_post.std(axis=0)

print(f"  k:     mean={k_mean.mean():.3f}, SD(across subj)={k_mean.std():.3f}")
print(f"  β:     mean={beta_mean.mean():.3f}, SD(across subj)={beta_mean.std():.3f}")
print(f"  α:     mean={alpha_mean.mean():.3f}, SD(across subj)={alpha_mean.std():.3f}")
print(f"  δ:     mean={delta_mean.mean():.3f}, SD(across subj)={delta_mean.std():.3f}")

# Check σ_delta
sigma_delta_samples = np.array(post_joint["sigma_delta"])
print(f"\n  σ_δ posterior: mean={sigma_delta_samples.mean():.4f}, "
      f"SD={sigma_delta_samples.std():.4f}, "
      f"P(>0.05)={np.mean(sigma_delta_samples > 0.05):.3f}")

delta_subj_sd = delta_mean.std()
print(f"  SD of δ posterior means: {delta_subj_sd:.4f}")

if sigma_delta_samples.mean() < 0.05:
    print("  ⚠ σ_δ appears collapsed — individual δ differences may not be recovered")
else:
    print("  ✓ σ_δ is non-trivial — individual differences in δ recovered")

# ============================================================
# 14. Post-hoc validation: do model-estimated correlations
#     match empirical correlations of posterior means?
# ============================================================
print("\n" + "=" * 60)
print("VALIDATION: Model ρ vs empirical r (posterior means)")
print("=" * 60)

empirical_corrs = []
for i in range(N_PARAMS):
    for j in range(i + 1, N_PARAMS):
        vals_i = [k_mean, beta_mean, alpha_mean, delta_mean][i]
        vals_j = [k_mean, beta_mean, alpha_mean, delta_mean][j]
        r, p = stats.pearsonr(vals_i, vals_j)
        name = f"{PARAM_NAMES[i]} × {PARAM_NAMES[j]}"
        model_rho = corr_df[(corr_df["param_1"] == PARAM_NAMES[i]) &
                            (corr_df["param_2"] == PARAM_NAMES[j])]["rho_mean"].values[0]
        print(f"  {name:<20s}  model ρ={model_rho:+.3f}  empirical r={r:+.3f}  (p={p:.4f})")
        empirical_corrs.append({
            "pair": name,
            "model_rho": model_rho,
            "empirical_r": round(r, 4),
            "empirical_p": round(p, 4),
        })

# ============================================================
# 15. Key question: does δ have meaningful individual variation?
# ============================================================
print("\n" + "=" * 60)
print("δ INDIVIDUAL DIFFERENCES DIAGNOSTICS")
print("=" * 60)

# % positive
pct_pos = (delta_mean > 0).mean() * 100
print(f"  % subjects with δ > 0: {pct_pos:.1f}%")

# Population effect
mu_delta_m, mu_delta_s = post_summary(post_joint, "mu_delta")
p_mu_pos = float(np.mean(np.array(post_joint["mu_delta"]) > 0))
print(f"  μ_δ: {mu_delta_m:+.4f} (±{mu_delta_s:.4f}), P(μ_δ>0) = {p_mu_pos:.4f}")

# Shrinkage diagnostic: compare model σ_δ to empirical SD of OLS δ
# Quick OLS per subject for comparison
lam_pop = pop["lambda"]
merged_copy = merged.copy()
# S for vigor uses chosen-option distance
d_chosen = merged_copy["choice"] * merged_copy["distance_H"] + (1 - merged_copy["choice"]) * merged_copy["distance_L"]
merged_copy["S"] = (1 - merged_copy["threat"]) + merged_copy["threat"] / (1 + lam_pop * d_chosen)
merged_copy["one_minus_S"] = 1 - merged_copy["S"]

ols_deltas = []
for s in subj_ids:
    sub = merged_copy[merged_copy["subj"] == s]
    if len(sub) < 10:
        ols_deltas.append(np.nan)
        continue
    X = np.column_stack([np.ones(len(sub)), sub["one_minus_S"].values])
    y = sub["excess"].values
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    ols_deltas.append(beta_ols[1])

ols_deltas = np.array(ols_deltas)
print(f"\n  OLS δ: mean={np.nanmean(ols_deltas):.4f}, SD={np.nanstd(ols_deltas):.4f}")
print(f"  Model δ: mean={delta_mean.mean():.4f}, SD={delta_mean.std():.4f}")
shrinkage = 1 - (delta_mean.std() / np.nanstd(ols_deltas)) if np.nanstd(ols_deltas) > 0 else np.nan
print(f"  Shrinkage: {shrinkage:.1%}")
print(f"  (0% = no shrinkage, 100% = fully collapsed to population mean)")

# ============================================================
# 16. Save outputs
# ============================================================
print("\nSaving outputs ...", flush=True)

# 16a. Correlation matrix
corr_df.to_csv(STAT_DIR / "joint_correlated_correlations.csv", index=False)
print(f"  Saved: {STAT_DIR}/joint_correlated_correlations.csv")

# 16b. Subject parameters
subj_df = pd.DataFrame({
    "subj": subj_ids,
    "k": k_mean, "k_sd": k_sd,
    "beta": beta_mean, "beta_sd": beta_sd,
    "alpha": alpha_mean, "alpha_sd": alpha_sd,
    "delta": delta_mean, "delta_sd": delta_sd,
    "delta_ols": ols_deltas,
})
subj_df.to_csv(STAT_DIR / "joint_correlated_subjects.csv", index=False)
print(f"  Saved: {STAT_DIR}/joint_correlated_subjects.csv")

# 16c. Population parameters
pop_df = pd.DataFrame([pop])
pop_df["lambda_choice"] = lam_choice_mean
pop_df["lambda_choice_sd"] = lam_choice_sd
pop_df["elbo_joint"] = final_elbo_joint
pop_df["elbo_choice"] = final_elbo_choice
pop_df["elbo_diff"] = final_elbo_joint - final_elbo_choice
pop_df["n_trials"] = len(merged)
pop_df["n_subjects"] = N_subj
pop_df.to_csv(STAT_DIR / "joint_correlated_population.csv", index=False)
print(f"  Saved: {STAT_DIR}/joint_correlated_population.csv")

# 16d. Full omega samples for downstream analysis
omega_flat = pd.DataFrame({
    f"rho_{PARAM_NAMES[i]}_{PARAM_NAMES[j]}": omega_samples[:, i, j]
    for i in range(N_PARAMS) for j in range(i + 1, N_PARAMS)
})
omega_flat.to_csv(STAT_DIR / "joint_correlated_omega_samples.csv", index=False)
print(f"  Saved: {STAT_DIR}/joint_correlated_omega_samples.csv")

# 16e. ELBO history
pd.DataFrame({
    "step": list(range(1, N_STEPS + 1)),
    "elbo_joint": elbo_history,
    "elbo_choice": elbo_history_choice,
}).to_csv(STAT_DIR / "joint_correlated_elbo_history.csv", index=False)
print(f"  Saved: {STAT_DIR}/joint_correlated_elbo_history.csv")

# ============================================================
# 17. Print summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  ELBO — Joint: {final_elbo_joint:.1f}, Choice-only: {final_elbo_choice:.1f}, "
      f"Δ: {final_elbo_joint - final_elbo_choice:+.1f}")
print(f"  λ (fixed from choice): {pop['lambda']:.3f} (±{pop['lambda_sd']:.3f})")
print(f"  μ_δ = {pop['mu_delta']:+.4f}, σ_δ = {pop['sigma_delta']:.4f}")
print(f"  δ shrinkage: {shrinkage:.1%}")
print()

# Key correlation for the paper: β × δ
bd = corr_df[(corr_df["param_1"] == "log_beta") & (corr_df["param_2"] == "delta")]
if len(bd) > 0:
    rho_bd = bd.iloc[0]["rho_mean"]
    lo_bd = bd.iloc[0]["rho_2.5"]
    hi_bd = bd.iloc[0]["rho_97.5"]
    print(f"  ★ KEY: ρ(β, δ) = {rho_bd:+.3f} [{lo_bd:+.3f}, {hi_bd:+.3f}]")
    print(f"         Threat bias in choice correlates with vigor mobilization under danger")

kd = corr_df[(corr_df["param_1"] == "log_k") & (corr_df["param_2"] == "delta")]
if len(kd) > 0:
    rho_kd = kd.iloc[0]["rho_mean"]
    lo_kd = kd.iloc[0]["rho_2.5"]
    hi_kd = kd.iloc[0]["rho_97.5"]
    print(f"  ★ KEY: ρ(k, δ) = {rho_kd:+.3f} [{lo_kd:+.3f}, {hi_kd:+.3f}]")
    print(f"         Effort sensitivity negatively predicts vigor mobilization")

print("\n" + "=" * 60)
print("DONE.")
print("=" * 60)
