"""
run_joint_choice_vigor_model.py
================================
Joint choice-vigor SVI model for EffortForagingUnderThreat (N=293, exploratory).

Model specification:
  Survival:  S = (1-T) + T / (1 + lambda * D)
  Choice:    SV_H = R_H * S - k_i * E_H - beta_i * (1-S)
             SV_L = R_L * S - k_i * E_L - beta_i * (1-S)
             choice ~ Bernoulli(sigmoid(tau * (SV_H - SV_L)))
  Vigor:     excess_ij = alpha_i + delta_i * (1-S_ij) + eps_ij
             eps_ij ~ Normal(0, sigma_excess)
  where excess_ij = vigor_norm_ij - effort_chosen_ij

Population parameters: lambda, tau, sigma_excess
  mu_k, sigma_k, mu_beta, sigma_beta
  mu_alpha, sigma_alpha, mu_delta, sigma_delta

Per-subject (non-centered):
  k_i     = mu_k     + sigma_k     * k_raw_i
  beta_i  = exp(mu_beta_log  + sigma_beta  * beta_raw_i)  [log-normal, must be positive]
  alpha_i = mu_alpha + sigma_alpha * alpha_raw_i
  delta_i = mu_delta + sigma_delta * delta_raw_i

Priors:
  lambda   ~ HalfNormal(2.0)
  tau      ~ HalfNormal(2.0)
  mu_k     ~ Normal(0, 3)
  sigma_k  ~ HalfNormal(2.0)
  mu_beta_log ~ Normal(0, 2)
  sigma_beta  ~ HalfNormal(2.0)
  mu_alpha ~ Normal(0, 2)
  sigma_alpha ~ HalfNormal(2.0)
  mu_delta ~ Normal(0, 2)
  sigma_delta ~ HalfNormal(2.0)  [WIDE prior to prevent collapse]
  sigma_excess ~ HalfNormal(1.0)

Fit: SVI with AutoNormal guide, 20000 steps, Adam lr=0.003

Outputs:
  results/stats/joint_model_population.csv
  results/stats/joint_model_subjects.csv
  results/joint_model_text.md
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Set PATH for local binaries
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import numpyro.optim as optim

numpyro.set_platform("cpu")

print("="*60)
print("Joint Choice-Vigor SVI Model")
print("="*60)
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
# 1. Load data
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
# 3. Identify probe trial global indices per subject
#    and map behavioral trials to sequential index 1-45
# ============================================================
feelings = pd.read_csv(STAGE5 / "feelings.csv")
probe_per_subj = feelings.groupby("subj")["trialNumber"].apply(set).reset_index()
probe_per_subj.columns = ["subj", "probe_set"]

# Merge and filter out probe trials
vigor_trial = vigor_trial.merge(probe_per_subj, on="subj", how="left")
vigor_trial["is_probe"] = vigor_trial.apply(
    lambda r: r["global_trial"] in r["probe_set"], axis=1
)
vigor_beh = vigor_trial[~vigor_trial["is_probe"]].copy()
vigor_beh = vigor_beh.drop(columns=["probe_set", "is_probe"])

# Rank-order within subject to get sequential trial index (1-45)
vigor_beh = vigor_beh.sort_values(["subj", "global_trial"])
vigor_beh["trial"] = vigor_beh.groupby("subj").cumcount() + 1

print(f"  Behavioral vigor: {len(vigor_beh):,} rows")
n_per_subj = vigor_beh.groupby("subj").size()
print(f"  Trials per subject: {n_per_subj.min()}–{n_per_subj.max()} (expected 45)")

# ============================================================
# 4. Merge behavior + vigor
# ============================================================
print("\nMerging behavior and vigor ...", flush=True)

# behavior.csv: subj (1-293), trial (1-45), threat, effort_L, effort_H, distance_H, choice
# vigor_beh: subj, global_trial, vigor_norm, trial (1-45 sequential)

# For subjects with != 45 behavioral vigor trials, we'll use inner join
# (drop trials missing from either source)
merged = beh.merge(
    vigor_beh[["subj", "trial", "vigor_norm"]],
    on=["subj", "trial"],
    how="inner"
)
print(f"  Merged: {len(merged):,} trials (of {len(beh):,} behavioral)")
print(f"  Subjects: {merged['subj'].nunique()}")

# Effort chosen: choice=1 → effort_H, choice=0 → effort_L
# E_L = 0.4 always
merged["effort_chosen"] = (
    merged["choice"] * merged["effort_H"] +
    (1 - merged["choice"]) * merged["effort_L"]
)

# Excess vigor = vigor - demand (effort chosen)
merged["excess"] = merged["vigor_norm"] - merged["effort_chosen"]

print(f"  Excess vigor: mean={merged['excess'].mean():.3f}, SD={merged['excess'].std():.3f}")

# ============================================================
# 5. Build arrays for JAX
# ============================================================
print("\nBuilding JAX arrays ...", flush=True)

# Re-index subjects to 0-based
subj_ids = sorted(merged["subj"].unique())
subj_map = {s: i for i, s in enumerate(subj_ids)}
merged["subj_idx"] = merged["subj"].map(subj_map)
N_subj = len(subj_ids)

# Data arrays
subj_idx = jnp.array(merged["subj_idx"].values, dtype=jnp.int32)
threat    = jnp.array(merged["threat"].values, dtype=jnp.float32)
dist_H    = jnp.array(merged["distance_H"].values, dtype=jnp.float32)  # 1/2/3
effort_H  = jnp.array(merged["effort_H"].values, dtype=jnp.float32)
effort_L  = jnp.array(merged["effort_L"].values, dtype=jnp.float32)
choice    = jnp.array(merged["choice"].values, dtype=jnp.int32)
vigor_obs = jnp.array(merged["vigor_norm"].values, dtype=jnp.float32)
effort_ch = jnp.array(merged["effort_chosen"].values, dtype=jnp.float32)
excess_obs = jnp.array(merged["excess"].values, dtype=jnp.float32)

# Constants
R_H = 5.0
R_L = 1.0

print(f"  N subjects: {N_subj}")
print(f"  N trials:   {len(merged):,}")
print(f"  threat unique: {sorted(merged['threat'].unique())}")
print(f"  distance_H unique: {sorted(merged['distance_H'].unique())}")

# ============================================================
# 6. Define joint model
# ============================================================

def joint_model(subj_idx, threat, dist_H, effort_H, effort_L, effort_ch,
                excess_obs, choice_obs, N_subj,
                R_H=5.0, R_L=1.0):
    """
    Joint choice-vigor model with non-centered parameterization.
    """
    # ---- Population hyperparameters ----
    lam       = numpyro.sample("lambda",       dist.HalfNormal(2.0))
    tau       = numpyro.sample("tau",          dist.HalfNormal(2.0))

    mu_k      = numpyro.sample("mu_k",         dist.Normal(0.0, 3.0))
    sigma_k   = numpyro.sample("sigma_k",      dist.HalfNormal(2.0))

    mu_beta_log  = numpyro.sample("mu_beta_log",  dist.Normal(0.0, 2.0))
    sigma_beta   = numpyro.sample("sigma_beta",   dist.HalfNormal(2.0))

    mu_alpha  = numpyro.sample("mu_alpha",     dist.Normal(0.0, 2.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(2.0))

    mu_delta  = numpyro.sample("mu_delta",     dist.Normal(0.0, 2.0))
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(2.0))  # wide prior

    sigma_excess = numpyro.sample("sigma_excess", dist.HalfNormal(1.0))

    # ---- Per-subject non-centered variables ----
    with numpyro.plate("subjects", N_subj):
        k_raw     = numpyro.sample("k_raw",     dist.Normal(0.0, 1.0))
        beta_raw  = numpyro.sample("beta_raw",  dist.Normal(0.0, 1.0))
        alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 1.0))
        delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0))

    # Non-centered transformation
    k_i     = mu_k + sigma_k * k_raw             # can be negative
    beta_i  = jnp.exp(mu_beta_log + sigma_beta * beta_raw)  # positive
    alpha_i = mu_alpha + sigma_alpha * alpha_raw
    delta_i = mu_delta + sigma_delta * delta_raw

    # Index to trial level
    k_t     = k_i[subj_idx]
    beta_t  = beta_i[subj_idx]
    alpha_t = alpha_i[subj_idx]
    delta_t = delta_i[subj_idx]

    # ---- Survival function ----
    # S = (1-T) + T / (1 + lambda * D)
    S = (1.0 - threat) + threat / (1.0 + lam * dist_H)

    # ---- Choice model ----
    SV_H = R_H * S - k_t * effort_H - beta_t * (1.0 - S)
    SV_L = R_L * S - k_t * effort_L - beta_t * (1.0 - S)
    delta_SV = tau * (SV_H - SV_L)
    choice_p = jax.nn.sigmoid(delta_SV)

    with numpyro.plate("trials_choice", len(choice_obs)):
        numpyro.sample("choice", dist.Bernoulli(probs=choice_p), obs=choice_obs)

    # ---- Vigor model ----
    # excess = alpha_i + delta_i * (1-S) + eps
    mu_excess = alpha_t + delta_t * (1.0 - S)

    with numpyro.plate("trials_vigor", len(excess_obs)):
        numpyro.sample("excess", dist.Normal(mu_excess, sigma_excess), obs=excess_obs)


# ============================================================
# 7. Define choice-only model (for comparison)
# ============================================================

def choice_only_model(subj_idx, threat, dist_H, effort_H, effort_L,
                      choice_obs, N_subj, R_H=5.0, R_L=1.0):
    """Choice-only model with same structure as joint (for ELBO comparison)."""
    lam       = numpyro.sample("lambda",       dist.HalfNormal(2.0))
    tau       = numpyro.sample("tau",          dist.HalfNormal(2.0))

    mu_k      = numpyro.sample("mu_k",         dist.Normal(0.0, 3.0))
    sigma_k   = numpyro.sample("sigma_k",      dist.HalfNormal(2.0))

    mu_beta_log  = numpyro.sample("mu_beta_log",  dist.Normal(0.0, 2.0))
    sigma_beta   = numpyro.sample("sigma_beta",   dist.HalfNormal(2.0))

    with numpyro.plate("subjects", N_subj):
        k_raw    = numpyro.sample("k_raw",    dist.Normal(0.0, 1.0))
        beta_raw = numpyro.sample("beta_raw", dist.Normal(0.0, 1.0))

    k_i    = mu_k + sigma_k * k_raw
    beta_i = jnp.exp(mu_beta_log + sigma_beta * beta_raw)

    k_t    = k_i[subj_idx]
    beta_t = beta_i[subj_idx]

    S = (1.0 - threat) + threat / (1.0 + lam * dist_H)

    SV_H = R_H * S - k_t * effort_H - beta_t * (1.0 - S)
    SV_L = R_L * S - k_t * effort_L - beta_t * (1.0 - S)
    delta_SV = tau * (SV_H - SV_L)
    choice_p = jax.nn.sigmoid(delta_SV)

    with numpyro.plate("trials", len(choice_obs)):
        numpyro.sample("choice", dist.Bernoulli(probs=choice_p), obs=choice_obs)


# ============================================================
# 8. Fit joint model
# ============================================================
print("\n" + "="*60)
print("Fitting Joint Model (SVI, 20000 steps, lr=0.003)")
print("="*60, flush=True)

rng_key = jax.random.PRNGKey(42)
rng_key, rng_init, rng_run = jax.random.split(rng_key, 3)

guide_joint = AutoNormal(joint_model)
optimizer_joint = optim.Adam(step_size=0.003)
svi_joint = SVI(
    joint_model, guide_joint, optimizer_joint,
    loss=Trace_ELBO()
)

# Model args
joint_args = (
    subj_idx, threat, dist_H, effort_H, effort_L, effort_ch,
    excess_obs, choice, N_subj
)
joint_kwargs = {"R_H": R_H, "R_L": R_L}

# Initialize
print("  Initializing ...", flush=True)
svi_state_joint = svi_joint.init(rng_init, *joint_args, **joint_kwargs)

# Run
N_STEPS = 20000
PRINT_EVERY = 2000
print(f"  Running {N_STEPS} steps ...", flush=True)
t0 = time.time()

elbo_history_joint = []
for step in range(N_STEPS):
    svi_state_joint, loss = svi_joint.update(svi_state_joint, *joint_args, **joint_kwargs)
    elbo_history_joint.append(-float(loss))
    if (step + 1) % PRINT_EVERY == 0:
        elapsed = time.time() - t0
        print(f"  Step {step+1:5d}/{N_STEPS}  ELBO={-float(loss):.1f}  ({elapsed:.0f}s)", flush=True)

final_elbo_joint = elbo_history_joint[-1]
print(f"\n  Joint model final ELBO: {final_elbo_joint:.1f}", flush=True)

# Extract params
params_joint = svi_joint.get_params(svi_state_joint)

# ============================================================
# 9. Fit choice-only model
# ============================================================
print("\n" + "="*60)
print("Fitting Choice-Only Model (SVI, 20000 steps, lr=0.003)")
print("="*60, flush=True)

rng_key, rng_init2, rng_run2 = jax.random.split(rng_key, 3)

guide_choice = AutoNormal(choice_only_model)
optimizer_choice = optim.Adam(step_size=0.003)
svi_choice = SVI(
    choice_only_model, guide_choice, optimizer_choice,
    loss=Trace_ELBO()
)

choice_args = (subj_idx, threat, dist_H, effort_H, effort_L, choice, N_subj)
choice_kwargs = {"R_H": R_H, "R_L": R_L}

svi_state_choice = svi_choice.init(rng_init2, *choice_args, **choice_kwargs)

t0 = time.time()
elbo_history_choice = []
for step in range(N_STEPS):
    svi_state_choice, loss = svi_choice.update(svi_state_choice, *choice_args, **choice_kwargs)
    elbo_history_choice.append(-float(loss))
    if (step + 1) % PRINT_EVERY == 0:
        elapsed = time.time() - t0
        print(f"  Step {step+1:5d}/{N_STEPS}  ELBO={-float(loss):.1f}  ({elapsed:.0f}s)", flush=True)

final_elbo_choice = elbo_history_choice[-1]
print(f"\n  Choice-only model final ELBO: {final_elbo_choice:.1f}", flush=True)

params_choice = svi_choice.get_params(svi_state_choice)

# ============================================================
# 10. Extract population parameters from joint model
# ============================================================
print("\nExtracting population parameters ...", flush=True)

def get_loc(params, name):
    """Get AutoNormal loc for a parameter."""
    key = f"{name}_auto_loc"
    return float(params[key]) if key in params else None

def get_scale(params, name):
    """Get AutoNormal scale for a parameter."""
    key = f"{name}_auto_scale"
    return float(params[key]) if key in params else None

# Population parameters (deterministic transforms of AutoNormal locs)
pop_params = {}

# lambda: HalfNormal → loc/scale in unconstrained space
# AutoNormal works in unconstrained space via transforms
# Extract by sampling from posterior
print("  Sampling from posterior for population parameters ...", flush=True)
n_posterior_samples = 2000
rng_key, rng_post = jax.random.split(rng_key)
posterior_samples_joint = guide_joint.sample_posterior(
    rng_post, params_joint, sample_shape=(n_posterior_samples,)
)
posterior_samples_choice = guide_choice.sample_posterior(
    rng_post, params_choice, sample_shape=(n_posterior_samples,)
)

# Population parameter means from joint model
def post_mean(name):
    vals = posterior_samples_joint[name]
    return float(jnp.mean(vals))

def post_sd(name):
    vals = posterior_samples_joint[name]
    return float(jnp.std(vals))

def post_mean_choice(name):
    vals = posterior_samples_choice[name]
    return float(jnp.mean(vals))

# Compute mu_k, sigma_k, mu_beta (on log scale), etc.
pop_pop_params = {
    "lambda_joint": post_mean("lambda"),
    "lambda_joint_sd": post_sd("lambda"),
    "lambda_choice": post_mean_choice("lambda"),
    "lambda_choice_sd": float(jnp.std(posterior_samples_choice["lambda"])),
    "tau_joint": post_mean("tau"),
    "tau_joint_sd": post_sd("tau"),
    "mu_k_joint": post_mean("mu_k"),
    "mu_k_joint_sd": post_sd("mu_k"),
    "sigma_k_joint": post_mean("sigma_k"),
    "sigma_k_joint_sd": post_sd("sigma_k"),
    "mu_beta_log_joint": post_mean("mu_beta_log"),
    "mu_beta_log_joint_sd": post_sd("mu_beta_log"),
    "sigma_beta_joint": post_mean("sigma_beta"),
    "sigma_beta_joint_sd": post_sd("sigma_beta"),
    "mu_alpha_joint": post_mean("mu_alpha"),
    "mu_alpha_joint_sd": post_sd("mu_alpha"),
    "sigma_alpha_joint": post_mean("sigma_alpha"),
    "sigma_alpha_joint_sd": post_sd("sigma_alpha"),
    "mu_delta_joint": post_mean("mu_delta"),
    "mu_delta_joint_sd": post_sd("mu_delta"),
    "sigma_delta_joint": post_mean("sigma_delta"),
    "sigma_delta_joint_sd": post_sd("sigma_delta"),
    "sigma_excess_joint": post_mean("sigma_excess"),
    "sigma_excess_joint_sd": post_sd("sigma_excess"),
    "elbo_joint": final_elbo_joint,
    "elbo_choice": final_elbo_choice,
    "elbo_diff_joint_minus_choice": final_elbo_joint - final_elbo_choice,
    "n_trials": len(merged),
    "n_subjects": N_subj,
}

print("\n  Population parameters (joint model):")
for k, v in pop_pop_params.items():
    if isinstance(v, float):
        print(f"    {k}: {v:.4f}")
    else:
        print(f"    {k}: {v}")

# ============================================================
# 11. Extract per-subject parameters
# ============================================================
print("\nExtracting per-subject parameters ...", flush=True)

# Subject-level: k_i, beta_i, alpha_i, delta_i
# Posterior mean over samples
k_raw_post     = np.array(posterior_samples_joint["k_raw"])      # (n_post, N_subj)
beta_raw_post  = np.array(posterior_samples_joint["beta_raw"])
alpha_raw_post = np.array(posterior_samples_joint["alpha_raw"])
delta_raw_post = np.array(posterior_samples_joint["delta_raw"])

mu_k_post        = np.array(posterior_samples_joint["mu_k"])[:, None]
sigma_k_post     = np.array(posterior_samples_joint["sigma_k"])[:, None]
mu_beta_log_post = np.array(posterior_samples_joint["mu_beta_log"])[:, None]
sigma_beta_post  = np.array(posterior_samples_joint["sigma_beta"])[:, None]
mu_alpha_post    = np.array(posterior_samples_joint["mu_alpha"])[:, None]
sigma_alpha_post = np.array(posterior_samples_joint["sigma_alpha"])[:, None]
mu_delta_post    = np.array(posterior_samples_joint["mu_delta"])[:, None]
sigma_delta_post = np.array(posterior_samples_joint["sigma_delta"])[:, None]

# Transform to actual parameters
k_post     = mu_k_post + sigma_k_post * k_raw_post
beta_post  = np.exp(mu_beta_log_post + sigma_beta_post * beta_raw_post)
alpha_post = mu_alpha_post + sigma_alpha_post * alpha_raw_post
delta_post = mu_delta_post + sigma_delta_post * delta_raw_post

# Posterior means per subject
k_mean     = k_post.mean(axis=0)
k_sd       = k_post.std(axis=0)
beta_mean  = beta_post.mean(axis=0)
beta_sd    = beta_post.std(axis=0)
alpha_mean = alpha_post.mean(axis=0)
alpha_sd   = alpha_post.std(axis=0)
delta_mean = delta_post.mean(axis=0)
delta_sd   = delta_post.std(axis=0)

subj_df = pd.DataFrame({
    "subj": subj_ids,
    "k_joint": k_mean,
    "k_joint_sd": k_sd,
    "beta_joint": beta_mean,
    "beta_joint_sd": beta_sd,
    "alpha_joint": alpha_mean,
    "alpha_joint_sd": alpha_sd,
    "delta_joint": delta_mean,
    "delta_joint_sd": delta_sd,
})

print(f"  k_joint:     mean={k_mean.mean():.3f}, SD={k_mean.std():.3f}")
print(f"  beta_joint:  mean={beta_mean.mean():.3f}, SD={beta_mean.std():.3f}")
print(f"  alpha_joint: mean={alpha_mean.mean():.3f}, SD={alpha_mean.std():.3f}")
print(f"  delta_joint: mean={delta_mean.mean():.3f}, SD={delta_mean.std():.3f}")
print(f"  sigma_delta: {pop_pop_params['sigma_delta_joint']:.4f}")

# ============================================================
# 12. Diagnose sigma_delta collapse
# ============================================================
print("\nDiagnosing sigma_delta ...", flush=True)
sigma_delta_samples = np.array(posterior_samples_joint["sigma_delta"])
print(f"  sigma_delta posterior: mean={sigma_delta_samples.mean():.4f}, "
      f"SD={sigma_delta_samples.std():.4f}, "
      f"P(>0.1)={np.mean(sigma_delta_samples > 0.1):.3f}")

delta_individual_sd = float(delta_mean.std())
print(f"  SD of delta_i posterior means: {delta_individual_sd:.4f}")

sigma_delta_collapsed = (sigma_delta_samples.mean() < 0.1)
if sigma_delta_collapsed:
    print("  WARNING: sigma_delta has collapsed — individual differences in delta not recovered.")
    print("  => delta is a population-level effect; use OLS for individual estimates.")
else:
    print("  sigma_delta is non-trivial — individual differences in delta recovered.")

# ============================================================
# 13. OLS fallback for delta if collapsed
# ============================================================
if sigma_delta_collapsed:
    print("\nRunning OLS for individual delta estimates (fallback) ...", flush=True)
    from sklearn.linear_model import LinearRegression

    # For each subject: regress excess on (1-S) to get alpha_ols + delta_ols * (1-S)
    # S = (1-T) + T/(1+lambda*D) using population lambda from joint model
    lam_pop = pop_pop_params["lambda_joint"]

    merged_copy = merged.copy()
    merged_copy["S"] = (
        (1 - merged_copy["threat"]) +
        merged_copy["threat"] / (1 + lam_pop * merged_copy["distance_H"])
    )
    merged_copy["one_minus_S"] = 1 - merged_copy["S"]

    ols_results = []
    for s in subj_ids:
        sub = merged_copy[merged_copy["subj"] == s]
        if len(sub) < 10:
            ols_results.append({"subj": s, "alpha_ols": np.nan, "delta_ols": np.nan,
                                 "delta_ols_se": np.nan, "delta_ols_p": np.nan})
            continue
        X = sub[["one_minus_S"]].values
        y = sub["excess"].values
        from sklearn.preprocessing import StandardScaler
        # Simple OLS with intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])
        # OLS closed form
        beta_ols, res, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
        alpha_ols = beta_ols[0]
        delta_ols = beta_ols[1]
        # SE via residuals
        y_hat = X_with_const @ beta_ols
        resid = y - y_hat
        n, p = len(y), 2
        s2 = np.sum(resid**2) / (n - p)
        XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        se = np.sqrt(np.diag(XtX_inv * s2))
        t_delta = delta_ols / se[1]
        p_delta = 2 * (1 - stats.t.cdf(abs(t_delta), df=n-p))
        ols_results.append({
            "subj": s,
            "alpha_ols": alpha_ols,
            "delta_ols": delta_ols,
            "delta_ols_se": se[1],
            "delta_ols_p": p_delta,
        })

    ols_df = pd.DataFrame(ols_results)
    subj_df = subj_df.merge(ols_df, on="subj", how="left")

    print(f"  OLS delta: mean={ols_df['delta_ols'].mean():.3f}, SD={ols_df['delta_ols'].std():.3f}")
    n_sig = (ols_df["delta_ols_p"] < 0.05).sum()
    print(f"  Subjects with p<0.05 for delta: {n_sig}/{len(ols_df)}")

# ============================================================
# 14. Cross-domain correlations
# ============================================================
print("\nComputing cross-domain correlations ...", flush=True)

param_cols = ["k_joint", "beta_joint", "alpha_joint", "delta_joint"]
if sigma_delta_collapsed and "delta_ols" in subj_df.columns:
    param_cols_ext = param_cols + ["delta_ols"]
else:
    param_cols_ext = param_cols

corr_results = []
pairs = [
    ("k_joint", "delta_joint", "k × delta"),
    ("beta_joint", "delta_joint", "beta × delta"),
    ("alpha_joint", "k_joint", "alpha × k"),
    ("alpha_joint", "beta_joint", "alpha × beta"),
    ("k_joint", "beta_joint", "k × beta"),
    ("alpha_joint", "delta_joint", "alpha × delta"),
]
if sigma_delta_collapsed and "delta_ols" in subj_df.columns:
    pairs += [
        ("k_joint", "delta_ols", "k × delta_ols"),
        ("beta_joint", "delta_ols", "beta × delta_ols"),
        ("alpha_joint", "delta_ols", "alpha × delta_ols"),
    ]

valid = subj_df.dropna(subset=["k_joint", "beta_joint", "alpha_joint"])
for col1, col2, label in pairs:
    if col1 not in valid.columns or col2 not in valid.columns:
        continue
    sub_valid = valid.dropna(subset=[col1, col2])
    r, p = stats.pearsonr(sub_valid[col1], sub_valid[col2])
    corr_results.append({
        "pair": label,
        "r": round(r, 4),
        "p": round(p, 4),
        "n": len(sub_valid),
        "sig": p < 0.05,
    })
    print(f"  {label}: r={r:.3f}, p={p:.3f}, n={len(sub_valid)}")

corr_df = pd.DataFrame(corr_results)

# ============================================================
# 15. Save outputs
# ============================================================
print("\nSaving outputs ...", flush=True)

# 15a. Population parameters
pop_out = pd.DataFrame([{
    "parameter": k, "value": v
} for k, v in pop_pop_params.items()])
pop_out.to_csv(STAT_DIR / "joint_model_population.csv", index=False)
print(f"  Saved: {STAT_DIR}/joint_model_population.csv")

# 15b. Subject parameters
subj_df.to_csv(STAT_DIR / "joint_model_subjects.csv", index=False)
print(f"  Saved: {STAT_DIR}/joint_model_subjects.csv")

# 15c. ELBO history
elbo_hist_df = pd.DataFrame({
    "step": list(range(1, N_STEPS+1)),
    "elbo_joint": elbo_history_joint,
    "elbo_choice": elbo_history_choice,
})
elbo_hist_df.to_csv(STAT_DIR / "joint_model_elbo_history.csv", index=False)
print(f"  Saved: {STAT_DIR}/joint_model_elbo_history.csv")

# ============================================================
# 16. Write text summary
# ============================================================
print("\nWriting text summary ...", flush=True)

lam_j = pop_pop_params["lambda_joint"]
lam_c = pop_pop_params["lambda_choice"]
lam_diff = lam_j - lam_c
tau_j = pop_pop_params["tau_joint"]
mu_k = pop_pop_params["mu_k_joint"]
sig_k = pop_pop_params["sigma_k_joint"]
mu_beta_log = pop_pop_params["mu_beta_log_joint"]
sig_beta = pop_pop_params["sigma_beta_joint"]
mu_alpha = pop_pop_params["mu_alpha_joint"]
sig_alpha = pop_pop_params["sigma_alpha_joint"]
mu_delta = pop_pop_params["mu_delta_joint"]
sig_delta = pop_pop_params["sigma_delta_joint"]
sig_excess = pop_pop_params["sigma_excess_joint"]
elbo_j = pop_pop_params["elbo_joint"]
elbo_c = pop_pop_params["elbo_choice"]
elbo_diff = pop_pop_params["elbo_diff_joint_minus_choice"]

# Correlation table
corr_table = corr_df.to_string(index=False) if len(corr_df) > 0 else "(none computed)"

delta_diagnosis = (
    "sigma_delta collapsed (< 0.1) — individual differences in delta not recovered by SVI. "
    "delta is treated as a population-level effect. "
    "OLS per-subject estimates provided as fallback (delta_ols)."
    if sigma_delta_collapsed else
    "sigma_delta is non-trivial — individual differences in delta were recovered."
)

md_text = f"""# Joint Choice-Vigor SVI Model — Results Summary

**Date:** 2026-03-20
**Dataset:** Exploratory N=293, {len(merged):,} trials
**Model:** Joint choice + vigor SVI (AutoNormal, 20,000 steps, Adam lr=0.003)

---

## Model Specification

```
Survival:  S = (1-T) + T / (1 + λ·D)

Choice:    SV_H = R_H·S - k_i·E_H - β_i·(1-S)
           SV_L = R_L·S - k_i·E_L - β_i·(1-S)
           choice ~ Bernoulli(sigmoid(τ·(SV_H - SV_L)))

Vigor:     excess_ij = α_i + δ_i·(1-S_ij) + ε_ij
           ε_ij ~ Normal(0, σ_excess)
           where excess_ij = vigor_norm_ij - effort_chosen_ij
```

Constants: R_H=5, R_L=1, E_L=0.4

---

## Population Parameters

| Parameter | Joint model | Choice-only | Δ |
|-----------|-------------|-------------|---|
| λ (hazard sensitivity) | {lam_j:.3f} | {lam_c:.3f} | {lam_diff:+.3f} |
| τ (inverse temperature) | {tau_j:.3f} | — | — |

| Hyperparameter | Mean | SD |
|---------------|------|----|
| μ_k | {mu_k:.3f} | {pop_pop_params["mu_k_joint_sd"]:.3f} |
| σ_k | {sig_k:.3f} | {pop_pop_params["sigma_k_joint_sd"]:.3f} |
| μ_β (log) | {mu_beta_log:.3f} | {pop_pop_params["mu_beta_log_joint_sd"]:.3f} |
| σ_β | {sig_beta:.3f} | {pop_pop_params["sigma_beta_joint_sd"]:.3f} |
| μ_α | {mu_alpha:.3f} | {pop_pop_params["mu_alpha_joint_sd"]:.3f} |
| σ_α | {sig_alpha:.3f} | {pop_pop_params["sigma_alpha_joint_sd"]:.3f} |
| μ_δ | {mu_delta:.3f} | {pop_pop_params["mu_delta_joint_sd"]:.3f} |
| σ_δ | {sig_delta:.3f} | {pop_pop_params["sigma_delta_joint_sd"]:.3f} |
| σ_excess | {sig_excess:.3f} | {pop_pop_params["sigma_excess_joint_sd"]:.3f} |

---

## Model Comparison (ELBO)

| Model | ELBO |
|-------|------|
| Joint (choice + vigor) | {elbo_j:.1f} |
| Choice-only | {elbo_c:.1f} |
| ΔELBO (joint − choice-only) | {elbo_diff:+.1f} |

**Note on λ comparison:** Joint model λ={lam_j:.3f}, choice-only λ={lam_c:.3f} (Δ={lam_diff:+.3f}).
{"The joint model λ is very close to the choice-only λ, indicating that the vigor observations do not substantially shift the survival function estimate from choice data alone." if abs(lam_diff) < 0.3 else "The joint model λ differs from the choice-only λ, suggesting vigor observations provide complementary constraints on the survival function."}

---

## σ_δ Diagnosis

{delta_diagnosis}

Per-subject δ posterior SD: {delta_individual_sd:.4f}
σ_δ posterior mean: {sig_delta:.4f} (prior: HalfNormal(2.0))

---

## Cross-Domain Correlations

{corr_table}

---

## Subject-Level Parameter Summary

| Parameter | Mean | SD | Min | Max |
|-----------|------|----|-----|-----|
| k_i | {subj_df["k_joint"].mean():.3f} | {subj_df["k_joint"].std():.3f} | {subj_df["k_joint"].min():.3f} | {subj_df["k_joint"].max():.3f} |
| β_i | {subj_df["beta_joint"].mean():.3f} | {subj_df["beta_joint"].std():.3f} | {subj_df["beta_joint"].min():.3f} | {subj_df["beta_joint"].max():.3f} |
| α_i | {subj_df["alpha_joint"].mean():.3f} | {subj_df["alpha_joint"].std():.3f} | {subj_df["alpha_joint"].min():.3f} | {subj_df["alpha_joint"].max():.3f} |
| δ_i | {subj_df["delta_joint"].mean():.3f} | {subj_df["delta_joint"].std():.3f} | {subj_df["delta_joint"].min():.3f} | {subj_df["delta_joint"].max():.3f} |

---

## Notes

- Vigor measure: mean vigor_norm per behavioral trial (from smoothed_vigor_ts.parquet)
- Demand: effort of chosen option (E_H or E_L); excess = vigor_norm − effort_chosen
- Probe trials excluded from vigor alignment using feelings.csv (N≈36/subject)
- Non-centered parameterization used for all per-subject parameters
- β_i is log-normal (positive constraint)
- Priors widened for σ_δ: HalfNormal(2.0) vs previous HalfNormal(1.0)
- If σ_δ collapsed, per-subject δ estimates via OLS are the recommended individual measure
"""

out_path = RESULTS_DIR / "joint_model_text.md"
out_path.write_text(md_text)
print(f"  Saved: {out_path}")

print("\n" + "="*60)
print("DONE.")
print("="*60)
print(f"  joint_model_population.csv: {STAT_DIR}/joint_model_population.csv")
print(f"  joint_model_subjects.csv:   {STAT_DIR}/joint_model_subjects.csv")
print(f"  joint_model_text.md:        {RESULTS_DIR}/joint_model_text.md")
print()
print(f"  Final ELBO — Joint: {elbo_j:.1f}, Choice-only: {elbo_c:.1f}")
print(f"  λ — Joint: {lam_j:.3f}, Choice-only: {lam_c:.3f}")
print(f"  σ_δ: {sig_delta:.4f}  ({'collapsed' if sigma_delta_collapsed else 'non-trivial'})")
