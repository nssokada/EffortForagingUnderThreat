"""
run_vigor_hbm.py
================
Bayesian Hierarchical Vigor Model (NB16 equivalent)

Two-window model:
  pre_enc_rate  ~ Normal(α_i, σ_pre)                      # [enc-2, enc]
  terminal_rate ~ Normal(γ_i + ρ_i · attack_ij, σ_term)   # [trialEnd-2, trialEnd]

Uses smoothed_vigor_ts.parquet (vigor_norm) for rate computation.

Outputs:
  results/stats/vigor_hbm_posteriors.csv
  results/stats/vigor_hbm_population.csv
  results/model_fits/exploratory/vigor_hbm_idata.nc
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

# Try to import arviz
try:
    import arviz as az
    HAS_ARVIZ = True
    print("arviz available")
except ImportError:
    HAS_ARVIZ = False
    print("arviz not available — skipping idata.nc save")

# ============================================================
# Paths
# ============================================================
ROOT = Path("/workspace")
VIGOR_PROCESSED = ROOT / "data/exploratory_350/processed/vigor_processed"
VIGOR_PREP = ROOT / "data/exploratory_350/processed/vigor_prep"
STAT_DIR = ROOT / "results/stats"
MODEL_DIR = ROOT / "results/model_fits/exploratory"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Load data
# ============================================================
print("Loading smoothed_vigor_ts.parquet ...", flush=True)
ts = pd.read_parquet(VIGOR_PROCESSED / "smoothed_vigor_ts.parquet")
print(f"  Loaded: {len(ts):,} rows, {ts['subj'].nunique()} subjects, {ts['trial'].nunique()} trials")

# Load trial_events for metadata (isAttackTrial, encounterTime, trialEndTime)
print("Loading trial_events.parquet ...", flush=True)
te = pd.read_parquet(VIGOR_PREP / "trial_events.parquet")
print(f"  Loaded: {len(te):,} trial-level rows")

# ============================================================
# 2. Compute window-averaged vigor rates
# ============================================================
print("\nComputing pre-encounter and terminal window rates ...", flush=True)

def compute_window_rates(ts_df):
    """
    For each (subj, trial), compute:
      pre_enc_rate  = mean vigor_norm in [max(0, enc-2), enc]
      terminal_rate = mean vigor_norm in [max(0, end-2), end]
    """
    results = []
    for (subj, trial), grp in ts_df.groupby(["subj", "trial"], sort=True):
        enc = grp["encounterTime"].iloc[0]
        tend = grp["trialEndTime"].iloc[0]
        isattack = int(grp["isAttackTrial"].iloc[0])

        # Pre-encounter window
        pre_win = grp[(grp["t"] >= max(0.0, enc - 2.0)) & (grp["t"] <= enc)]
        pre_rate = pre_win["vigor_norm"].mean() if len(pre_win) > 0 else np.nan

        # Terminal window
        term_win = grp[(grp["t"] >= max(0.0, tend - 2.0)) & (grp["t"] <= tend)]
        term_rate = term_win["vigor_norm"].mean() if len(term_win) > 0 else np.nan

        results.append({
            "subj": subj,
            "trial": trial,
            "isAttackTrial": isattack,
            "encounterTime": enc,
            "trialEndTime": tend,
            "pre_enc_rate": pre_rate,
            "terminal_rate": term_rate,
        })
    return pd.DataFrame(results)

trial_rates = compute_window_rates(ts)
print(f"  Trial rates computed: {len(trial_rates)} rows")
print(f"  pre_enc_rate NaN: {trial_rates['pre_enc_rate'].isna().sum()}")
print(f"  terminal_rate NaN: {trial_rates['terminal_rate'].isna().sum()}")

# ============================================================
# 3. Merge with trial metadata & filter
# ============================================================
# The ts already has all columns we need; trial_rates is derived from it
# Drop trials with NaN rates
model_data = trial_rates.dropna(subset=["pre_enc_rate", "terminal_rate"]).copy()
print(f"\nAfter dropping NaN: {len(model_data)} trials across {model_data['subj'].nunique()} subjects")

# Contiguous subject index
subj_ids = sorted(model_data["subj"].unique())
subj_to_idx = {s: i for i, s in enumerate(subj_ids)}
model_data["subj_idx"] = model_data["subj"].map(subj_to_idx)
n_subj = len(subj_ids)
n_obs = len(model_data)
print(f"n_subj={n_subj}, n_obs={n_obs}")

# Summary stats
print(f"\npre_enc_rate: mean={model_data['pre_enc_rate'].mean():.4f}, "
      f"sd={model_data['pre_enc_rate'].std():.4f}, "
      f"range=[{model_data['pre_enc_rate'].min():.4f}, {model_data['pre_enc_rate'].max():.4f}]")
print(f"terminal_rate: mean={model_data['terminal_rate'].mean():.4f}, "
      f"sd={model_data['terminal_rate'].std():.4f}, "
      f"range=[{model_data['terminal_rate'].min():.4f}, {model_data['terminal_rate'].max():.4f}]")
print(f"Attack trials: {model_data['isAttackTrial'].sum()} / {len(model_data)}")

# ============================================================
# 4. JAX arrays
# ============================================================
subj_idx_jax = jnp.array(model_data["subj_idx"].values)
attack_jax    = jnp.array(model_data["isAttackTrial"].values, dtype=float)
rate_pre_jax  = jnp.array(model_data["pre_enc_rate"].values)
rate_term_jax = jnp.array(model_data["terminal_rate"].values)

# ============================================================
# 5. Model definition
# ============================================================
def vigor_model(subj_idx, attack, rate_pre=None, rate_term=None, n_subj=293):
    """
    Two-window hierarchical Bayesian vigor model.

    pre_enc_rate_ij  ~ Normal(α_i, σ_pre)                     [enc-2, enc]
    terminal_rate_ij ~ Normal(γ_i + ρ_i · attack_ij, σ_term)  [trialEnd-2, trialEnd]

    α_i ~ Normal(μ_α, σ_α)     tonic vigor
    γ_i ~ Normal(μ_γ, σ_γ)     nuisance terminal baseline
    ρ_i ~ Normal(μ_ρ, σ_ρ)     phasic attack boost
    """
    # --- Population hyperpriors ---
    mu_alpha  = numpyro.sample("mu_alpha",  dist.Normal(0.5, 0.5))
    mu_gamma  = numpyro.sample("mu_gamma",  dist.Normal(0.5, 0.5))
    mu_rho    = numpyro.sample("mu_rho",    dist.Normal(0.0, 0.5))

    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(0.5))
    sigma_gamma = numpyro.sample("sigma_gamma", dist.HalfNormal(0.5))
    sigma_rho   = numpyro.sample("sigma_rho",   dist.HalfNormal(0.5))

    # Observation noise
    sigma_pre  = numpyro.sample("sigma_pre",  dist.HalfNormal(0.5))
    sigma_term = numpyro.sample("sigma_term", dist.HalfNormal(0.5))

    # --- Subject-level parameters (non-centered) ---
    with numpyro.plate("subjects", n_subj):
        alpha_z = numpyro.sample("alpha_z", dist.Normal(0, 1))
        gamma_z = numpyro.sample("gamma_z", dist.Normal(0, 1))
        rho_z   = numpyro.sample("rho_z",   dist.Normal(0, 1))

    alpha = numpyro.deterministic("alpha", mu_alpha + sigma_alpha * alpha_z)
    gamma = numpyro.deterministic("gamma", mu_gamma + sigma_gamma * gamma_z)
    rho   = numpyro.deterministic("rho",   mu_rho   + sigma_rho   * rho_z)

    # --- Likelihoods ---
    # Pre-encounter: [enc-2, enc] → α
    mu_pre = alpha[subj_idx]
    with numpyro.plate("obs_pre_enc", len(subj_idx)):
        obs_pre = numpyro.sample("obs_pre", dist.Normal(mu_pre, sigma_pre), obs=rate_pre)

    # Terminal: [trialEnd-2, trialEnd] → γ + ρ·attack
    mu_term = gamma[subj_idx] + rho[subj_idx] * attack
    with numpyro.plate("obs_terminal", len(subj_idx)):
        obs_term = numpyro.sample("obs_term", dist.Normal(mu_term, sigma_term), obs=rate_term)

# ============================================================
# 6. Run MCMC
# ============================================================
print("\n" + "="*60)
print("Running MCMC (4 chains × 1000 warmup + 1000 samples) ...")
print("="*60, flush=True)

kernel = NUTS(vigor_model, target_accept_prob=0.9)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=4,
            progress_bar=True)

rng_key = jax.random.PRNGKey(42)
mcmc.run(rng_key,
         subj_idx=subj_idx_jax,
         attack=attack_jax,
         rate_pre=rate_pre_jax,
         rate_term=rate_term_jax,
         n_subj=n_subj)

print("\nMCMC complete.", flush=True)
mcmc.print_summary(exclude_deterministic=True)

# ============================================================
# 7. Extract posteriors
# ============================================================
posterior = mcmc.get_samples()

# Population parameters
mu_alpha_samples  = np.array(posterior["mu_alpha"])
mu_rho_samples    = np.array(posterior["mu_rho"])
mu_gamma_samples  = np.array(posterior["mu_gamma"])
sigma_alpha_s     = np.array(posterior["sigma_alpha"])
sigma_gamma_s     = np.array(posterior["sigma_gamma"])
sigma_rho_s       = np.array(posterior["sigma_rho"])
sigma_pre_s       = np.array(posterior["sigma_pre"])
sigma_term_s      = np.array(posterior["sigma_term"])

# Subject-level
alpha_samples = np.array(posterior["alpha"])  # (n_samples, n_subj)
rho_samples   = np.array(posterior["rho"])
gamma_samples = np.array(posterior["gamma"])

# ============================================================
# 8. Convergence diagnostics
# ============================================================
n_divergent = 0
max_rhat_alpha = np.nan
max_rhat_rho = np.nan

if HAS_ARVIZ:
    idata = az.from_numpyro(mcmc)
    # Save NetCDF
    nc_path = MODEL_DIR / "vigor_hbm_idata.nc"
    idata.to_netcdf(str(nc_path))
    print(f"\nSaved idata: {nc_path}")

    # Diagnostics
    pop_params = ["mu_alpha", "mu_gamma", "mu_rho",
                  "sigma_alpha", "sigma_gamma", "sigma_rho",
                  "sigma_pre", "sigma_term"]
    pop_summary = az.summary(idata, var_names=pop_params)
    print("\n=== Population parameters ===")
    print(pop_summary.round(4))

    all_rhat = az.rhat(idata)
    if "alpha" in all_rhat:
        max_rhat_alpha = float(np.array(all_rhat["alpha"]).max())
    if "rho" in all_rhat:
        max_rhat_rho = float(np.array(all_rhat["rho"]).max())
    print(f"\nMax Rhat — alpha: {max_rhat_alpha:.4f}, rho: {max_rhat_rho:.4f}")

    # Divergences
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_divergent = int(idata.sample_stats["diverging"].values.sum())
    print(f"Divergences: {n_divergent}")
else:
    # Basic Rhat from raw chains
    print("\narviz not available; skipping full diagnostics")

# ============================================================
# 9. Population-level sprint test
# ============================================================
prob_rho_pos = float((mu_rho_samples > 0).mean())
hdi_lo_rho = float(np.percentile(mu_rho_samples, 2.5))
hdi_hi_rho = float(np.percentile(mu_rho_samples, 97.5))
hdi_lo_alpha = float(np.percentile(mu_alpha_samples, 2.5))
hdi_hi_alpha = float(np.percentile(mu_alpha_samples, 97.5))

print(f"\n=== Group-level tonic vigor (μ_α) ===")
print(f"  Mean:  {mu_alpha_samples.mean():.4f}")
print(f"  SD:    {mu_alpha_samples.std():.4f}")
print(f"  95% CI: [{hdi_lo_alpha:.4f}, {hdi_hi_alpha:.4f}]")

print(f"\n=== Group-level sprint effect (μ_ρ) ===")
print(f"  Mean:  {mu_rho_samples.mean():.4f}")
print(f"  SD:    {mu_rho_samples.std():.4f}")
print(f"  95% CI: [{hdi_lo_rho:.4f}, {hdi_hi_rho:.4f}]")
print(f"  P(μ_ρ > 0): {prob_rho_pos:.4f}")

# ============================================================
# 10. Subject-level summary + shrinkage
# ============================================================
alpha_mean = alpha_samples.mean(axis=0)
alpha_sd   = alpha_samples.std(axis=0)
rho_mean   = rho_samples.mean(axis=0)
rho_sd     = rho_samples.std(axis=0)
gamma_mean = gamma_samples.mean(axis=0)
gamma_sd   = gamma_samples.std(axis=0)

# OLS estimates for shrinkage comparison
ols_alpha = model_data.groupby("subj_idx")["pre_enc_rate"].mean().reindex(range(n_subj)).values
ols_rho_attack = (model_data[model_data["isAttackTrial"]==1]
                  .groupby("subj_idx")["terminal_rate"].mean()
                  .reindex(range(n_subj)).values)
ols_rho_noatk  = (model_data[model_data["isAttackTrial"]==0]
                  .groupby("subj_idx")["terminal_rate"].mean()
                  .reindex(range(n_subj)).values)
ols_rho = ols_rho_attack - ols_rho_noatk

# Shrinkage Bayes factor proxy: ratio of posterior SD to prior SD
sb_alpha = 1.0 - (alpha_sd.mean() / float(sigma_alpha_s.mean()))
sb_rho   = 1.0 - (rho_sd.mean() / float(sigma_rho_s.mean()))
print(f"\nShrinkage (Bayes) — α: {sb_alpha:.3f}, ρ: {sb_rho:.3f}")

# Build subject summary DataFrame
subj_summary = pd.DataFrame({
    "subj":       subj_ids,
    "subj_idx":   list(range(n_subj)),
    "alpha_bayes": alpha_mean,
    "alpha_sd":    alpha_sd,
    "rho_bayes":   rho_mean,
    "rho_sd":      rho_sd,
    "gamma_bayes": gamma_mean,
    "gamma_sd":    gamma_sd,
    "alpha_ols":   ols_alpha,
    "rho_ols":     ols_rho,
})

# ============================================================
# 11. Validation
# ============================================================
print("\n=== Validation ===")

# α-OLS correlation
r_ao, p_ao = stats.pearsonr(subj_summary["alpha_bayes"].values,
                             subj_summary["alpha_ols"].values)
print(f"α_bayes vs α_OLS: r={r_ao:.3f}, p={p_ao:.4f}")

# α-ρ independence
r_ar, p_ar = stats.pearsonr(subj_summary["alpha_bayes"].values,
                             subj_summary["rho_bayes"].values)
print(f"α-ρ correlation: r={r_ar:.3f}, p={p_ar:.4f}")

# ============================================================
# 12. Split-half reliability
# ============================================================
print("\n=== Split-half reliability ===")

odd_trials  = model_data[model_data["trial"] % 2 == 1]
even_trials = model_data[model_data["trial"] % 2 == 0]

# α split-half (pre-enc rate)
alpha_odd  = odd_trials.groupby("subj_idx")["pre_enc_rate"].mean().reindex(range(n_subj))
alpha_even = even_trials.groupby("subj_idx")["pre_enc_rate"].mean().reindex(range(n_subj))
valid_a    = (~alpha_odd.isna()) & (~alpha_even.isna())
r_alpha_sh, _ = stats.pearsonr(alpha_odd[valid_a], alpha_even[valid_a])
r_alpha_sb = (2 * r_alpha_sh) / (1 + r_alpha_sh)  # Spearman-Brown correction

# ρ split-half (terminal attack contrast)
def rho_ols_from_data(df):
    atk  = df[df["isAttackTrial"]==1].groupby("subj_idx")["terminal_rate"].mean()
    noatk = df[df["isAttackTrial"]==0].groupby("subj_idx")["terminal_rate"].mean()
    return (atk - noatk).reindex(range(n_subj))

rho_odd   = rho_ols_from_data(odd_trials)
rho_even  = rho_ols_from_data(even_trials)
valid_r   = (~rho_odd.isna()) & (~rho_even.isna())
r_rho_sh, _ = stats.pearsonr(rho_odd[valid_r], rho_even[valid_r])
r_rho_sb = (2 * r_rho_sh) / (1 + r_rho_sh)

print(f"α split-half r = {r_alpha_sh:.3f}, Spearman-Brown = {r_alpha_sb:.3f}")
print(f"ρ split-half r = {r_rho_sh:.3f}, Spearman-Brown = {r_rho_sb:.3f}")

# ============================================================
# 13. Save outputs
# ============================================================
print("\n=== Saving outputs ===")

# Per-subject posteriors
subj_summary.to_csv(STAT_DIR / "vigor_hbm_posteriors.csv", index=False)
print(f"Saved: {STAT_DIR / 'vigor_hbm_posteriors.csv'}")
print(f"  Columns: {subj_summary.columns.tolist()}")

# Population hyperparameters
pop_df = pd.DataFrame([{
    "mu_alpha_mean":  float(mu_alpha_samples.mean()),
    "mu_alpha_hdi_lo": hdi_lo_alpha,
    "mu_alpha_hdi_hi": hdi_hi_alpha,
    "mu_rho_mean":    float(mu_rho_samples.mean()),
    "mu_rho_hdi_lo":  hdi_lo_rho,
    "mu_rho_hdi_hi":  hdi_hi_rho,
    "p_rho_pos":      prob_rho_pos,
    "sigma_alpha":    float(sigma_alpha_s.mean()),
    "sigma_gamma":    float(sigma_gamma_s.mean()),
    "sigma_rho":      float(sigma_rho_s.mean()),
    "sigma_pre":      float(sigma_pre_s.mean()),
    "sigma_term":     float(sigma_term_s.mean()),
    "r_alpha_split_half": r_alpha_sh,
    "r_alpha_sb":         r_alpha_sb,
    "r_rho_split_half":   r_rho_sh,
    "r_rho_sb":           r_rho_sb,
    "r_alpha_ols":        r_ao,
    "r_alpha_rho":        r_ar,
    "shrinkage_alpha":    sb_alpha,
    "shrinkage_rho":      sb_rho,
    "n_subj":         n_subj,
    "n_obs":          n_obs,
    "n_divergent":    n_divergent,
    "max_rhat_alpha": max_rhat_alpha,
    "max_rhat_rho":   max_rhat_rho,
}])
pop_df.to_csv(STAT_DIR / "vigor_hbm_population.csv", index=False)
print(f"Saved: {STAT_DIR / 'vigor_hbm_population.csv'}")

print("\nDone.")
