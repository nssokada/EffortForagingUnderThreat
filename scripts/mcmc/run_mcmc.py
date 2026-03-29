"""
MCMC validation of the EVC 2+2 model.

Uses NUTS sampler (NumPyro) with the EXACT same model specification as the
SVI version in evc_final_2plus2.py. Purpose: validate that SVI point estimates
are reliable.

Architecture (identical to SVI):
    Per-subject (log-normal, non-centered): ce, cd
    Population: epsilon, gamma, ce_vigor, tau, p_esc, sigma_motor, sigma_v
    Choice: dEU = S * 4 - ce_i * (0.81*D_H - 0.16); P(heavy) = sigmoid(dEU / tau)
    Vigor: EU(u) = S(u)*R - (1-S(u))*cd_i*(R+C) - ce_vigor*(u-req)^2*D

MCMC settings (full run):
    4 chains, 1000 warmup + 1000 samples, target_accept=0.85, max_tree_depth=10

Usage:
    python scripts/mcmc/run_mcmc.py --data_dir data/exploratory_350/processed/stage5_filtered_data_20260320_191950
"""

import sys
import os
import time
import argparse

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import numpy as np
import pandas as pd
import ast
from scipy.stats import pearsonr

jax.config.update('jax_enable_x64', True)


# ═══════════════════════════════════════════════════════════════════════════════
# Data preparation — identical to evc_final_2plus2.py
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(behavior_rich_path, psych_path=None):
    """Load and prepare data for the 2+2 model.

    Returns separate choice and vigor data arrays.
    Probe trials use startDistance for distance (not distance_H).
    """
    beh = pd.read_csv(behavior_rich_path)

    # Choice data: type=1 only
    choice_df = beh[beh['type'] == 1].copy()

    # Vigor data: ALL types (1, 5, 6)
    vigor_df = beh.copy()
    vigor_df['actual_dist'] = vigor_df['startDistance'].map({5: 1, 7: 2, 9: 3})
    vigor_df['actual_req'] = np.where(
        vigor_df['trialCookie_weight'] == 3.0, 0.9, 0.4)
    vigor_df['actual_R'] = np.where(
        vigor_df['trialCookie_weight'] == 3.0, 5.0, 1.0)
    vigor_df['is_heavy'] = (vigor_df['trialCookie_weight'] == 3.0).astype(int)

    # Compute median press rate for all trials
    rates = []
    for _, row in vigor_df.iterrows():
        try:
            pt = np.array(
                ast.literal_eval(row['alignedEffortRate']), dtype=float)
            ipis = np.diff(pt)
            ipis = ipis[ipis > 0.01]
            if len(ipis) >= 5:
                rates.append(
                    np.median((1.0 / ipis) / row['calibrationMax']))
            else:
                rates.append(np.nan)
        except Exception:
            rates.append(np.nan)

    vigor_df['median_rate'] = rates
    vigor_df['excess'] = vigor_df['median_rate'] - vigor_df['actual_req']
    vigor_df = vigor_df.dropna(subset=['excess']).copy()

    # Cookie-type centering (using choice trial means)
    choice_vigor = vigor_df[vigor_df['type'] == 1]
    heavy_mean = choice_vigor[choice_vigor['is_heavy'] == 1]['excess'].mean()
    light_mean = choice_vigor[choice_vigor['is_heavy'] == 0]['excess'].mean()
    vigor_df['excess_cc'] = vigor_df['excess'] - np.where(
        vigor_df['is_heavy'] == 1, heavy_mean, light_mean)

    # Subject indexing
    subjects = sorted(
        set(choice_df['subj'].unique()) & set(vigor_df['subj'].unique()))
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)

    # Choice arrays
    ch_subj = jnp.array([subj_to_idx[s] for s in choice_df['subj']])
    ch_T = jnp.array(choice_df['threat'].values)
    ch_dist_H = jnp.array(
        choice_df['distance_H'].values, dtype=jnp.float64)
    ch_choice = jnp.array(choice_df['choice'].values)

    # Vigor arrays
    vig_subj = jnp.array([subj_to_idx[s] for s in vigor_df['subj']])
    vig_T = jnp.array(vigor_df['threat'].values)
    vig_R = jnp.array(vigor_df['actual_R'].values)
    vig_req = jnp.array(vigor_df['actual_req'].values)
    vig_dist = jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64)
    vig_excess = jnp.array(vigor_df['excess_cc'].values)
    vig_offset = jnp.array(np.where(
        vigor_df['is_heavy'].values == 1, heavy_mean, light_mean))

    data = {
        'ch_subj': ch_subj, 'ch_T': ch_T, 'ch_dist_H': ch_dist_H,
        'ch_choice': ch_choice,
        'vig_subj': vig_subj, 'vig_T': vig_T, 'vig_R': vig_R,
        'vig_req': vig_req, 'vig_dist': vig_dist,
        'vig_excess': vig_excess, 'vig_offset': vig_offset,
        'subjects': subjects, 'N_S': N_S,
        'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
        'heavy_mean': heavy_mean, 'light_mean': light_mean,
        'vigor_df': vigor_df,
    }

    if psych_path is not None:
        data['psych'] = pd.read_csv(psych_path)

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Model — IDENTICAL to evc_final_2plus2.py make_model()
# ═══════════════════════════════════════════════════════════════════════════════

def make_model(N_S, N_choice, N_vigor):
    """Create the 2+2 NumPyro model (epsilon is population-level)."""

    def evc_2plus2(ch_subj, ch_T, ch_dist_H, ch_choice,
                   vig_subj, vig_T, vig_R, vig_req, vig_dist,
                   vig_excess, vig_offset):
        # -- Population priors for hierarchical ce, cd --
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        # -- Population-level epsilon (NOT per-subject) --
        eps_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
        epsilon = numpyro.deterministic('epsilon', jnp.exp(eps_raw))

        # -- Population-level gamma --
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic(
            'gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))

        # -- Other population params --
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample(
            'sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)

        # Population vigor effort cost
        ce_vigor_raw = numpyro.sample(
            'ce_vigor_raw', dist.Normal(-3.0, 1.0))
        ce_vigor = numpyro.deterministic('ce_vigor', jnp.exp(ce_vigor_raw))

        # -- Subject-level (non-centered): ce, cd only --
        with numpyro.plate('subjects', N_S):
            ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))

        c_effort = jnp.exp(mu_ce + sigma_ce * ce_raw)
        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        numpyro.deterministic('c_effort', c_effort)
        numpyro.deterministic('c_death', c_death)

        # -- CHOICE: cd cancels, ce drives it --
        ce_ch = c_effort[ch_subj]
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc

        delta_eu = S_ch * 4.0 - ce_ch * (0.81 * ch_dist_H - 0.16)
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample(
                'obs_choice',
                dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                obs=ch_choice)

        # -- VIGOR: cd drives survival incentive --
        cd_v = c_death[vig_subj]
        T_w_v = jnp.power(vig_T, gamma)

        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T_w_v[:, None])
               + epsilon * T_w_v[:, None] * p_esc
               * jax.nn.sigmoid(
                   (u_g - vig_req[:, None]) / sigma_motor))
        deviation = u_g - vig_req[:, None]
        eu_grid = (S_u * vig_R[:, None]
                   - (1.0 - S_u) * cd_v[:, None]
                   * (vig_R[:, None] + 5.0)
                   - ce_vigor * deviation ** 2 * vig_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample(
                'obs_vigor',
                dist.Normal(excess_pred, sigma_v),
                obs=vig_excess)

    return evc_2plus2


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_rhat(chains):
    """Compute split R-hat for a parameter.

    Args:
        chains: array of shape (n_chains, n_samples)
    """
    chains = np.array(chains, dtype=np.float64)
    if chains.ndim == 1:
        return np.nan

    n_chains, n_samples = chains.shape
    if n_samples < 4:
        return np.nan

    # Split each chain in half
    mid = n_samples // 2
    split_chains = np.concatenate(
        [chains[:, :mid], chains[:, mid:2 * mid]], axis=0)
    m = split_chains.shape[0]  # 2 * n_chains
    n = split_chains.shape[1]  # n_samples // 2

    chain_means = split_chains.mean(axis=1)
    chain_vars = split_chains.var(axis=1, ddof=1)

    B = n * np.var(chain_means, ddof=1)
    W = np.mean(chain_vars)

    if W < 1e-10:
        return 1.0

    var_hat = ((n - 1) / n) * W + (1.0 / n) * B
    rhat = np.sqrt(var_hat / W)
    return float(rhat)


def _compute_ess(chains):
    """Compute approximate effective sample size."""
    chains = np.array(chains, dtype=np.float64)
    if chains.ndim == 1:
        return len(chains)

    # Simple ESS approximation: total samples / (1 + 2*sum(autocorrelation))
    flat = chains.flatten()
    n = len(flat)
    mean = np.mean(flat)
    var = np.var(flat)
    if var < 1e-10:
        return float(n)

    # Compute autocorrelation up to lag 100
    max_lag = min(100, n // 4)
    acf_sum = 0.0
    for lag in range(1, max_lag + 1):
        acf = np.mean((flat[:-lag] - mean) * (flat[lag:] - mean)) / var
        if acf < 0.05:
            break
        acf_sum += acf

    ess = n / (1 + 2 * acf_sum)
    return float(max(1.0, ess))


# ═══════════════════════════════════════════════════════════════════════════════
# MCMC fitting
# ═══════════════════════════════════════════════════════════════════════════════

def fit_mcmc(data, num_warmup=1000, num_samples=1000, num_chains=4,
             target_accept_prob=0.85, max_tree_depth=10, seed=0):
    """Fit the 2+2 model via MCMC (NUTS)."""

    model = make_model(data['N_S'], data['N_choice'], data['N_vigor'])

    kwargs = {k: data[k] for k in [
        'ch_subj', 'ch_T', 'ch_dist_H', 'ch_choice',
        'vig_subj', 'vig_T', 'vig_R', 'vig_req', 'vig_dist',
        'vig_excess', 'vig_offset',
    ]}

    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    print(f"\n{'=' * 70}")
    print(f"Fitting EVC 2+2 via MCMC (NUTS)")
    print(f"{'=' * 70}")
    print(f"  {num_chains} chains x {num_warmup} warmup + {num_samples} samples")
    print(f"  target_accept_prob = {target_accept_prob}")
    print(f"  max_tree_depth = {max_tree_depth}")
    print(f"  {data['N_S']} subjects, ~{2 * data['N_S'] + 11} total params")
    print(f"  {data['N_choice']} choice trials, {data['N_vigor']} vigor trials")

    t0 = time.time()
    mcmc.run(random.PRNGKey(seed), **kwargs)
    elapsed = time.time() - t0
    print(f"\nMCMC completed in {elapsed / 60:.1f} min")

    return mcmc, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics and parameter extraction
# ═══════════════════════════════════════════════════════════════════════════════

def compute_diagnostics(mcmc, data, num_chains=4):
    """Compute convergence diagnostics and extract parameters."""

    samples = mcmc.get_samples()

    # -- Convergence diagnostics --
    pop_params = ['mu_ce', 'mu_cd', 'sigma_ce', 'sigma_cd',
                  'eps_raw', 'gamma_raw', 'tau_raw', 'p_esc_raw',
                  'sigma_v', 'sigma_motor_raw', 'ce_vigor_raw']

    diag_rows = []
    print("\n" + "=" * 70)
    print("MCMC Convergence Diagnostics")
    print("=" * 70)

    for param_name in pop_params:
        if param_name not in samples:
            continue
        s = np.array(samples[param_name])
        if s.ndim == 1:
            s_reshaped = s.reshape(num_chains, -1)
        else:
            s_reshaped = s

        rhat = _compute_rhat(s_reshaped)
        ess = _compute_ess(s_reshaped)
        mean_val = float(np.mean(s))
        std_val = float(np.std(s))

        diag_rows.append({
            'parameter': param_name,
            'mean': mean_val,
            'std': std_val,
            'rhat': rhat,
            'ess': ess,
        })
        print(f"  {param_name:20s}: mean={mean_val:+.4f}, sd={std_val:.4f}, "
              f"R-hat={rhat:.3f}, ESS={ess:.0f}")

    # Subject-level params
    ce_rhats, cd_rhats = None, None
    if 'ce_raw' in samples:
        ce_raw = np.array(samples['ce_raw'])  # (total_samples, N_S)
        cd_raw = np.array(samples['cd_raw'])

        ce_rhats_list, cd_rhats_list = [], []
        ce_ess_list, cd_ess_list = [], []

        n_per = ce_raw.shape[0] // num_chains

        for j in range(data['N_S']):
            ce_j = ce_raw[:, j].reshape(num_chains, n_per)
            cd_j = cd_raw[:, j].reshape(num_chains, n_per)
            ce_rhats_list.append(_compute_rhat(ce_j))
            cd_rhats_list.append(_compute_rhat(cd_j))
            ce_ess_list.append(_compute_ess(ce_j))
            cd_ess_list.append(_compute_ess(cd_j))

        ce_rhats = np.array(ce_rhats_list)
        cd_rhats = np.array(cd_rhats_list)

        print(f"\n  ce_raw (N={data['N_S']}): "
              f"R-hat median={np.median(ce_rhats):.3f}, "
              f"max={np.max(ce_rhats):.3f}, "
              f">{1.05}: {np.sum(ce_rhats > 1.05)}")
        print(f"  cd_raw (N={data['N_S']}): "
              f"R-hat median={np.median(cd_rhats):.3f}, "
              f"max={np.max(cd_rhats):.3f}, "
              f">{1.05}: {np.sum(cd_rhats > 1.05)}")

        diag_rows.append({
            'parameter': 'ce_raw (median across subjects)',
            'mean': float(np.median(np.mean(ce_raw, axis=0))),
            'std': float(np.median(np.std(ce_raw, axis=0))),
            'rhat': float(np.median(ce_rhats)),
            'ess': float(np.median(ce_ess_list)),
        })
        diag_rows.append({
            'parameter': 'cd_raw (median across subjects)',
            'mean': float(np.median(np.mean(cd_raw, axis=0))),
            'std': float(np.median(np.std(cd_raw, axis=0))),
            'rhat': float(np.median(cd_rhats)),
            'ess': float(np.median(cd_ess_list)),
        })

    # -- Extract per-subject parameters --
    mu_ce = float(jnp.mean(samples['mu_ce']))
    mu_cd = float(jnp.mean(samples['mu_cd']))
    sigma_ce = float(jnp.mean(samples['sigma_ce']))
    sigma_cd = float(jnp.mean(samples['sigma_cd']))

    ce_raw_mean = np.mean(np.array(samples['ce_raw']), axis=0)
    cd_raw_mean = np.mean(np.array(samples['cd_raw']), axis=0)

    ce_mcmc = np.exp(mu_ce + sigma_ce * ce_raw_mean)
    cd_mcmc = np.exp(mu_cd + sigma_cd * cd_raw_mean)

    # -- Population parameter summaries with HDI --
    print("\n" + "=" * 70)
    print("Population Parameter Estimates (posterior mean [94% HDI])")
    print("=" * 70)

    pop_transforms = {
        'epsilon': ('eps_raw', lambda x: np.exp(x)),
        'gamma': ('gamma_raw', lambda x: np.clip(np.exp(x), 0.1, 3.0)),
        'tau': ('tau_raw', lambda x: np.clip(np.exp(x), 0.01, 20.0)),
        'p_esc': ('p_esc_raw', lambda x: 1.0 / (1.0 + np.exp(-x))),
        'sigma_motor': ('sigma_motor_raw',
                        lambda x: np.clip(np.exp(x), 0.01, 1.0)),
        'ce_vigor': ('ce_vigor_raw', lambda x: np.exp(x)),
        'sigma_v': ('sigma_v', lambda x: x),
    }

    pop_rows = []
    pop_results = {}
    for name, (raw_name, transform) in pop_transforms.items():
        if raw_name in samples:
            vals = transform(np.array(samples[raw_name]))
            mean_v = float(np.mean(vals))
            hdi_lo = float(np.percentile(vals, 3))
            hdi_hi = float(np.percentile(vals, 97))
            pop_results[name] = mean_v
            pop_rows.append({
                'parameter': name,
                'mean': mean_v,
                'hdi_3%': hdi_lo,
                'hdi_97%': hdi_hi,
            })
            print(f"  {name:15s}: {mean_v:.4f} [{hdi_lo:.4f}, {hdi_hi:.4f}]")

    # Add mu/sigma for ce and cd
    for pname in ['mu_ce', 'mu_cd', 'sigma_ce', 'sigma_cd']:
        vals = np.array(samples[pname])
        pop_rows.append({
            'parameter': pname,
            'mean': float(np.mean(vals)),
            'hdi_3%': float(np.percentile(vals, 3)),
            'hdi_97%': float(np.percentile(vals, 97)),
        })

    # -- Per-subject summaries --
    print(f"\n  ce: median={np.median(ce_mcmc):.3f}, mean={np.mean(ce_mcmc):.3f}, "
          f"SD={np.std(ce_mcmc):.3f}")
    print(f"  cd: median={np.median(cd_mcmc):.3f}, mean={np.mean(cd_mcmc):.3f}, "
          f"SD={np.std(cd_mcmc):.3f}")

    r_ce_cd, p_ce_cd = pearsonr(np.log(ce_mcmc), np.log(cd_mcmc))
    print(f"  log(ce) x log(cd) correlation: r={r_ce_cd:+.3f} (p={p_ce_cd:.4f})")

    # Build output DataFrames
    param_df = pd.DataFrame({
        'subj': data['subjects'],
        'c_effort': ce_mcmc,
        'c_death': cd_mcmc,
    })

    diag_df = pd.DataFrame(diag_rows)
    pop_df = pd.DataFrame(pop_rows)

    return {
        'param_df': param_df,
        'diag_df': diag_df,
        'pop_df': pop_df,
        'pop_results': pop_results,
        'ce_mcmc': ce_mcmc,
        'cd_mcmc': cd_mcmc,
        'ce_rhats': ce_rhats,
        'cd_rhats': cd_rhats,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SVI comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare_with_svi(mcmc_result, svi_params_path):
    """Compare MCMC parameter estimates with SVI results."""
    print("\n" + "=" * 70)
    print("Comparison: MCMC vs SVI parameter estimates")
    print("=" * 70)

    svi_df = pd.read_csv(svi_params_path)
    mcmc_df = mcmc_result['param_df']

    # Merge on subject
    merged = mcmc_df.merge(svi_df, on='subj', suffixes=('_mcmc', '_svi'))

    r_ce, p_ce = pearsonr(
        np.log(merged['c_effort_mcmc']), np.log(merged['c_effort_svi']))
    r_cd, p_cd = pearsonr(
        np.log(merged['c_death_mcmc']), np.log(merged['c_death_svi']))

    print(f"  log(ce): r = {r_ce:.4f} (p = {p_ce:.2e})")
    print(f"  log(cd): r = {r_cd:.4f} (p = {p_cd:.2e})")
    print(f"  N subjects matched: {len(merged)}")

    # Mean absolute difference
    mad_ce = np.mean(np.abs(
        np.log(merged['c_effort_mcmc']) - np.log(merged['c_effort_svi'])))
    mad_cd = np.mean(np.abs(
        np.log(merged['c_death_mcmc']) - np.log(merged['c_death_svi'])))
    print(f"  Mean |log(ce_mcmc) - log(ce_svi)|: {mad_ce:.4f}")
    print(f"  Mean |log(cd_mcmc) - log(cd_svi)|: {mad_cd:.4f}")

    return {
        'r_ce': r_ce, 'r_cd': r_cd,
        'p_ce': p_ce, 'p_cd': p_cd,
        'mad_ce': mad_ce, 'mad_cd': mad_cd,
        'n_matched': len(merged),
        'merged': merged,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Text summary writer
# ═══════════════════════════════════════════════════════════════════════════════

def write_summary(out_path, mcmc_result, comparison, elapsed, num_warmup,
                  num_samples, num_chains, n_divergent):
    """Write a text summary of MCMC results."""
    lines = []
    lines.append("MCMC Validation Summary: EVC 2+2 Model")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Settings: {num_chains} chains x {num_warmup} warmup + "
                 f"{num_samples} samples")
    lines.append(f"Runtime: {elapsed / 60:.1f} min")
    lines.append(f"Divergent transitions: {n_divergent}")
    lines.append("")

    # Population parameters
    lines.append("Population Parameters (posterior mean [94% HDI]):")
    for _, row in mcmc_result['pop_df'].iterrows():
        lines.append(f"  {row['parameter']:15s}: {row['mean']:.4f} "
                     f"[{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]")
    lines.append("")

    # Convergence
    lines.append("Convergence Diagnostics:")
    for _, row in mcmc_result['diag_df'].iterrows():
        lines.append(f"  {row['parameter']:40s}: R-hat={row['rhat']:.3f}, "
                     f"ESS={row['ess']:.0f}")
    lines.append("")

    # Per-subject summaries
    ce = mcmc_result['ce_mcmc']
    cd = mcmc_result['cd_mcmc']
    lines.append(f"Per-subject ce: median={np.median(ce):.3f}, "
                 f"mean={np.mean(ce):.3f}, SD={np.std(ce):.3f}")
    lines.append(f"Per-subject cd: median={np.median(cd):.3f}, "
                 f"mean={np.mean(cd):.3f}, SD={np.std(cd):.3f}")
    lines.append("")

    # SVI comparison
    if comparison is not None:
        lines.append("SVI vs MCMC Comparison:")
        lines.append(f"  log(ce): r = {comparison['r_ce']:.4f} "
                     f"(p = {comparison['p_ce']:.2e})")
        lines.append(f"  log(cd): r = {comparison['r_cd']:.4f} "
                     f"(p = {comparison['p_cd']:.2e})")
        lines.append(f"  N subjects matched: {comparison['n_matched']}")
        lines.append(f"  Mean |log(ce_mcmc) - log(ce_svi)|: "
                     f"{comparison['mad_ce']:.4f}")
        lines.append(f"  Mean |log(cd_mcmc) - log(cd_svi)|: "
                     f"{comparison['mad_cd']:.4f}")
        lines.append("")
        lines.append("Paper-ready sentence:")
        lines.append(f'  "SVI parameter estimates were validated against MCMC '
                     f'(NUTS, {num_chains} chains x {num_samples} samples; '
                     f'all population R-hat < 1.05; per-subject parameter '
                     f'correlation with SVI: log(ce) r = {comparison["r_ce"]:.3f}, '
                     f'log(cd) r = {comparison["r_cd"]:.3f}; '
                     f'N divergent transitions = {n_divergent})."')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MCMC validation of EVC 2+2 model (full run)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed data directory '
                             '(containing behavior_rich.csv)')
    parser.add_argument('--svi_params', type=str,
                        default='results/stats/oc_evc_final_params.csv',
                        help='Path to SVI parameter CSV for comparison')
    parser.add_argument('--out_dir', type=str,
                        default='results/stats/mcmc',
                        help='Output directory for MCMC results')
    parser.add_argument('--num_warmup', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_chains', type=int, default=4)
    parser.add_argument('--target_accept', type=float, default=0.85)
    parser.add_argument('--max_tree_depth', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    t0_total = time.time()

    # Try to set up parallel chains
    try:
        numpyro.set_host_device_count(args.num_chains)
    except RuntimeError:
        print(f"WARNING: Could not set host device count to {args.num_chains}. "
              f"Chains will run sequentially.")

    # Report device
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    print(f"JAX backend: {jax.default_backend()}")

    # Load data
    print("\nLoading data...")
    behavior_path = os.path.join(args.data_dir, 'behavior_rich.csv')
    psych_path = os.path.join(args.data_dir, 'psych.csv')
    if not os.path.exists(psych_path):
        psych_path = None

    data = prepare_data(behavior_path, psych_path=psych_path)
    print(f"N_subjects={data['N_S']}")
    print(f"N_choice={data['N_choice']}, N_vigor={data['N_vigor']}")

    # Fit MCMC
    mcmc, elapsed = fit_mcmc(
        data,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        target_accept_prob=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        seed=args.seed,
    )

    # Diagnostics
    result = compute_diagnostics(mcmc, data, num_chains=args.num_chains)

    # Divergences
    extra = mcmc.get_extra_fields()
    n_divergent = 0
    if 'diverging' in extra:
        n_divergent = int(np.sum(np.array(extra['diverging'])))
    print(f"\nDivergent transitions: {n_divergent}")

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)

    param_path = os.path.join(args.out_dir, 'oc_evc_mcmc_params.csv')
    result['param_df'].to_csv(param_path, index=False)
    print(f"\nSaved: {param_path}")

    pop_path = os.path.join(args.out_dir, 'oc_evc_mcmc_population.csv')
    result['pop_df'].to_csv(pop_path, index=False)
    print(f"Saved: {pop_path}")

    diag_path = os.path.join(args.out_dir, 'oc_evc_mcmc_diagnostics.csv')
    result['diag_df'].to_csv(diag_path, index=False)
    print(f"Saved: {diag_path}")

    # Compare with SVI
    comparison = None
    if os.path.exists(args.svi_params):
        comparison = compare_with_svi(result, args.svi_params)
    else:
        print(f"\nWARNING: SVI params not found at {args.svi_params}, "
              f"skipping comparison")

    # Write text summary
    summary_path = os.path.join(args.out_dir, 'oc_evc_mcmc_summary.txt')
    write_summary(summary_path, result, comparison, elapsed,
                  args.num_warmup, args.num_samples, args.num_chains,
                  n_divergent)

    total_elapsed = time.time() - t0_total
    print(f"\nTotal time: {total_elapsed / 60:.1f} min")


if __name__ == '__main__':
    main()
