"""
MCMC validation of the EVC 2+2 model.

Uses NUTS sampler (NumPyro) with the EXACT same model specification as the
SVI version in evc_final_2plus2.py. Purpose: validate that SVI point estimates
are reliable, as requested by R1.

Architecture (identical to SVI):
    Per-subject (log-normal, non-centered): ce, cd
    Population: epsilon, gamma, ce_vigor, tau, p_esc, sigma_motor, sigma_v
    Choice: dEU = S * 4 - ce_i * (0.81*D_H - 0.16)
    Vigor: EU(u) = S(u)*R - (1-S(u))*cd_i*(R+C) - ce_vigor*(u-req)^2*D

MCMC settings:
    4 chains, 500 warmup + 500 samples, target_accept=0.8, max_tree_depth=10
"""

import sys
import os
import time

# Add parent dirs to path so we can import the SVI module directly
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, '..', '..'))
sys.path.insert(0, _this_dir)

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

jax.config.update('jax_enable_x64', True)

# Import data preparation and model from SVI script
from evc_final_2plus2 import prepare_data, make_model


def fit_mcmc(data, num_warmup=500, num_samples=500, num_chains=4,
             target_accept_prob=0.8, max_tree_depth=10, seed=0):
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

    print(f"\nFitting EVC 2+2 via MCMC (NUTS)")
    print(f"  {num_chains} chains x {num_warmup} warmup + {num_samples} samples")
    print(f"  target_accept_prob = {target_accept_prob}")
    print(f"  max_tree_depth = {max_tree_depth}")
    print(f"  {data['N_S']} subjects, ~{2*data['N_S'] + 11} total params")
    print(f"  {data['N_choice']} choice trials, {data['N_vigor']} vigor trials")

    t0 = time.time()
    mcmc.run(random.PRNGKey(seed), **kwargs)
    elapsed = time.time() - t0
    print(f"\nMCMC completed in {elapsed/60:.1f} min")

    return mcmc, elapsed


def compute_diagnostics(mcmc, data):
    """Compute convergence diagnostics and extract parameters."""

    samples = mcmc.get_samples()
    extra_fields = mcmc.get_extra_fields()

    # ── Convergence diagnostics ──
    summary = {}
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
        s = samples[param_name]
        if s.ndim == 1:
            # Reshape to (num_chains, num_samples)
            n_chains = mcmc._num_chains if hasattr(mcmc, '_num_chains') else 4
            s_reshaped = s.reshape(n_chains, -1)
        else:
            s_reshaped = s

        # R-hat (split R-hat)
        rhat = _compute_rhat(s_reshaped)
        ess = _compute_ess(s_reshaped)
        mean_val = float(jnp.mean(s))
        std_val = float(jnp.std(s))

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
    if 'ce_raw' in samples:
        ce_raw = np.array(samples['ce_raw'])  # (total_samples, N_S)
        cd_raw = np.array(samples['cd_raw'])

        ce_rhats, cd_rhats = [], []
        ce_ess_list, cd_ess_list = [], []

        n_chains = mcmc._num_chains if hasattr(mcmc, '_num_chains') else 4
        n_per = ce_raw.shape[0] // n_chains

        for j in range(data['N_S']):
            ce_j = ce_raw[:, j].reshape(n_chains, n_per)
            cd_j = cd_raw[:, j].reshape(n_chains, n_per)
            ce_rhats.append(_compute_rhat(ce_j))
            cd_rhats.append(_compute_rhat(cd_j))
            ce_ess_list.append(_compute_ess(ce_j))
            cd_ess_list.append(_compute_ess(cd_j))

        ce_rhats = np.array(ce_rhats)
        cd_rhats = np.array(cd_rhats)

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

    # ── Extract per-subject parameters ──
    mu_ce = float(jnp.mean(samples['mu_ce']))
    mu_cd = float(jnp.mean(samples['mu_cd']))
    sigma_ce = float(jnp.mean(samples['sigma_ce']))
    sigma_cd = float(jnp.mean(samples['sigma_cd']))

    ce_raw_mean = np.mean(np.array(samples['ce_raw']), axis=0)
    cd_raw_mean = np.mean(np.array(samples['cd_raw']), axis=0)

    ce_mcmc = np.exp(mu_ce + sigma_ce * ce_raw_mean)
    cd_mcmc = np.exp(mu_cd + sigma_cd * cd_raw_mean)

    # ── Population parameter summaries ──
    print("\n" + "=" * 70)
    print("Population Parameter Estimates (posterior mean [94% HDI])")
    print("=" * 70)

    pop_transforms = {
        'epsilon': ('eps_raw', lambda x: np.exp(x)),
        'gamma': ('gamma_raw', lambda x: np.clip(np.exp(x), 0.1, 3.0)),
        'tau': ('tau_raw', lambda x: np.clip(np.exp(x), 0.01, 20.0)),
        'p_esc': ('p_esc_raw', lambda x: 1.0 / (1.0 + np.exp(-x))),
        'sigma_motor': ('sigma_motor_raw', lambda x: np.clip(np.exp(x), 0.01, 1.0)),
        'ce_vigor': ('ce_vigor_raw', lambda x: np.exp(x)),
        'sigma_v': ('sigma_v', lambda x: x),
    }

    pop_results = {}
    for name, (raw_name, transform) in pop_transforms.items():
        if raw_name in samples:
            vals = transform(np.array(samples[raw_name]))
            mean_v = float(np.mean(vals))
            hdi_lo = float(np.percentile(vals, 3))
            hdi_hi = float(np.percentile(vals, 97))
            pop_results[name] = mean_v
            print(f"  {name:15s}: {mean_v:.4f} [{hdi_lo:.4f}, {hdi_hi:.4f}]")

    # ── Per-subject summaries ──
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

    return {
        'param_df': param_df,
        'diag_df': diag_df,
        'pop_results': pop_results,
        'ce_mcmc': ce_mcmc,
        'cd_mcmc': cd_mcmc,
        'ce_rhats': ce_rhats if 'ce_raw' in samples else None,
        'cd_rhats': cd_rhats if 'ce_raw' in samples else None,
    }


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
    }


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
    split_chains = np.concatenate([chains[:, :mid], chains[:, mid:2*mid]], axis=0)
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


if __name__ == '__main__':
    t0_total = time.time()

    DATA_DIR = ('data/exploratory_350/processed/'
                'stage5_filtered_data_20260320_191950')
    SVI_PARAMS = 'results/stats/oc_evc_final_params.csv'

    # Set number of chains for parallel execution
    numpyro.set_host_device_count(4)

    print("Loading data...")
    data = prepare_data(
        f'{DATA_DIR}/behavior_rich.csv',
        psych_path=f'{DATA_DIR}/psych.csv',
    )
    print(f"N_subjects={data['N_S']}")
    print(f"N_choice={data['N_choice']}, N_vigor={data['N_vigor']}")

    # ── Fit MCMC ──
    # 200+200 to fit within time constraints (CPU only, ~600 params)
    # Can increase to 500+500 with GPU
    mcmc, elapsed = fit_mcmc(
        data,
        num_warmup=200,
        num_samples=200,
        num_chains=4,
        target_accept_prob=0.8,
        max_tree_depth=10,
        seed=0,
    )

    # ── Diagnostics ──
    result = compute_diagnostics(mcmc, data)

    # ── Save per-subject params ──
    out_dir = 'results/stats'
    os.makedirs(out_dir, exist_ok=True)

    param_path = os.path.join(out_dir, 'oc_evc_mcmc_params.csv')
    result['param_df'].to_csv(param_path, index=False)
    print(f"\nSaved: {param_path}")

    diag_path = os.path.join(out_dir, 'oc_evc_mcmc_diagnostics.csv')
    result['diag_df'].to_csv(diag_path, index=False)
    print(f"Saved: {diag_path}")

    # ── Compare with SVI ──
    if os.path.exists(SVI_PARAMS):
        comparison = compare_with_svi(result, SVI_PARAMS)
    else:
        print(f"\nWARNING: SVI params not found at {SVI_PARAMS}, skipping comparison")

    total_elapsed = time.time() - t0_total
    print(f"\nTotal time: {total_elapsed/60:.1f} min")

    # ── Print divergences ──
    extra = mcmc.get_extra_fields()
    if 'diverging' in extra:
        n_div = int(np.sum(np.array(extra['diverging'])))
        print(f"\nDivergent transitions: {n_div}")
    else:
        print("\nDivergence info not available")
