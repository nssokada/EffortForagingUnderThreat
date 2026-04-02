"""
Parameter recovery simulation for the EVC+gamma model.

Generates synthetic datasets from known parameters, re-fits via SVI,
and evaluates whether the model can recover the true generating parameters.

Output:
  - results/stats/evc_parameter_recovery.csv
  - results/figs/paper/fig_s_parameter_recovery.png
"""

import sys
sys.path.insert(0, '/workspace')

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from jax import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

jax.config.update('jax_enable_x64', True)

# Import model definition
import importlib.util
spec = importlib.util.spec_from_file_location("oc_evc_gamma", "/workspace/scripts/modeling/oc_evc_gamma.py")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
make_model = _mod.make_model


# ── Empirical distribution parameters (from N=293 fit) ──────────────────────

GAMMA_TRUE = 0.283  # population probability weighting

# Log-normal population parameters estimated from fitted params
MU_CE = -5.6501
SIGMA_CE = 0.9751
MU_CD = -0.6354
SIGMA_CD = 0.7701
MU_EPS = -1.9241
SIGMA_EPS = 0.8185

# Population parameters (fixed for simulation)
TAU = 0.5
P_ESC = 0.6
SIGMA_MOTOR = 0.15
SIGMA_V = 0.3

# Cookie-type centering offsets (from empirical data)
HEAVY_MEAN_EXCESS = 0.104
LIGHT_MEAN_EXCESS = 0.543


# ── Task design ──────────────────────────────────────────────────────────────

def make_task_design():
    """Create the 45-trial task design: 9 conditions x 5 reps.

    Returns dict with arrays of length 45:
        T: threat probability {0.1, 0.5, 0.9}
        dist_H: heavy cookie distance {1, 2, 3}
    """
    threats = [0.1, 0.5, 0.9]
    distances = [1.0, 2.0, 3.0]
    reps = 5

    T_list = []
    dist_H_list = []
    for t in threats:
        for d in distances:
            for _ in range(reps):
                T_list.append(t)
                dist_H_list.append(d)

    return {
        'T': np.array(T_list),
        'dist_H': np.array(dist_H_list),
    }


# ── Generative model ────────────────────────────────────────────────────────

def simulate_data(N_sim, rng_seed=0):
    """Simulate synthetic choice + vigor data from known parameters.

    Uses the EXACT same generative process as the EVC+gamma model:
      - Choice: P(heavy) = sigmoid((EU_H - EU_L) / tau)
      - Vigor: u* from 30-point grid soft-argmax, then excess_cc = u* - req - offset + N(0, sigma_v)

    Returns:
        data dict compatible with model fitting
        true_params DataFrame with ground-truth subject parameters
    """
    rng = np.random.RandomState(rng_seed)

    # Draw subject parameters from population distribution
    log_ce = rng.normal(MU_CE, SIGMA_CE, N_sim)
    log_cd = rng.normal(MU_CD, SIGMA_CD, N_sim)
    log_eps = rng.normal(MU_EPS, SIGMA_EPS, N_sim)

    c_effort = np.exp(log_ce)
    c_death = np.exp(log_cd)
    epsilon = np.exp(log_eps)

    gamma = GAMMA_TRUE

    true_params = pd.DataFrame({
        'subj': np.arange(N_sim),
        'c_effort': c_effort,
        'c_death': c_death,
        'epsilon': epsilon,
    })

    # Task design
    design = make_task_design()
    n_trials_per_subj = len(design['T'])  # 45

    # Build trial-level arrays
    all_subj_idx = []
    all_T = []
    all_dist_H = []
    all_choice = []
    all_excess_cc = []
    all_chosen_R = []
    all_chosen_req = []
    all_chosen_dist = []
    all_chosen_offset = []

    for s in range(N_sim):
        ce_s = c_effort[s]
        cd_s = c_death[s]
        eps_s = epsilon[s]

        for t_idx in range(n_trials_per_subj):
            T_val = design['T'][t_idx]
            dH = design['dist_H'][t_idx]

            # Weighted threat
            T_w = T_val ** gamma

            # ── Choice model ──
            # S_full and S_stop for choice (binary: full-speed vs stop)
            S_full = (1.0 - T_w) + eps_s * T_w * P_ESC
            S_stop = 1.0 - T_w

            # Heavy option
            eu_H_full = S_full * 5 - (1 - S_full) * cd_s * 10 - ce_s * 0.81 * dH
            eu_H_stop = S_stop * 5 - (1 - S_stop) * cd_s * 10
            eu_H = max(eu_H_full, eu_H_stop)

            # Light option (distance always 1)
            eu_L_full = S_full * 1 - (1 - S_full) * cd_s * 6 - ce_s * 0.16
            eu_L_stop = S_stop * 1 - (1 - S_stop) * cd_s * 6
            eu_L = max(eu_L_full, eu_L_stop)

            logit = np.clip((eu_H - eu_L) / TAU, -20, 20)
            p_H = 1.0 / (1.0 + np.exp(-logit))

            # Sample choice
            choice = int(rng.rand() < p_H)

            # ── Vigor model ──
            chosen_R = 5.0 if choice == 1 else 1.0
            chosen_req = 0.9 if choice == 1 else 0.4
            chosen_dist = dH if choice == 1 else 1.0
            chosen_offset = HEAVY_MEAN_EXCESS if choice == 1 else LIGHT_MEAN_EXCESS

            # Grid optimization (30 points)
            u_grid = np.linspace(0.1, 1.5, 30)

            S_u = ((1.0 - T_w)
                   + eps_s * T_w * P_ESC
                   * (1.0 / (1.0 + np.exp(-(u_grid - chosen_req) / SIGMA_MOTOR))))

            eu_grid = (S_u * chosen_R
                       - (1.0 - S_u) * cd_s * (chosen_R + 5.0)
                       - ce_s * u_grid ** 2 * chosen_dist)

            # Soft argmax
            weights = np.exp(eu_grid * 10.0)
            weights = weights / weights.sum()
            u_star = np.sum(weights * u_grid)

            excess_pred = u_star - chosen_req - chosen_offset
            excess_obs = excess_pred + rng.normal(0, SIGMA_V)

            all_subj_idx.append(s)
            all_T.append(T_val)
            all_dist_H.append(dH)
            all_choice.append(choice)
            all_excess_cc.append(excess_obs)
            all_chosen_R.append(chosen_R)
            all_chosen_req.append(chosen_req)
            all_chosen_dist.append(chosen_dist)
            all_chosen_offset.append(chosen_offset)

    data = {
        'subj_idx': jnp.array(all_subj_idx),
        'T': jnp.array(all_T),
        'dist_H': jnp.array(all_dist_H, dtype=jnp.float64),
        'choice': jnp.array(all_choice),
        'excess_cc': jnp.array(all_excess_cc),
        'chosen_R': jnp.array(all_chosen_R),
        'chosen_req': jnp.array(all_chosen_req),
        'chosen_dist': jnp.array(all_chosen_dist),
        'chosen_offset': jnp.array(all_chosen_offset),
        'N_S': N_sim,
        'N_T': N_sim * n_trials_per_subj,
    }

    return data, true_params


# ── Fitting synthetic data ───────────────────────────────────────────────────

def fit_synthetic(data, n_steps=35000, lr=0.002, seed=42):
    """Fit EVC+gamma to synthetic data via SVI (same settings as real fit)."""
    model = make_model(data['N_S'])

    kwargs = {k: data[k] for k in [
        'subj_idx', 'T', 'dist_H', 'choice', 'excess_cc',
        'chosen_R', 'chosen_req', 'chosen_dist', 'chosen_offset',
    ]}

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    losses = []
    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        losses.append(float(loss))
        if (i + 1) % 10000 == 0:
            print(f"    Step {i+1}: loss={loss:.1f}")

    params_fit = svi.get_params(state)

    return {
        'params': params_fit,
        'losses': losses,
        'guide': guide,
        'model': model,
        'kwargs': kwargs,
        'data': data,
    }


def extract_recovered_params(fit_result, n_samples=500, seed=44):
    """Extract posterior mean parameters from a fitted model."""
    guide = fit_result['guide']
    params_fit = fit_result['params']

    obs_kwargs = {k: v for k, v in fit_result['kwargs'].items()
                  if k not in ['choice', 'excess_cc']}

    pred = Predictive(fit_result['model'], guide=guide,
                      params=params_fit, num_samples=n_samples)
    samples = pred(random.PRNGKey(seed), **obs_kwargs)

    ce = np.array(samples['c_effort']).mean(0)
    cd = np.array(samples['c_death']).mean(0)
    eps = np.array(samples['epsilon']).mean(0)
    gamma_val = float(np.array(samples['gamma']).mean())

    # Also get SDs for coverage
    ce_sd = np.array(samples['c_effort']).std(0)
    cd_sd = np.array(samples['c_death']).std(0)
    eps_sd = np.array(samples['epsilon']).std(0)
    gamma_sd = float(np.array(samples['gamma']).std())

    recovered = pd.DataFrame({
        'subj': np.arange(len(ce)),
        'c_effort': ce, 'c_death': cd, 'epsilon': eps,
        'c_effort_sd': ce_sd, 'c_death_sd': cd_sd, 'epsilon_sd': eps_sd,
    })

    return recovered, gamma_val, gamma_sd


# ── Recovery metrics ─────────────────────────────────────────────────────────

def compute_recovery_metrics(true_params, recovered_params,
                              true_gamma, recovered_gamma, recovered_gamma_sd):
    """Compute recovery statistics for each parameter."""
    metrics = {}

    for param in ['c_effort', 'c_death', 'epsilon']:
        true_vals = true_params[param].values
        rec_vals = recovered_params[param].values
        rec_sd = recovered_params[f'{param}_sd'].values

        # Pearson r (raw scale)
        r, p = pearsonr(true_vals, rec_vals)

        # Pearson r (log scale — natural scale for log-normal params)
        r_log, p_log = pearsonr(np.log(true_vals), np.log(rec_vals))

        # Mean absolute error (on log scale for better interpretability)
        log_mae = np.mean(np.abs(np.log(true_vals) - np.log(rec_vals)))

        # Coverage: fraction where true is within 1 SD of recovered
        within_1sd = np.mean(np.abs(true_vals - rec_vals) <= rec_sd)

        # Bias: mean(recovered - true) on log scale
        log_bias = np.mean(np.log(rec_vals) - np.log(true_vals))

        metrics[param] = {
            'r': r, 'p': p,
            'r_log': r_log, 'p_log': p_log,
            'log_mae': log_mae,
            'log_bias': log_bias,
            'coverage_1sd': within_1sd,
        }

    # Gamma recovery
    metrics['gamma'] = {
        'true': true_gamma,
        'recovered': recovered_gamma,
        'recovered_sd': recovered_gamma_sd,
        'abs_error': abs(recovered_gamma - true_gamma),
        'within_1sd': abs(recovered_gamma - true_gamma) <= recovered_gamma_sd,
    }

    return metrics


# ── Main simulation ──────────────────────────────────────────────────────────

def run_parameter_recovery(n_datasets=5, n_sim=50, n_steps=35000):
    """Run the full parameter recovery simulation."""

    all_results = []
    all_true = []
    all_recovered = []

    for ds in range(n_datasets):
        print(f"\n{'='*60}")
        print(f"Dataset {ds+1}/{n_datasets}")
        print(f"{'='*60}")

        t0 = time.time()

        # Simulate
        print(f"  Simulating {n_sim} subjects...")
        data, true_params = simulate_data(n_sim, rng_seed=ds * 100)
        print(f"  Generated {data['N_T']} trials")

        # Fit
        print(f"  Fitting (SVI, {n_steps} steps)...")
        fit_result = fit_synthetic(data, n_steps=n_steps, lr=0.002, seed=ds * 10 + 42)

        # Extract recovered
        recovered, gamma_rec, gamma_sd = extract_recovered_params(fit_result)

        # Compute metrics
        metrics = compute_recovery_metrics(
            true_params, recovered, GAMMA_TRUE, gamma_rec, gamma_sd
        )

        elapsed = time.time() - t0
        print(f"\n  Results (took {elapsed:.1f}s):")
        print(f"  gamma: true={GAMMA_TRUE:.3f}, recovered={gamma_rec:.3f} +/- {gamma_sd:.3f}")

        for param in ['c_effort', 'c_death', 'epsilon']:
            m = metrics[param]
            print(f"  {param:12s}: r={m['r']:.3f}, r_log={m['r_log']:.3f}, log_MAE={m['log_mae']:.3f}, "
                  f"log_bias={m['log_bias']:+.3f}, coverage={m['coverage_1sd']:.2f}")

        # Store
        result_row = {'dataset': ds, 'elapsed_s': elapsed}
        for param in ['c_effort', 'c_death', 'epsilon']:
            for k, v in metrics[param].items():
                result_row[f'{param}_{k}'] = v
        result_row['gamma_true'] = GAMMA_TRUE
        result_row['gamma_recovered'] = gamma_rec
        result_row['gamma_sd'] = gamma_sd
        result_row['gamma_abs_error'] = metrics['gamma']['abs_error']
        all_results.append(result_row)

        # Store per-subject for scatter plot (last dataset)
        true_params['dataset'] = ds
        recovered['dataset'] = ds
        all_true.append(true_params)
        all_recovered.append(recovered)

    # ── Aggregate results ────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean +/- SD across {n_datasets} datasets)")
    print(f"{'='*60}")

    for param in ['c_effort', 'c_death', 'epsilon']:
        r_mean = results_df[f'{param}_r'].mean()
        r_sd = results_df[f'{param}_r'].std()
        rlog_mean = results_df[f'{param}_r_log'].mean()
        rlog_sd = results_df[f'{param}_r_log'].std()
        mae_mean = results_df[f'{param}_log_mae'].mean()
        cov_mean = results_df[f'{param}_coverage_1sd'].mean()
        print(f"  {param:12s}: r={r_mean:.3f}+/-{r_sd:.3f}, "
              f"r_log={rlog_mean:.3f}+/-{rlog_sd:.3f}, "
              f"log_MAE={mae_mean:.3f}, coverage={cov_mean:.2f}")

    gamma_rec_mean = results_df['gamma_recovered'].mean()
    gamma_rec_sd = results_df['gamma_recovered'].std()
    print(f"  gamma: true={GAMMA_TRUE:.3f}, recovered={gamma_rec_mean:.3f}+/-{gamma_rec_sd:.3f}")

    # Save results
    out_path = '/workspace/results/stats/evc_parameter_recovery.csv'
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── Create figure ────────────────────────────────────────────────────
    true_all = pd.concat(all_true, ignore_index=True)
    rec_all = pd.concat(all_recovered, ignore_index=True)

    make_recovery_figure(true_all, rec_all, results_df)

    return results_df


def make_recovery_figure(true_all, rec_all, results_df):
    """Create 3-panel scatter plot: true vs recovered for each parameter."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    param_labels = {
        'c_effort': r'$c_{\mathrm{effort}}$',
        'c_death': r'$c_{\mathrm{death}}$',
        'epsilon': r'$\varepsilon$',
    }

    for i, param in enumerate(['c_effort', 'c_death', 'epsilon']):
        ax = axes[i]

        true_vals = true_all[param].values
        rec_vals = rec_all[param].values

        # Use log scale for better visualization
        log_true = np.log10(true_vals)
        log_rec = np.log10(rec_vals)

        # Color by dataset
        colors = plt.cm.Set2(true_all['dataset'].values / 5.0)

        ax.scatter(log_true, log_rec, alpha=0.5, s=20, c=colors, edgecolors='none')

        # Identity line
        lims = [min(log_true.min(), log_rec.min()) - 0.2,
                max(log_true.max(), log_rec.max()) + 0.2]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='identity')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Aggregate r (use log-scale r since axes are log)
        r_mean = results_df[f'{param}_r_log'].mean()
        r_sd = results_df[f'{param}_r_log'].std()

        ax.set_xlabel(f'True {param_labels[param]} (log10)')
        ax.set_ylabel(f'Recovered {param_labels[param]} (log10)')
        ax.set_title(f'{param_labels[param]}')

        # Annotate
        ax.text(0.05, 0.95, f'r = {r_mean:.3f} ({r_sd:.3f})',
                transform=ax.transAxes, va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_aspect('equal')

    fig.suptitle('Parameter Recovery: EVC+gamma model (5 synthetic datasets, N=50 each)',
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out_path = '/workspace/results/figs/paper/fig_s_parameter_recovery.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure: {out_path}")
    plt.close()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("EVC+gamma Parameter Recovery Simulation")
    print(f"  Population: mu_ce={MU_CE:.3f}, sigma_ce={SIGMA_CE:.3f}")
    print(f"  Population: mu_cd={MU_CD:.3f}, sigma_cd={SIGMA_CD:.3f}")
    print(f"  Population: mu_eps={MU_EPS:.3f}, sigma_eps={SIGMA_EPS:.3f}")
    print(f"  gamma={GAMMA_TRUE}")
    print(f"  tau={TAU}, p_esc={P_ESC}, sigma_motor={SIGMA_MOTOR}, sigma_v={SIGMA_V}")

    results = run_parameter_recovery(n_datasets=5, n_sim=50, n_steps=35000)
