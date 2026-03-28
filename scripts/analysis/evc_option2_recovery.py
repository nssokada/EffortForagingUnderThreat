"""
Parameter recovery for the EVC Option 2 model.

Simulates 5 datasets × 50 subjects × 81 trials each, refits the model,
and computes Pearson r between log(true) and log(recovered) for ce, cd, eps.

Usage:
    python scripts/analysis/evc_option2_recovery.py
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
import os
import time

jax.config.update('jax_enable_x64', True)

# ── Empirical distribution from fitted N=293 ──
# From results/stats/oc_evc_option2_final_params.csv
MU_CE = -0.5050      # log-mean of c_effort
SD_CE = 0.6716       # log-sd of c_effort
MU_CD = 2.9721       # log-mean of c_death (clean, cd < 1000)
SD_CD = 1.3658       # log-sd of c_death (clean)
MU_EPS = -2.1441     # log-mean of epsilon
SD_EPS = 0.05        # use wider SD than empirical (0.0006) for identifiability test

# Population parameters from the fitted model
CE_VIGOR = 0.05      # population vigor effort cost
GAMMA = 0.21         # probability weighting
TAU = 0.48           # choice temperature
P_ESC = 0.6          # escape probability at full speed
SIGMA_MOTOR = 0.15   # motor noise around speed threshold
SIGMA_V = 0.3        # vigor observation noise

# Task design
THREATS = [0.1, 0.5, 0.9]
DISTANCES = [1, 2, 3]       # coded as 1,2,3
R_H = 5.0
R_L = 1.0
C = 5.0
REQ_H = 0.9
REQ_L = 0.4

N_SUBJECTS = 50
N_DATASETS = 5
N_SVI_STEPS = 25000


def compute_survival(T, eps, u=None, req=None):
    """Compute survival probability."""
    T_w = T ** GAMMA
    if u is not None and req is not None:
        # Vigor: speed-dependent
        return (1.0 - T_w) + eps * T_w * P_ESC * 1.0 / (1.0 + np.exp(-(u - req) / SIGMA_MOTOR))
    else:
        # Choice: binary (assume full speed)
        return (1.0 - T_w) + eps * T_w * P_ESC


def compute_optimal_vigor(cd, eps, ce_vigor, T, R, req, D):
    """Compute optimal vigor u* via grid search."""
    u_grid = np.linspace(0.1, 1.5, 30)
    T_w = T ** GAMMA

    S_u = (1.0 - T_w) + eps * T_w * P_ESC / (1.0 + np.exp(-(u_grid - req) / SIGMA_MOTOR))
    deviation = u_grid - req
    eu_grid = S_u * R - (1.0 - S_u) * cd * (R + C) - ce_vigor * deviation**2 * D

    # Soft argmax with temperature 10
    weights = np.exp(eu_grid * 10.0 - np.max(eu_grid * 10.0))
    weights = weights / weights.sum()
    u_star = np.sum(weights * u_grid)
    return u_star


def simulate_dataset(seed, n_subjects=N_SUBJECTS):
    """Simulate one full dataset with true parameters."""
    rng = np.random.RandomState(seed)

    # Draw true subject parameters from empirical distribution
    true_ce = np.exp(rng.normal(MU_CE, SD_CE, n_subjects))
    true_cd = np.exp(rng.normal(MU_CD, SD_CD, n_subjects))
    true_eps = np.exp(rng.normal(MU_EPS, SD_EPS, n_subjects))

    # Clip extremes (matching what real fitting encounters)
    true_ce = np.clip(true_ce, 0.05, 10.0)
    true_cd = np.clip(true_cd, 0.1, 500.0)
    true_eps = np.clip(true_eps, 0.05, 0.5)

    # ── Choice trials: 3T × 3D × 5 reps = 45 ──
    choice_records = []
    for subj in range(n_subjects):
        for T in THREATS:
            for D in DISTANCES:
                for rep in range(5):
                    ce = true_ce[subj]
                    eps = true_eps[subj]
                    S = compute_survival(T, eps)

                    # ΔEU = S × (R_H - R_L) - ce × (req_H² × D_H - req_L² × D_L)
                    # With same distance for both cookies:
                    # ΔEU = S × 4 - ce × (0.81*D - 0.16)
                    delta_eu = S * 4.0 - ce * (0.81 * D - 0.16)
                    logit = np.clip(delta_eu / TAU, -20, 20)
                    p_heavy = 1.0 / (1.0 + np.exp(-logit))
                    choice = int(rng.random() < p_heavy)

                    choice_records.append({
                        'subj_idx': subj, 'T': T, 'D': D,
                        'choice': choice, 'is_heavy': choice,
                    })

    choice_df = pd.DataFrame(choice_records)

    # ── Vigor for choice trials ──
    vigor_records = []
    for _, row in choice_df.iterrows():
        subj = int(row['subj_idx'])
        T = row['T']
        D = row['D']
        cd = true_cd[subj]
        eps = true_eps[subj]

        # The chosen cookie determines R and req
        if row['is_heavy']:
            R, req = R_H, REQ_H
        else:
            R, req = R_L, REQ_L

        u_star = compute_optimal_vigor(cd, eps, CE_VIGOR, T, R, req, D)
        excess = u_star - req + rng.normal(0, SIGMA_V)

        vigor_records.append({
            'subj_idx': subj, 'T': T, 'D': D, 'R': R,
            'req': req, 'excess': excess, 'is_heavy': int(row['is_heavy']),
            'trial_type': 'choice',
        })

    # ── Probe trials: 3T × 3D × 2 cookie types × 2 reps = 36, vigor only ──
    for subj in range(n_subjects):
        for T in THREATS:
            for D in DISTANCES:
                for cookie_type in [0, 1]:  # 0=light, 1=heavy
                    for rep in range(2):
                        cd = true_cd[subj]
                        eps = true_eps[subj]

                        if cookie_type == 1:
                            R, req = R_H, REQ_H
                        else:
                            R, req = R_L, REQ_L

                        u_star = compute_optimal_vigor(cd, eps, CE_VIGOR, T, R, req, D)
                        excess = u_star - req + rng.normal(0, SIGMA_V)

                        vigor_records.append({
                            'subj_idx': subj, 'T': T, 'D': D, 'R': R,
                            'req': req, 'excess': excess,
                            'is_heavy': cookie_type,
                            'trial_type': 'probe',
                        })

    vigor_df = pd.DataFrame(vigor_records)

    # Cookie-type centering (using choice trial means, as in real pipeline)
    choice_vigor = vigor_df[vigor_df['trial_type'] == 'choice']
    heavy_mean = choice_vigor[choice_vigor['is_heavy'] == 1]['excess'].mean()
    light_mean = choice_vigor[choice_vigor['is_heavy'] == 0]['excess'].mean()
    vigor_df['excess_cc'] = vigor_df['excess'] - np.where(
        vigor_df['is_heavy'] == 1, heavy_mean, light_mean)
    vigor_df['offset'] = np.where(
        vigor_df['is_heavy'] == 1, heavy_mean, light_mean)

    # Build JAX arrays
    ch_subj = jnp.array(choice_df['subj_idx'].values)
    ch_T = jnp.array(choice_df['T'].values)
    ch_dist_H = jnp.array(choice_df['D'].values, dtype=jnp.float64)
    ch_choice = jnp.array(choice_df['choice'].values)

    vig_subj = jnp.array(vigor_df['subj_idx'].values)
    vig_T = jnp.array(vigor_df['T'].values)
    vig_R = jnp.array(vigor_df['R'].values)
    vig_req = jnp.array(vigor_df['req'].values)
    vig_dist = jnp.array(vigor_df['D'].values, dtype=jnp.float64)
    vig_excess = jnp.array(vigor_df['excess_cc'].values)
    vig_offset = jnp.array(vigor_df['offset'].values)

    data = {
        'ch_subj': ch_subj, 'ch_T': ch_T, 'ch_dist_H': ch_dist_H,
        'ch_choice': ch_choice,
        'vig_subj': vig_subj, 'vig_T': vig_T, 'vig_R': vig_R,
        'vig_req': vig_req, 'vig_dist': vig_dist,
        'vig_excess': vig_excess, 'vig_offset': vig_offset,
        'N_S': n_subjects,
        'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
    }

    true_params = {
        'ce': true_ce, 'cd': true_cd, 'eps': true_eps,
    }

    return data, true_params


def make_recovery_model(N_S, N_choice, N_vigor):
    """NumPyro model for recovery (same architecture as Option 2)."""

    def evc_option2(ch_subj, ch_T, ch_dist_H, ch_choice,
                    vig_subj, vig_T, vig_R, vig_req, vig_dist,
                    vig_excess, vig_offset):
        # Population priors
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        mu_eps = numpyro.sample('mu_eps', dist.Normal(-0.5, 0.5))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))
        sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.3))

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample(
            'sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic(
            'gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))

        ce_vigor_raw = numpyro.sample(
            'ce_vigor_raw', dist.Normal(-3.0, 1.0))
        ce_vigor = numpyro.deterministic('ce_vigor', jnp.exp(ce_vigor_raw))

        # Subject-level (non-centered)
        with numpyro.plate('subjects', N_S):
            ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))
            eps_raw = numpyro.sample('eps_raw', dist.Normal(0.0, 1.0))

        c_effort = jnp.exp(mu_ce + sigma_ce * ce_raw)
        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        epsilon = jnp.exp(mu_eps + sigma_eps * eps_raw)
        numpyro.deterministic('c_effort', c_effort)
        numpyro.deterministic('c_death', c_death)
        numpyro.deterministic('epsilon', epsilon)

        # ── CHOICE ──
        ce_ch = c_effort[ch_subj]
        eps_ch = epsilon[ch_subj]
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + eps_ch * T_w_ch * p_esc

        delta_eu = S_ch * 4.0 - ce_ch * (0.81 * ch_dist_H - 0.16)
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample(
                'obs_choice',
                dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                obs=ch_choice)

        # ── VIGOR ──
        cd_v = c_death[vig_subj]
        eps_v = epsilon[vig_subj]
        T_w_v = jnp.power(vig_T, gamma)

        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T_w_v[:, None])
               + eps_v[:, None] * T_w_v[:, None] * p_esc
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

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample(
                'obs_vigor',
                dist.Normal(excess_pred, sigma_v),
                obs=vig_excess)

    return evc_option2


def fit_recovery(data, seed=42, n_steps=N_SVI_STEPS, lr=0.002):
    """Fit the model on synthetic data."""
    model = make_recovery_model(data['N_S'], data['N_choice'], data['N_vigor'])

    kwargs = {k: data[k] for k in [
        'ch_subj', 'ch_T', 'ch_dist_H', 'ch_choice',
        'vig_subj', 'vig_T', 'vig_R', 'vig_req', 'vig_dist',
        'vig_excess', 'vig_offset',
    ]}

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        if (i + 1) % 5000 == 0:
            print(f"    Step {i+1}: loss={float(loss):.1f}")

    params_fit = svi.get_params(state)

    # Extract posterior means
    pred = Predictive(
        model, guide=guide, params=params_fit,
        num_samples=300,
        return_sites=['c_effort', 'c_death', 'epsilon', 'gamma', 'ce_vigor'])
    samples = pred(random.PRNGKey(seed + 100), **kwargs)

    rec_ce = np.array(samples['c_effort']).mean(0)
    rec_cd = np.array(samples['c_death']).mean(0)
    rec_eps = np.array(samples['epsilon']).mean(0)
    rec_gamma = float(np.array(samples['gamma']).mean())

    return rec_ce, rec_cd, rec_eps, rec_gamma


def main():
    t0 = time.time()
    os.makedirs('/workspace/results/figs/paper', exist_ok=True)
    os.makedirs('/workspace/results/stats', exist_ok=True)

    all_results = []
    all_true_ce, all_rec_ce = [], []
    all_true_cd, all_rec_cd = [], []
    all_true_eps, all_rec_eps = [], []
    gamma_true_list, gamma_rec_list = [], []

    for ds in range(N_DATASETS):
        seed = 1000 + ds * 100
        print(f"\n{'='*60}")
        print(f"Dataset {ds+1}/{N_DATASETS} (seed={seed})")
        print(f"{'='*60}")

        print("  Simulating...")
        data, true_params = simulate_dataset(seed, N_SUBJECTS)
        print(f"  N_choice={data['N_choice']}, N_vigor={data['N_vigor']}")

        print("  Fitting...")
        rec_ce, rec_cd, rec_eps, rec_gamma = fit_recovery(
            data, seed=seed + 1, n_steps=N_SVI_STEPS)

        # Per-dataset correlations
        r_ce, p_ce = pearsonr(np.log(true_params['ce']), np.log(rec_ce))
        r_cd, p_cd = pearsonr(np.log(true_params['cd']), np.log(rec_cd))
        r_eps, p_eps = pearsonr(np.log(true_params['eps']), np.log(rec_eps))

        print(f"  Recovery r: ce={r_ce:.3f}, cd={r_cd:.3f}, eps={r_eps:.3f}")
        print(f"  Gamma: true={GAMMA:.3f}, recovered={rec_gamma:.3f}")

        all_true_ce.extend(true_params['ce'])
        all_rec_ce.extend(rec_ce)
        all_true_cd.extend(true_params['cd'])
        all_rec_cd.extend(rec_cd)
        all_true_eps.extend(true_params['eps'])
        all_rec_eps.extend(rec_eps)
        gamma_true_list.append(GAMMA)
        gamma_rec_list.append(rec_gamma)

        for i in range(N_SUBJECTS):
            all_results.append({
                'dataset': ds + 1,
                'subject': i + 1,
                'true_ce': true_params['ce'][i],
                'rec_ce': float(rec_ce[i]),
                'true_cd': true_params['cd'][i],
                'rec_cd': float(rec_cd[i]),
                'true_eps': true_params['eps'][i],
                'rec_eps': float(rec_eps[i]),
                'true_gamma': GAMMA,
                'rec_gamma': rec_gamma,
            })

    # ── Aggregate statistics ──
    all_true_ce = np.array(all_true_ce)
    all_rec_ce = np.array(all_rec_ce)
    all_true_cd = np.array(all_true_cd)
    all_rec_cd = np.array(all_rec_cd)
    all_true_eps = np.array(all_true_eps)
    all_rec_eps = np.array(all_rec_eps)

    r_ce_all, p_ce_all = pearsonr(np.log(all_true_ce), np.log(all_rec_ce))
    r_cd_all, p_cd_all = pearsonr(np.log(all_true_cd), np.log(all_rec_cd))
    r_eps_all, p_eps_all = pearsonr(np.log(all_true_eps), np.log(all_rec_eps))
    gamma_rec_mean = np.mean(gamma_rec_list)

    print(f"\n{'='*60}")
    print(f"AGGREGATE RECOVERY ({N_DATASETS} datasets × {N_SUBJECTS} subjects)")
    print(f"{'='*60}")
    print(f"  c_effort:  r={r_ce_all:.3f}  (p={p_ce_all:.2e})  "
          f"{'PASS' if r_ce_all > 0.5 else 'FAIL'}")
    print(f"  c_death:   r={r_cd_all:.3f}  (p={p_cd_all:.2e})  "
          f"{'PASS' if r_cd_all > 0.5 else 'FAIL'}")
    print(f"  epsilon:   r={r_eps_all:.3f}  (p={p_eps_all:.2e})  "
          f"{'PASS' if r_eps_all > 0.5 else 'FAIL'}")
    print(f"  gamma:     true={GAMMA:.3f}, recovered={gamma_rec_mean:.3f}  "
          f"(bias={gamma_rec_mean - GAMMA:+.3f})  "
          f"{'PASS' if abs(gamma_rec_mean - GAMMA) / GAMMA < 0.3 else 'FAIL'}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

    # ── Save CSV ──
    results_df = pd.DataFrame(all_results)
    csv_path = '/workspace/results/stats/evc_option2_recovery.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Figure: 3-panel scatter ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    params = [
        ('c_effort', all_true_ce, all_rec_ce, r_ce_all, p_ce_all),
        ('c_death', all_true_cd, all_rec_cd, r_cd_all, p_cd_all),
        ('epsilon', all_true_eps, all_rec_eps, r_eps_all, p_eps_all),
    ]

    for ax, (name, true, rec, r_val, p_val) in zip(axes, params):
        log_true = np.log(true)
        log_rec = np.log(rec)
        ax.scatter(log_true, log_rec, alpha=0.3, s=15, color='steelblue')

        # Identity line
        lo = min(log_true.min(), log_rec.min())
        hi = max(log_true.max(), log_rec.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', lw=1, alpha=0.5)

        # Regression line
        m, b = np.polyfit(log_true, log_rec, 1)
        x_line = np.linspace(lo - margin, hi + margin, 100)
        ax.plot(x_line, m * x_line + b, 'r-', lw=1.5, alpha=0.7)

        ax.set_xlabel(f'log(true {name})', fontsize=11)
        ax.set_ylabel(f'log(recovered {name})', fontsize=11)
        status = 'PASS' if r_val > 0.5 else 'FAIL'
        ax.set_title(f'{name}\nr = {r_val:.3f} [{status}]', fontsize=12)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)

    plt.tight_layout()
    fig_path = '/workspace/results/figs/paper/fig_s_option2_recovery.png'
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
