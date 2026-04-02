#!/usr/bin/env python3
"""
Parameter Recovery for the EVC-LQR model.

Simulates 50 subjects x 45 trials x 5 datasets from the fitted population distribution,
then re-fits each dataset and checks recovery of c_death, epsilon, and gamma.

Output:
  results/stats/evc_lqr_recovery.csv
  results/figs/paper/fig_s_lqr_recovery.png
"""

import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/scripts')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from jax import random

jax.config.update('jax_enable_x64', True)

np.random.seed(42)

# ── Load fitted params to get population distribution ────────────────────────
params = pd.read_csv('/workspace/results/stats/oc_evc_lqr_final_params.csv')
pop = pd.read_csv('/workspace/results/stats/oc_evc_lqr_final_population.csv')

# Population params from fitted model
CE_POP = float(pop['c_effort'].iloc[0])  # 0.007
GAMMA_TRUE = float(pop['gamma'].iloc[0])  # 0.318
TAU = 0.5
P_ESC = 0.6
SIGMA_MOTOR = 0.15
SIGMA_V = 0.3

# Log-space distribution of fitted params
log_cd = np.log(params['c_death'].values)
log_eps = np.log(params['epsilon'].values)
MU_CD = log_cd.mean()
SIGMA_CD = log_cd.std()
MU_EPS = log_eps.mean()
SIGMA_EPS = log_eps.std()

print(f"Population parameters from fit:")
print(f"  c_effort = {CE_POP:.4f}")
print(f"  gamma = {GAMMA_TRUE:.3f}")
print(f"  mu_cd = {MU_CD:.3f}, sigma_cd = {SIGMA_CD:.3f}")
print(f"  mu_eps = {MU_EPS:.3f}, sigma_eps = {SIGMA_EPS:.3f}")
print(f"  tau = {TAU}, p_esc = {P_ESC}, sigma_motor = {SIGMA_MOTOR}")

# ── Task design ──────────────────────────────────────────────────────────────
THREATS = [0.1, 0.5, 0.9]
DISTANCES = [1.0, 2.0, 3.0]
N_TRIALS_PER_COND = 5  # 3T x 3D x 5 = 45 trials

# ── Simulation function ─────────────────────────────────────────────────────

def simulate_dataset(n_subj=50, seed=0):
    """Simulate choices and vigor for n_subj subjects."""
    rng = np.random.RandomState(seed)

    # Sample subject parameters
    cd = np.exp(rng.normal(MU_CD, SIGMA_CD, n_subj))
    eps = np.exp(rng.normal(MU_EPS, SIGMA_EPS, n_subj))

    # Generate trial structure
    conditions = []
    for t in THREATS:
        for d in DISTANCES:
            for _ in range(N_TRIALS_PER_COND):
                conditions.append((t, d))
    n_trials = len(conditions)  # 45

    records = []
    for s in range(n_subj):
        cd_s = cd[s]
        eps_s = eps[s]

        for trial_idx, (T, dist_H) in enumerate(conditions):
            T_w = T ** GAMMA_TRUE

            # Choice: compute EU for heavy vs light
            S_full = (1.0 - T_w) + eps_s * T_w * P_ESC
            S_stop = 1.0 - T_w

            # Heavy (req=0.9, R=5, C=5+5=10)
            eu_H_full = S_full * 5.0 - (1 - S_full) * cd_s * 10.0 - CE_POP * 0.81 * dist_H
            eu_H_stop = S_stop * 5.0 - (1 - S_stop) * cd_s * 10.0
            eu_H = max(eu_H_full, eu_H_stop)

            # Light (req=0.4, R=1, C=1+5=6, dist=1)
            eu_L_full = S_full * 1.0 - (1 - S_full) * cd_s * 6.0 - CE_POP * 0.16
            eu_L_stop = S_stop * 1.0 - (1 - S_stop) * cd_s * 6.0
            eu_L = max(eu_L_full, eu_L_stop)

            logit = np.clip((eu_H - eu_L) / TAU, -20, 20)
            p_H = expit(logit)
            choice = int(rng.random() < p_H)

            # Vigor: compute u* for chosen option
            chosen_R = 5.0 if choice == 1 else 1.0
            chosen_req = 0.9 if choice == 1 else 0.4
            chosen_dist = dist_H if choice == 1 else 1.0

            u_grid = np.linspace(0.1, 1.5, 30)
            S_u = ((1.0 - T_w) + eps_s * T_w * P_ESC
                   * expit((u_grid - chosen_req) / SIGMA_MOTOR))

            deviation = u_grid - chosen_req
            effort_vigor = CE_POP * deviation ** 2 * chosen_dist

            eu_grid = (S_u * chosen_R
                       - (1.0 - S_u) * cd_s * (chosen_R + 5.0)
                       - effort_vigor)

            weights = np.exp(eu_grid * 10.0)
            weights = weights / weights.sum()
            u_star = (weights * u_grid).sum()

            # Add noise
            excess = u_star - chosen_req + rng.normal(0, SIGMA_V)

            records.append({
                'subj': s, 'threat': T, 'distance_H': dist_H,
                'choice': choice, 'excess_cc': excess,
                'true_cd': cd_s, 'true_eps': eps_s,
                'chosen_R': chosen_R, 'chosen_req': chosen_req,
                'chosen_dist': chosen_dist,
            })

    df = pd.DataFrame(records)

    # Cookie-type centering
    heavy_mask = df['chosen_req'] == 0.9
    heavy_mean = df.loc[heavy_mask, 'excess_cc'].mean()
    light_mean = df.loc[~heavy_mask, 'excess_cc'].mean()
    df['excess_cc'] = df['excess_cc'] - np.where(heavy_mask, heavy_mean, light_mean)
    df['chosen_offset'] = np.where(heavy_mask, heavy_mean, light_mean)

    return df, cd, eps


# ── Model for recovery ──────────────────────────────────────────────────────

def make_recovery_model(N_S):
    def model(subj_idx, T, dist_H, choice=None, excess_cc=None,
              chosen_R=None, chosen_req=None, chosen_dist=None, chosen_offset=None):
        mu_ce_raw = numpyro.sample('mu_ce_raw', dist.Normal(0.0, 1.0))
        c_effort = numpyro.deterministic('c_effort', jnp.clip(jnp.exp(mu_ce_raw), 1e-6, 100.0))

        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))
        mu_eps = numpyro.sample('mu_eps', dist.Normal(-0.5, 0.5))
        sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.3))

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)

        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)

        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)

        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))

        with numpyro.plate('subjects', N_S):
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))
            eps_raw = numpyro.sample('eps_raw', dist.Normal(0.0, 1.0))

        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        epsilon = jnp.exp(mu_eps + sigma_eps * eps_raw)
        numpyro.deterministic('c_death', c_death)
        numpyro.deterministic('epsilon', epsilon)

        cd_i = c_death[subj_idx]
        eps_i = epsilon[subj_idx]
        T_w = jnp.power(T, gamma)

        S_full = (1.0 - T_w) + eps_i * T_w * p_esc
        S_stop = 1.0 - T_w

        eu_H_full = S_full * 5.0 - (1.0 - S_full) * cd_i * 10.0 - c_effort * 0.81 * dist_H
        eu_H_stop = S_stop * 5.0 - (1.0 - S_stop) * cd_i * 10.0
        eu_H = jnp.maximum(eu_H_full, eu_H_stop)

        eu_L_full = S_full * 1.0 - (1.0 - S_full) * cd_i * 6.0 - c_effort * 0.16
        eu_L_stop = S_stop * 1.0 - (1.0 - S_stop) * cd_i * 6.0
        eu_L = jnp.maximum(eu_L_full, eu_L_stop)

        logit = jnp.clip((eu_H - eu_L) / tau, -20.0, 20.0)
        p_H = jax.nn.sigmoid(logit)

        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]

        S_u = ((1.0 - T_w[:, None])
               + eps_i[:, None] * T_w[:, None] * p_esc
               * jax.nn.sigmoid((u_g - chosen_req[:, None]) / sigma_motor))

        deviation = u_g - chosen_req[:, None]
        effort_vigor = c_effort * deviation ** 2 * chosen_dist[:, None]

        eu_grid = (S_u * chosen_R[:, None]
                   - (1.0 - S_u) * cd_i[:, None] * (chosen_R[:, None] + 5.0)
                   - effort_vigor)

        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - chosen_req - chosen_offset

        N_T = subj_idx.shape[0]
        with numpyro.plate('trials', N_T):
            numpyro.sample('obs_choice', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)), obs=choice)
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v), obs=excess_cc)

    return model


def fit_recovery(sim_df, n_steps=25000, lr=0.002, seed=42):
    """Fit the recovery model to simulated data."""
    subjects = sorted(sim_df['subj'].unique())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)

    subj_idx = jnp.array([subj_to_idx[s] for s in sim_df['subj']])

    kwargs = {
        'subj_idx': subj_idx,
        'T': jnp.array(sim_df['threat'].values),
        'dist_H': jnp.array(sim_df['distance_H'].values, dtype=jnp.float64),
        'choice': jnp.array(sim_df['choice'].values),
        'excess_cc': jnp.array(sim_df['excess_cc'].values),
        'chosen_R': jnp.array(sim_df['chosen_R'].values),
        'chosen_req': jnp.array(sim_df['chosen_req'].values),
        'chosen_dist': jnp.array(sim_df['chosen_dist'].values),
        'chosen_offset': jnp.array(sim_df['chosen_offset'].values),
    }

    model = make_recovery_model(N_S)
    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        if (i + 1) % 10000 == 0:
            print(f"    Step {i+1}: loss={float(loss):.1f}")

    params_fit = svi.get_params(state)

    # Extract posterior means
    obs_kwargs = {k: v for k, v in kwargs.items() if k not in ['choice', 'excess_cc']}
    pred = Predictive(model, guide=guide, params=params_fit, num_samples=200)
    samples = pred(random.PRNGKey(seed + 1), **obs_kwargs)

    cd_rec = np.array(samples['c_death']).mean(0)
    eps_rec = np.array(samples['epsilon']).mean(0)
    ce_rec = float(np.array(samples['c_effort']).mean())
    gamma_rec = float(np.array(samples['gamma']).mean())

    return cd_rec, eps_rec, ce_rec, gamma_rec


# ── Run recovery ─────────────────────────────────────────────────────────────
N_DATASETS = 5
N_SUBJ = 50

all_results = []
all_true_cd = []
all_rec_cd = []
all_true_eps = []
all_rec_eps = []
all_gamma_true = []
all_gamma_rec = []

print(f"\n{'='*60}")
print(f"PARAMETER RECOVERY: {N_DATASETS} datasets x {N_SUBJ} subjects x 45 trials")
print(f"{'='*60}")

for ds in range(N_DATASETS):
    print(f"\n--- Dataset {ds+1}/{N_DATASETS} ---")
    sim_df, true_cd, true_eps = simulate_dataset(n_subj=N_SUBJ, seed=ds * 100)
    print(f"  Simulated {len(sim_df)} trials, P(heavy)={sim_df['choice'].mean():.3f}")

    cd_rec, eps_rec, ce_rec, gamma_rec = fit_recovery(sim_df, n_steps=25000, seed=ds * 100 + 1)

    r_cd, p_cd = pearsonr(np.log(true_cd), np.log(cd_rec))
    r_eps, p_eps = pearsonr(np.log(true_eps), np.log(eps_rec))

    print(f"  Recovery: c_death r={r_cd:.3f} (p={p_cd:.2e})")
    print(f"  Recovery: epsilon r={r_eps:.3f} (p={p_eps:.2e})")
    print(f"  Recovery: c_effort={ce_rec:.4f} (true={CE_POP:.4f})")
    print(f"  Recovery: gamma={gamma_rec:.3f} (true={GAMMA_TRUE:.3f})")

    all_true_cd.extend(true_cd)
    all_rec_cd.extend(cd_rec)
    all_true_eps.extend(true_eps)
    all_rec_eps.extend(eps_rec)
    all_gamma_true.append(GAMMA_TRUE)
    all_gamma_rec.append(gamma_rec)

    all_results.append({
        'dataset': ds + 1,
        'r_cd': r_cd, 'p_cd': p_cd,
        'r_eps': r_eps, 'p_eps': p_eps,
        'ce_true': CE_POP, 'ce_rec': ce_rec,
        'gamma_true': GAMMA_TRUE, 'gamma_rec': gamma_rec,
        'n_subj': N_SUBJ, 'n_trials': len(sim_df),
    })

# ── Overall recovery ─────────────────────────────────────────────────────────
all_true_cd = np.array(all_true_cd)
all_rec_cd = np.array(all_rec_cd)
all_true_eps = np.array(all_true_eps)
all_rec_eps = np.array(all_rec_eps)

r_cd_all, p_cd_all = pearsonr(np.log(all_true_cd), np.log(all_rec_cd))
r_eps_all, p_eps_all = pearsonr(np.log(all_true_eps), np.log(all_rec_eps))
gamma_mean_rec = np.mean(all_gamma_rec)

print(f"\n{'='*60}")
print(f"OVERALL RECOVERY ({N_DATASETS} datasets pooled)")
print(f"{'='*60}")
print(f"  c_death:  r = {r_cd_all:.3f} (p = {p_cd_all:.2e})")
print(f"  epsilon:  r = {r_eps_all:.3f} (p = {p_eps_all:.2e})")
print(f"  gamma: true={GAMMA_TRUE:.3f}, recovered={gamma_mean_rec:.3f}")
print(f"  c_effort: true={CE_POP:.4f}, mean recovered={np.mean([r['ce_rec'] for r in all_results]):.4f}")

# Add overall row
all_results.append({
    'dataset': 'overall',
    'r_cd': r_cd_all, 'p_cd': p_cd_all,
    'r_eps': r_eps_all, 'p_eps': p_eps_all,
    'ce_true': CE_POP, 'ce_rec': np.mean([r['ce_rec'] for r in all_results[:-1]]) if len(all_results) > 1 else np.nan,
    'gamma_true': GAMMA_TRUE, 'gamma_rec': gamma_mean_rec,
    'n_subj': N_SUBJ * N_DATASETS, 'n_trials': N_SUBJ * 45 * N_DATASETS,
})

# Save CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv('/workspace/results/stats/evc_lqr_recovery.csv', index=False)
print(f"\nSaved: /workspace/results/stats/evc_lqr_recovery.csv")

# ── Create figure ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'axes.spines.right': False,
    'axes.spines.top': False,
})

# Panel A: c_death recovery
ax1.scatter(np.log(all_true_cd), np.log(all_rec_cd), s=15, alpha=0.4,
            color='#457B9D', edgecolor='white', linewidth=0.3)
lim = [min(np.log(all_true_cd).min(), np.log(all_rec_cd).min()) - 0.2,
       max(np.log(all_true_cd).max(), np.log(all_rec_cd).max()) + 0.2]
ax1.plot(lim, lim, '--', color='#D1D5DB', linewidth=1, zorder=0)
ax1.set_xlim(lim)
ax1.set_ylim(lim)
ax1.set_xlabel('True log(c_death)', fontsize=12)
ax1.set_ylabel('Recovered log(c_death)', fontsize=12)
ax1.set_title(f'A  c_death recovery (r = {r_cd_all:.3f})', fontsize=13, fontweight='bold', loc='left')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Panel B: epsilon recovery
ax2.scatter(np.log(all_true_eps), np.log(all_rec_eps), s=15, alpha=0.4,
            color='#E63946', edgecolor='white', linewidth=0.3)
lim = [min(np.log(all_true_eps).min(), np.log(all_rec_eps).min()) - 0.2,
       max(np.log(all_true_eps).max(), np.log(all_rec_eps).max()) + 0.2]
ax2.plot(lim, lim, '--', color='#D1D5DB', linewidth=1, zorder=0)
ax2.set_xlim(lim)
ax2.set_ylim(lim)
ax2.set_xlabel('True log(epsilon)', fontsize=12)
ax2.set_ylabel('Recovered log(epsilon)', fontsize=12)
ax2.set_title(f'B  epsilon recovery (r = {r_eps_all:.3f})', fontsize=13, fontweight='bold', loc='left')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('/workspace/results/figs/paper/fig_s_lqr_recovery.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print(f"Saved: /workspace/results/figs/paper/fig_s_lqr_recovery.png")
print("\nDone!")
