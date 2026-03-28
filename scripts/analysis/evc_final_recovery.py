"""
Parameter recovery for the EVC 2+2 model (population epsilon).

Simulates 3 datasets x 50 subjects x 81 trials each, refits the model,
and computes Pearson r between log(true) and log(recovered) for ce and cd.

Usage:
    python scripts/analysis/evc_final_recovery.py
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

# ── From fitted N=293 model ──
# Read actual fitted params
fitted_params = pd.read_csv('/workspace/results/stats/oc_evc_final_params.csv')
fitted_pop = pd.read_csv('/workspace/results/stats/oc_evc_final_population.csv')

log_ce = np.log(fitted_params['c_effort'].values)
log_cd = np.log(fitted_params['c_death'].values)
# Remove extreme cd values for simulation distribution
mask_cd = fitted_params['c_death'] < 1000
log_cd_clean = np.log(fitted_params.loc[mask_cd, 'c_death'].values)

MU_CE = log_ce.mean()
SD_CE = log_ce.std()
MU_CD = log_cd_clean.mean()
SD_CD = log_cd_clean.std()

# Population parameters from the fitted model
EPSILON = float(fitted_pop['epsilon'].iloc[0])
GAMMA = float(fitted_pop['gamma'].iloc[0])
CE_VIGOR = float(fitted_pop['ce_vigor'].iloc[0])
TAU = float(fitted_pop['tau'].iloc[0])
P_ESC = float(fitted_pop['p_esc'].iloc[0])
SIGMA_MOTOR = float(fitted_pop['sigma_motor'].iloc[0])
SIGMA_V = float(fitted_pop['sigma_v'].iloc[0])

print(f"Simulation parameters:")
print(f"  MU_CE={MU_CE:.3f}, SD_CE={SD_CE:.3f}")
print(f"  MU_CD={MU_CD:.3f}, SD_CD={SD_CD:.3f}")
print(f"  EPSILON={EPSILON:.4f}, GAMMA={GAMMA:.3f}")
print(f"  CE_VIGOR={CE_VIGOR:.4f}, TAU={TAU:.3f}")
print(f"  P_ESC={P_ESC:.3f}, SIGMA_MOTOR={SIGMA_MOTOR:.3f}, SIGMA_V={SIGMA_V:.3f}")

# Task design
THREATS = [0.1, 0.5, 0.9]
DISTANCES = [1, 2, 3]
R_H, R_L, C = 5.0, 1.0, 5.0
REQ_H, REQ_L = 0.9, 0.4

N_SUBJECTS = 50
N_DATASETS = 3
N_SVI_STEPS = 25000


def compute_survival(T):
    T_w = T ** GAMMA
    return (1.0 - T_w) + EPSILON * T_w * P_ESC


def compute_optimal_vigor(cd, T, R, req, D):
    u_grid = np.linspace(0.1, 1.5, 30)
    T_w = T ** GAMMA
    S_u = (1.0 - T_w) + EPSILON * T_w * P_ESC / (1.0 + np.exp(-(u_grid - req) / SIGMA_MOTOR))
    deviation = u_grid - req
    eu_grid = S_u * R - (1.0 - S_u) * cd * (R + C) - CE_VIGOR * deviation**2 * D
    weights = np.exp(eu_grid * 10.0 - np.max(eu_grid * 10.0))
    weights = weights / weights.sum()
    return np.sum(weights * u_grid)


def simulate_dataset(seed, n_subjects=N_SUBJECTS):
    rng = np.random.RandomState(seed)

    true_ce = np.exp(rng.normal(MU_CE, SD_CE, n_subjects))
    true_cd = np.exp(rng.normal(MU_CD, SD_CD, n_subjects))
    true_ce = np.clip(true_ce, 0.05, 10.0)
    true_cd = np.clip(true_cd, 0.1, 500.0)

    # Choice trials: 3T x 3D x 5 reps = 45
    choice_records = []
    for subj in range(n_subjects):
        for T in THREATS:
            for D in DISTANCES:
                for rep in range(5):
                    ce = true_ce[subj]
                    S = compute_survival(T)
                    delta_eu = S * 4.0 - ce * (0.81 * D - 0.16)
                    logit = np.clip(delta_eu / TAU, -20, 20)
                    p_heavy = 1.0 / (1.0 + np.exp(-logit))
                    choice = int(rng.random() < p_heavy)
                    choice_records.append({
                        'subj_idx': subj, 'T': T, 'D': D,
                        'choice': choice, 'is_heavy': choice,
                    })
    choice_df = pd.DataFrame(choice_records)

    # Vigor for choice trials
    vigor_records = []
    for _, row in choice_df.iterrows():
        subj = int(row['subj_idx'])
        T, D = row['T'], row['D']
        cd = true_cd[subj]
        R, req = (R_H, REQ_H) if row['is_heavy'] else (R_L, REQ_L)
        u_star = compute_optimal_vigor(cd, T, R, req, D)
        excess = u_star - req + rng.normal(0, SIGMA_V)
        vigor_records.append({
            'subj_idx': subj, 'T': T, 'D': D, 'R': R,
            'req': req, 'excess': excess, 'is_heavy': int(row['is_heavy']),
            'trial_type': 'choice',
        })

    # Probe trials: 3T x 3D x 2 cookie x 2 reps = 36
    for subj in range(n_subjects):
        for T in THREATS:
            for D in DISTANCES:
                for cookie_type in [0, 1]:
                    for rep in range(2):
                        cd = true_cd[subj]
                        R, req = (R_H, REQ_H) if cookie_type else (R_L, REQ_L)
                        u_star = compute_optimal_vigor(cd, T, R, req, D)
                        excess = u_star - req + rng.normal(0, SIGMA_V)
                        vigor_records.append({
                            'subj_idx': subj, 'T': T, 'D': D, 'R': R,
                            'req': req, 'excess': excess,
                            'is_heavy': cookie_type, 'trial_type': 'probe',
                        })

    vigor_df = pd.DataFrame(vigor_records)

    # Cookie-type centering
    choice_vigor = vigor_df[vigor_df['trial_type'] == 'choice']
    heavy_mean = choice_vigor[choice_vigor['is_heavy'] == 1]['excess'].mean()
    light_mean = choice_vigor[choice_vigor['is_heavy'] == 0]['excess'].mean()
    vigor_df['excess_cc'] = vigor_df['excess'] - np.where(
        vigor_df['is_heavy'] == 1, heavy_mean, light_mean)
    vigor_df['offset'] = np.where(
        vigor_df['is_heavy'] == 1, heavy_mean, light_mean)

    data = {
        'ch_subj': jnp.array(choice_df['subj_idx'].values),
        'ch_T': jnp.array(choice_df['T'].values),
        'ch_dist_H': jnp.array(choice_df['D'].values, dtype=jnp.float64),
        'ch_choice': jnp.array(choice_df['choice'].values),
        'vig_subj': jnp.array(vigor_df['subj_idx'].values),
        'vig_T': jnp.array(vigor_df['T'].values),
        'vig_R': jnp.array(vigor_df['R'].values),
        'vig_req': jnp.array(vigor_df['req'].values),
        'vig_dist': jnp.array(vigor_df['D'].values, dtype=jnp.float64),
        'vig_excess': jnp.array(vigor_df['excess_cc'].values),
        'vig_offset': jnp.array(vigor_df['offset'].values),
        'N_S': n_subjects,
        'N_choice': len(choice_df),
        'N_vigor': len(vigor_df),
    }

    return data, {'ce': true_ce, 'cd': true_cd}


def make_recovery_model(N_S, N_choice, N_vigor):
    def evc_2plus2(ch_subj, ch_T, ch_dist_H, ch_choice,
                   vig_subj, vig_T, vig_R, vig_req, vig_dist,
                   vig_excess, vig_offset):
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        eps_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
        epsilon = numpyro.deterministic('epsilon', jnp.exp(eps_raw))

        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)
        ce_vigor_raw = numpyro.sample('ce_vigor_raw', dist.Normal(-3.0, 1.0))
        ce_vigor = numpyro.deterministic('ce_vigor', jnp.exp(ce_vigor_raw))

        with numpyro.plate('subjects', N_S):
            ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))

        c_effort = jnp.exp(mu_ce + sigma_ce * ce_raw)
        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        numpyro.deterministic('c_effort', c_effort)
        numpyro.deterministic('c_death', c_death)

        ce_ch = c_effort[ch_subj]
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc
        delta_eu = S_ch * 4.0 - ce_ch * (0.81 * ch_dist_H - 0.16)
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        cd_v = c_death[vig_subj]
        T_w_v = jnp.power(vig_T, gamma)
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T_w_v[:, None])
               + epsilon * T_w_v[:, None] * p_esc
               * jax.nn.sigmoid((u_g - vig_req[:, None]) / sigma_motor))
        deviation = u_g - vig_req[:, None]
        eu_grid = (S_u * vig_R[:, None]
                   - (1.0 - S_u) * cd_v[:, None] * (vig_R[:, None] + 5.0)
                   - ce_vigor * deviation**2 * vig_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - vig_req - vig_offset

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)

    return evc_2plus2


def fit_recovery(data, seed=42, n_steps=N_SVI_STEPS, lr=0.002):
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
    pred = Predictive(
        model, guide=guide, params=params_fit, num_samples=300,
        return_sites=['c_effort', 'c_death', 'epsilon', 'gamma'])
    samples = pred(random.PRNGKey(seed + 100), **kwargs)

    rec_ce = np.array(samples['c_effort']).mean(0)
    rec_cd = np.array(samples['c_death']).mean(0)
    rec_eps = float(np.array(samples['epsilon']).mean())
    rec_gamma = float(np.array(samples['gamma']).mean())

    return rec_ce, rec_cd, rec_eps, rec_gamma


def main():
    t0 = time.time()
    os.makedirs('/workspace/results/figs/paper', exist_ok=True)
    os.makedirs('/workspace/results/stats', exist_ok=True)

    all_results = []
    all_true_ce, all_rec_ce = [], []
    all_true_cd, all_rec_cd = [], []
    eps_rec_list, gamma_rec_list = [], []

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
            data, seed=seed+1, n_steps=N_SVI_STEPS)

        r_ce, p_ce = pearsonr(np.log(true_params['ce']), np.log(rec_ce))
        r_cd, p_cd = pearsonr(np.log(true_params['cd']), np.log(rec_cd))

        print(f"  Recovery r: ce={r_ce:.3f}, cd={r_cd:.3f}")
        print(f"  Epsilon: true={EPSILON:.4f}, rec={rec_eps:.4f}")
        print(f"  Gamma: true={GAMMA:.3f}, rec={rec_gamma:.3f}")

        all_true_ce.extend(true_params['ce'])
        all_rec_ce.extend(rec_ce)
        all_true_cd.extend(true_params['cd'])
        all_rec_cd.extend(rec_cd)
        eps_rec_list.append(rec_eps)
        gamma_rec_list.append(rec_gamma)

        for i in range(N_SUBJECTS):
            all_results.append({
                'dataset': ds+1, 'subject': i+1,
                'true_ce': true_params['ce'][i], 'rec_ce': float(rec_ce[i]),
                'true_cd': true_params['cd'][i], 'rec_cd': float(rec_cd[i]),
                'true_eps': EPSILON, 'rec_eps': rec_eps,
                'true_gamma': GAMMA, 'rec_gamma': rec_gamma,
            })

    # Aggregate
    all_true_ce = np.array(all_true_ce)
    all_rec_ce = np.array(all_rec_ce)
    all_true_cd = np.array(all_true_cd)
    all_rec_cd = np.array(all_rec_cd)

    r_ce_all, p_ce_all = pearsonr(np.log(all_true_ce), np.log(all_rec_ce))
    r_cd_all, p_cd_all = pearsonr(np.log(all_true_cd), np.log(all_rec_cd))
    eps_rec_mean = np.mean(eps_rec_list)
    gamma_rec_mean = np.mean(gamma_rec_list)

    print(f"\n{'='*60}")
    print(f"AGGREGATE RECOVERY ({N_DATASETS} datasets x {N_SUBJECTS} subjects)")
    print(f"{'='*60}")
    print(f"  c_effort:  r={r_ce_all:.3f}  (p={p_ce_all:.2e})  "
          f"{'PASS' if r_ce_all > 0.7 else 'FAIL'}")
    print(f"  c_death:   r={r_cd_all:.3f}  (p={p_cd_all:.2e})  "
          f"{'PASS' if r_cd_all > 0.7 else 'FAIL'}")
    print(f"  epsilon:   true={EPSILON:.4f}, rec={eps_rec_mean:.4f}  "
          f"(bias={eps_rec_mean - EPSILON:+.4f})  "
          f"{'PASS' if abs(eps_rec_mean - EPSILON) / max(EPSILON, 0.001) < 0.3 else 'FAIL'}")
    print(f"  gamma:     true={GAMMA:.3f}, rec={gamma_rec_mean:.3f}  "
          f"(bias={gamma_rec_mean - GAMMA:+.3f})  "
          f"{'PASS' if abs(gamma_rec_mean - GAMMA) / GAMMA < 0.3 else 'FAIL'}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

    # Save CSV
    results_df = pd.DataFrame(all_results)
    csv_path = '/workspace/results/stats/evc_final_recovery.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Figure: 2-panel scatter (ce and cd)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    params_list = [
        ('c_effort', all_true_ce, all_rec_ce, r_ce_all, p_ce_all),
        ('c_death', all_true_cd, all_rec_cd, r_cd_all, p_cd_all),
    ]

    for ax, (name, true, rec, r_val, p_val) in zip(axes, params_list):
        log_true = np.log(true)
        log_rec = np.log(rec)
        ax.scatter(log_true, log_rec, alpha=0.3, s=15, color='steelblue')

        lo = min(log_true.min(), log_rec.min())
        hi = max(log_true.max(), log_rec.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin],
                'k--', lw=1, alpha=0.5)

        m, b = np.polyfit(log_true, log_rec, 1)
        x_line = np.linspace(lo-margin, hi+margin, 100)
        ax.plot(x_line, m*x_line + b, 'r-', lw=1.5, alpha=0.7)

        ax.set_xlabel(f'log(true {name})', fontsize=11)
        ax.set_ylabel(f'log(recovered {name})', fontsize=11)
        status = 'PASS' if r_val > 0.7 else 'FAIL'
        ax.set_title(f'{name}\nr = {r_val:.3f} [{status}]', fontsize=12)
        ax.set_xlim(lo-margin, hi+margin)
        ax.set_ylim(lo-margin, hi+margin)

    plt.tight_layout()
    fig_path = '/workspace/results/figs/paper/fig_s_final_recovery.png'
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
