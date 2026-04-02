#!/usr/bin/env python3
"""
model_comparison_v2.py — Full 8-model choice comparison (Plan v2)

Phase A: req·T vs req²·D effort parameterization within M1/M2
Phase B: All 8 models with winning effort term
Sequential: vigor prediction, clinical, bridge analyses

Uses NumPyro SVI with ClippedAdam + early stopping.
"""

import sys, time, warnings
warnings.filterwarnings("ignore")

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
from scipy.special import expit
from pathlib import Path

jax.config.update('jax_enable_x64', True)

DATA_DIR = Path("data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("results/stats/model_comparison_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_choice_data():
    beh = pd.read_csv(DATA_DIR / "behavior.csv")
    # Exclude calibration outliers
    exclude = [154, 197, 208]
    beh = beh[~beh['subj'].isin(exclude)].copy()

    subjects = sorted(beh['subj'].unique())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)

    # Trial variables
    beh['req_H'] = beh['effort_H']  # 0.6, 0.8, 1.0
    beh['req_L'] = 0.4
    beh['D_H'] = beh['distance_H']  # 1, 2, 3
    beh['T_H'] = beh['distance_H'].map({1: 5.0, 2: 7.0, 3: 9.0})  # exposure seconds
    beh['T_L'] = 5.0
    beh['p'] = beh['threat']  # 0.1, 0.5, 0.9

    # Effort terms
    beh['effort_reqT'] = beh['req_H'] * beh['T_H'] - beh['req_L'] * beh['T_L']
    beh['effort_lqr'] = beh['req_H']**2 * beh['D_H'] - beh['req_L']**2 * 1

    ch_subj = jnp.array([subj_to_idx[s] for s in beh['subj']])

    data = {
        'ch_subj': ch_subj,
        'ch_choice': jnp.array(beh['choice'].values),
        'p': jnp.array(beh['p'].values),
        'T_H': jnp.array(beh['T_H'].values),
        'T_L': jnp.array(beh['T_L'].values),
        'D_H': jnp.array(beh['D_H'].values, dtype=jnp.float64),
        'req_H': jnp.array(beh['req_H'].values),
        'req_L': jnp.array(beh['req_L'].values * np.ones(len(beh))),
        'effort_reqT': jnp.array(beh['effort_reqT'].values),
        'effort_lqr': jnp.array(beh['effort_lqr'].values),
        'subjects': subjects, 'N_S': N_S, 'N': len(beh),
        'beh': beh,
    }
    return data


# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════

def make_model(model_name, N_S, N, effort_key='effort_lqr'):
    """Create a NumPyro model for the given specification."""

    def model_fn(ch_subj, ch_choice, p, T_H, T_L, D_H, req_H, req_L,
                 effort_reqT, effort_lqr):
        effort = effort_lqr if effort_key == 'effort_lqr' else effort_reqT

        # ── Shared hierarchical structure ──
        # λ (effort cost) — all models
        mu_lam = numpyro.sample('mu_lam', dist.Normal(0.0, 1.0))
        sig_lam = numpyro.sample('sig_lam', dist.HalfNormal(0.5))
        # β (inverse temperature) — all models
        mu_beta = numpyro.sample('mu_beta', dist.Normal(0.5, 1.0))
        sig_beta = numpyro.sample('sig_beta', dist.HalfNormal(0.5))

        with numpyro.plate('subjects', N_S):
            lam_raw = numpyro.sample('lam_raw', dist.Normal(0.0, 1.0))
            beta_raw = numpyro.sample('beta_raw', dist.Normal(0.0, 1.0))

        lam = jnp.exp(mu_lam + sig_lam * lam_raw)
        beta = jnp.exp(mu_beta + sig_beta * beta_raw)
        numpyro.deterministic('lam', lam)
        numpyro.deterministic('beta', beta)

        lam_t = lam[ch_subj]
        beta_t = beta[ch_subj]

        # ── Model-specific ΔV ──
        if model_name == 'M1':
            # ΔV = 4 - λ·effort(D)
            dv = 4.0 - lam_t * effort

        elif model_name == 'M2':
            mu_gam = numpyro.sample('mu_gam', dist.Normal(0.0, 1.0))
            sig_gam = numpyro.sample('sig_gam', dist.HalfNormal(0.5))
            with numpyro.plate('subjects_gam', N_S):
                gam_raw = numpyro.sample('gam_raw', dist.Normal(0.0, 1.0))
            gam = jnp.exp(mu_gam + sig_gam * gam_raw)
            numpyro.deterministic('gam', gam)
            dv = 4.0 - lam_t * effort - gam[ch_subj] * p

        elif model_name == 'M3':
            # ΔV = 5·exp(-p·T_H) - exp(-p·T_L) - λ·effort
            dv = 5.0 * jnp.exp(-p * T_H) - jnp.exp(-p * T_L) - lam_t * effort

        elif model_name == 'M4':
            # Rate of return: [exp(-p·T_H)·5 - req_H·λ]/T_H - [exp(-p·T_L)·1 - req_L·λ]/T_L
            rate_H = (jnp.exp(-p * T_H) * 5.0 - req_H * lam_t) / T_H
            rate_L = (jnp.exp(-p * T_L) * 1.0 - req_L * lam_t) / T_L
            dv = rate_H - rate_L

        elif model_name == 'M5':
            mu_kap = numpyro.sample('mu_kap', dist.Normal(-1.0, 1.0))
            sig_kap = numpyro.sample('sig_kap', dist.HalfNormal(0.5))
            mu_alp = numpyro.sample('mu_alp', dist.Normal(0.0, 0.5))
            sig_alp = numpyro.sample('sig_alp', dist.HalfNormal(0.3))
            with numpyro.plate('subjects_ka', N_S):
                kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
                alp_raw = numpyro.sample('alp_raw', dist.Normal(0.0, 1.0))
            kap = jnp.exp(mu_kap + sig_kap * kap_raw)
            alp = jnp.exp(mu_alp + sig_alp * alp_raw)
            numpyro.deterministic('kap', kap)
            numpyro.deterministic('alp', alp)
            p_dist = jnp.power(jnp.clip(p, 0.01, 0.99), alp[ch_subj])
            dv = (5.0 * jnp.exp(-kap[ch_subj] * p_dist * T_H)
                  - jnp.exp(-kap[ch_subj] * p_dist * T_L)
                  - lam_t * effort)

        elif model_name == 'M6':
            mu_kap = numpyro.sample('mu_kap', dist.Normal(-1.0, 1.0))
            sig_kap = numpyro.sample('sig_kap', dist.HalfNormal(0.5))
            mu_alp = numpyro.sample('mu_alp', dist.Normal(0.0, 0.5))
            sig_alp = numpyro.sample('sig_alp', dist.HalfNormal(0.3))
            with numpyro.plate('subjects_ka', N_S):
                kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
                alp_raw = numpyro.sample('alp_raw', dist.Normal(0.0, 1.0))
            kap = jnp.exp(mu_kap + sig_kap * kap_raw)
            alp = jnp.exp(mu_alp + sig_alp * alp_raw)
            numpyro.deterministic('kap', kap)
            numpyro.deterministic('alp', alp)
            p_dist = jnp.power(jnp.clip(p, 0.01, 0.99), alp[ch_subj])
            rate_H = (jnp.exp(-kap[ch_subj] * p_dist * T_H) * 5.0 - req_H * lam_t) / T_H
            rate_L = (jnp.exp(-kap[ch_subj] * p_dist * T_L) * 1.0 - req_L * lam_t) / T_L
            dv = rate_H - rate_L

        elif model_name == 'M7':
            mu_kap = numpyro.sample('mu_kap', dist.Normal(-1.0, 1.0))
            sig_kap = numpyro.sample('sig_kap', dist.HalfNormal(0.5))
            mu_alp = numpyro.sample('mu_alp', dist.Normal(0.0, 0.5))
            sig_alp = numpyro.sample('sig_alp', dist.HalfNormal(0.3))
            mu_psi = numpyro.sample('mu_psi', dist.Normal(0.0, 1.0))
            sig_psi = numpyro.sample('sig_psi', dist.HalfNormal(0.5))
            with numpyro.plate('subjects_kap', N_S):
                kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
                alp_raw = numpyro.sample('alp_raw', dist.Normal(0.0, 1.0))
                psi_raw = numpyro.sample('psi_raw', dist.Normal(0.0, 1.0))
            kap = jnp.exp(mu_kap + sig_kap * kap_raw)
            alp = jnp.exp(mu_alp + sig_alp * alp_raw)
            psi = jnp.exp(mu_psi + sig_psi * psi_raw)
            numpyro.deterministic('kap', kap)
            numpyro.deterministic('alp', alp)
            numpyro.deterministic('psi', psi)
            p_dist = jnp.power(jnp.clip(p, 0.01, 0.99), alp[ch_subj])
            S_H = jnp.exp(-kap[ch_subj] * p_dist * T_H)
            S_L = jnp.exp(-kap[ch_subj] * p_dist * T_L)
            C = 5.0
            ev_H = S_H * 5.0 - (1 - S_H) * psi[ch_subj] * (5.0 + C) - lam_t * effort
            ev_L = S_L * 1.0 - (1 - S_L) * psi[ch_subj] * (1.0 + C)
            dv = ev_H - ev_L

        elif model_name == 'M8':
            mu_gam = numpyro.sample('mu_gam', dist.Normal(0.0, 1.0))
            sig_gam = numpyro.sample('sig_gam', dist.HalfNormal(0.5))
            mu_del = numpyro.sample('mu_del', dist.Normal(0.0, 1.0))
            sig_del = numpyro.sample('sig_del', dist.HalfNormal(0.5))
            with numpyro.plate('subjects_gd', N_S):
                gam_raw = numpyro.sample('gam_raw', dist.Normal(0.0, 1.0))
                del_raw = numpyro.sample('del_raw', dist.Normal(0.0, 1.0))
            gam = jnp.exp(mu_gam + sig_gam * gam_raw)
            delta = jnp.exp(mu_del + sig_del * del_raw)
            numpyro.deterministic('gam', gam)
            numpyro.deterministic('delta', delta)
            dv = 4.0 - lam_t * effort - gam[ch_subj] * p - delta[ch_subj] * p * D_H

        # ── Choice likelihood ──
        logit = jnp.clip(beta_t * dv, -20, 20)
        p_heavy = jax.nn.sigmoid(logit)

        with numpyro.plate('trials', N):
            numpyro.sample('obs', dist.Bernoulli(probs=jnp.clip(p_heavy, 1e-6, 1-1e-6)),
                          obs=ch_choice)

    return model_fn


def fit_model(model_name, data, effort_key='effort_lqr', n_steps=40000, lr=0.001, seed=42):
    """Fit a single model via SVI with early stopping."""
    model_fn = make_model(model_name, data['N_S'], data['N'], effort_key)

    kwargs = {k: data[k] for k in ['ch_subj', 'ch_choice', 'p', 'T_H', 'T_L',
                                     'D_H', 'req_H', 'req_L', 'effort_reqT', 'effort_lqr']}

    guide = AutoNormal(model_fn)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
    svi = SVI(model_fn, guide, optimizer, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    best_loss = float('inf')
    best_params = None
    best_step = 0

    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        lv = float(loss)
        if lv < best_loss:
            best_loss = lv
            best_params = svi.get_params(state)
            best_step = i + 1

    # Count parameters
    n_per_subj_map = {'M1': 2, 'M2': 3, 'M3': 2, 'M4': 2,
                      'M5': 4, 'M6': 4, 'M7': 5, 'M8': 4}
    n_pop_map = {'M1': 4, 'M2': 6, 'M3': 4, 'M4': 4,
                 'M5': 8, 'M6': 8, 'M7': 10, 'M8': 8}
    n_per = n_per_subj_map.get(model_name, 2)
    n_pop = n_pop_map.get(model_name, 4)
    n_params = n_per * data['N_S'] + n_pop
    bic_approx = 2 * best_loss + n_params * np.log(data['N'])

    # Extract parameters and evaluate
    param_sites = ['lam', 'beta']
    if model_name == 'M2': param_sites.append('gam')
    if model_name in ['M5', 'M6']: param_sites += ['kap', 'alp']
    if model_name == 'M7': param_sites += ['kap', 'alp', 'psi']
    if model_name == 'M8': param_sites += ['gam', 'delta']

    pred = Predictive(model_fn, guide=guide, params=best_params,
                      num_samples=200, return_sites=param_sites)
    samples = pred(random.PRNGKey(seed + 1), **kwargs)

    param_means = {}
    for site in param_sites:
        param_means[site] = np.array(samples[site]).mean(0)

    # Choice predictions using extracted params
    beh = data['beh']
    ch_subj_np = np.array(data['ch_subj'])
    ch_choice_np = np.array(data['ch_choice'])
    p_np = np.array(data['p'])
    T_H_np = np.array(data['T_H'])
    T_L_np = np.array(data['T_L'])
    D_H_np = np.array(data['D_H'])
    req_H_np = np.array(data['req_H'])
    req_L_np = np.array(data['req_L'])
    effort_np = np.array(data[effort_key])

    lam_v = param_means['lam']
    beta_v = param_means['beta']

    # Compute ΔV per trial
    if model_name == 'M1':
        dv = 4.0 - lam_v[ch_subj_np] * effort_np
    elif model_name == 'M2':
        gam_v = param_means['gam']
        dv = 4.0 - lam_v[ch_subj_np] * effort_np - gam_v[ch_subj_np] * p_np
    elif model_name == 'M3':
        dv = 5.0 * np.exp(-p_np * T_H_np) - np.exp(-p_np * T_L_np) - lam_v[ch_subj_np] * effort_np
    elif model_name == 'M4':
        rH = (np.exp(-p_np * T_H_np) * 5 - req_H_np * lam_v[ch_subj_np]) / T_H_np
        rL = (np.exp(-p_np * T_L_np) * 1 - req_L_np * lam_v[ch_subj_np]) / T_L_np
        dv = rH - rL
    elif model_name in ['M5', 'M6']:
        kap_v = param_means['kap']; alp_v = param_means['alp']
        p_distorted = np.clip(p_np, 0.01, 0.99) ** alp_v[ch_subj_np]
        if model_name == 'M5':
            dv = (5 * np.exp(-kap_v[ch_subj_np] * p_distorted * T_H_np)
                  - np.exp(-kap_v[ch_subj_np] * p_distorted * T_L_np)
                  - lam_v[ch_subj_np] * effort_np)
        else:
            rH = (np.exp(-kap_v[ch_subj_np] * p_distorted * T_H_np) * 5 - req_H_np * lam_v[ch_subj_np]) / T_H_np
            rL = (np.exp(-kap_v[ch_subj_np] * p_distorted * T_L_np) * 1 - req_L_np * lam_v[ch_subj_np]) / T_L_np
            dv = rH - rL
    elif model_name == 'M7':
        kap_v = param_means['kap']; alp_v = param_means['alp']; psi_v = param_means['psi']
        p_distorted = np.clip(p_np, 0.01, 0.99) ** alp_v[ch_subj_np]
        S_H = np.exp(-kap_v[ch_subj_np] * p_distorted * T_H_np)
        S_L = np.exp(-kap_v[ch_subj_np] * p_distorted * T_L_np)
        ev_H = S_H * 5 - (1 - S_H) * psi_v[ch_subj_np] * 10 - lam_v[ch_subj_np] * effort_np
        ev_L = S_L * 1 - (1 - S_L) * psi_v[ch_subj_np] * 6
        dv = ev_H - ev_L
    elif model_name == 'M8':
        gam_v = param_means['gam']; delta_v = param_means['delta']
        dv = 4.0 - lam_v[ch_subj_np] * effort_np - gam_v[ch_subj_np] * p_np - delta_v[ch_subj_np] * p_np * D_H_np

    p_heavy = expit(np.clip(beta_v[ch_subj_np] * dv, -20, 20))
    acc = ((p_heavy >= 0.5).astype(int) == ch_choice_np).mean()

    # Per-subject choice r²
    cdf = pd.DataFrame({'subj': ch_subj_np, 'choice': ch_choice_np, 'p_H': p_heavy})
    sc = cdf.groupby('subj').agg(o=('choice', 'mean'), p=('p_H', 'mean')).reset_index()
    r_choice, _ = pearsonr(sc['o'], sc['p'])

    return {
        'model': model_name, 'effort': effort_key,
        'best_loss': best_loss, 'best_step': best_step,
        'bic_approx': bic_approx, 'n_params': n_params,
        'accuracy': acc, 'choice_r': r_choice, 'choice_r2': r_choice**2,
        'param_means': param_means, 'params_fit': best_params,
        'guide': guide, 'model_fn': model_fn,
        'dv': dv, 'p_heavy': p_heavy,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()
    data = load_choice_data()
    print(f"N={data['N_S']} subjects, {data['N']} trials")
    print(f"Effort req·T range: [{data['beh']['effort_reqT'].min():.1f}, {data['beh']['effort_reqT'].max():.1f}]")
    print(f"Effort req²·D range: [{data['beh']['effort_lqr'].min():.2f}, {data['beh']['effort_lqr'].max():.2f}]")

    # ═══════════════════════════════════════════════════════════
    # PHASE A: EFFORT PARAMETERIZATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE A: EFFORT PARAMETERIZATION (M1/M2 × req·T/req²·D)")
    print("=" * 70)

    phase_a_results = []
    for mname in ['M1', 'M2']:
        for ek in ['effort_reqT', 'effort_lqr']:
            label = f"{mname}_{ek.split('_')[1]}"
            print(f"\n  Fitting {label}...")
            res = fit_model(mname, data, effort_key=ek)
            phase_a_results.append(res)
            print(f"    Loss={res['best_loss']:.1f} (step {res['best_step']}), "
                  f"BIC={res['bic_approx']:.0f}, Acc={res['accuracy']:.3f}, r²={res['choice_r2']:.3f}")

    # Select winner
    phase_a_df = pd.DataFrame([{
        'model': r['model'], 'effort': r['effort'],
        'loss': r['best_loss'], 'bic': r['bic_approx'],
        'acc': r['accuracy'], 'r2': r['choice_r2'],
    } for r in phase_a_results]).sort_values('bic')

    print(f"\n{'=' * 70}")
    print("PHASE A RESULTS")
    print(f"{'=' * 70}")
    print(phase_a_df.to_string(index=False))

    # Pick effort term from M2 (the model where it matters most)
    m2_results = [r for r in phase_a_results if r['model'] == 'M2']
    best_m2 = min(m2_results, key=lambda r: r['bic_approx'])
    winning_effort = best_m2['effort']
    print(f"\nWinning effort parameterization: {winning_effort}")
    print(f"  (from M2: ΔBIC = {m2_results[0]['bic_approx'] - m2_results[1]['bic_approx']:.1f})")

    # ═══════════════════════════════════════════════════════════
    # PHASE B: ALL 8 MODELS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE B: ALL 8 MODELS (effort={winning_effort})")
    print("=" * 70)

    all_results = []
    for mname in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']:
        print(f"\n  Fitting {mname}...")
        try:
            res = fit_model(mname, data, effort_key=winning_effort)
            all_results.append(res)
            print(f"    Loss={res['best_loss']:.1f} (step {res['best_step']}), "
                  f"BIC={res['bic_approx']:.0f}, Acc={res['accuracy']:.3f}, r²={res['choice_r2']:.3f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            all_results.append({
                'model': mname, 'effort': winning_effort,
                'best_loss': float('inf'), 'bic_approx': float('inf'),
                'accuracy': 0, 'choice_r': 0, 'choice_r2': 0,
                'best_step': 0, 'n_params': 0,
                'param_means': {}, 'dv': None, 'p_heavy': None,
            })

    # ── Comparison table ──
    comp_df = pd.DataFrame([{
        'Model': r['model'],
        'Params': r.get('n_params', 0),
        'Loss': r['best_loss'],
        'BIC_approx': r['bic_approx'],
        'Accuracy': r['accuracy'],
        'Choice_r2': r['choice_r2'],
        'Best_step': r.get('best_step', 0),
    } for r in all_results]).sort_values('BIC_approx')

    best_bic = comp_df['BIC_approx'].min()
    comp_df['ΔBIC'] = comp_df['BIC_approx'] - best_bic

    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON (sorted by BIC_approx)")
    print(f"{'=' * 70}")
    print(comp_df.to_string(index=False))

    winner_name = comp_df.iloc[0]['Model']
    winner = [r for r in all_results if r['model'] == winner_name][0]
    print(f"\n*** WINNING MODEL: {winner_name} ***")

    # ── Posterior predictive check ──
    print(f"\n{'=' * 70}")
    print(f"POSTERIOR PREDICTIVE CHECK ({winner_name})")
    print(f"{'=' * 70}")

    beh = data['beh'].copy()
    beh['p_heavy_pred'] = winner['p_heavy']

    print(f"\n{'':>12} {'D=1':>16} {'D=2':>16} {'D=3':>16}")
    for T in [0.1, 0.5, 0.9]:
        row = f"T={T:.0%}  "
        for D in [1, 2, 3]:
            sub = beh[(beh['threat'].round(1) == T) & (beh['distance_H'] == D)]
            obs = sub['choice'].mean()
            pred = sub['p_heavy_pred'].mean()
            diff = pred - obs
            flag = " *" if abs(diff) > 0.10 else ""
            row += f"  {obs:.3f}/{pred:.3f}{flag}"
        print(row)

    # ── Save results ──
    comp_df.to_csv(OUT_DIR / 'model_comparison.csv', index=False)
    phase_a_df.to_csv(OUT_DIR / 'phase_a_effort.csv', index=False)

    # Save winning model parameters
    if winner['param_means']:
        param_rows = []
        for i, s in enumerate(data['subjects']):
            row = {'subj': s}
            for pname, vals in winner['param_means'].items():
                if len(vals) == data['N_S']:
                    row[pname] = vals[i]
            param_rows.append(row)
        pd.DataFrame(param_rows).to_csv(OUT_DIR / f'{winner_name}_params.csv', index=False)

    # Save per-trial predictions
    beh[['subj', 'trial', 'threat', 'distance_H', 'choice', 'p_heavy_pred']].to_csv(
        OUT_DIR / f'{winner_name}_predictions.csv', index=False)

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed:.1f} min")
    print(f"Results saved to {OUT_DIR}")
