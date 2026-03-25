"""
Effort-Scaled Model Comparison
==============================
Tests whether scaling E by distance (E = effort_rate × distance) improves
the choice model, and whether combining E-scaling with α-in-S (vigor-informed
survival) gives the best fit.

Models tested:
  L3_add:         SV = R·S - k·E        - β·(1-S),  S = (1-T)+T/(1+λD)         [current winner]
  L4a_add:        SV = R·S - k·E        - β·(1-S),  S = (1-T)+T/(1+λD/α)       [α in S]
  L3_add_Escaled: SV = R·S - k·(E·D)    - β·(1-S),  S = (1-T)+T/(1+λD)         [E scaled]
  L4a_add_Escaled:SV = R·S - k·(E·D)    - β·(1-S),  S = (1-T)+T/(1+λD/α)       [E scaled + α in S]

Usage: python scripts/run_effort_scaled_comparison.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from jax import random

numpyro.set_platform('cpu')

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path('/workspace')
DATA_DIR = ROOT / 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
STAT_DIR = ROOT / 'results/stats'

# ── Load data ────────────────────────────────────────────────────────────────
print('Loading data...', flush=True)
behavior = pd.read_csv(DATA_DIR / 'behavior.csv')
vigor_params = pd.read_csv(STAT_DIR / 'vigor_hbm_posteriors.csv')

df = behavior.merge(vigor_params[['subj', 'alpha_bayes']], on='subj')

subj_ids = sorted(df['subj'].unique())
subj_to_idx = {s: i for i, s in enumerate(subj_ids)}
n_subj = len(subj_ids)

# JAX arrays
si    = jnp.array(df['subj'].map(subj_to_idx).values, dtype=jnp.int32)
T     = jnp.array(df['threat'].values,                 dtype=jnp.float32)
eH    = jnp.array(df['effort_H'].values,               dtype=jnp.float32)
eL    = jnp.array(df['effort_L'].values,               dtype=jnp.float32)
dH    = jnp.array(df['distance_H'].values,             dtype=jnp.float32)
dL    = jnp.array(df['distance_L'].values,             dtype=jnp.float32)
ch    = jnp.array(df['choice'].values,                 dtype=jnp.float32)
alpha = jnp.array(df['alpha_bayes'].values,            dtype=jnp.float32)

# Scaled effort: rate × distance (total physical cost proxy)
eH_scaled = eH * dH  # {0.6, 1.6, 3.0}
eL_scaled = eL * dL  # {0.4} (always 0.4 × 1 = 0.4)

R_H = 5.0; R_L = 1.0

print(f'N = {n_subj} subjects, {len(df)} trials', flush=True)
print(f'E_H (rate only): {sorted(df["effort_H"].unique())}', flush=True)
print(f'E_H_scaled (rate × dist): {sorted((df["effort_H"] * df["distance_H"]).unique())}', flush=True)

# ── Model definitions ────────────────────────────────────────────────────────

def L3_add(si, T, eH, eL, dH, dL, alpha, ch=None):
    """Current winner: S = (1-T)+T/(1+λD), SV = R·S - k·E - β·(1-S)"""
    tau = numpyro.sample('tau', dist.LogNormal(0, 1))
    lam = numpyro.sample('lam', dist.LogNormal(0, 1))
    mu_logk = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk = numpyro.sample('sd_logk', dist.HalfNormal(1))
    mu_logb = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb = numpyro.sample('sd_logb', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
        logb = numpyro.sample('logb', dist.Normal(mu_logb, sd_logb))
    k = jnp.exp(logk[si]); beta = jnp.exp(logb[si])
    SH = (1 - T) + T / (1.0 + lam * dH)
    SL = (1 - T) + T / (1.0 + lam * dL)
    SVH = R_H * SH - k * eH - beta * (1 - SH)
    SVL = R_L * SL - k * eL - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def L4a_add(si, T, eH, eL, dH, dL, alpha, ch=None):
    """α in survival: S = (1-T)+T/(1+λD/α), SV = R·S - k·E - β·(1-S)"""
    tau = numpyro.sample('tau', dist.LogNormal(0, 1))
    lam = numpyro.sample('lam', dist.LogNormal(0, 1))
    mu_logk = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk = numpyro.sample('sd_logk', dist.HalfNormal(1))
    mu_logb = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb = numpyro.sample('sd_logb', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
        logb = numpyro.sample('logb', dist.Normal(mu_logb, sd_logb))
    k = jnp.exp(logk[si]); beta = jnp.exp(logb[si])
    SH = (1 - T) + T / (1.0 + lam * dH / alpha)
    SL = (1 - T) + T / (1.0 + lam * dL / alpha)
    SVH = R_H * SH - k * eH - beta * (1 - SH)
    SVL = R_L * SL - k * eL - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def L3_add_Escaled(si, T, eH, eL, dH, dL, alpha, ch=None):
    """E scaled by distance: SV = R·S - k·(E·D) - β·(1-S)"""
    tau = numpyro.sample('tau', dist.LogNormal(0, 1))
    lam = numpyro.sample('lam', dist.LogNormal(0, 1))
    mu_logk = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk = numpyro.sample('sd_logk', dist.HalfNormal(1))
    mu_logb = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb = numpyro.sample('sd_logb', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
        logb = numpyro.sample('logb', dist.Normal(mu_logb, sd_logb))
    k = jnp.exp(logk[si]); beta = jnp.exp(logb[si])
    SH = (1 - T) + T / (1.0 + lam * dH)
    SL = (1 - T) + T / (1.0 + lam * dL)
    # E scaled: effort_rate × distance
    SVH = R_H * SH - k * (eH * dH) - beta * (1 - SH)
    SVL = R_L * SL - k * (eL * dL) - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def L4a_add_Escaled(si, T, eH, eL, dH, dL, alpha, ch=None):
    """E scaled + α in survival: full vigor-informed model"""
    tau = numpyro.sample('tau', dist.LogNormal(0, 1))
    lam = numpyro.sample('lam', dist.LogNormal(0, 1))
    mu_logk = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk = numpyro.sample('sd_logk', dist.HalfNormal(1))
    mu_logb = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb = numpyro.sample('sd_logb', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
        logb = numpyro.sample('logb', dist.Normal(mu_logb, sd_logb))
    k = jnp.exp(logk[si]); beta = jnp.exp(logb[si])
    SH = (1 - T) + T / (1.0 + lam * dH / alpha)
    SL = (1 - T) + T / (1.0 + lam * dL / alpha)
    SVH = R_H * SH - k * (eH * dH) - beta * (1 - SH)
    SVL = R_L * SL - k * (eL * dL) - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


# ── Registry ─────────────────────────────────────────────────────────────────
models = {
    'L3_add':           L3_add,
    'L4a_add':          L4a_add,
    'L3_add_Escaled':   L3_add_Escaled,
    'L4a_add_Escaled':  L4a_add_Escaled,
}

# All have 2 per-subject params (k, β) and 6 population params (tau, lam, mu/sd for k and β)
n_per_subj = {k: 2 for k in models}
n_pop = {k: 6 for k in models}

# ── SVI fitting ──────────────────────────────────────────────────────────────
data = dict(si=si, T=T, eH=eH, eL=eL, dH=dH, dL=dL, alpha=alpha, ch=ch)

def fit_svi(model, name, n_steps=15000, lr=0.003):
    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam(lr), loss=Trace_ELBO())
    state = svi.init(random.PRNGKey(42), **data)
    update_fn = jax.jit(svi.update)

    losses = []
    for i in range(n_steps):
        state, loss = update_fn(state, **data)
        losses.append(float(loss))
        if (i + 1) % 5000 == 0:
            running_elbo = -np.mean(losses[-500:])
            print(f'  {name}: step {i+1}/{n_steps}, ELBO={running_elbo:.1f}', flush=True)

    elbo = -np.mean(losses[-200:])
    print(f'  {name}: FINAL ELBO = {elbo:.1f}', flush=True)
    return {
        'elbo':   elbo,
        'losses': losses,
        'params': svi.get_params(state),
        'guide':  guide,
        'svi':    svi,
    }

# ── Fit all 4 models ─────────────────────────────────────────────────────────
results = {}
for name, model_fn in models.items():
    print(f'\nFitting {name}...', flush=True)
    results[name] = fit_svi(model_fn, name)

# ── Comparison table ─────────────────────────────────────────────────────────
n_total = len(df)
rows = []
for name in models:
    n_sp = n_per_subj[name] * n_subj + n_pop[name]
    elbo = results[name]['elbo']
    bic  = -2 * elbo + n_sp * np.log(n_total)
    rows.append({'model': name, 'ELBO': elbo, 'total_params': n_sp, 'BIC': bic})

comp = pd.DataFrame(rows).sort_values('ELBO', ascending=False)
comp['dELBO'] = comp['ELBO'] - comp['ELBO'].max()
comp['dBIC']  = comp['BIC'] - comp['BIC'].min()

print('\n' + '=' * 80)
print('EFFORT-SCALED MODEL COMPARISON')
print('=' * 80)
print(comp[['model', 'ELBO', 'dELBO', 'BIC', 'dBIC']].to_string(index=False))

# ── Key comparisons ──────────────────────────────────────────────────────────
def get_elbo(name):
    return comp[comp['model'] == name]['ELBO'].values[0]

print(f'\n--- Key comparisons (ΔELBO, positive = second model wins) ---')
pairs = [
    ('L3_add',         'L3_add_Escaled',  'Effect of E scaling (no α)'),
    ('L4a_add',        'L4a_add_Escaled', 'Effect of E scaling (with α)'),
    ('L3_add',         'L4a_add',         'Effect of α in S (no E scaling)'),
    ('L3_add_Escaled', 'L4a_add_Escaled', 'Effect of α in S (with E scaling)'),
    ('L3_add',         'L4a_add_Escaled', 'Full improvement: E scaling + α'),
]
for a, b, label in pairs:
    delta = get_elbo(b) - get_elbo(a)
    print(f'  {label}: {delta:+.1f}')

# ── Extract parameters from each model ───────────────────────────────────────
print('\n--- Population parameters ---')
for name in models:
    r = results[name]
    predictive = Predictive(r['guide'], params=r['params'], num_samples=500)
    samples = predictive(random.PRNGKey(99), **data)

    lam_samples = np.array(samples['lam'])
    tau_samples = np.array(samples['tau'])
    logk_samples = np.array(samples['logk'])
    logb_samples = np.array(samples['logb'])

    lam_mean = lam_samples.mean()
    tau_mean = tau_samples.mean()
    k_mean = np.exp(logk_samples.mean(axis=0)).mean()
    beta_mean = np.exp(logb_samples.mean(axis=0)).mean()

    print(f'\n  {name}:')
    print(f'    λ = {lam_mean:.2f} (±{lam_samples.std():.2f})')
    print(f'    τ = {tau_mean:.3f}')
    print(f'    k (mean across subj) = {k_mean:.4f}')
    print(f'    β (mean across subj) = {beta_mean:.4f}')

    # S values at key conditions
    def S_val(T_val, D_val, alpha_val=0.54):
        if 'Escaled' not in name and '4a' not in name:
            return (1 - T_val) + T_val / (1 + lam_mean * D_val)
        elif '4a' in name:
            return (1 - T_val) + T_val / (1 + lam_mean * D_val / alpha_val)
        else:
            return (1 - T_val) + T_val / (1 + lam_mean * D_val)

    print(f'    S(T=0.9, D=1) = {S_val(0.9, 1):.3f}')
    print(f'    S(T=0.9, D=3) = {S_val(0.9, 3):.3f}')
    print(f'    S distance gradient at T=0.9: {S_val(0.9, 1) - S_val(0.9, 3):.4f}')

# ── Save comparison ──────────────────────────────────────────────────────────
comp.to_csv(STAT_DIR / 'effort_scaled_comparison.csv', index=False)
print(f'\nSaved: {STAT_DIR}/effort_scaled_comparison.csv')
print('\nDone.')
