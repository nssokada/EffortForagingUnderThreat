"""
Unified Model Comparison — Choice Modeling Pipeline
====================================================
Fits 12+ models via SVI (NumPyro), comparing levels from simple effort-only
to mechanistic survival-function models.

Winning model: L3_add
  SV = R·S - k·E - β·(1-S)
  S  = (1-T) + T/(1+λD)
  k, β per-subject; λ, τ population-level

Also tests L4a_add (α from vigor enters survival):
  S  = (1-T) + T/(1+λ·D/α)

Usage:
  export PATH="$HOME/.local/bin:$PATH"
  python3 scripts/run_unified_model_comparison.py
"""

import sys
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

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path('/workspace')
DATA_DIR = ROOT / 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
STAT_DIR = ROOT / 'results/stats'
FIG_DIR  = ROOT / 'results/figs'
STAT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
print('Loading data...', flush=True)
behavior = pd.read_csv(DATA_DIR / 'behavior.csv')
vigor_params = pd.read_csv(STAT_DIR / 'vigor_hbm_posteriors.csv')

df = behavior.merge(vigor_params[['subj', 'alpha_bayes']], on='subj')

subj_ids = sorted(df['subj'].unique())
subj_to_idx = {s: i for i, s in enumerate(subj_ids)}
n_subj = len(subj_ids)

# JAX arrays
si    = jnp.array(df['subj'].map(subj_to_idx).values,  dtype=jnp.int32)
T     = jnp.array(df['threat'].values,                  dtype=jnp.float32)
eH    = jnp.array(df['effort_H'].values,                dtype=jnp.float32)
eL    = jnp.array(df['effort_L'].values,                dtype=jnp.float32)
dH    = jnp.array(df['distance_H'].values,              dtype=jnp.float32)
dL    = jnp.array(df['distance_L'].values,              dtype=jnp.float32)
ch    = jnp.array(df['choice'].values,                  dtype=jnp.float32)
alpha = jnp.array(df['alpha_bayes'].values,             dtype=jnp.float32)

R_H = 5.0; R_L = 1.0

print(f'N = {n_subj} subjects, {len(df)} trials', flush=True)
print(f'Effort L unique:   {sorted(df["effort_L"].unique())}', flush=True)
print(f'Distance L unique: {sorted(df["distance_L"].unique())}', flush=True)

# ─── Model definitions ────────────────────────────────────────────────────────
# All models share the same signature: (si, T, eH, eL, dH, dL, alpha, ch=None)

def level_0(si, T, eH, eL, dH, dL, alpha, ch=None):
    """Effort only: SV = R·exp(-k·E)"""
    tau = numpyro.sample('tau', dist.LogNormal(0, 1))
    mu_logk = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk = numpyro.sample('sd_logk', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
    k = jnp.exp(logk[si])
    SVH = R_H * jnp.exp(-k * eH)
    SVL = R_L * jnp.exp(-k * eL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def level_1(si, T, eH, eL, dH, dL, alpha, ch=None):
    """+ additive threat-distance penalty: SV = R·exp(-k·E) - β·T·D"""
    tau = numpyro.sample('tau', dist.LogNormal(0, 1))
    mu_logk = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk = numpyro.sample('sd_logk', dist.HalfNormal(1))
    mu_logb = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb = numpyro.sample('sd_logb', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
        logb = numpyro.sample('logb', dist.Normal(mu_logb, sd_logb))
    k = jnp.exp(logk[si]); beta = jnp.exp(logb[si])
    SVH = R_H * jnp.exp(-k * eH) - beta * T * dH
    SVL = R_L * jnp.exp(-k * eL) - beta * T * dL
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def level_2(si, T, eH, eL, dH, dL, alpha, ch=None):
    """+ T×D quadratic interaction: SV = R·exp(-k·E) - β·T·D - γ·T·D²"""
    tau = numpyro.sample('tau', dist.LogNormal(0, 1))
    mu_logk = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk = numpyro.sample('sd_logk', dist.HalfNormal(1))
    mu_logb = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb = numpyro.sample('sd_logb', dist.HalfNormal(1))
    gamma = numpyro.sample('gamma', dist.Normal(0, 1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
        logb = numpyro.sample('logb', dist.Normal(mu_logb, sd_logb))
    k = jnp.exp(logk[si]); beta = jnp.exp(logb[si])
    SVH = R_H * jnp.exp(-k * eH) - beta * T * dH - gamma * T * dH**2
    SVL = R_L * jnp.exp(-k * eL) - beta * T * dL - gamma * T * dL**2
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def level_3(si, T, eH, eL, dH, dL, alpha, ch=None):
    """Mechanistic survival (exponential kernel): S = (1-T) + T·exp(-λD)
    Multiplicative: SV = R·exp(-k·E)·S - β·(1-S)"""
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
    SH = (1 - T) + T * jnp.exp(-lam * dH)
    SL = (1 - T) + T * jnp.exp(-lam * dL)
    SVH = R_H * jnp.exp(-k * eH) * SH - beta * (1 - SH)
    SVL = R_L * jnp.exp(-k * eL) * SL - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def level_3b(si, T, eH, eL, dH, dL, alpha, ch=None):
    """Mechanistic survival with per-subject z: S = (1-T) + T·exp(-D^z_i)"""
    tau = numpyro.sample('tau', dist.LogNormal(0, 1))
    mu_logk = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk = numpyro.sample('sd_logk', dist.HalfNormal(1))
    mu_logz = numpyro.sample('mu_logz', dist.Normal(-1, 1))
    sd_logz = numpyro.sample('sd_logz', dist.HalfNormal(1))
    mu_logb = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb = numpyro.sample('sd_logb', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
        logz = numpyro.sample('logz', dist.Normal(mu_logz, sd_logz))
        logb = numpyro.sample('logb', dist.Normal(mu_logb, sd_logb))
    k = jnp.exp(logk[si]); z = jnp.exp(logz[si]); beta = jnp.exp(logb[si])
    SH = (1 - T) + T * jnp.exp(-jnp.power(dH, z))
    SL = (1 - T) + T * jnp.exp(-jnp.power(dL, z))
    SVH = R_H * jnp.exp(-k * eH) * SH - beta * (1 - SH)
    SVL = R_L * jnp.exp(-k * eL) * SL - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


# ── ADDITIVE EFFORT MODELS (the real winning family) ─────────────────────────

def L3_add(si, T, eH, eL, dH, dL, alpha, ch=None):
    """WINNING MODEL: hyperbolic survival, additive effort, no alpha.
    S = (1-T) + T/(1+λD)
    SV = R·S - k·E - β·(1-S)"""
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
    """Unified: α (from vigor) enters hyperbolic survival kernel.
    S = (1-T) + T/(1+λ·D/α)
    SV = R·S - k·E - β·(1-S)"""
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
    # α scales the effective distance in survival (normalised: D/α)
    SH = (1 - T) + T / (1.0 + lam * dH / alpha)
    SL = (1 - T) + T / (1.0 + lam * dL / alpha)
    SVH = R_H * SH - k * eH - beta * (1 - SH)
    SVL = R_L * SL - k * eL - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def L4a_add_norm(si, T, eH, eL, dH, dL, alpha, ch=None):
    """Unified: α enters survival, D normalised to [0,1] (D/3).
    S = (1-T) + T/(1+λ·(D/3)/α)
    SV = R·S - k·E - β·(1-S)"""
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
    dH_n = dH / 3.0; dL_n = dL / 3.0
    SH = (1 - T) + T / (1.0 + lam * dH_n / alpha)
    SL = (1 - T) + T / (1.0 + lam * dL_n / alpha)
    SVH = R_H * SH - k * eH - beta * (1 - SH)
    SVL = R_L * SL - k * eL - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def L4a_hyp(si, T, eH, eL, dH, dL, alpha, ch=None):
    """Unified: α in survival, multiplicative effort (original L4a style).
    S = (1-T) + T/(1+λ·D/α)
    SV = R·exp(-k·E)·S - β·(1-S)"""
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
    SVH = R_H * jnp.exp(-k * eH) * SH - beta * (1 - SH)
    SVL = R_L * jnp.exp(-k * eL) * SL - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def L4c_add(si, T, eH, eL, dH, dL, alpha, ch=None):
    """α moderates effort cost only (E/α): SV = R·S - k·(E/α) - β·(1-S)
    S = (1-T) + T/(1+λD) [no α in survival]"""
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
    SVH = R_H * SH - k * eH / alpha - beta * (1 - SH)
    SVL = R_L * SL - k * eL / alpha - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def L4d_add(si, T, eH, eL, dH, dL, alpha, ch=None):
    """α in BOTH survival and effort: SV = R·S(α) - k·(E/α) - β·(1-S(α))"""
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
    SVH = R_H * SH - k * eH / alpha - beta * (1 - SH)
    SVL = R_L * SL - k * eL / alpha - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


# ─── Model registry ───────────────────────────────────────────────────────────
models = {
    'L0_effort':      level_0,
    'L1_threat':      level_1,
    'L2_TxD':         level_2,
    'L3_survival':    level_3,
    'L3b_surv_zi':    level_3b,
    'L3_add':         L3_add,
    'L4a_add':        L4a_add,
    'L4a_add_norm':   L4a_add_norm,
    'L4a_hyp':        L4a_hyp,
    'L4c_add':        L4c_add,
    'L4d_add':        L4d_add,
}

# Parameter counts for BIC
n_per_subj = {
    'L0_effort':    1,
    'L1_threat':    2,
    'L2_TxD':       2,
    'L3_survival':  2,
    'L3b_surv_zi':  3,
    'L3_add':       2,
    'L4a_add':      2,
    'L4a_add_norm': 2,
    'L4a_hyp':      2,
    'L4c_add':      2,
    'L4d_add':      2,
}
n_pop = {
    'L0_effort':    3,
    'L1_threat':    5,
    'L2_TxD':       6,
    'L3_survival':  6,
    'L3b_surv_zi':  7,
    'L3_add':       6,
    'L4a_add':      6,
    'L4a_add_norm': 6,
    'L4a_hyp':      6,
    'L4c_add':      6,
    'L4d_add':      6,
}

print(f'Defined {len(models)} models', flush=True)

# ─── SVI fitting function ─────────────────────────────────────────────────────
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
        if i == 0:
            print(f'  {name}: compiled, step 0 loss={loss:.1f}', flush=True)
        if (i + 1) % 3000 == 0:
            running_elbo = -np.mean(losses[-500:])
            print(f'  {name}: step {i+1}/{n_steps}, running ELBO={running_elbo:.1f}', flush=True)

    elbo = -np.mean(losses[-200:])
    print(f'  {name}: FINAL ELBO = {elbo:.1f}', flush=True)
    return {
        'elbo':   elbo,
        'losses': losses,
        'params': svi.get_params(state),
        'guide':  guide,
        'svi':    svi,
    }


# ─── Fit all models ───────────────────────────────────────────────────────────
results = {}
for name, model_fn in models.items():
    print(f'\nFitting {name}...', flush=True)
    results[name] = fit_svi(model_fn, name)

# ─── Model comparison table ───────────────────────────────────────────────────
n_total = len(df)
rows = []
for name in models:
    n_sp  = n_per_subj[name] * n_subj + n_pop[name]
    elbo  = results[name]['elbo']
    bic   = -2 * elbo + n_sp * np.log(n_total)
    rows.append({
        'model':          name,
        'ELBO':           elbo,
        'per_subj_params': n_per_subj[name],
        'total_params':   n_sp,
        'BIC':            bic,
    })

comp = pd.DataFrame(rows).sort_values('ELBO', ascending=False)
comp['dELBO'] = comp['ELBO'] - comp['ELBO'].max()
comp['dBIC']  = comp['BIC'] - comp['BIC'].min()

print('\n' + '='*90, flush=True)
print('MODEL COMPARISON', flush=True)
print('='*90, flush=True)
print(comp[['model', 'ELBO', 'dELBO', 'per_subj_params', 'total_params', 'BIC', 'dBIC']].to_string(index=False), flush=True)

best_elbo = comp.iloc[0]['model']
best_bic  = comp.loc[comp['BIC'].idxmin(), 'model']
print(f'\nBest by ELBO: {best_elbo}', flush=True)
print(f'Best by BIC:  {best_bic}', flush=True)

# Key comparisons
def get_elbo(name):
    return comp[comp['model'] == name]['ELBO'].values[0]

print(f'\n--- Key comparisons (ΔELBO) ---', flush=True)
pairs = [
    ('L0_effort',   'L1_threat',   'L0→L1 (add threat)'),
    ('L1_threat',   'L2_TxD',      'L1→L2 (T×D)'),
    ('L2_TxD',      'L3_survival', 'L2→L3 (exponential survival)'),
    ('L3_survival', 'L3_add',      'L3_survival→L3_add (additive effort)'),
    ('L3_survival', 'L3b_surv_zi', 'L3→L3b (per-subj z)'),
    ('L3_add',      'L4a_add',     'L3_add→L4a_add (α in S)'),
    ('L3_add',      'L4c_add',     'L3_add→L4c_add (α in E)'),
    ('L3_add',      'L4d_add',     'L3_add→L4d_add (α in E+S)'),
    ('L4a_hyp',     'L4a_add',     'L4a_hyp(mult)→L4a_add (additive)'),
]
for a, b, label in pairs:
    if a in comp['model'].values and b in comp['model'].values:
        delta = get_elbo(b) - get_elbo(a)
        print(f'  {label}: {delta:+.1f}', flush=True)

# ─── Extract parameters from winning model (L3_add) ──────────────────────────
# Always use L3_add as the primary model for downstream analyses
best_model = 'L3_add'
print(f'\n=== Extracting parameters from {best_model} ===', flush=True)

r = results[best_model]
predictive = Predictive(r['guide'], params=r['params'], num_samples=500)
samples = predictive(random.PRNGKey(99), **data)

# Population parameters
print(f'\nPopulation parameters:', flush=True)
for p in ['tau', 'lam', 'mu_logk', 'sd_logk', 'mu_logb', 'sd_logb']:
    if p in samples:
        v = np.array(samples[p])
        print(f'  {p:12s}: {v.mean():.4f} ± {v.std():.4f}', flush=True)

# Per-subject parameters
logk = np.array(samples['logk'])
logb = np.array(samples['logb'])
k_est    = np.exp(logk.mean(axis=0))
beta_est = np.exp(logb.mean(axis=0))

# α from vigor (not from choice model)
alpha_est = vigor_params.set_index('subj').loc[subj_ids, 'alpha_bayes'].values

print(f'\nPer-subject parameter summary:', flush=True)
for arr, lab in [(k_est, 'k'), (beta_est, 'β'), (alpha_est, 'α')]:
    print(f'  {lab}: mean={arr.mean():.4f}, sd={arr.std():.4f}, '
          f'range=[{arr.min():.4f}, {arr.max():.4f}]', flush=True)

print(f'\nParameter correlations:', flush=True)
for (a, la), (b, lb) in [
    ((k_est, 'k'),    (alpha_est, 'α')),
    ((k_est, 'k'),    (beta_est,  'β')),
    ((beta_est, 'β'), (alpha_est, 'α')),
]:
    r_val, p_val = stats.pearsonr(a, b)
    sig = ' *' if p_val < 0.05 else ''
    print(f'  {la}-{lb}: r={r_val:+.4f}, p={p_val:.4f}{sig}', flush=True)

# ─── Save outputs ─────────────────────────────────────────────────────────────
# 1. Model comparison table
comp.to_csv(STAT_DIR / 'unified_model_comparison.csv', index=False)
print(f'\nSaved: {STAT_DIR}/unified_model_comparison.csv', flush=True)

# 2. Subject-level parameters from L3_add (+ alpha from vigor)
param_df = pd.DataFrame({
    'subj':  subj_ids,
    'k':     k_est,
    'beta':  beta_est,
    'alpha': alpha_est,
})
param_df.to_csv(STAT_DIR / 'unified_3param_clean.csv', index=False)
print(f'Saved: {STAT_DIR}/unified_3param_clean.csv', flush=True)

print('\nDone.', flush=True)
