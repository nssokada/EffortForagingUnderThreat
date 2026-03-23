"""
01_choice_model.py — Choice model fitting and comparison
=========================================================
Fits L3_add (winning model) via SVI and compares against key alternatives.

Model L3_add:
    S  = (1-T) + T/(1+λD)
    SV = R·S - k·E - β·(1-S)
    choice ~ Bernoulli(sigmoid(τ · (SV_H - SV_L)))

k and β are log-normal per-subject (non-centered).
λ and τ are population-level.

Outputs:
    results/stats/paper/choice_model_comparison.csv
    results/stats/paper/choice_params.csv

Usage:
    export PATH="$HOME/.local/bin:$PATH"
    python3 notebooks/06_paper_pipeline/01_choice_model.py
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

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path('/workspace')
DATA_DIR = ROOT / 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
OUT_DIR  = ROOT / 'results/stats/paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_H = 5.0
R_L = 1.0

# ── Load data ──────────────────────────────────────────────────────────────────
print('=' * 70)
print('STEP 1: Loading data')
print('=' * 70)

behavior = pd.read_csv(DATA_DIR / 'behavior.csv')
print(f'Loaded behavior.csv: {len(behavior)} trials, {behavior["subj"].nunique()} subjects')

# Subject index mapping
subj_ids    = sorted(behavior['subj'].unique())
subj_to_idx = {s: i for i, s in enumerate(subj_ids)}
n_subj      = len(subj_ids)

# JAX arrays
si  = jnp.array(behavior['subj'].map(subj_to_idx).values, dtype=jnp.int32)
T   = jnp.array(behavior['threat'].values,                 dtype=jnp.float32)
eH  = jnp.array(behavior['effort_H'].values,              dtype=jnp.float32)
eL  = jnp.array(behavior['effort_L'].values,              dtype=jnp.float32)
dH  = jnp.array(behavior['distance_H'].values,            dtype=jnp.float32)
dL  = jnp.array(behavior['distance_L'].values,            dtype=jnp.float32)
ch  = jnp.array(behavior['choice'].values,                dtype=jnp.float32)

n_trials = len(behavior)
print(f'N subjects = {n_subj}, N trials = {n_trials}')
print(f'effort_H range:   [{float(eH.min()):.2f}, {float(eH.max()):.2f}]')
print(f'distance_H range: [{float(dH.min()):.1f}, {float(dH.max()):.1f}]')
print(f'threat values:    {sorted(behavior["threat"].unique())}')
print(f'choice mean:      {float(ch.mean()):.3f} (prop. chose high-effort)')

# ── Model definitions ──────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 2: Defining models')
print('=' * 70)

data = dict(si=si, T=T, eH=eH, eL=eL, dH=dH, dL=dL, ch=ch)


def L3_add(si, T, eH, eL, dH, dL, ch=None):
    """WINNING MODEL: additive effort, hyperbolic survival, no alpha.
    S  = (1-T) + T/(1+λD)
    SV = R·S - k·E - β·(1-S)
    """
    tau      = numpyro.sample('tau',     dist.LogNormal(0, 1))
    lam      = numpyro.sample('lam',     dist.LogNormal(0, 1))
    mu_logk  = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk  = numpyro.sample('sd_logk', dist.HalfNormal(1))
    mu_logb  = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb  = numpyro.sample('sd_logb', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
        logb = numpyro.sample('logb', dist.Normal(mu_logb, sd_logb))
    k    = jnp.exp(logk[si])
    beta = jnp.exp(logb[si])
    SH   = (1 - T) + T / (1.0 + lam * dH)
    SL   = (1 - T) + T / (1.0 + lam * dL)
    SVH  = R_H * SH - k * eH - beta * (1 - SH)
    SVL  = R_L * SL - k * eL - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def L3_mult(si, T, eH, eL, dH, dL, ch=None):
    """Multiplicative effort: SV = R·exp(-k·E)·S - β·(1-S)
    S = (1-T) + T/(1+λD)  [hyperbolic survival, multiplicative effort]
    """
    tau      = numpyro.sample('tau',     dist.LogNormal(0, 1))
    lam      = numpyro.sample('lam',     dist.LogNormal(0, 1))
    mu_logk  = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk  = numpyro.sample('sd_logk', dist.HalfNormal(1))
    mu_logb  = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb  = numpyro.sample('sd_logb', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
        logb = numpyro.sample('logb', dist.Normal(mu_logb, sd_logb))
    k    = jnp.exp(logk[si])
    beta = jnp.exp(logb[si])
    SH   = (1 - T) + T / (1.0 + lam * dH)
    SL   = (1 - T) + T / (1.0 + lam * dL)
    SVH  = R_H * jnp.exp(-k * eH) * SH - beta * (1 - SH)
    SVL  = R_L * jnp.exp(-k * eL) * SL - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def L3_exp_survival(si, T, eH, eL, dH, dL, ch=None):
    """Exponential survival kernel: S = (1-T) + T·exp(-λD)
    Additive effort: SV = R·S - k·E - β·(1-S)
    """
    tau      = numpyro.sample('tau',     dist.LogNormal(0, 1))
    lam      = numpyro.sample('lam',     dist.LogNormal(0, 1))
    mu_logk  = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk  = numpyro.sample('sd_logk', dist.HalfNormal(1))
    mu_logb  = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb  = numpyro.sample('sd_logb', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
        logb = numpyro.sample('logb', dist.Normal(mu_logb, sd_logb))
    k    = jnp.exp(logk[si])
    beta = jnp.exp(logb[si])
    SH   = (1 - T) + T * jnp.exp(-lam * dH)
    SL   = (1 - T) + T * jnp.exp(-lam * dL)
    SVH  = R_H * SH - k * eH - beta * (1 - SH)
    SVL  = R_L * SL - k * eL - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def L0_effort_only(si, T, eH, eL, dH, dL, ch=None):
    """Effort only: SV = R·exp(-k·E)  (no threat)"""
    tau      = numpyro.sample('tau',     dist.LogNormal(0, 1))
    mu_logk  = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk  = numpyro.sample('sd_logk', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logk = numpyro.sample('logk', dist.Normal(mu_logk, sd_logk))
    k   = jnp.exp(logk[si])
    SVH = R_H * jnp.exp(-k * eH)
    SVL = R_L * jnp.exp(-k * eL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


def L0_threat_only(si, T, eH, eL, dH, dL, ch=None):
    """Threat only: SV = R·S,  S = (1-T) + T/(1+λD)  (no effort cost)"""
    tau      = numpyro.sample('tau', dist.LogNormal(0, 1))
    lam      = numpyro.sample('lam', dist.LogNormal(0, 1))
    mu_logb  = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb  = numpyro.sample('sd_logb', dist.HalfNormal(1))
    with numpyro.plate('subj', n_subj):
        logb = numpyro.sample('logb', dist.Normal(mu_logb, sd_logb))
    beta = jnp.exp(logb[si])
    SH   = (1 - T) + T / (1.0 + lam * dH)
    SL   = (1 - T) + T / (1.0 + lam * dL)
    SVH  = R_H * SH - beta * (1 - SH)
    SVL  = R_L * SL - beta * (1 - SL)
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)


models = {
    'L3_add':          (L3_add,          2, 6),  # (fn, per_subj_params, pop_params)
    'L3_mult':         (L3_mult,         2, 6),
    'L3_exp_survival': (L3_exp_survival, 2, 6),
    'L0_effort_only':  (L0_effort_only,  1, 3),
    'L0_threat_only':  (L0_threat_only,  1, 5),
}

print(f'Defined {len(models)} models: {list(models.keys())}')

# ── SVI fitting function ───────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 3: Fitting models via SVI (AutoNormal guide)')
print('=' * 70)


def fit_svi(model_fn, name, n_steps=15000, lr=0.003, seed=42):
    """Fit model via SVI with AutoNormal guide. Returns ELBO and fitted state."""
    guide    = AutoNormal(model_fn)
    svi      = SVI(model_fn, guide, Adam(lr), loss=Trace_ELBO())
    state    = svi.init(random.PRNGKey(seed), **data)
    update   = jax.jit(svi.update)
    losses   = []
    for i in range(n_steps):
        state, loss = update(state, **data)
        losses.append(float(loss))
        if i == 0:
            print(f'  {name}: compiled. step 0 loss={loss:.1f}', flush=True)
        if (i + 1) % 5000 == 0:
            elbo_run = -np.mean(losses[-500:])
            print(f'  {name}: step {i+1}/{n_steps}, running ELBO={elbo_run:.1f}', flush=True)

    elbo = -np.mean(losses[-200:])
    print(f'  {name}: FINAL ELBO = {elbo:.1f}', flush=True)
    return {
        'elbo':   elbo,
        'losses': losses,
        'params': svi.get_params(state),
        'guide':  guide,
        'svi':    svi,
        'state':  state,
    }


fit_results = {}
for name, (fn, _, _) in models.items():
    print(f'\nFitting {name}...')
    fit_results[name] = fit_svi(fn, name)

# ── Model comparison table ─────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 4: Model comparison')
print('=' * 70)

rows = []
for name, (fn, n_sp, n_pp) in models.items():
    elbo       = fit_results[name]['elbo']
    total_pars = n_sp * n_subj + n_pp
    bic        = -2 * elbo + total_pars * np.log(n_trials)
    rows.append({
        'model':         name,
        'ELBO':          elbo,
        'n_subj_params': n_sp,
        'n_pop_params':  n_pp,
        'total_params':  total_pars,
        'BIC':           bic,
    })

comp = pd.DataFrame(rows).sort_values('ELBO', ascending=False).reset_index(drop=True)
comp['dELBO'] = comp['ELBO'] - comp['ELBO'].max()
comp['dBIC']  = comp['BIC']  - comp['BIC'].min()

print('\nModel comparison (sorted by ELBO):')
print(comp[['model', 'ELBO', 'dELBO', 'total_params', 'BIC', 'dBIC']].to_string(index=False))

best_by_elbo = comp.iloc[0]['model']
best_by_bic  = comp.loc[comp['BIC'].idxmin(), 'model']
print(f'\nBest by ELBO: {best_by_elbo}')
print(f'Best by BIC:  {best_by_bic}')

# Key pairwise comparisons
def get_elbo(name):
    return comp.loc[comp['model'] == name, 'ELBO'].values[0]

print('\nKey pairwise ΔELBO (positive = second model better):')
pairs = [
    ('L3_add', 'L3_mult',         'additive vs multiplicative effort'),
    ('L3_add', 'L3_exp_survival', 'hyperbolic vs exponential survival'),
    ('L0_effort_only', 'L3_add',  'effort-only vs full L3_add'),
    ('L0_threat_only', 'L3_add',  'threat-only vs full L3_add'),
]
for a, b, label in pairs:
    if a in comp['model'].values and b in comp['model'].values:
        delta = get_elbo(b) - get_elbo(a)
        print(f'  {label}: {delta:+.1f}')

# ── Extract per-subject parameters from L3_add ────────────────────────────────
print('\n' + '=' * 70)
print('STEP 5: Extract per-subject parameters from L3_add')
print('=' * 70)

r        = fit_results['L3_add']
pred     = Predictive(r['guide'], params=r['params'], num_samples=500)
samples  = pred(random.PRNGKey(99), **data)

# Population parameters
print('\nPopulation parameters (L3_add):')
for p in ['tau', 'lam', 'mu_logk', 'sd_logk', 'mu_logb', 'sd_logb']:
    if p in samples:
        v = np.array(samples[p])
        print(f'  {p:12s}: mean={v.mean():.4f}, SD={v.std():.4f}')

# Per-subject posterior means
logk_arr  = np.array(samples['logk'])   # (n_samples, n_subj)
logb_arr  = np.array(samples['logb'])   # (n_samples, n_subj)
k_subj    = np.exp(logk_arr.mean(axis=0))
beta_subj = np.exp(logb_arr.mean(axis=0))
k_sd      = (logk_arr.std(axis=0) * np.exp(logk_arr.mean(axis=0)))  # delta-method approx
beta_sd   = (logb_arr.std(axis=0) * np.exp(logb_arr.mean(axis=0)))

print('\nPer-subject parameter distributions (L3_add):')
print(f'  k    : mean={k_subj.mean():.3f}, SD={k_subj.std():.3f}, '
      f'range=[{k_subj.min():.3f}, {k_subj.max():.3f}]')
print(f'  β    : mean={beta_subj.mean():.3f}, SD={beta_subj.std():.3f}, '
      f'range=[{beta_subj.min():.3f}, {beta_subj.max():.3f}]')

# k-β correlation
r_val, p_val = stats.pearsonr(k_subj, beta_subj)
print(f'\nk-β correlation: r={r_val:+.4f}, p={p_val:.4f}'
      + (' *' if p_val < 0.05 else ' n.s.'))

# ── Model-free accuracy (PPC) ──────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 6: Posterior predictive check (L3_add accuracy)')
print('=' * 70)

# Sample predictions
ppc_pred = Predictive(r['guide'], params=r['params'], num_samples=200)
ppc      = ppc_pred(random.PRNGKey(77), **{k: v for k, v in data.items() if k != 'ch'})

# logit of choice: compute manually using posterior samples
logk_ppc   = np.array(ppc['logk'])   # (200, n_subj)
logb_ppc   = np.array(ppc['logb'])
tau_ppc    = np.array(ppc['tau'])    # (200,)
lam_ppc    = np.array(ppc['lam'])    # (200,)

si_np  = np.array(si)
T_np   = np.array(T)
eH_np  = np.array(eH)
eL_np  = np.array(eL)
dH_np  = np.array(dH)
dL_np  = np.array(dL)
ch_np  = np.array(ch)

p_high_samples = []
for s_idx in range(200):
    k_s    = np.exp(logk_ppc[s_idx])[si_np]
    b_s    = np.exp(logb_ppc[s_idx])[si_np]
    tau_s  = tau_ppc[s_idx]
    lam_s  = lam_ppc[s_idx]
    SH_s   = (1 - T_np) + T_np / (1 + lam_s * dH_np)
    SL_s   = (1 - T_np) + T_np / (1 + lam_s * dL_np)
    SVH_s  = R_H * SH_s - k_s * eH_np - b_s * (1 - SH_s)
    SVL_s  = R_L * SL_s - k_s * eL_np - b_s * (1 - SL_s)
    p_s    = 1 / (1 + np.exp(-tau_s * (SVH_s - SVL_s)))
    p_high_samples.append(p_s)

p_high_mean  = np.mean(p_high_samples, axis=0)
pred_choice  = (p_high_mean > 0.5).astype(float)
accuracy     = np.mean(pred_choice == ch_np)
print(f'L3_add predictive accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)')

# ── Save outputs ───────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 7: Saving outputs')
print('=' * 70)

# 1. Model comparison
comp.to_csv(OUT_DIR / 'choice_model_comparison.csv', index=False)
print(f'Saved: {OUT_DIR}/choice_model_comparison.csv')

# 2. Per-subject parameters
param_df = pd.DataFrame({
    'subj':    subj_ids,
    'k':       k_subj,
    'k_sd':    k_sd,
    'beta':    beta_subj,
    'beta_sd': beta_sd,
})
param_df.to_csv(OUT_DIR / 'choice_params.csv', index=False)
print(f'Saved: {OUT_DIR}/choice_params.csv')

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'Winning model by ELBO: {best_by_elbo}')
print(f'Winning model by BIC:  {best_by_bic}')
print(f'L3_add predictive accuracy: {accuracy*100:.1f}%')
print(f'k:    mean={k_subj.mean():.3f} ± {k_subj.std():.3f}')
print(f'β:    mean={beta_subj.mean():.3f} ± {beta_subj.std():.3f}')
print(f'k-β:  r={r_val:+.4f}, p={p_val:.4f}')
print('\nDone.')
