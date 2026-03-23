"""
05_joint_model.py — Joint SVI model (choice + vigor)
=====================================================
Fits a joint model where choice and vigor share the same survival signal S.

The joint model:
  Shared survival: S = (1-T) + T/(1+λD)
  Choice:          p(high) = sigmoid(τ · (SV_H - SV_L))
                   SV = R·S - k·E - β·(1-S)
  Vigor (additive): vigor_norm ~ Normal(μ_v + δ·(1-S), σ_v)
                    where μ_v = α_v + b_demand·E

This tests whether k and β from joint estimation match separate estimates
and whether joint fitting improves parameter identifiability.

Outputs:
    results/stats/paper/joint_model_params.csv

Usage:
    export PATH="$HOME/.local/bin:$PATH"
    python3 notebooks/06_paper_pipeline/05_joint_model.py
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

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path('/workspace')
DATA_DIR  = ROOT / 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
VIGOR_DIR = ROOT / 'data/exploratory_350/processed/vigor_processed'
STAT_DIR  = ROOT / 'results/stats'
OUT_DIR   = ROOT / 'results/stats/paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_H = 5.0
R_L = 1.0

# ── Load data ──────────────────────────────────────────────────────────────────
print('=' * 70)
print('STEP 1: Loading data')
print('=' * 70)

behavior = pd.read_csv(DATA_DIR / 'behavior.csv')
print(f'behavior.csv: {len(behavior)} trials, {behavior["subj"].nunique()} subjects')

print('Loading smoothed_vigor_ts.parquet...')
vigor_ts = pd.read_parquet(VIGOR_DIR / 'smoothed_vigor_ts.parquet')

# Compute trial-level mean vigor
trial_vigor = (
    vigor_ts
    .groupby(['subj', 'trial'])['vigor_norm']
    .mean()
    .reset_index()
    .rename(columns={'vigor_norm': 'mean_vigor'})
)
print(f'Trial-level vigor computed: {len(trial_vigor)} rows')

# Merge: behavior trial is 1-indexed; vigor trial is 0-indexed global event index
behavior_v = behavior.copy()
behavior_v['trial_0'] = behavior_v['trial'] - 1

merged = behavior_v.merge(
    trial_vigor.rename(columns={'trial': 'trial_0'}),
    on=['subj', 'trial_0'],
    how='inner'
)
merged = merged.dropna(subset=['mean_vigor']).reset_index(drop=True)
print(f'Merged and cleaned: {len(merged)} trials, {merged["subj"].nunique()} subjects')

# ── Subject index ──────────────────────────────────────────────────────────────
subj_ids    = sorted(merged['subj'].unique())
subj_to_idx = {s: i for i, s in enumerate(subj_ids)}
n_subj      = len(subj_ids)

# JAX arrays
si     = jnp.array(merged['subj'].map(subj_to_idx).values, dtype=jnp.int32)
T      = jnp.array(merged['threat'].values,                dtype=jnp.float32)
eH     = jnp.array(merged['effort_H'].values,             dtype=jnp.float32)
eL     = jnp.array(merged['effort_L'].values,             dtype=jnp.float32)
dH     = jnp.array(merged['distance_H'].values,           dtype=jnp.float32)
dL     = jnp.array(merged['distance_L'].values,           dtype=jnp.float32)
ch     = jnp.array(merged['choice'].values,               dtype=jnp.float32)
vigor  = jnp.array(merged['mean_vigor'].values,           dtype=jnp.float32)
# Chosen effort and distance (for vigor model)
e_chosen = jnp.where(ch == 1, eH, eL)

n_trials = len(merged)
print(f'\nN subjects = {n_subj}, N trials = {n_trials}')

# ── Joint model definition ─────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 2: Define joint choice + vigor model')
print('=' * 70)


def joint_model(si, T, eH, eL, dH, dL, e_chosen, ch=None, vigor=None):
    """Joint model: shared S drives both choice and vigor.

    Choice:
        S_H = (1-T) + T/(1+λ·dH)
        SV_H = R_H·S_H - k_i·eH - β_i·(1-S_H)
        p(H) = sigmoid(τ · (SV_H - SV_L))

    Vigor:
        S_chosen = (1-T) + T/(1+λ·d_chosen)  [same λ]
        danger = 1 - S_chosen
        mean_vigor ~ Normal(μ_v_i + δ_i·danger + b_e·e_chosen, σ_v)
    """
    # Population-level choice params
    tau     = numpyro.sample('tau',     dist.LogNormal(0, 1))
    lam     = numpyro.sample('lam',     dist.LogNormal(0, 1))
    mu_logk = numpyro.sample('mu_logk', dist.Normal(0, 1))
    sd_logk = numpyro.sample('sd_logk', dist.HalfNormal(1))
    mu_logb = numpyro.sample('mu_logb', dist.Normal(0, 1))
    sd_logb = numpyro.sample('sd_logb', dist.HalfNormal(1))

    # Population-level vigor params
    mu_logalpha = numpyro.sample('mu_logalpha', dist.Normal(0, 1))
    sd_alpha    = numpyro.sample('sd_alpha',    dist.HalfNormal(0.5))
    mu_delta    = numpyro.sample('mu_delta',    dist.Normal(0, 0.5))
    sd_delta    = numpyro.sample('sd_delta',    dist.HalfNormal(0.3))
    b_effort    = numpyro.sample('b_effort',    dist.Normal(0, 0.5))
    sigma_v     = numpyro.sample('sigma_v',     dist.HalfNormal(0.5))

    with numpyro.plate('subj', n_subj):
        # Choice parameters (non-centered)
        logk_raw = numpyro.sample('logk_raw', dist.Normal(0, 1))
        logb_raw = numpyro.sample('logb_raw', dist.Normal(0, 1))
        logk     = mu_logk + sd_logk * logk_raw
        logb     = mu_logb + sd_logb * logb_raw

        # Vigor parameters (non-centered)
        alpha_raw = numpyro.sample('alpha_raw', dist.Normal(0, 1))
        delta_raw = numpyro.sample('delta_raw', dist.Normal(0, 1))
        alpha_v   = numpyro.deterministic('alpha_v',
                        jnp.exp(mu_logalpha + sd_alpha * alpha_raw))
        delta_i   = numpyro.deterministic('delta_i',
                        mu_delta + sd_delta * delta_raw)

    k_i     = jnp.exp(logk[si])
    beta_i  = jnp.exp(logb[si])
    alpha_i = alpha_v[si]
    delta_s = delta_i[si]

    # Survival for choice options
    SH = (1 - T) + T / (1.0 + lam * dH)
    SL = (1 - T) + T / (1.0 + lam * dL)

    # Subjective values
    SVH = R_H * SH - k_i * eH - beta_i * (1 - SH)
    SVL = R_L * SL - k_i * eL - beta_i * (1 - SL)

    # Choice likelihood
    numpyro.sample('ch', dist.Bernoulli(logits=tau * (SVH - SVL)), obs=ch)

    # Survival for chosen option (same λ)
    # d_chosen approximated from e_chosen (effort ≈ distance_H scaled; use S from chosen effort proxy)
    # Use S from choice model: for chosen option, use S based on e_chosen as proxy for D
    # Since effort_H ∈ {0.5, 0.8, 1.0} and distance_H ∈ {1,2,3}, we directly recompute S
    # from e_chosen by mapping back: distance proxy = e_chosen / effort_scale
    # Simpler: use trial-level danger from SH when choice=1, SL when choice=0
    S_chosen = jnp.where(ch == 1, SH, SL)
    danger_i = 1 - S_chosen

    # Vigor likelihood
    mu_vigor = alpha_i + delta_s * danger_i + b_effort * e_chosen
    numpyro.sample('vigor', dist.Normal(mu_vigor, sigma_v), obs=vigor)


print('Joint model defined.')
print('Parameters: τ, λ (pop choice), μ_k, σ_k, μ_β, σ_β,')
print('            μ_α_v, σ_α_v, μ_δ, σ_δ, b_effort, σ_v (pop vigor)')
print('            k_i, β_i, α_v_i, δ_i (per-subject)')

# ── Fit joint model ───────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 3: Fit joint model via SVI (AutoNormal guide)')
print('=' * 70)

data_joint = dict(
    si=si, T=T, eH=eH, eL=eL, dH=dH, dL=dL,
    e_chosen=e_chosen, ch=ch, vigor=vigor
)

guide_joint = AutoNormal(joint_model)
svi_joint   = SVI(joint_model, guide_joint, Adam(0.003), loss=Trace_ELBO())
state_joint = svi_joint.init(random.PRNGKey(42), **data_joint)
update_fn   = jax.jit(svi_joint.update)

N_STEPS = 20000
losses  = []
for i in range(N_STEPS):
    state_joint, loss = update_fn(state_joint, **data_joint)
    losses.append(float(loss))
    if i == 0:
        print(f'  Joint model: compiled. step 0 loss={loss:.1f}', flush=True)
    if (i + 1) % 5000 == 0:
        elbo_run = -np.mean(losses[-500:])
        print(f'  step {i+1}/{N_STEPS}, running ELBO={elbo_run:.1f}', flush=True)

elbo_joint = -np.mean(losses[-200:])
print(f'\nJoint model FINAL ELBO = {elbo_joint:.1f}')

# ── Extract parameters ─────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 4: Extract population and per-subject parameters')
print('=' * 70)

params_joint = svi_joint.get_params(state_joint)
pred         = Predictive(guide_joint, params=params_joint, num_samples=500)
samples      = pred(random.PRNGKey(99), **{k: v for k, v in data_joint.items()
                                           if k not in ('ch', 'vigor')})

# Population parameters
print('\nPopulation parameters (joint model):')
pop_params = ['tau', 'lam', 'mu_logk', 'sd_logk', 'mu_logb', 'sd_logb',
              'mu_logalpha', 'sd_alpha', 'mu_delta', 'sd_delta', 'b_effort', 'sigma_v']
pop_summary = {}
for p in pop_params:
    if p in samples:
        v = np.array(samples[p])
        pop_summary[p] = {'mean': float(v.mean()), 'sd': float(v.std())}
        print(f'  {p:15s}: {v.mean():.4f} ± {v.std():.4f}')

# Per-subject parameters
logk_arr    = np.array(samples.get('logk_raw', np.zeros((500, n_subj))))
logb_arr    = np.array(samples.get('logb_raw', np.zeros((500, n_subj))))
alpha_v_arr = np.array(samples.get('alpha_v', np.zeros((500, n_subj))))
delta_arr   = np.array(samples.get('delta_i', np.zeros((500, n_subj))))

mu_logk_s = np.array(samples['mu_logk']).mean() if 'mu_logk' in samples else 0.0
sd_logk_s = np.array(samples['sd_logk']).mean() if 'sd_logk' in samples else 1.0
mu_logb_s = np.array(samples['mu_logb']).mean() if 'mu_logb' in samples else 0.0
sd_logb_s = np.array(samples['sd_logb']).mean() if 'sd_logb' in samples else 1.0

k_joint    = np.exp(mu_logk_s + sd_logk_s * logk_arr.mean(axis=0))
beta_joint = np.exp(mu_logb_s + sd_logb_s * logb_arr.mean(axis=0))
alpha_v_joint = alpha_v_arr.mean(axis=0)
delta_joint   = delta_arr.mean(axis=0)

print('\nPer-subject parameter distributions (joint model):')
print(f'  k:       mean={k_joint.mean():.3f}, SD={k_joint.std():.3f}')
print(f'  β:       mean={beta_joint.mean():.3f}, SD={beta_joint.std():.3f}')
print(f'  α_v:     mean={alpha_v_joint.mean():.4f}, SD={alpha_v_joint.std():.4f}')
print(f'  δ:       mean={delta_joint.mean():.4f}, SD={delta_joint.std():.4f}')

# ── Compare with separate estimates ───────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 5: Compare joint vs. separate estimates')
print('=' * 70)

# Load separate estimates
params_sep_path = OUT_DIR / 'choice_params.csv'
vigor_sep_path  = OUT_DIR / 'vigor_params.csv'

if params_sep_path.exists() and vigor_sep_path.exists():
    params_sep = pd.read_csv(params_sep_path)
    vigor_sep  = pd.read_csv(vigor_sep_path)

    # Align subjects
    sep_subj   = sorted(params_sep['subj'].unique())
    joint_subj = subj_ids

    # Subjects in both
    common = sorted(set(sep_subj) & set(joint_subj) & set(vigor_sep['subj'].unique()))
    print(f'Common subjects for comparison: {len(common)}')

    sep_idx   = [sep_subj.index(s) if s in sep_subj else -1 for s in common]
    joint_idx = [joint_subj.index(s) for s in common]

    k_sep_arr    = params_sep.set_index('subj').loc[common, 'k'].values
    beta_sep_arr = params_sep.set_index('subj').loc[common, 'beta'].values

    k_j_arr    = k_joint[[joint_subj.index(s) for s in common]]
    beta_j_arr = beta_joint[[joint_subj.index(s) for s in common]]

    r_k,    p_k    = stats.pearsonr(np.log(k_sep_arr    + 1e-6), np.log(k_j_arr    + 1e-6))
    r_beta, p_beta = stats.pearsonr(np.log(beta_sep_arr + 1e-6), np.log(beta_j_arr + 1e-6))

    print(f'k   (separate vs joint): r={r_k:.4f}, p={p_k:.4f}')
    print(f'β   (separate vs joint): r={r_beta:.4f}, p={p_beta:.4f}')

    # Vigor comparison
    delta_sep_arr   = vigor_sep.set_index('subj').loc[common, 'delta'].values
    delta_j_arr     = delta_joint[[joint_subj.index(s) for s in common]]
    alpha_sep_arr   = vigor_sep.set_index('subj').loc[common, 'alpha_v'].values
    alpha_j_arr     = alpha_v_joint[[joint_subj.index(s) for s in common]]

    r_delta, p_delta = stats.pearsonr(delta_sep_arr, delta_j_arr)
    r_alpha, p_alpha = stats.pearsonr(alpha_sep_arr, alpha_j_arr)
    print(f'δ   (separate vs joint): r={r_delta:.4f}, p={p_delta:.4f}')
    print(f'α_v (separate vs joint): r={r_alpha:.4f}, p={p_alpha:.4f}')
else:
    print('Separate estimate files not found — skipping comparison.')
    print(f'  Missing: {params_sep_path if not params_sep_path.exists() else vigor_sep_path}')
    common = []

# ── Save outputs ───────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 6: Saving outputs')
print('=' * 70)

joint_df = pd.DataFrame({
    'subj':    subj_ids,
    'k_joint':      k_joint,
    'beta_joint':   beta_joint,
    'alpha_v_joint': alpha_v_joint,
    'delta_joint':  delta_joint,
})
joint_df.to_csv(OUT_DIR / 'joint_model_params.csv', index=False)
print(f'Saved: {OUT_DIR}/joint_model_params.csv')

# Population summary
pop_df = pd.DataFrame([
    {'param': k, 'mean': v['mean'], 'sd': v['sd']}
    for k, v in pop_summary.items()
])
pop_df.to_csv(OUT_DIR / 'joint_model_population.csv', index=False)
print(f'Saved: {OUT_DIR}/joint_model_population.csv')

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'Joint model ELBO: {elbo_joint:.1f}')
print(f'N subjects: {n_subj}, N trials: {n_trials}')
print(f'k (joint):     mean={k_joint.mean():.3f} ± {k_joint.std():.3f}')
print(f'β (joint):     mean={beta_joint.mean():.3f} ± {beta_joint.std():.3f}')
print(f'α_v (joint):   mean={alpha_v_joint.mean():.4f} ± {alpha_v_joint.std():.4f}')
print(f'δ (joint):     mean={delta_joint.mean():.4f} ± {delta_joint.std():.4f}')
if common:
    print(f'Convergence with separate: k r={r_k:.3f}, β r={r_beta:.3f}, '
          f'δ r={r_delta:.3f}, α_v r={r_alpha:.3f}')
print('\nDone.')
