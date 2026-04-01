"""
5-Model Comparison: Theoretical motivation for the λ-ω (avoidance-activation) model.

Models:
    M1: Effort-only      — λ only, no threat in choice, no vigor
    M2: Threat-only       — no individual λ, S governs choice, ω in vigor
    M3: Single-channel    — one param θ enters both choice AND vigor
    M4: Independent       — λ + ω but no shared S (additive choice, linear T)
    M5: Full (λ,ω,S)     — two channels with shared survival function

Each model tested via SVI (AutoNormal, 30k steps).
Comparison: ELBO, BIC, choice accuracy, choice r², vigor r².
"""

import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from jax import random
import numpy as np
import pandas as pd
import ast
from scipy.stats import pearsonr
from scipy.special import expit
from pathlib import Path

# ============================================================
# Data preparation (adapted from evc_final_2plus2.py)
# ============================================================

EXCLUDE = [154, 197, 208]
DATA_DIR = Path("data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("results/stats/model_comparison_5models")


def prepare_data():
    """Load and prepare choice + vigor arrays.

    Uses precomputed trial_vigor.csv for vigor data (fast).
    If not found, falls back to parsing alignedEffortRate from behavior_rich.csv.
    Run scripts/preprocessing/precompute_vigor.py first to generate trial_vigor.csv.
    """
    vigor_path = DATA_DIR / "trial_vigor.csv"
    if vigor_path.exists():
        print("  Loading precomputed trial_vigor.csv...")
        vigor_df = pd.read_csv(vigor_path)
        vigor_df = vigor_df.dropna(subset=['excess']).copy()

        # Choice data: type=1 from vigor_df (it has all columns we need)
        beh = pd.read_csv(DATA_DIR / "behavior_rich.csv",
                          usecols=['subj', 'type', 'threat', 'distance_H', 'choice'],
                          low_memory=False)
        beh = beh[~beh['subj'].isin(EXCLUDE)]
        choice_df = beh[beh['type'] == 1].copy()
    else:
        print("  trial_vigor.csv not found — parsing alignedEffortRate (slow)...")
        beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
        beh = beh[~beh['subj'].isin(EXCLUDE)].copy()

        choice_df = beh[beh['type'] == 1].copy()

        vigor_df = beh.copy()
        vigor_df['actual_dist'] = vigor_df['startDistance'].map({5: 1, 7: 2, 9: 3})
        vigor_df['actual_req'] = np.where(
            vigor_df['trialCookie_weight'] == 3.0, 0.9, 0.4)
        vigor_df['actual_R'] = np.where(
            vigor_df['trialCookie_weight'] == 3.0, 5.0, 1.0)
        vigor_df['is_heavy'] = (vigor_df['trialCookie_weight'] == 3.0).astype(int)

        rates = []
        for _, row in vigor_df.iterrows():
            try:
                pt = np.array(ast.literal_eval(row['alignedEffortRate']), dtype=float)
                ipis = np.diff(pt)
                ipis = ipis[ipis > 0.01]
                if len(ipis) >= 5:
                    rates.append(np.median((1.0 / ipis) / row['calibrationMax']))
                else:
                    rates.append(np.nan)
            except Exception:
                rates.append(np.nan)

        vigor_df['median_rate'] = rates
        vigor_df['excess'] = vigor_df['median_rate'] - vigor_df['actual_req']
        vigor_df = vigor_df.dropna(subset=['excess']).copy()

    # Cookie-type centering
    choice_vigor = vigor_df[vigor_df['type'] == 1]
    heavy_mean = choice_vigor[choice_vigor['is_heavy'] == 1]['excess'].mean()
    light_mean = choice_vigor[choice_vigor['is_heavy'] == 0]['excess'].mean()
    vigor_df['excess_cc'] = vigor_df['excess'] - np.where(
        vigor_df['is_heavy'] == 1, heavy_mean, light_mean)

    subjects = sorted(
        set(choice_df['subj'].unique()) & set(vigor_df['subj'].unique()))
    si = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)

    data = {
        'ch_subj': jnp.array([si[s] for s in choice_df['subj']]),
        'ch_T': jnp.array(choice_df['threat'].values),
        'ch_dist_H': jnp.array(choice_df['distance_H'].values, dtype=jnp.float64),
        'ch_choice': jnp.array(choice_df['choice'].values),
        'vig_subj': jnp.array([si[s] for s in vigor_df['subj']]),
        'vig_T': jnp.array(vigor_df['threat'].values),
        'vig_R': jnp.array(vigor_df['actual_R'].values),
        'vig_req': jnp.array(vigor_df['actual_req'].values),
        'vig_dist': jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64),
        'vig_excess': jnp.array(vigor_df['excess_cc'].values),
        'vig_offset': jnp.array(np.where(
            vigor_df['is_heavy'].values == 1, heavy_mean, light_mean)),
        'subjects': subjects, 'N_S': N_S,
        'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
        'heavy_mean': heavy_mean, 'light_mean': light_mean,
    }
    return data


# ============================================================
# Shared vigor computation helper
# ============================================================

def vigor_eu_block(omega_v, T_w_v, epsilon, p_esc, sigma_motor, ce_vigor,
                   vig_R, vig_req, vig_dist):
    """Compute optimal vigor u* from EU grid search."""
    u_grid = jnp.linspace(0.1, 1.5, 30)
    u_g = u_grid[None, :]
    S_u = ((1.0 - T_w_v[:, None])
           + epsilon * T_w_v[:, None] * p_esc
           * jax.nn.sigmoid((u_g - vig_req[:, None]) / sigma_motor))
    deviation = u_g - vig_req[:, None]
    eu_grid = (S_u * vig_R[:, None]
               - (1.0 - S_u) * omega_v[:, None] * (vig_R[:, None] + 5.0)
               - ce_vigor * deviation ** 2 * vig_dist[:, None])
    weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
    u_star = jnp.sum(weights * u_g, axis=1)
    return u_star


# ============================================================
# Common population vigor priors
# ============================================================

def sample_vigor_pop_params():
    """Sample shared population params for vigor models."""
    eps_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
    epsilon = numpyro.deterministic('epsilon', jnp.exp(eps_raw))
    gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
    gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
    p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
    p_esc = jax.nn.sigmoid(p_esc_raw)
    sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
    sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)
    ce_vigor_raw = numpyro.sample('ce_vigor_raw', dist.Normal(-3.0, 1.0))
    ce_vigor = numpyro.deterministic('ce_vigor', jnp.exp(ce_vigor_raw))
    sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
    return epsilon, gamma, p_esc, sigma_motor, ce_vigor, sigma_v


# ============================================================
# Model definitions
# ============================================================

def make_model_m1(N_S, N_choice, N_vigor):
    """M1: Effort-only. λ in choice, no threat, no vigor model."""
    def model(ch_subj, ch_T, ch_dist_H, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_excess, vig_offset):
        mu_lam = numpyro.sample('mu_lam', dist.Normal(0.0, 1.0))
        sigma_lam = numpyro.sample('sigma_lam', dist.HalfNormal(0.5))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)

        with numpyro.plate('subjects', N_S):
            lam_raw = numpyro.sample('lam_raw', dist.Normal(0.0, 1.0))
        lam = jnp.exp(mu_lam + sigma_lam * lam_raw)
        numpyro.deterministic('lam', lam)

        # Choice: no threat term
        effort_cost = 0.81 * ch_dist_H - 0.16
        delta_eu = 4.0 - lam[ch_subj] * effort_cost
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)
        # No vigor likelihood
    return model


def make_model_m2(N_S, N_choice, N_vigor):
    """M2: Threat-only. S governs choice (no per-subject choice param), ω in vigor."""
    def model(ch_subj, ch_T, ch_dist_H, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_excess, vig_offset):
        # Population params
        epsilon, gamma, p_esc, sigma_motor, ce_vigor, sigma_v = sample_vigor_pop_params()
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)

        # Per-subject: omega only
        mu_om = numpyro.sample('mu_om', dist.Normal(0.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(0.5))
        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
        omega = jnp.exp(mu_om + sigma_om * om_raw)
        numpyro.deterministic('omega', omega)

        # Choice: S only, no per-subject param
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc
        delta_eu = S_ch * 4.0
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # Vigor
        T_w_v = jnp.power(vig_T, gamma)
        u_star = vigor_eu_block(omega[vig_subj], T_w_v, epsilon, p_esc,
                                sigma_motor, ce_vigor, vig_R, vig_req, vig_dist)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)
    return model


def make_model_m3(N_S, N_choice, N_vigor):
    """M3: Single-channel. One param θ enters both choice AND vigor."""
    def model(ch_subj, ch_T, ch_dist_H, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_excess, vig_offset):
        epsilon, gamma, p_esc, sigma_motor, ce_vigor, sigma_v = sample_vigor_pop_params()
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)

        # Per-subject: one param for both
        mu_theta = numpyro.sample('mu_theta', dist.Normal(0.0, 1.0))
        sigma_theta = numpyro.sample('sigma_theta', dist.HalfNormal(0.5))
        with numpyro.plate('subjects', N_S):
            theta_raw = numpyro.sample('theta_raw', dist.Normal(0.0, 1.0))
        theta = jnp.exp(mu_theta + sigma_theta * theta_raw)
        numpyro.deterministic('theta', theta)

        # Choice: θ as effort cost (same role as λ)
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc
        effort_cost = 0.81 * ch_dist_H - 0.16
        delta_eu = S_ch * 4.0 - theta[ch_subj] * effort_cost
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # Vigor: θ as capture cost (same role as ω)
        T_w_v = jnp.power(vig_T, gamma)
        u_star = vigor_eu_block(theta[vig_subj], T_w_v, epsilon, p_esc,
                                sigma_motor, ce_vigor, vig_R, vig_req, vig_dist)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)
    return model


def make_model_m4(N_S, N_choice, N_vigor):
    """M4: Independent channels. λ + ω but no shared S in choice (additive T)."""
    def model(ch_subj, ch_T, ch_dist_H, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_excess, vig_offset):
        # No gamma/epsilon in choice — vigor uses linear T
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)
        ce_vigor_raw = numpyro.sample('ce_vigor_raw', dist.Normal(-3.0, 1.0))
        ce_vigor = numpyro.deterministic('ce_vigor', jnp.exp(ce_vigor_raw))
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        beta_pop = numpyro.sample('beta_pop', dist.Normal(0.0, 5.0))

        # Per-subject: λ and ω (independent)
        mu_lam = numpyro.sample('mu_lam', dist.Normal(0.0, 1.0))
        sigma_lam = numpyro.sample('sigma_lam', dist.HalfNormal(0.5))
        mu_om = numpyro.sample('mu_om', dist.Normal(0.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(0.5))

        with numpyro.plate('subjects', N_S):
            lam_raw = numpyro.sample('lam_raw', dist.Normal(0.0, 1.0))
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
        lam = jnp.exp(mu_lam + sigma_lam * lam_raw)
        omega = jnp.exp(mu_om + sigma_om * om_raw)
        numpyro.deterministic('lam', lam)
        numpyro.deterministic('omega', omega)

        # Choice: additive, no S — just λ·effort + β_pop·T
        effort_cost = 0.81 * ch_dist_H - 0.16
        delta_eu = 4.0 - lam[ch_subj] * effort_cost - beta_pop * ch_T
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # Vigor: S(u) with linear T (no gamma)
        # epsilon fixed to a reasonable constant since it's not shared with choice
        eps_vig = numpyro.sample('eps_vig_raw', dist.Normal(-1.0, 0.5))
        epsilon_vig = jnp.exp(eps_vig)
        T_w_v = vig_T  # No gamma — linear T
        u_star = vigor_eu_block(omega[vig_subj], T_w_v, epsilon_vig, p_esc,
                                sigma_motor, ce_vigor, vig_R, vig_req, vig_dist)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)
    return model


def make_model_m5(N_S, N_choice, N_vigor):
    """M5: Full model. λ + ω with shared S (γ, ε) across choice and vigor."""
    def model(ch_subj, ch_T, ch_dist_H, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_excess, vig_offset):
        epsilon, gamma, p_esc, sigma_motor, ce_vigor, sigma_v = sample_vigor_pop_params()
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)

        # Per-subject: λ (avoidance) and ω (activation)
        mu_lam = numpyro.sample('mu_lam', dist.Normal(0.0, 1.0))
        sigma_lam = numpyro.sample('sigma_lam', dist.HalfNormal(0.5))
        mu_om = numpyro.sample('mu_om', dist.Normal(0.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(0.5))

        with numpyro.plate('subjects', N_S):
            lam_raw = numpyro.sample('lam_raw', dist.Normal(0.0, 1.0))
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
        lam = jnp.exp(mu_lam + sigma_lam * lam_raw)
        omega = jnp.exp(mu_om + sigma_om * om_raw)
        numpyro.deterministic('lam', lam)
        numpyro.deterministic('omega', omega)

        # Choice: S × ΔR - λ × effort
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc
        effort_cost = 0.81 * ch_dist_H - 0.16
        delta_eu = S_ch * 4.0 - lam[ch_subj] * effort_cost
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # Vigor: EU(u) with shared S
        T_w_v = jnp.power(vig_T, gamma)
        u_star = vigor_eu_block(omega[vig_subj], T_w_v, epsilon, p_esc,
                                sigma_motor, ce_vigor, vig_R, vig_req, vig_dist)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)
    return model


# ============================================================
# Generic fit and evaluate
# ============================================================

KWARGS_KEYS = ['ch_subj', 'ch_T', 'ch_dist_H', 'ch_choice',
               'vig_subj', 'vig_T', 'vig_R', 'vig_req', 'vig_dist',
               'vig_excess', 'vig_offset']


def fit_model(name, model_fn, data, n_steps=30000, lr=0.001, seed=42):
    """Fit a model via SVI, return best params and loss."""
    kwargs = {k: data[k] for k in KWARGS_KEYS}
    guide = AutoNormal(model_fn)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
    svi = SVI(model_fn, guide, optimizer, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    best_loss = float('inf')
    best_params = None
    t0 = time.time()

    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        l = float(loss)
        if l < best_loss:
            best_loss = l
            best_params = svi.get_params(state)
        if (i + 1) % 10000 == 0:
            print(f"    {name} step {i+1}: loss={l:.1f} (best={best_loss:.1f})")

    elapsed = time.time() - t0
    print(f"    {name} done in {elapsed:.0f}s, best_loss={best_loss:.1f}")

    return {
        'name': name, 'best_loss': best_loss, 'best_params': best_params,
        'guide': guide, 'model_fn': model_fn, 'kwargs': kwargs,
    }


def evaluate_model(fit_result, data, n_samples=300, seed=44):
    """Extract predictions, compute metrics."""
    guide = fit_result['guide']
    model_fn = fit_result['model_fn']
    params_fit = fit_result['best_params']
    kwargs = fit_result['kwargs']
    name = fit_result['name']

    # Determine which sites to return
    return_sites = []
    # Check for per-subject params by model name
    if name == 'M1':
        return_sites = ['lam']
    elif name == 'M2':
        return_sites = ['omega', 'epsilon', 'gamma', 'excess_pred']
    elif name == 'M3':
        return_sites = ['theta', 'epsilon', 'gamma', 'excess_pred']
    elif name == 'M4':
        return_sites = ['lam', 'omega', 'beta_pop', 'excess_pred']
    elif name == 'M5':
        return_sites = ['lam', 'omega', 'epsilon', 'gamma', 'excess_pred']

    # Always try to get these
    return_sites.extend(['tau_raw', 'p_esc_raw', 'sigma_motor_raw',
                         'ce_vigor', 'sigma_v'])
    return_sites = list(set(return_sites))

    pred = Predictive(model_fn, guide=guide, params=params_fit,
                      num_samples=n_samples, return_sites=return_sites)
    samples = pred(random.PRNGKey(seed), **kwargs)

    # -- Choice predictions --
    ch_subj_np = np.array(data['ch_subj'])
    ch_T_np = np.array(data['ch_T'])
    ch_dist_np = np.array(data['ch_dist_H'])
    ch_choice_np = np.array(data['ch_choice'])

    tau_val = float(np.exp(np.array(samples['tau_raw']).mean()))

    # Reconstruct P(heavy) per model
    if name == 'M1':
        lam = np.array(samples['lam']).mean(0)
        effort = 0.81 * ch_dist_np - 0.16
        delta_eu = 4.0 - lam[ch_subj_np] * effort
        p_H = expit(np.clip(delta_eu / tau_val, -20, 20))

    elif name == 'M2':
        eps_val = float(np.array(samples['epsilon']).mean())
        gamma_val = float(np.array(samples['gamma']).mean())
        p_esc_val = float(expit(np.array(samples['p_esc_raw']).mean()))
        T_w = ch_T_np ** gamma_val
        S = (1 - T_w) + eps_val * T_w * p_esc_val
        delta_eu = S * 4.0
        p_H = expit(np.clip(delta_eu / tau_val, -20, 20))

    elif name == 'M3':
        theta = np.array(samples['theta']).mean(0)
        eps_val = float(np.array(samples['epsilon']).mean())
        gamma_val = float(np.array(samples['gamma']).mean())
        p_esc_val = float(expit(np.array(samples['p_esc_raw']).mean()))
        T_w = ch_T_np ** gamma_val
        S = (1 - T_w) + eps_val * T_w * p_esc_val
        effort = 0.81 * ch_dist_np - 0.16
        delta_eu = S * 4.0 - theta[ch_subj_np] * effort
        p_H = expit(np.clip(delta_eu / tau_val, -20, 20))

    elif name == 'M4':
        lam = np.array(samples['lam']).mean(0)
        beta_pop_val = float(np.array(samples['beta_pop']).mean())
        effort = 0.81 * ch_dist_np - 0.16
        delta_eu = 4.0 - lam[ch_subj_np] * effort - beta_pop_val * ch_T_np
        p_H = expit(np.clip(delta_eu / tau_val, -20, 20))

    elif name == 'M5':
        lam = np.array(samples['lam']).mean(0)
        eps_val = float(np.array(samples['epsilon']).mean())
        gamma_val = float(np.array(samples['gamma']).mean())
        p_esc_val = float(expit(np.array(samples['p_esc_raw']).mean()))
        T_w = ch_T_np ** gamma_val
        S = (1 - T_w) + eps_val * T_w * p_esc_val
        effort = 0.81 * ch_dist_np - 0.16
        delta_eu = S * 4.0 - lam[ch_subj_np] * effort
        p_H = expit(np.clip(delta_eu / tau_val, -20, 20))

    # Choice accuracy
    choice_acc = ((p_H >= 0.5).astype(int) == ch_choice_np).mean()

    # Per-subject choice r
    ch_df = pd.DataFrame({'subj': ch_subj_np, 'choice': ch_choice_np, 'p_H': p_H})
    sc = ch_df.groupby('subj').agg(o=('choice', 'mean'), p=('p_H', 'mean'))
    try:
        r_choice, _ = pearsonr(sc['o'], sc['p'])
    except Exception:
        r_choice = np.nan

    # Vigor r (if modeled)
    r_vigor = np.nan
    if name != 'M1':
        if 'excess_pred' in samples:
            ep = np.array(samples['excess_pred']).mean(0)
        else:
            ep = None

        if ep is not None:
            vig_excess_np = np.array(data['vig_excess'])
            r_vigor, _ = pearsonr(ep, vig_excess_np)

    return {
        'choice_acc': choice_acc,
        'choice_r': r_choice,
        'choice_r2': r_choice ** 2,
        'vigor_r': r_vigor,
        'vigor_r2': r_vigor ** 2 if not np.isnan(r_vigor) else np.nan,
    }


# ============================================================
# Main
# ============================================================

MODEL_SPECS = [
    ('M1', make_model_m1, 1, 'Effort-only (λ)'),
    ('M2', make_model_m2, 1, 'Threat-only (ω)'),
    ('M3', make_model_m3, 1, 'Single-channel (θ)'),
    ('M4', make_model_m4, 2, 'Independent (λ+ω, no shared S)'),
    ('M5', make_model_m5, 2, 'Full model (λ+ω, shared S)'),
]

# Parameter counts (hierarchical params + population params)
# N_S = per-subject raw params (each has mu + sigma at population level)
PARAM_COUNTS = {
    'M1': lambda N_S: N_S + 3,       # lam_raw + mu_lam, sigma_lam, tau_raw
    'M2': lambda N_S: N_S + 9,       # om_raw + mu_om, sigma_om, eps, gamma, tau, p_esc, sigma_motor, ce_vigor, sigma_v
    'M3': lambda N_S: N_S + 9,       # theta_raw + mu_theta, sigma_theta, eps, gamma, tau, p_esc, sigma_motor, ce_vigor, sigma_v
    'M4': lambda N_S: 2*N_S + 11,    # lam+om raw + mu_lam, sigma_lam, mu_om, sigma_om, beta_pop, tau, p_esc, sigma_motor, ce_vigor, sigma_v, eps_vig
    'M5': lambda N_S: 2*N_S + 11,    # lam+om raw + mu_lam, sigma_lam, mu_om, sigma_om, eps, gamma, tau, p_esc, sigma_motor, ce_vigor, sigma_v
}


if __name__ == '__main__':
    t_start = time.time()

    print("=" * 70)
    print("5-MODEL COMPARISON: Avoidance-Activation Framework")
    print("=" * 70)

    # Prepare data
    print("\nPreparing data...")
    data = prepare_data()
    N_S = data['N_S']
    print(f"  {N_S} subjects, {data['N_choice']} choice trials, {data['N_vigor']} vigor trials")

    # Fit and evaluate all models
    results = []

    for name, make_fn, n_per_subj, description in MODEL_SPECS:
        print(f"\n--- {name}: {description} ---")
        model_fn = make_fn(N_S, data['N_choice'], data['N_vigor'])

        fit_result = fit_model(name, model_fn, data, n_steps=30000, lr=0.001)
        metrics = evaluate_model(fit_result, data)

        # BIC
        n_params = PARAM_COUNTS[name](N_S)
        if name == 'M1':
            n_obs = data['N_choice']
        else:
            n_obs = data['N_choice'] + data['N_vigor']
        bic = 2 * fit_result['best_loss'] + n_params * np.log(n_obs)

        row = {
            'Model': name,
            'Description': description,
            'n_per_subj': n_per_subj,
            'n_params': n_params,
            'ELBO': -fit_result['best_loss'],
            'BIC': bic,
            'choice_acc': metrics['choice_acc'],
            'choice_r': metrics['choice_r'],
            'choice_r2': metrics['choice_r2'],
            'vigor_r': metrics['vigor_r'],
            'vigor_r2': metrics['vigor_r2'],
        }
        results.append(row)

        print(f"  ELBO={-fit_result['best_loss']:.1f}, BIC={bic:.0f}")
        print(f"  Choice: acc={metrics['choice_acc']:.3f}, r²={metrics['choice_r2']:.3f}")
        if not np.isnan(metrics['vigor_r']):
            print(f"  Vigor:  r²={metrics['vigor_r2']:.3f}")

    # Build comparison table
    df = pd.DataFrame(results)
    # ΔBIC relative to M5
    bic_m5 = df.loc[df['Model'] == 'M5', 'BIC'].values[0]
    df['dBIC'] = df['BIC'] - bic_m5

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(df[['Model', 'Description', 'n_per_subj', 'n_params', 'ELBO',
              'BIC', 'dBIC', 'choice_acc', 'choice_r2', 'vigor_r2']].to_string(index=False))

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "comparison_table.csv", index=False)
    print(f"\nSaved to {OUT_DIR / 'comparison_table.csv'}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")
