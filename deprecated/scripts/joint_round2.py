"""
Joint Optimal Control — Round 2

Key fixes from Round 1 diagnostics:
1. Exponential S(u, T, D) = exp(-h·T^γ·D/u) — distance enters survival
2. Reference-u choice evaluation — evaluate W at u=req, not at u* (avoids scale mismatch)
3. Upweighted choice likelihood (2x)
4. Per-subject baseline for vigor
5. Wider ω priors

Models:
  V6:  Exponential S, grid-search choice (V_H(u*_H) vs V_L(u*_L))
  V7:  Exponential S, reference-u choice (W_H(req_H) vs W_L(req_L))
  V8:  V7 + per-subject vigor baseline
  V9:  V7 + 2x choice weight
  V10: Shared ω only (κ population), exponential S, reference-u
"""

import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

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
from scipy.stats import pearsonr
from scipy.special import expit
from pathlib import Path

EXCLUDE = [154, 197, 208]
DATA_DIR = Path("data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("results/stats/joint_optimal")
C_PENALTY = 5.0


def prepare_data():
    beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)]
    choice_df = beh[beh['type'] == 1].copy()
    vigor_df = pd.read_csv(DATA_DIR / "trial_vigor.csv")
    vigor_df = vigor_df[~vigor_df['subj'].isin(EXCLUDE)]
    vigor_df = vigor_df.dropna(subset=['median_rate']).copy()

    subjects = sorted(set(choice_df['subj'].unique()) & set(vigor_df['subj'].unique()))
    si = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)

    # Check empirical survival rates
    surv = beh.groupby(['threat', 'distance_H'])['isAttackTrial'].mean()
    print("  Attack rates by threat × distance:")
    for t in [0.1, 0.5, 0.9]:
        for d in [1, 2, 3]:
            try:
                rate = surv.loc[(t, d)]
                print(f"    T={t}, D={d}: attack={rate:.3f}, survival≈{1-rate:.3f}")
            except:
                pass

    data = {
        'ch_subj': jnp.array([si[s] for s in choice_df['subj']]),
        'ch_T': jnp.array(choice_df['threat'].values),
        'ch_D_H': jnp.array(choice_df['distance_H'].values, dtype=jnp.float64),
        'ch_D_L': jnp.ones(len(choice_df)),
        'ch_choice': jnp.array(choice_df['choice'].values),
        'vig_subj': jnp.array([si[s] for s in vigor_df['subj']]),
        'vig_T': jnp.array(vigor_df['threat'].values),
        'vig_R': jnp.array(vigor_df['actual_R'].values),
        'vig_req': jnp.array(vigor_df['actual_req'].values),
        'vig_dist': jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64),
        'vig_rate': jnp.array(vigor_df['median_rate'].values),
        'vig_cookie': jnp.array(vigor_df['is_heavy'].values, dtype=jnp.float64),
        'subjects': subjects, 'N_S': N_S,
        'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
    }
    print(f"  {N_S} subjects, {data['N_choice']} choice, {data['N_vigor']} vigor")
    return data


# ============================================================
# Exponential survival: S = exp(-h · T^γ · D / u)
# ============================================================

def exp_survival(u, T, D, gamma, hazard_scale):
    """S(u, T, D) = exp(-h · T^γ · D / max(u, 0.1))"""
    T_w = jnp.power(T, gamma)
    return jnp.exp(-hazard_scale * T_w * D / jnp.clip(u, 0.1, None))


def vigor_eu_exp(omega, kappa, T, D, R, req, gamma, hazard_scale, u_grid):
    """Find u* from EU grid search with exponential S."""
    u_g = u_grid[None, :]
    S = exp_survival(u_g, T[:, None], D[:, None], gamma, hazard_scale)
    W = (S * R[:, None]
         - (1.0 - S) * omega[:, None] * (R[:, None] + C_PENALTY)
         - kappa[:, None] * (u_g - req[:, None]) ** 2 * D[:, None])
    weights = jax.nn.softmax(W * 20.0, axis=1)
    u_star = jnp.sum(weights * u_g, axis=1)
    V_star = jnp.sum(weights * W, axis=1)
    return u_star, V_star


# ============================================================
# V6: Exponential S, grid-search choice
# ============================================================

def make_v6(N_S, N_ch, N_vig):
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist, vig_rate, vig_cookie):
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        hz_raw = numpyro.sample('hazard_raw', dist.Normal(-1.0, 1.0))
        hazard = numpyro.deterministic('hazard', jnp.exp(hz_raw))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 50.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        mu_om = numpyro.sample('mu_om', dist.Normal(1.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(1.0))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-2.0, 1.0))
        sigma_kap = numpyro.sample('sigma_kap', dist.HalfNormal(0.5))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
        omega = jnp.exp(mu_om + sigma_om * om_raw)
        kappa = jnp.exp(mu_kap + sigma_kap * kap_raw)
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)

        u_grid = jnp.linspace(0.1, 1.5, 30)

        # Choice via grid search
        _, V_H = vigor_eu_exp(omega[ch_subj], kappa[ch_subj], ch_T, ch_D_H,
                              jnp.full(N_ch, 5.0), jnp.full(N_ch, 0.9),
                              gamma, hazard, u_grid)
        _, V_L = vigor_eu_exp(omega[ch_subj], kappa[ch_subj], ch_T, ch_D_L,
                              jnp.full(N_ch, 1.0), jnp.full(N_ch, 0.4),
                              gamma, hazard, u_grid)
        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice', N_ch):
            numpyro.sample('obs_ch', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # Vigor
        u_star, _ = vigor_eu_exp(omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
                                 vig_R, vig_req, gamma, hazard, u_grid)
        rate_pred = u_star + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)
        with numpyro.plate('vigor', N_vig):
            numpyro.sample('obs_vig', dist.Normal(rate_pred, sigma_v), obs=vig_rate)
    return model


# ============================================================
# V7: Exponential S, reference-u choice (W at u=req)
# ============================================================

def make_v7(N_S, N_ch, N_vig):
    """Choice evaluates W at the required pressing rate, not at u*.
    This avoids the scale mismatch: choice uses the 'default effort' value,
    vigor uses the 'optimized effort' value. Same W function, same parameters.
    """
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist, vig_rate, vig_cookie):
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        hz_raw = numpyro.sample('hazard_raw', dist.Normal(-1.0, 1.0))
        hazard = numpyro.deterministic('hazard', jnp.exp(hz_raw))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 50.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        mu_om = numpyro.sample('mu_om', dist.Normal(1.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(1.0))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-2.0, 1.0))
        sigma_kap = numpyro.sample('sigma_kap', dist.HalfNormal(0.5))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
        omega = jnp.exp(mu_om + sigma_om * om_raw)
        kappa = jnp.exp(mu_kap + sigma_kap * kap_raw)
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)

        u_grid = jnp.linspace(0.1, 1.5, 30)

        # Choice: evaluate W at reference rates (u = req)
        # Heavy at u=0.9: S_H = exp(-h·T^γ·D_H/0.9), cost_H = 0
        # Light at u=0.4: S_L = exp(-h·T^γ·1/0.4), cost_L = 0
        S_H = exp_survival(jnp.full(N_ch, 0.9), ch_T, ch_D_H, gamma, hazard)
        S_L = exp_survival(jnp.full(N_ch, 0.4), ch_T, ch_D_L, gamma, hazard)
        V_H = S_H * 5.0 - (1.0 - S_H) * omega[ch_subj] * 10.0
        V_L = S_L * 1.0 - (1.0 - S_L) * omega[ch_subj] * 6.0

        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice', N_ch):
            numpyro.sample('obs_ch', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # Vigor: full EU optimization
        u_star, _ = vigor_eu_exp(omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
                                 vig_R, vig_req, gamma, hazard, u_grid)
        rate_pred = u_star + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)
        with numpyro.plate('vigor', N_vig):
            numpyro.sample('obs_vig', dist.Normal(rate_pred, sigma_v), obs=vig_rate)
    return model


# ============================================================
# V8: V7 + per-subject vigor baseline
# ============================================================

def make_v8(N_S, N_ch, N_vig):
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist, vig_rate, vig_cookie):
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        hz_raw = numpyro.sample('hazard_raw', dist.Normal(-1.0, 1.0))
        hazard = numpyro.deterministic('hazard', jnp.exp(hz_raw))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 50.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        mu_om = numpyro.sample('mu_om', dist.Normal(1.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(1.0))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-2.0, 1.0))
        sigma_kap = numpyro.sample('sigma_kap', dist.HalfNormal(0.5))
        mu_base = numpyro.sample('mu_base', dist.Normal(0.0, 0.3))
        sigma_base = numpyro.sample('sigma_base', dist.HalfNormal(0.2))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
            base_raw = numpyro.sample('base_raw', dist.Normal(0.0, 1.0))
        omega = jnp.exp(mu_om + sigma_om * om_raw)
        kappa = jnp.exp(mu_kap + sigma_kap * kap_raw)
        baseline = mu_base + sigma_base * base_raw
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)

        u_grid = jnp.linspace(0.1, 1.5, 30)

        # Choice at reference u (same as V7)
        S_H = exp_survival(jnp.full(N_ch, 0.9), ch_T, ch_D_H, gamma, hazard)
        S_L = exp_survival(jnp.full(N_ch, 0.4), ch_T, ch_D_L, gamma, hazard)
        V_H = S_H * 5.0 - (1.0 - S_H) * omega[ch_subj] * 10.0
        V_L = S_L * 1.0 - (1.0 - S_L) * omega[ch_subj] * 6.0
        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice', N_ch):
            numpyro.sample('obs_ch', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # Vigor with baseline
        u_star, _ = vigor_eu_exp(omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
                                 vig_R, vig_req, gamma, hazard, u_grid)
        rate_pred = u_star + baseline[vig_subj] + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)
        with numpyro.plate('vigor', N_vig):
            numpyro.sample('obs_vig', dist.Normal(rate_pred, sigma_v), obs=vig_rate)
    return model


# ============================================================
# V9: V7 with 2x choice weight
# ============================================================

def make_v9(N_S, N_ch, N_vig):
    """Same as V7 but choice likelihood is upweighted 2x."""
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist, vig_rate, vig_cookie):
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        hz_raw = numpyro.sample('hazard_raw', dist.Normal(-1.0, 1.0))
        hazard = numpyro.deterministic('hazard', jnp.exp(hz_raw))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 50.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        mu_om = numpyro.sample('mu_om', dist.Normal(1.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(1.0))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-2.0, 1.0))
        sigma_kap = numpyro.sample('sigma_kap', dist.HalfNormal(0.5))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
        omega = jnp.exp(mu_om + sigma_om * om_raw)
        kappa = jnp.exp(mu_kap + sigma_kap * kap_raw)
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)

        u_grid = jnp.linspace(0.1, 1.5, 30)

        # Choice at reference u — UPWEIGHTED 2x via scale
        S_H = exp_survival(jnp.full(N_ch, 0.9), ch_T, ch_D_H, gamma, hazard)
        S_L = exp_survival(jnp.full(N_ch, 0.4), ch_T, ch_D_L, gamma, hazard)
        V_H = S_H * 5.0 - (1.0 - S_H) * omega[ch_subj] * 10.0
        V_L = S_L * 1.0 - (1.0 - S_L) * omega[ch_subj] * 6.0
        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        # Upweight: observe choice TWICE (equivalent to 2x log-likelihood)
        with numpyro.plate('choice1', N_ch):
            numpyro.sample('obs_ch1', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)
        with numpyro.plate('choice2', N_ch):
            numpyro.sample('obs_ch2', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # Vigor
        u_star, _ = vigor_eu_exp(omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
                                 vig_R, vig_req, gamma, hazard, u_grid)
        rate_pred = u_star + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)
        with numpyro.plate('vigor', N_vig):
            numpyro.sample('obs_vig', dist.Normal(rate_pred, sigma_v), obs=vig_rate)
    return model


# ============================================================
# V10: Only ω per subject, κ population, exponential S, reference-u
# ============================================================

def make_v10(N_S, N_ch, N_vig):
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist, vig_rate, vig_cookie):
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        hz_raw = numpyro.sample('hazard_raw', dist.Normal(-1.0, 1.0))
        hazard = numpyro.deterministic('hazard', jnp.exp(hz_raw))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 50.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        kap_raw = numpyro.sample('kap_pop_raw', dist.Normal(-2.0, 1.0))
        kappa_pop = jnp.exp(kap_raw)

        mu_om = numpyro.sample('mu_om', dist.Normal(1.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(1.0))
        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
        omega = jnp.exp(mu_om + sigma_om * om_raw)
        numpyro.deterministic('omega', omega)

        u_grid = jnp.linspace(0.1, 1.5, 30)
        kappa_vig = jnp.full(N_vig, kappa_pop)

        # Choice at reference u
        S_H = exp_survival(jnp.full(N_ch, 0.9), ch_T, ch_D_H, gamma, hazard)
        S_L = exp_survival(jnp.full(N_ch, 0.4), ch_T, ch_D_L, gamma, hazard)
        V_H = S_H * 5.0 - (1.0 - S_H) * omega[ch_subj] * 10.0
        V_L = S_L * 1.0 - (1.0 - S_L) * omega[ch_subj] * 6.0
        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice', N_ch):
            numpyro.sample('obs_ch', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        u_star, _ = vigor_eu_exp(omega[vig_subj], kappa_vig, vig_T, vig_dist,
                                 vig_R, vig_req, gamma, hazard, u_grid)
        rate_pred = u_star + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)
        with numpyro.plate('vigor', N_vig):
            numpyro.sample('obs_vig', dist.Normal(rate_pred, sigma_v), obs=vig_rate)
    return model


# ============================================================
# Fitting and evaluation
# ============================================================

KWARGS_KEYS = ['ch_subj', 'ch_T', 'ch_D_H', 'ch_D_L', 'ch_choice',
               'vig_subj', 'vig_T', 'vig_R', 'vig_req', 'vig_dist',
               'vig_rate', 'vig_cookie']


def fit_model(name, model_fn, data, n_steps=30000, lr=0.001, seed=42):
    kwargs = {k: data[k] for k in KWARGS_KEYS}
    guide = AutoNormal(model_fn)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
    svi = SVI(model_fn, guide, optimizer, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)
    best_loss, best_params = float('inf'), None
    t0 = time.time()
    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        l = float(loss)
        if l < best_loss and not np.isnan(l):
            best_loss = l; best_params = svi.get_params(state)
        if (i + 1) % 10000 == 0:
            print(f"    {name} step {i+1}: loss={l:.1f} (best={best_loss:.1f})")
    print(f"    {name} done in {time.time()-t0:.0f}s, best={best_loss:.1f}")
    return {'name': name, 'best_loss': best_loss, 'best_params': best_params,
            'guide': guide, 'model_fn': model_fn, 'kwargs': kwargs}


def evaluate(fit, data, n_samples=300):
    guide, model_fn = fit['guide'], fit['model_fn']
    sites = ['omega', 'kappa', 'rate_pred', 'gamma', 'tau_raw', 'hazard',
             'b_cookie', 'sigma_v']
    pred = Predictive(model_fn, guide=guide, params=fit['best_params'],
                      num_samples=n_samples, return_sites=sites)
    samples = pred(random.PRNGKey(44), **fit['kwargs'])

    # Vigor r²
    vig_rate_np = np.array(data['vig_rate'])
    rp = np.array(samples['rate_pred']).mean(0) if 'rate_pred' in samples else None
    r_vig = pearsonr(rp, vig_rate_np)[0] if rp is not None else np.nan

    # Choice: reconstruct P(H) from params
    omega = np.array(samples['omega']).mean(0) if 'omega' in samples else None
    gamma_val = float(np.array(samples['gamma']).mean()) if 'gamma' in samples else 1.0
    hazard_val = float(np.array(samples['hazard']).mean()) if 'hazard' in samples else 0.5
    tau_val = float(np.exp(np.array(samples['tau_raw']).mean())) if 'tau_raw' in samples else 1.0

    ch_subj = np.array(data['ch_subj'])
    ch_T = np.array(data['ch_T'])
    ch_D_H = np.array(data['ch_D_H'])
    ch_choice = np.array(data['ch_choice'])

    if omega is not None:
        S_H = np.exp(-hazard_val * ch_T**gamma_val * ch_D_H / 0.9)
        S_L = np.exp(-hazard_val * ch_T**gamma_val * 1.0 / 0.4)
        V_H = S_H * 5.0 - (1 - S_H) * omega[ch_subj] * 10.0
        V_L = S_L * 1.0 - (1 - S_L) * omega[ch_subj] * 6.0
        p_H = expit(np.clip((V_H - V_L) / tau_val, -20, 20))
        acc = ((p_H >= 0.5).astype(int) == ch_choice).mean()
        ch_df = pd.DataFrame({'s': ch_subj, 'c': ch_choice, 'p': p_H})
        sc = ch_df.groupby('s').agg(o=('c','mean'), p=('p','mean'))
        try: r_ch = pearsonr(sc['o'], sc['p'])[0]
        except: r_ch = np.nan
    else:
        acc, r_ch = np.nan, np.nan

    return {'choice_acc': acc, 'choice_r2': r_ch**2 if not np.isnan(r_ch) else np.nan,
            'vigor_r2': r_vig**2 if not np.isnan(r_vig) else np.nan,
            'omega': omega, 'gamma': gamma_val, 'hazard': hazard_val, 'tau': tau_val}


# ============================================================
# Main
# ============================================================

MODELS = [
    ('V6', make_v6, 2, 'ExpS + grid choice'),
    ('V7', make_v7, 2, 'ExpS + ref-u choice'),
    ('V8', make_v8, 3, 'V7 + per-subj baseline'),
    ('V9', make_v9, 2, 'V7 + 2x choice weight'),
    ('V10', make_v10, 1, 'ω only + ref-u'),
]

PCOUNTS = {
    'V6': lambda N: 2*N + 8, 'V7': lambda N: 2*N + 8,
    'V8': lambda N: 3*N + 10, 'V9': lambda N: 2*N + 8,
    'V10': lambda N: N + 7,
}

if __name__ == '__main__':
    t0 = time.time()
    print("=" * 70)
    print("ROUND 2: Joint models with exponential S(u,T,D)")
    print("=" * 70)
    data = prepare_data()
    N_S = data['N_S']
    results = []

    for name, make_fn, nps, desc in MODELS:
        print(f"\n{'='*50}\n--- {name}: {desc} ---\n{'='*50}")
        model_fn = make_fn(N_S, data['N_choice'], data['N_vigor'])
        fit = fit_model(name, model_fn, data, n_steps=30000)
        if fit['best_params'] is None:
            print(f"  {name} FAILED"); continue
        m = evaluate(fit, data)
        n_p = PCOUNTS[name](N_S)
        bic = 2*fit['best_loss'] + n_p*np.log(data['N_choice']+data['N_vigor'])
        row = {'Model': name, 'Desc': desc, 'nps': nps, 'ELBO': -fit['best_loss'],
               'BIC': bic, 'ch_acc': m['choice_acc'], 'ch_r2': m['choice_r2'],
               'vig_r2': m['vigor_r2']}
        results.append(row)
        print(f"  ELBO={-fit['best_loss']:.1f} BIC={bic:.0f}")
        print(f"  Choice: acc={m['choice_acc']:.3f} r²={m['choice_r2']:.3f}")
        print(f"  Vigor: r²={m['vigor_r2']:.3f}")
        print(f"  γ={m['gamma']:.2f} h={m['hazard']:.3f} τ={m['tau']:.2f}")
        if m['omega'] is not None:
            print(f"  ω: mean={m['omega'].mean():.2f} SD={m['omega'].std():.2f}")

    if results:
        df = pd.DataFrame(results)
        best = df['BIC'].min(); df['dBIC'] = df['BIC'] - best
        print("\n" + "="*70 + "\nROUND 2 COMPARISON\n" + "="*70)
        print(df.to_string(index=False))
        print("\nM5 benchmark: acc=0.794 ch_r²=0.952 vig_r²=0.496 BIC=16191")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "joint_round2.csv", index=False)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
