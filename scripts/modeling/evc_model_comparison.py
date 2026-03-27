"""
EVC Model Comparison: 6 model variants to justify each component.

Models:
  1. EVC_base         — 2 subject params (c_effort, c_death), no epsilon, no gamma
  2. EVC_eps          — 3 subject params (c_effort, c_death, epsilon), no gamma
  3. EVC_gamma        — 3 subject params + population gamma (CURRENT BEST)
  4. EVC_gamma_ind    — 4 subject params (c_effort, c_death, epsilon, gamma_i)
  5. EVC_linear_effort — Same as EVC_gamma but linear effort cost
  6. EVC_no_sigmoid   — Same as EVC_gamma but linear ramp instead of sigmoid

Output: results/stats/evc_model_comparison.csv
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import jax
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

jax.config.update('jax_enable_x64', True)


# ── Shared data preparation (identical to oc_evc_gamma.py) ───────────────────

def prepare_data(behavior_rich_path):
    """Load and prepare data for EVC models."""
    beh = pd.read_csv(behavior_rich_path)
    beh_c = beh[beh['type'] == 1].copy()

    rates = []
    for _, row in beh_c.iterrows():
        try:
            press_times = np.array(ast.literal_eval(row['alignedEffortRate']), dtype=float)
        except Exception:
            rates.append(np.nan)
            continue
        ipis = np.diff(press_times)
        ipis = ipis[ipis > 0.01]
        if len(ipis) < 5:
            rates.append(np.nan)
            continue
        rates.append(np.median((1.0 / ipis) / row['calibrationMax']))

    beh_c['median_rate'] = rates
    beh_c['req_rate'] = np.where(beh_c['trialCookie_weight'] == 3.0, 0.9, 0.4)
    beh_c['excess'] = beh_c['median_rate'] - beh_c['req_rate']
    beh_c = beh_c.dropna(subset=['excess']).copy()

    heavy_mean = beh_c[beh_c['trialCookie_weight'] == 3.0]['excess'].mean()
    light_mean = beh_c[beh_c['trialCookie_weight'] == 1.0]['excess'].mean()
    beh_c['excess_cc'] = beh_c['excess'] - np.where(
        beh_c['trialCookie_weight'] == 3.0, heavy_mean, light_mean
    )

    subjects = sorted(beh_c['subj'].unique())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)
    N_T = len(beh_c)

    subj_idx = jnp.array([subj_to_idx[s] for s in beh_c['subj']])
    T = jnp.array(beh_c['threat'].values)
    dist_H = jnp.array(beh_c['distance_H'].values, dtype=jnp.float64)
    choice = jnp.array(beh_c['choice'].values)
    excess_cc = jnp.array(beh_c['excess_cc'].values)
    chosen_R = jnp.where(choice == 1, 5.0, 1.0)
    chosen_req = jnp.where(choice == 1, 0.9, 0.4)
    chosen_dist = jnp.where(choice == 1, dist_H, 1.0)
    chosen_offset = jnp.where(choice == 1, heavy_mean, light_mean)

    data = {
        'subj_idx': subj_idx, 'T': T, 'dist_H': dist_H,
        'choice': choice, 'excess_cc': excess_cc,
        'chosen_R': chosen_R, 'chosen_req': chosen_req,
        'chosen_dist': chosen_dist, 'chosen_offset': chosen_offset,
        'subjects': subjects, 'N_S': N_S, 'N_T': N_T,
        'heavy_mean': heavy_mean, 'light_mean': light_mean,
        'beh_c': beh_c,
    }
    return data


# ── Model 1: EVC_base — no epsilon, no gamma ────────────────────────────────

def make_model_base(N_S):
    def model(subj_idx, T, dist_H, choice=None, excess_cc=None,
              chosen_R=None, chosen_req=None, chosen_dist=None, chosen_offset=None):
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)

        with numpyro.plate('subjects', N_S):
            ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))

        c_effort = jnp.exp(mu_ce + sigma_ce * ce_raw)
        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        numpyro.deterministic('c_effort', c_effort)
        numpyro.deterministic('c_death', c_death)

        ce_i = c_effort[subj_idx]
        cd_i = c_death[subj_idx]

        # No gamma, no epsilon: S = (1-T) + T * p_esc
        S_full = (1.0 - T) + T * p_esc
        S_stop = 1.0 - T

        eu_H_full = S_full * 5 - (1 - S_full) * cd_i * 10 - ce_i * 0.81 * dist_H
        eu_H_stop = S_stop * 5 - (1 - S_stop) * cd_i * 10
        eu_H = jnp.maximum(eu_H_full, eu_H_stop)

        eu_L_full = S_full * 1 - (1 - S_full) * cd_i * 6 - ce_i * 0.16
        eu_L_stop = S_stop * 1 - (1 - S_stop) * cd_i * 6
        eu_L = jnp.maximum(eu_L_full, eu_L_stop)

        logit = jnp.clip((eu_H - eu_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        # Vigor
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T[:, None])
               + T[:, None] * p_esc
               * jax.nn.sigmoid((u_g - chosen_req[:, None]) / sigma_motor))
        eu_grid = (S_u * chosen_R[:, None]
                   - (1.0 - S_u) * cd_i[:, None] * (chosen_R[:, None] + 5.0)
                   - ce_i[:, None] * u_g ** 2 * chosen_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - chosen_req - chosen_offset
        numpyro.deterministic('excess_pred', excess_pred)

        N_T = subj_idx.shape[0]
        with numpyro.plate('trials', N_T):
            numpyro.sample('obs_choice', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)), obs=choice)
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v), obs=excess_cc)
    return model


# ── Model 2: EVC_eps — epsilon but no gamma ──────────────────────────────────

def make_model_eps(N_S):
    def model(subj_idx, T, dist_H, choice=None, excess_cc=None,
              chosen_R=None, chosen_req=None, chosen_dist=None, chosen_offset=None):
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)
        mu_eps = numpyro.sample('mu_eps', dist.Normal(-0.5, 0.5))
        sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.3))

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

        ce_i = c_effort[subj_idx]
        cd_i = c_death[subj_idx]
        eps_i = epsilon[subj_idx]

        # No gamma: T_w = T (no probability weighting)
        S_full = (1.0 - T) + eps_i * T * p_esc
        S_stop = 1.0 - T

        eu_H_full = S_full * 5 - (1 - S_full) * cd_i * 10 - ce_i * 0.81 * dist_H
        eu_H_stop = S_stop * 5 - (1 - S_stop) * cd_i * 10
        eu_H = jnp.maximum(eu_H_full, eu_H_stop)

        eu_L_full = S_full * 1 - (1 - S_full) * cd_i * 6 - ce_i * 0.16
        eu_L_stop = S_stop * 1 - (1 - S_stop) * cd_i * 6
        eu_L = jnp.maximum(eu_L_full, eu_L_stop)

        logit = jnp.clip((eu_H - eu_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        # Vigor
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T[:, None])
               + eps_i[:, None] * T[:, None] * p_esc
               * jax.nn.sigmoid((u_g - chosen_req[:, None]) / sigma_motor))
        eu_grid = (S_u * chosen_R[:, None]
                   - (1.0 - S_u) * cd_i[:, None] * (chosen_R[:, None] + 5.0)
                   - ce_i[:, None] * u_g ** 2 * chosen_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - chosen_req - chosen_offset
        numpyro.deterministic('excess_pred', excess_pred)

        N_T = subj_idx.shape[0]
        with numpyro.plate('trials', N_T):
            numpyro.sample('obs_choice', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)), obs=choice)
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v), obs=excess_cc)
    return model


# ── Model 3: EVC_gamma — current best (3 subject + pop gamma) ───────────────

def make_model_gamma(N_S):
    def model(subj_idx, T, dist_H, choice=None, excess_cc=None,
              chosen_R=None, chosen_req=None, chosen_dist=None, chosen_offset=None):
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)
        mu_eps = numpyro.sample('mu_eps', dist.Normal(-0.5, 0.5))
        sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.3))
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))

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

        ce_i = c_effort[subj_idx]
        cd_i = c_death[subj_idx]
        eps_i = epsilon[subj_idx]

        T_w = jnp.power(T, gamma)

        S_full = (1.0 - T_w) + eps_i * T_w * p_esc
        S_stop = 1.0 - T_w

        eu_H_full = S_full * 5 - (1 - S_full) * cd_i * 10 - ce_i * 0.81 * dist_H
        eu_H_stop = S_stop * 5 - (1 - S_stop) * cd_i * 10
        eu_H = jnp.maximum(eu_H_full, eu_H_stop)

        eu_L_full = S_full * 1 - (1 - S_full) * cd_i * 6 - ce_i * 0.16
        eu_L_stop = S_stop * 1 - (1 - S_stop) * cd_i * 6
        eu_L = jnp.maximum(eu_L_full, eu_L_stop)

        logit = jnp.clip((eu_H - eu_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T_w[:, None])
               + eps_i[:, None] * T_w[:, None] * p_esc
               * jax.nn.sigmoid((u_g - chosen_req[:, None]) / sigma_motor))
        eu_grid = (S_u * chosen_R[:, None]
                   - (1.0 - S_u) * cd_i[:, None] * (chosen_R[:, None] + 5.0)
                   - ce_i[:, None] * u_g ** 2 * chosen_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - chosen_req - chosen_offset
        numpyro.deterministic('excess_pred', excess_pred)

        N_T = subj_idx.shape[0]
        with numpyro.plate('trials', N_T):
            numpyro.sample('obs_choice', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)), obs=choice)
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v), obs=excess_cc)
    return model


# ── Model 4: EVC_gamma_ind — individual gamma per subject ────────────────────

def make_model_gamma_ind(N_S):
    def model(subj_idx, T, dist_H, choice=None, excess_cc=None,
              chosen_R=None, chosen_req=None, chosen_dist=None, chosen_offset=None):
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)
        mu_eps = numpyro.sample('mu_eps', dist.Normal(-0.5, 0.5))
        sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.3))

        # Individual gamma: log-normal, non-centered
        mu_gamma = numpyro.sample('mu_gamma', dist.Normal(0.0, 0.5))
        sigma_gamma = numpyro.sample('sigma_gamma', dist.HalfNormal(0.3))

        with numpyro.plate('subjects', N_S):
            ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))
            eps_raw = numpyro.sample('eps_raw', dist.Normal(0.0, 1.0))
            gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 1.0))

        c_effort = jnp.exp(mu_ce + sigma_ce * ce_raw)
        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        epsilon = jnp.exp(mu_eps + sigma_eps * eps_raw)
        gamma_i = jnp.clip(jnp.exp(mu_gamma + sigma_gamma * gamma_raw), 0.1, 3.0)
        numpyro.deterministic('c_effort', c_effort)
        numpyro.deterministic('c_death', c_death)
        numpyro.deterministic('epsilon', epsilon)
        numpyro.deterministic('gamma_i', gamma_i)

        ce_i = c_effort[subj_idx]
        cd_i = c_death[subj_idx]
        eps_i = epsilon[subj_idx]
        gam_i = gamma_i[subj_idx]

        T_w = jnp.power(T, gam_i)

        S_full = (1.0 - T_w) + eps_i * T_w * p_esc
        S_stop = 1.0 - T_w

        eu_H_full = S_full * 5 - (1 - S_full) * cd_i * 10 - ce_i * 0.81 * dist_H
        eu_H_stop = S_stop * 5 - (1 - S_stop) * cd_i * 10
        eu_H = jnp.maximum(eu_H_full, eu_H_stop)

        eu_L_full = S_full * 1 - (1 - S_full) * cd_i * 6 - ce_i * 0.16
        eu_L_stop = S_stop * 1 - (1 - S_stop) * cd_i * 6
        eu_L = jnp.maximum(eu_L_full, eu_L_stop)

        logit = jnp.clip((eu_H - eu_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T_w[:, None])
               + eps_i[:, None] * T_w[:, None] * p_esc
               * jax.nn.sigmoid((u_g - chosen_req[:, None]) / sigma_motor))
        eu_grid = (S_u * chosen_R[:, None]
                   - (1.0 - S_u) * cd_i[:, None] * (chosen_R[:, None] + 5.0)
                   - ce_i[:, None] * u_g ** 2 * chosen_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - chosen_req - chosen_offset
        numpyro.deterministic('excess_pred', excess_pred)

        N_T = subj_idx.shape[0]
        with numpyro.plate('trials', N_T):
            numpyro.sample('obs_choice', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)), obs=choice)
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v), obs=excess_cc)
    return model


# ── Model 5: EVC_linear_effort — linear instead of quadratic effort cost ─────

def make_model_linear_effort(N_S):
    def model(subj_idx, T, dist_H, choice=None, excess_cc=None,
              chosen_R=None, chosen_req=None, chosen_dist=None, chosen_offset=None):
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)
        mu_eps = numpyro.sample('mu_eps', dist.Normal(-0.5, 0.5))
        sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.3))
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))

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

        ce_i = c_effort[subj_idx]
        cd_i = c_death[subj_idx]
        eps_i = epsilon[subj_idx]

        T_w = jnp.power(T, gamma)

        S_full = (1.0 - T_w) + eps_i * T_w * p_esc
        S_stop = 1.0 - T_w

        # LINEAR effort cost for choice: c_effort * u * D (u=0.9 for heavy, 0.4 for light)
        eu_H_full = S_full * 5 - (1 - S_full) * cd_i * 10 - ce_i * 0.9 * dist_H
        eu_H_stop = S_stop * 5 - (1 - S_stop) * cd_i * 10
        eu_H = jnp.maximum(eu_H_full, eu_H_stop)

        eu_L_full = S_full * 1 - (1 - S_full) * cd_i * 6 - ce_i * 0.4
        eu_L_stop = S_stop * 1 - (1 - S_stop) * cd_i * 6
        eu_L = jnp.maximum(eu_L_full, eu_L_stop)

        logit = jnp.clip((eu_H - eu_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        # Vigor: LINEAR effort cost
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T_w[:, None])
               + eps_i[:, None] * T_w[:, None] * p_esc
               * jax.nn.sigmoid((u_g - chosen_req[:, None]) / sigma_motor))
        eu_grid = (S_u * chosen_R[:, None]
                   - (1.0 - S_u) * cd_i[:, None] * (chosen_R[:, None] + 5.0)
                   - ce_i[:, None] * u_g * chosen_dist[:, None])  # LINEAR: u not u^2
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - chosen_req - chosen_offset
        numpyro.deterministic('excess_pred', excess_pred)

        N_T = subj_idx.shape[0]
        with numpyro.plate('trials', N_T):
            numpyro.sample('obs_choice', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)), obs=choice)
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v), obs=excess_cc)
    return model


# ── Model 6: EVC_no_sigmoid — linear ramp instead of sigmoid survival ────────

def make_model_no_sigmoid(N_S):
    def model(subj_idx, T, dist_H, choice=None, excess_cc=None,
              chosen_R=None, chosen_req=None, chosen_dist=None, chosen_offset=None):
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        mu_eps = numpyro.sample('mu_eps', dist.Normal(-0.5, 0.5))
        sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.3))
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))

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

        ce_i = c_effort[subj_idx]
        cd_i = c_death[subj_idx]
        eps_i = epsilon[subj_idx]

        T_w = jnp.power(T, gamma)

        # Choice: use linear ramp for S_full
        # S(u) = (1 - T_w) + eps * T_w * p_esc * clip((u - req*0.5) / (req*0.5), 0, 1)
        # For full speed (u=req), ramp = clip((req - req*0.5)/(req*0.5), 0, 1) = 1.0
        S_full = (1.0 - T_w) + eps_i * T_w * p_esc  # ramp=1 at full speed
        S_stop = 1.0 - T_w

        eu_H_full = S_full * 5 - (1 - S_full) * cd_i * 10 - ce_i * 0.81 * dist_H
        eu_H_stop = S_stop * 5 - (1 - S_stop) * cd_i * 10
        eu_H = jnp.maximum(eu_H_full, eu_H_stop)

        eu_L_full = S_full * 1 - (1 - S_full) * cd_i * 6 - ce_i * 0.16
        eu_L_stop = S_stop * 1 - (1 - S_stop) * cd_i * 6
        eu_L = jnp.maximum(eu_L_full, eu_L_stop)

        logit = jnp.clip((eu_H - eu_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        # Vigor: linear ramp instead of sigmoid
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        req = chosen_req[:, None]
        ramp = jnp.clip((u_g - req * 0.5) / (req * 0.5), 0.0, 1.0)
        S_u = (1.0 - T_w[:, None]) + eps_i[:, None] * T_w[:, None] * p_esc * ramp

        eu_grid = (S_u * chosen_R[:, None]
                   - (1.0 - S_u) * cd_i[:, None] * (chosen_R[:, None] + 5.0)
                   - ce_i[:, None] * u_g ** 2 * chosen_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - chosen_req - chosen_offset
        numpyro.deterministic('excess_pred', excess_pred)

        N_T = subj_idx.shape[0]
        with numpyro.plate('trials', N_T):
            numpyro.sample('obs_choice', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)), obs=choice)
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v), obs=excess_cc)
    return model


# ── Shared fitting function ──────────────────────────────────────────────────

def fit_model(model_fn, data, model_name, n_steps=35000, lr=0.002, seed=42, print_every=5000):
    """Fit a model via SVI and return results."""
    model = model_fn(data['N_S'])

    kwargs = {k: data[k] for k in [
        'subj_idx', 'T', 'dist_H', 'choice', 'excess_cc',
        'chosen_R', 'chosen_req', 'chosen_dist', 'chosen_offset',
    ]}

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    print(f"\n{'='*60}")
    print(f"Fitting {model_name} (SVI, {n_steps} steps, lr={lr})")
    print(f"  {data['N_S']} subjects, {data['N_T']} trials")

    t0 = time.time()
    losses = []
    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        losses.append(float(loss))
        if (i + 1) % print_every == 0:
            elapsed = time.time() - t0
            print(f"  Step {i+1}: loss={loss:.1f} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. Final loss: {losses[-1]:.1f}")

    params_fit = svi.get_params(state)
    return {
        'params': params_fit,
        'losses': losses,
        'guide': guide,
        'model': model,
        'kwargs': kwargs,
    }


def evaluate_model(fit_result, data, model_name, n_subj_params, n_pop_params,
                   n_samples=500, seed=44):
    """Evaluate a fitted model: choice accuracy, vigor r², BIC, gamma."""
    guide = fit_result['guide']
    params_fit = fit_result['params']

    obs_kwargs = {k: v for k, v in fit_result['kwargs'].items()
                  if k not in ['choice', 'excess_cc']}
    pred = Predictive(fit_result['model'], guide=guide,
                      params=params_fit, num_samples=n_samples)
    samples = pred(random.PRNGKey(seed), **obs_kwargs)

    ep = np.array(samples['excess_pred']).mean(0)
    eo = np.array(data['excess_cc'])
    ch = np.array(data['choice'])

    # Vigor r²
    r_vigor, _ = pearsonr(ep, eo)
    r2_vigor = r_vigor ** 2

    # Choice accuracy: predicted p(H) from obs_choice samples
    choice_pred = np.array(samples['obs_choice']).mean(0)
    choice_acc = np.mean((choice_pred > 0.5) == ch)

    # Parameter count and BIC
    n_total = n_subj_params * data['N_S'] + n_pop_params
    final_loss = fit_result['losses'][-1]
    bic = 2 * final_loss + n_total * np.log(data['N_T'])

    # Gamma value (if present)
    gamma_val = None
    if 'gamma' in samples:
        gamma_val = float(np.array(samples['gamma']).mean())
    elif 'gamma_i' in samples:
        gamma_val = float(np.array(samples['gamma_i']).mean())

    result = {
        'model': model_name,
        'n_subject_params': n_subj_params,
        'n_pop_params': n_pop_params,
        'n_total_params': n_total,
        'elbo': final_loss,
        'bic': bic,
        'choice_accuracy': choice_acc,
        'vigor_r2': r2_vigor,
        'gamma_value': gamma_val if gamma_val is not None else np.nan,
    }

    print(f"\n  {model_name} results:")
    print(f"    ELBO: {final_loss:.1f}")
    print(f"    BIC: {bic:.1f}")
    print(f"    Choice accuracy: {choice_acc:.3f}")
    print(f"    Vigor r²: {r2_vigor:.3f}")
    if gamma_val is not None:
        print(f"    Gamma: {gamma_val:.3f}")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_PATH = 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950/behavior_rich.csv'

    print("Loading data...")
    data = prepare_data(DATA_PATH)
    print(f"  N_subjects={data['N_S']}, N_trials={data['N_T']}")
    print(f"  Heavy mean excess: {data['heavy_mean']:.4f}")
    print(f"  Light mean excess: {data['light_mean']:.4f}")

    # Model specifications: (name, make_fn, n_subj_params, n_pop_params)
    # n_pop_params counts: mu_ce, mu_cd, sigma_ce, sigma_cd, tau_raw, p_esc_raw,
    #                      sigma_v, sigma_motor_raw = 8 base
    #                      + mu_eps, sigma_eps = 2 if epsilon
    #                      + gamma_raw = 1 if gamma pop
    #                      + mu_gamma, sigma_gamma = 2 if gamma ind
    models = [
        ('EVC_base',          make_model_base,          2, 8),
        ('EVC_eps',           make_model_eps,           3, 10),
        ('EVC_gamma',         make_model_gamma,         3, 11),
        ('EVC_gamma_ind',     make_model_gamma_ind,     4, 12),
        ('EVC_linear_effort', make_model_linear_effort, 3, 11),
        ('EVC_no_sigmoid',    make_model_no_sigmoid,    3, 10),  # no sigma_motor
    ]

    results = []
    total_start = time.time()

    for name, make_fn, n_sp, n_pp in models:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {name}")
        print(f"{'#'*60}")

        fit_result = fit_model(make_fn, data, name, n_steps=35000, lr=0.002, seed=42)
        result = evaluate_model(fit_result, data, name, n_sp, n_pp)
        results.append(result)

    total_elapsed = time.time() - total_start

    # ── Summary table ────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('bic')

    print(f"\n\n{'='*90}")
    print(f"MODEL COMPARISON RESULTS (sorted by BIC)")
    print(f"{'='*90}")
    print(f"{'Model':<22s} {'Subj':>4s} {'Pop':>4s} {'Total':>6s} "
          f"{'ELBO':>10s} {'BIC':>10s} {'ChoiceAcc':>10s} {'VigorR2':>8s} {'Gamma':>7s}")
    print(f"{'-'*90}")

    best_bic = results_df['bic'].min()
    for _, row in results_df.iterrows():
        delta = row['bic'] - best_bic
        gamma_str = f"{row['gamma_value']:.3f}" if not np.isnan(row['gamma_value']) else "  ---"
        marker = " ***" if delta == 0 else f" (+{delta:.0f})" if delta < 100 else f" (+{delta:.0f})"
        print(f"{row['model']:<22s} {row['n_subject_params']:>4d} {row['n_pop_params']:>4d} "
              f"{row['n_total_params']:>6d} {row['elbo']:>10.1f} {row['bic']:>10.1f} "
              f"{row['choice_accuracy']:>10.3f} {row['vigor_r2']:>8.3f} {gamma_str:>7s}{marker}")

    print(f"\nTotal runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Save
    out_path = 'results/stats/evc_model_comparison.csv'
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
