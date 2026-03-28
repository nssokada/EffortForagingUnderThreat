"""
Full 6-model comparison pipeline for EVC model.

All models use the SAME 81-trial data for both likelihoods.
They differ only in model structure.

Models:
  FINAL - EVC 2+2 (full model)
  M1    - Effort Only (no survival/threat)
  M2    - Threat Only (no effort cost)
  M3    - Separate Choice + Vigor (no shared structure)
  M4    - Population ce (ce not per-subject)
  M5    - No gamma (gamma=1, no probability weighting)
  M6    - Standard u² cost (not LQR deviation)
"""

import sys
sys.path.insert(0, '.')
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
import time

jax.config.update('jax_enable_x64', True)

# Import data preparation from the 81-trial model
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "evc_final_81trials",
    "/workspace/scripts/modeling/evc_final_81trials.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
prepare_data = _mod.prepare_data


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def make_model_final(N_S, N_trials):
    """FINAL: EVC 2+2 — full model."""
    def model(ch_subj, ch_T, ch_R_H, ch_R_L, ch_req_H, ch_req_L,
              ch_D_H, ch_D_L, ch_choice, ch_is_choice,
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

        # CHOICE
        ce_ch = c_effort[ch_subj]
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc
        delta_reward = S_ch * (ch_R_H - ch_R_L)
        delta_effort = ce_ch * (ch_req_H**2 * ch_D_H - ch_req_L**2 * ch_D_L)
        delta_eu = delta_reward - delta_effort
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice_trials', N_trials):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                           obs=ch_choice)

        # VIGOR
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
                   - ce_vigor * deviation ** 2 * vig_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)
        with numpyro.plate('vigor_trials', N_trials):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)
    return model


def make_model_m1(N_S, N_trials):
    """M1: Effort Only — no survival/threat."""
    def model(ch_subj, ch_T, ch_R_H, ch_R_L, ch_req_H, ch_req_L,
              ch_D_H, ch_D_L, ch_choice, ch_is_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_excess, vig_offset):
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        ce_vigor_raw = numpyro.sample('ce_vigor_raw', dist.Normal(-3.0, 1.0))
        ce_vigor = numpyro.deterministic('ce_vigor', jnp.exp(ce_vigor_raw))

        with numpyro.plate('subjects', N_S):
            ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
        c_effort = jnp.exp(mu_ce + sigma_ce * ce_raw)
        numpyro.deterministic('c_effort', c_effort)

        # CHOICE: no S term
        ce_ch = c_effort[ch_subj]
        delta_reward = ch_R_H - ch_R_L
        delta_effort = ce_ch * (ch_req_H**2 * ch_D_H - ch_req_L**2 * ch_D_L)
        delta_eu = delta_reward - delta_effort
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice_trials', N_trials):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                           obs=ch_choice)

        # VIGOR: EU(u) = R - ce_vigor × (u-req)² × D — no survival
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        deviation = u_g - vig_req[:, None]
        eu_grid = (vig_R[:, None]
                   - ce_vigor * deviation ** 2 * vig_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)
        with numpyro.plate('vigor_trials', N_trials):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)
    return model


def make_model_m2(N_S, N_trials):
    """M2: Threat Only — no effort cost."""
    def model(ch_subj, ch_T, ch_R_H, ch_R_L, ch_req_H, ch_req_L,
              ch_D_H, ch_D_L, ch_choice, ch_is_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_excess, vig_offset):
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
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

        with numpyro.plate('subjects', N_S):
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))
        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        numpyro.deterministic('c_death', c_death)

        # CHOICE: S × (R_H - R_L) — no effort
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc
        delta_eu = S_ch * (ch_R_H - ch_R_L)
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice_trials', N_trials):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                           obs=ch_choice)

        # VIGOR: S(u)×R - (1-S(u))×cd×(R+C) — no effort cost in vigor
        cd_v = c_death[vig_subj]
        T_w_v = jnp.power(vig_T, gamma)
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T_w_v[:, None])
               + epsilon * T_w_v[:, None] * p_esc
               * jax.nn.sigmoid((u_g - vig_req[:, None]) / sigma_motor))
        eu_grid = (S_u * vig_R[:, None]
                   - (1.0 - S_u) * cd_v[:, None] * (vig_R[:, None] + 5.0))
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)
        with numpyro.plate('vigor_trials', N_trials):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)
    return model


def make_model_m3(N_S, N_trials):
    """M3: Separate Choice + Vigor — no shared structure.
    Choice: standard EVC. Vigor: linear regression on danger."""
    def model(ch_subj, ch_T, ch_R_H, ch_R_L, ch_req_H, ch_req_L,
              ch_D_H, ch_D_L, ch_choice, ch_is_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_excess, vig_offset):
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        eps_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
        epsilon = numpyro.deterministic('epsilon', jnp.exp(eps_raw))
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))

        mu_alpha = numpyro.sample('mu_alpha', dist.Normal(0.0, 0.5))
        mu_delta = numpyro.sample('mu_delta', dist.Normal(0.0, 0.5))
        sigma_alpha = numpyro.sample('sigma_alpha', dist.HalfNormal(0.3))
        sigma_delta = numpyro.sample('sigma_delta', dist.HalfNormal(0.3))

        with numpyro.plate('subjects', N_S):
            ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
            alpha_raw = numpyro.sample('alpha_raw', dist.Normal(0.0, 1.0))
            delta_raw = numpyro.sample('delta_raw', dist.Normal(0.0, 1.0))

        c_effort = jnp.exp(mu_ce + sigma_ce * ce_raw)
        alpha = mu_alpha + sigma_alpha * alpha_raw
        delta = mu_delta + sigma_delta * delta_raw
        numpyro.deterministic('c_effort', c_effort)
        numpyro.deterministic('alpha', alpha)
        numpyro.deterministic('delta', delta)

        # CHOICE: same as FINAL
        ce_ch = c_effort[ch_subj]
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc
        delta_reward = S_ch * (ch_R_H - ch_R_L)
        delta_effort = ce_ch * (ch_req_H**2 * ch_D_H - ch_req_L**2 * ch_D_L)
        delta_eu_val = delta_reward - delta_effort
        logit = jnp.clip(delta_eu_val / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice_trials', N_trials):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                           obs=ch_choice)

        # VIGOR: excess = alpha + delta × (1-S) — linear, NOT EU optimization
        T_w_v = jnp.power(vig_T, gamma)
        S_v = (1.0 - T_w_v) + epsilon * T_w_v * p_esc
        danger = 1.0 - S_v
        excess_pred = alpha[vig_subj] + delta[vig_subj] * danger - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)
        with numpyro.plate('vigor_trials', N_trials):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)
    return model


def make_model_m4(N_S, N_trials):
    """M4: Population ce — ce is not per-subject."""
    def model(ch_subj, ch_T, ch_R_H, ch_R_L, ch_req_H, ch_req_L,
              ch_D_H, ch_D_L, ch_choice, ch_is_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_excess, vig_offset):
        ce_pop_raw = numpyro.sample('ce_pop_raw', dist.Normal(0.0, 1.0))
        ce_pop = numpyro.deterministic('ce_pop', jnp.exp(ce_pop_raw))

        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
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
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))
        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        numpyro.deterministic('c_death', c_death)

        # CHOICE: population ce
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc
        delta_reward = S_ch * (ch_R_H - ch_R_L)
        delta_effort = ce_pop * (ch_req_H**2 * ch_D_H - ch_req_L**2 * ch_D_L)
        delta_eu = delta_reward - delta_effort
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice_trials', N_trials):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                           obs=ch_choice)

        # VIGOR: same as FINAL
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
                   - ce_vigor * deviation ** 2 * vig_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)
        with numpyro.plate('vigor_trials', N_trials):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)
    return model


def make_model_m5(N_S, N_trials):
    """M5: No gamma — S = (1-T) + eps×T×p_esc."""
    def model(ch_subj, ch_T, ch_R_H, ch_R_L, ch_req_H, ch_req_L,
              ch_D_H, ch_D_L, ch_choice, ch_is_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_excess, vig_offset):
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        eps_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
        epsilon = numpyro.deterministic('epsilon', jnp.exp(eps_raw))

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

        # CHOICE: no gamma
        ce_ch = c_effort[ch_subj]
        S_ch = (1.0 - ch_T) + epsilon * ch_T * p_esc
        delta_reward = S_ch * (ch_R_H - ch_R_L)
        delta_effort = ce_ch * (ch_req_H**2 * ch_D_H - ch_req_L**2 * ch_D_L)
        delta_eu = delta_reward - delta_effort
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice_trials', N_trials):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                           obs=ch_choice)

        # VIGOR: no gamma
        cd_v = c_death[vig_subj]
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - vig_T[:, None])
               + epsilon * vig_T[:, None] * p_esc
               * jax.nn.sigmoid((u_g - vig_req[:, None]) / sigma_motor))
        deviation = u_g - vig_req[:, None]
        eu_grid = (S_u * vig_R[:, None]
                   - (1.0 - S_u) * cd_v[:, None] * (vig_R[:, None] + 5.0)
                   - ce_vigor * deviation ** 2 * vig_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)
        with numpyro.plate('vigor_trials', N_trials):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)
    return model


def make_model_m6(N_S, N_trials):
    """M6: Standard u² cost (not LQR deviation)."""
    def model(ch_subj, ch_T, ch_R_H, ch_R_L, ch_req_H, ch_req_L,
              ch_D_H, ch_D_L, ch_choice, ch_is_choice,
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

        # CHOICE: same as FINAL
        ce_ch = c_effort[ch_subj]
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc
        delta_reward = S_ch * (ch_R_H - ch_R_L)
        delta_effort = ce_ch * (ch_req_H**2 * ch_D_H - ch_req_L**2 * ch_D_L)
        delta_eu = delta_reward - delta_effort
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice_trials', N_trials):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                           obs=ch_choice)

        # VIGOR: u² × D instead of (u-req)² × D
        cd_v = c_death[vig_subj]
        T_w_v = jnp.power(vig_T, gamma)
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T_w_v[:, None])
               + epsilon * T_w_v[:, None] * p_esc
               * jax.nn.sigmoid((u_g - vig_req[:, None]) / sigma_motor))
        eu_grid = (S_u * vig_R[:, None]
                   - (1.0 - S_u) * cd_v[:, None] * (vig_R[:, None] + 5.0)
                   - ce_vigor * u_g ** 2 * vig_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)
        with numpyro.plate('vigor_trials', N_trials):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v),
                           obs=vig_excess)
    return model


# ============================================================
# GENERIC FIT + EVALUATE
# ============================================================

MODEL_SPECS = {
    'FINAL': {
        'make': make_model_final,
        'n_subj_params': 2,  # ce, cd
        'n_pop_params': 11,  # mu_ce, sigma_ce, mu_cd, sigma_cd, eps, gamma, tau, p_esc, sigma_motor, ce_vigor, sigma_v
        'description': 'EVC 2+2 (full model)',
    },
    'M1': {
        'make': make_model_m1,
        'n_subj_params': 1,  # ce only
        'n_pop_params': 5,   # mu_ce, sigma_ce, tau, ce_vigor, sigma_v
        'description': 'Effort Only',
    },
    'M2': {
        'make': make_model_m2,
        'n_subj_params': 1,  # cd only
        'n_pop_params': 8,   # mu_cd, sigma_cd, eps, gamma, tau, p_esc, sigma_motor, sigma_v
        'description': 'Threat Only',
    },
    'M3': {
        'make': make_model_m3,
        'n_subj_params': 3,  # ce, alpha, delta
        'n_pop_params': 11,  # mu_ce, sigma_ce, eps, gamma, tau, p_esc, mu_alpha, sigma_alpha, mu_delta, sigma_delta, sigma_v
        'description': 'Separate Choice + Vigor',
    },
    'M4': {
        'make': make_model_m4,
        'n_subj_params': 1,  # cd only
        'n_pop_params': 10,  # ce_pop, mu_cd, sigma_cd, eps, gamma, tau, p_esc, sigma_motor, ce_vigor, sigma_v
        'description': 'Population ce',
    },
    'M5': {
        'make': make_model_m5,
        'n_subj_params': 2,  # ce, cd
        'n_pop_params': 10,  # mu_ce, sigma_ce, mu_cd, sigma_cd, eps, tau, p_esc, sigma_motor, ce_vigor, sigma_v
        'description': 'No gamma (gamma=1)',
    },
    'M6': {
        'make': make_model_m6,
        'n_subj_params': 2,  # ce, cd
        'n_pop_params': 11,  # same as FINAL
        'description': 'Standard u² cost',
    },
}


def fit_model(name, data, n_steps=35000, lr=0.002, seed=42, print_every=5000):
    """Fit a named model."""
    spec = MODEL_SPECS[name]
    model_fn = spec['make'](data['N_S'], data['N_trials'])

    kwargs = {k: data[k] for k in [
        'ch_subj', 'ch_T', 'ch_R_H', 'ch_R_L',
        'ch_req_H', 'ch_req_L', 'ch_D_H', 'ch_D_L',
        'ch_choice', 'ch_is_choice',
        'vig_subj', 'vig_T', 'vig_R', 'vig_req', 'vig_dist',
        'vig_excess', 'vig_offset',
    ]}

    guide = AutoNormal(model_fn)
    svi = SVI(model_fn, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    print(f"\n{'=' * 60}")
    print(f"Fitting {name}: {spec['description']}")
    print(f"  SVI, {n_steps} steps, lr={lr}")
    print(f"  {data['N_S']} subjects, {spec['n_subj_params']} per-subj params")
    print(f"  {data['N_trials']} trials for both likelihoods")
    print(f"{'=' * 60}")

    losses = []
    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        losses.append(float(loss))
        if (i + 1) % print_every == 0:
            print(f"  Step {i + 1}: loss={loss:.1f}")

    params_fit = svi.get_params(state)

    return {
        'name': name,
        'params': params_fit, 'losses': losses,
        'guide': guide, 'model': model_fn, 'kwargs': kwargs,
        'data': data, 'spec': spec,
    }


def evaluate_model(fit_result, n_samples=500, seed=44):
    """Evaluate a fitted model: BIC, choice r², vigor r²."""
    name = fit_result['name']
    guide = fit_result['guide']
    model = fit_result['model']
    params_fit = fit_result['params']
    data = fit_result['data']
    kwargs = fit_result['kwargs']
    spec = fit_result['spec']

    # Determine return sites
    return_sites = ['excess_pred', 'tau_raw', 'sigma_v']
    if name in ('FINAL', 'M1', 'M3', 'M5', 'M6'):
        return_sites.append('c_effort')
    if name in ('FINAL', 'M2', 'M4', 'M5', 'M6'):
        return_sites.append('c_death')
    if name in ('FINAL', 'M2', 'M3', 'M4', 'M5', 'M6'):
        return_sites.append('epsilon')
    if name in ('FINAL', 'M2', 'M3', 'M4', 'M6'):
        return_sites.append('gamma')
    if name in ('FINAL', 'M1', 'M4', 'M5', 'M6'):
        return_sites.append('ce_vigor')
    if name in ('FINAL', 'M2', 'M3', 'M4', 'M5', 'M6'):
        return_sites.append('p_esc_raw')
    if name in ('FINAL', 'M2', 'M4', 'M5', 'M6'):
        return_sites.append('sigma_motor_raw')
    if name == 'M4':
        return_sites.append('ce_pop')
    if name == 'M3':
        return_sites.extend(['alpha', 'delta'])

    pred = Predictive(
        model, guide=guide, params=params_fit,
        num_samples=n_samples,
        return_sites=return_sites)
    samples = pred(random.PRNGKey(seed), **kwargs)

    ep = np.array(samples['excess_pred']).mean(0)

    # Vigor r²
    vig_excess_np = np.array(data['vig_excess'])
    r_vigor, _ = pearsonr(ep, vig_excess_np)

    # Per-subject vigor r²
    vig_subj_np = np.array(data['vig_subj'])
    vig_df = pd.DataFrame({
        'subj': vig_subj_np, 'obs': vig_excess_np, 'pred': ep
    })
    subj_vigor = vig_df.groupby('subj').agg(
        obs_mean=('obs', 'mean'), pred_mean=('pred', 'mean')
    ).reset_index()
    r_vigor_subj, _ = pearsonr(subj_vigor['obs_mean'], subj_vigor['pred_mean'])

    # BIC
    n_params = spec['n_subj_params'] * data['N_S'] + spec['n_pop_params']
    N_total = 2 * data['N_trials']
    final_loss = fit_result['losses'][-1]
    bic = 2 * final_loss + n_params * np.log(N_total)

    # Choice predictions for real choice trials
    ch_is_choice_np = np.array(data['ch_is_choice'])
    ch_choice_np = np.array(data['ch_choice'])
    ch_subj_np = np.array(data['ch_subj'])
    ch_T_np = np.array(data['ch_T'])
    ch_R_H_np = np.array(data['ch_R_H'])
    ch_R_L_np = np.array(data['ch_R_L'])
    ch_req_H_np = np.array(data['ch_req_H'])
    ch_req_L_np = np.array(data['ch_req_L'])
    ch_D_H_np = np.array(data['ch_D_H'])
    ch_D_L_np = np.array(data['ch_D_L'])
    tau_val = float(np.exp(np.array(samples['tau_raw']).mean()))

    # Model-specific choice prediction
    if name in ('FINAL', 'M5', 'M6'):
        ce = np.array(samples['c_effort']).mean(0)
        eps_val = float(np.array(samples['epsilon']).mean())
        if name == 'M5':
            T_w = ch_T_np
        else:
            gamma_val = float(np.array(samples['gamma']).mean())
            T_w = ch_T_np ** gamma_val
        p_esc_val = float(1.0 / (1.0 + np.exp(-np.array(samples['p_esc_raw']).mean())))
        S_ch = (1.0 - T_w) + eps_val * T_w * p_esc_val
        delta_reward = S_ch * (ch_R_H_np - ch_R_L_np)
        delta_effort = ce[ch_subj_np] * (ch_req_H_np**2 * ch_D_H_np - ch_req_L_np**2 * ch_D_L_np)
        delta_eu = delta_reward - delta_effort
    elif name == 'M1':
        ce = np.array(samples['c_effort']).mean(0)
        delta_reward = ch_R_H_np - ch_R_L_np
        delta_effort = ce[ch_subj_np] * (ch_req_H_np**2 * ch_D_H_np - ch_req_L_np**2 * ch_D_L_np)
        delta_eu = delta_reward - delta_effort
    elif name == 'M2':
        eps_val = float(np.array(samples['epsilon']).mean())
        gamma_val = float(np.array(samples['gamma']).mean())
        p_esc_val = float(1.0 / (1.0 + np.exp(-np.array(samples['p_esc_raw']).mean())))
        T_w = ch_T_np ** gamma_val
        S_ch = (1.0 - T_w) + eps_val * T_w * p_esc_val
        delta_eu = S_ch * (ch_R_H_np - ch_R_L_np)
    elif name == 'M3':
        ce = np.array(samples['c_effort']).mean(0)
        eps_val = float(np.array(samples['epsilon']).mean())
        gamma_val = float(np.array(samples['gamma']).mean())
        p_esc_val = float(1.0 / (1.0 + np.exp(-np.array(samples['p_esc_raw']).mean())))
        T_w = ch_T_np ** gamma_val
        S_ch = (1.0 - T_w) + eps_val * T_w * p_esc_val
        delta_reward = S_ch * (ch_R_H_np - ch_R_L_np)
        delta_effort = ce[ch_subj_np] * (ch_req_H_np**2 * ch_D_H_np - ch_req_L_np**2 * ch_D_L_np)
        delta_eu = delta_reward - delta_effort
    elif name == 'M4':
        eps_val = float(np.array(samples['epsilon']).mean())
        gamma_val = float(np.array(samples['gamma']).mean())
        p_esc_val = float(1.0 / (1.0 + np.exp(-np.array(samples['p_esc_raw']).mean())))
        T_w = ch_T_np ** gamma_val
        S_ch = (1.0 - T_w) + eps_val * T_w * p_esc_val
        delta_reward = S_ch * (ch_R_H_np - ch_R_L_np)
        ce_pop = float(np.array(samples['ce_pop']).mean())
        delta_effort = ce_pop * (ch_req_H_np**2 * ch_D_H_np - ch_req_L_np**2 * ch_D_L_np)
        delta_eu = delta_reward - delta_effort

    logit_ch = np.clip(delta_eu / tau_val, -20, 20)
    p_H = expit(logit_ch)

    real_mask = ch_is_choice_np == 1
    p_H_real = p_H[real_mask]
    choice_real = ch_choice_np[real_mask]
    subj_real = ch_subj_np[real_mask]

    choice_acc = ((p_H_real >= 0.5).astype(int) == choice_real).mean()

    ch_df = pd.DataFrame({
        'subj': subj_real, 'choice': choice_real, 'p_H': p_H_real
    })
    subj_choice = ch_df.groupby('subj').agg(
        obs_pH=('choice', 'mean'), pred_pH=('p_H', 'mean')
    ).reset_index()
    r_choice_subj, _ = pearsonr(subj_choice['obs_pH'], subj_choice['pred_pH'])

    print(f"\n  {name} Results:")
    print(f"    BIC: {bic:.0f}")
    print(f"    Final ELBO loss: {final_loss:.1f}")
    print(f"    Choice accuracy (45 real): {choice_acc:.3f}")
    print(f"    Per-subj choice r²: {r_choice_subj**2:.3f}")
    print(f"    Trial-level vigor r²: {r_vigor**2:.3f}")
    print(f"    Per-subj vigor r²: {r_vigor_subj**2:.3f}")
    print(f"    n_params: {n_params}")

    return {
        'name': name,
        'description': spec['description'],
        'bic': bic,
        'final_loss': final_loss,
        'choice_acc': choice_acc,
        'r_choice_subj': r_choice_subj,
        'r_vigor': r_vigor,
        'r_vigor_subj': r_vigor_subj,
        'n_subj_params': spec['n_subj_params'],
        'n_params': n_params,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    t0 = time.time()

    DATA_DIR = ('data/exploratory_350/processed/'
                'stage5_filtered_data_20260320_191950')

    print("Loading data...")
    data = prepare_data(
        f'{DATA_DIR}/behavior_rich.csv',
        psych_path=f'{DATA_DIR}/psych.csv',
    )
    print(f"N_subjects={data['N_S']}, N_trials={data['N_trials']}")

    results = []

    # Fit FINAL with 40k steps
    print("\n" + "=" * 70)
    print("STEP 1: Fitting FINAL model (40k steps)")
    print("=" * 70)
    fit_final = fit_model('FINAL', data, n_steps=40000, lr=0.002)
    eval_final = evaluate_model(fit_final)
    results.append(eval_final)

    # Save FINAL params separately
    evaluate_full = _mod.evaluate
    full_eval = evaluate_full(fit_final)
    full_eval['param_df'].to_csv('results/stats/oc_evc_final_81_params.csv', index=False)
    full_eval['pop_df'].to_csv('results/stats/oc_evc_final_81_population.csv', index=False)
    print("Saved FINAL params to results/stats/oc_evc_final_81_*.csv")

    # Fit comparison models (35k steps each)
    print("\n" + "=" * 70)
    print("STEP 2: Fitting comparison models (35k steps each)")
    print("=" * 70)
    for model_name in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
        try:
            # M2 needs lower lr to avoid divergence
            lr = 0.001 if model_name == 'M2' else 0.002
            fit_result = fit_model(model_name, data, n_steps=35000, lr=lr)
            # Check for NaN loss
            if np.isnan(fit_result['losses'][-1]):
                print(f"  WARNING: {model_name} diverged (NaN loss), retrying with lr=0.0005")
                fit_result = fit_model(model_name, data, n_steps=35000, lr=0.0005, seed=123)
            eval_result = evaluate_model(fit_result)
            results.append(eval_result)
        except Exception as e:
            print(f"  ERROR fitting/evaluating {model_name}: {e}")
            results.append({
                'name': model_name,
                'description': MODEL_SPECS[model_name]['description'],
                'bic': np.nan, 'final_loss': np.nan,
                'choice_acc': np.nan, 'r_choice_subj': np.nan,
                'r_vigor': np.nan, 'r_vigor_subj': np.nan,
                'n_subj_params': MODEL_SPECS[model_name]['n_subj_params'],
                'n_params': np.nan,
            })

    # STEP 3: Compile comparison table
    print("\n" + "=" * 70)
    print("STEP 3: Model Comparison Results")
    print("=" * 70)

    df = pd.DataFrame(results)
    final_bic = df[df['name'] == 'FINAL']['bic'].values[0]
    df['delta_bic'] = df['bic'] - final_bic
    df['choice_r2'] = df['r_choice_subj'] ** 2
    df['vigor_r2'] = df['r_vigor'] ** 2
    df['subj_vigor_r2'] = df['r_vigor_subj'] ** 2

    df = df[['name', 'description', 'n_subj_params', 'n_params',
             'bic', 'delta_bic', 'final_loss',
             'choice_acc', 'choice_r2', 'vigor_r2', 'subj_vigor_r2']]

    print("\n" + df.to_string(index=False))

    out_path = 'results/stats/evc_model_comparison_final.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # STEP 4: Compare with original fit
    print("\n" + "=" * 70)
    print("STEP 4: Comparison with original 45-choice-trial fit")
    print("=" * 70)
    try:
        orig_params = pd.read_csv('results/stats/oc_evc_final_params.csv')
        new_params = pd.read_csv('results/stats/oc_evc_final_81_params.csv')

        if 'c_effort' in orig_params.columns and 'c_effort' in new_params.columns:
            merged = orig_params.merge(new_params, on='subj', suffixes=('_orig', '_new'))
            r_ce, _ = pearsonr(merged['c_effort_orig'], merged['c_effort_new'])
            r_cd, _ = pearsonr(merged['c_death_orig'], merged['c_death_new'])
            print(f"  ce correlation (orig vs 81-trial): r={r_ce:.4f}")
            print(f"  cd correlation (orig vs 81-trial): r={r_cd:.4f}")
            print(f"  ce mean orig={merged['c_effort_orig'].mean():.4f}, new={merged['c_effort_new'].mean():.4f}")
            print(f"  cd mean orig={merged['c_death_orig'].mean():.4f}, new={merged['c_death_new'].mean():.4f}")

        orig_pop = pd.read_csv('results/stats/oc_evc_final_population.csv')
        new_pop = pd.read_csv('results/stats/oc_evc_final_81_population.csv')
        print(f"\n  Population params (orig -> new):")
        for col in orig_pop.columns:
            if col in new_pop.columns:
                print(f"    {col}: {orig_pop[col].values[0]:.4f} -> {new_pop[col].values[0]:.4f}")
    except Exception as e:
        print(f"  Could not compare with original: {e}")

    elapsed = time.time() - t0
    print(f"\nTotal pipeline time: {elapsed/60:.1f} min")
