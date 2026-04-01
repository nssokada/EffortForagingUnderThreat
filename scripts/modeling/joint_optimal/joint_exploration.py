"""
Joint Optimal Control Models: Bednekoff/Brown-style unified fitness optimization.

The core idea: a single fitness function W_j(u) jointly determines:
  1. Which cookie to pick (compare V_H = max_u W_H(u) vs V_L = max_u W_L(u))
  2. How hard to press (u* = argmax_u W_chosen(u))

W_j(u) = S_j(u)·R_j - (1 - S_j(u))·ω·(R_j + C) - κ·(u - req_j)²·D_j

where:
  S_j(u) = survival probability (depends on pressing rate, threat, distance)
  ω = subjective cost of capture (per subject)
  κ = effort cost sensitivity (per subject)
  R_j, req_j, D_j = reward, required rate, distance for cookie j

This is truly joint: the SAME ω and κ that drive vigor also drive choice,
through the SAME objective function. Unlike M5 where λ only affects choice
and ω only affects vigor.

Task structure:
  - Heavy: R=5, req=0.9, D_H ∈ {1,2,3}
  - Light: R=1, req=0.4, D_L = 1 (always)
  - Threat T ∈ {0.1, 0.5, 0.9}
  - Capture penalty C = 5

Multiple variants tested:
  V1: Basic joint model (ω, κ per subject)
  V2: V1 + per-subject baseline vigor (some people just press harder)
  V3: V1 + separate choice/vigor noise scaling
  V4: Simplified S (no sigmoid escape, just exponential)
  V5: κ population-level only (test if individual κ is needed)
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
C_PENALTY = 5.0  # capture penalty


# ============================================================
# Data preparation
# ============================================================

def prepare_data():
    """Load choice + vigor data for the joint model.

    Choice trials: need both cookies' properties for comparison.
    All trials: need chosen cookie's properties for vigor prediction.
    """
    beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)]

    # Choice data (type=1)
    choice_df = beh[beh['type'] == 1].copy()

    # Vigor data: load precomputed or parse
    vigor_path = DATA_DIR / "trial_vigor.csv"
    if vigor_path.exists():
        vigor_df = pd.read_csv(vigor_path)
        vigor_df = vigor_df[~vigor_df['subj'].isin(EXCLUDE)]
        vigor_df = vigor_df.dropna(subset=['median_rate']).copy()
    else:
        raise FileNotFoundError("Run precompute_vigor.py first")

    subjects = sorted(
        set(choice_df['subj'].unique()) & set(vigor_df['subj'].unique()))
    si = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)

    # Choice arrays
    ch_subj = jnp.array([si[s] for s in choice_df['subj']])
    ch_T = jnp.array(choice_df['threat'].values)
    ch_D_H = jnp.array(choice_df['distance_H'].values, dtype=jnp.float64)
    ch_D_L = jnp.ones_like(ch_D_H)  # light always at D=1
    ch_choice = jnp.array(choice_df['choice'].values)  # 1=heavy, 0=light

    # Vigor arrays (all trials)
    vig_subj = jnp.array([si[s] for s in vigor_df['subj']])
    vig_T = jnp.array(vigor_df['threat'].values)
    vig_R = jnp.array(vigor_df['actual_R'].values)
    vig_req = jnp.array(vigor_df['actual_req'].values)
    vig_dist = jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64)
    vig_rate = jnp.array(vigor_df['median_rate'].values)
    vig_cookie = jnp.array(vigor_df['is_heavy'].values, dtype=jnp.float64)

    data = {
        'ch_subj': ch_subj, 'ch_T': ch_T, 'ch_D_H': ch_D_H, 'ch_D_L': ch_D_L,
        'ch_choice': ch_choice,
        'vig_subj': vig_subj, 'vig_T': vig_T, 'vig_R': vig_R,
        'vig_req': vig_req, 'vig_dist': vig_dist, 'vig_rate': vig_rate,
        'vig_cookie': vig_cookie,
        'subjects': subjects, 'N_S': N_S,
        'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
    }
    print(f"  {N_S} subjects, {data['N_choice']} choice, {data['N_vigor']} vigor trials")
    return data


# ============================================================
# Core: compute W(u) and find optimal u*, V*
# ============================================================

def compute_fitness_grid(omega, kappa, T, D, R, req, gamma, epsilon, p_esc,
                         sigma_motor, u_grid):
    """Compute W(u) on a grid for given parameters.

    W(u) = S(u)·R - (1-S(u))·ω·(R+C) - κ·(u-req)²·D

    Returns: u_star (optimal rate), V_star (maximized fitness)
    """
    u_g = u_grid[None, :]  # (1, n_grid)

    # Survival: base survival + escape probability if pressing above req
    T_w = jnp.power(T[:, None], gamma)
    S = ((1.0 - T_w)
         + epsilon * T_w * p_esc
         * jax.nn.sigmoid((u_g - req[:, None]) / sigma_motor))

    # Fitness
    W = (S * R[:, None]
         - (1.0 - S) * omega[:, None] * (R[:, None] + C_PENALTY)
         - kappa[:, None] * (u_g - req[:, None]) ** 2 * D[:, None])

    # Soft argmax (temperature controls sharpness)
    softmax_temp = 20.0
    weights = jax.nn.softmax(W * softmax_temp, axis=1)
    u_star = jnp.sum(weights * u_g, axis=1)
    V_star = jnp.sum(weights * W, axis=1)

    return u_star, V_star


# ============================================================
# V1: Basic joint model (ω, κ per subject)
# ============================================================

def make_joint_v1(N_S, N_choice, N_vigor):
    """V1: Bednekoff joint optimization.
    Per-subject: ω (capture cost), κ (effort cost)
    Population: γ, ε, p_esc, σ_motor, τ, σ_v
    """
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_rate, vig_cookie):

        # Population params
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        eps_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
        epsilon = numpyro.deterministic('epsilon', jnp.exp(eps_raw))
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sm_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sm_raw), 0.01, 1.0)

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))

        # Per-subject: ω (capture cost) and κ (effort cost)
        mu_om = numpyro.sample('mu_om', dist.Normal(0.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(0.5))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-1.0, 1.0))
        sigma_kap = numpyro.sample('sigma_kap', dist.HalfNormal(0.5))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))

        omega = jnp.exp(mu_om + sigma_om * om_raw)
        kappa = jnp.exp(mu_kap + sigma_kap * kap_raw)
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)

        u_grid = jnp.linspace(0.1, 1.5, 30)

        # ---- CHOICE: compare W_H(u*_H) vs W_L(u*_L) ----
        R_H = jnp.full(N_choice, 5.0)
        R_L = jnp.full(N_choice, 1.0)
        req_H = jnp.full(N_choice, 0.9)
        req_L = jnp.full(N_choice, 0.4)

        _, V_H = compute_fitness_grid(
            omega[ch_subj], kappa[ch_subj], ch_T, ch_D_H,
            R_H, req_H, gamma, epsilon, p_esc, sigma_motor, u_grid)
        _, V_L = compute_fitness_grid(
            omega[ch_subj], kappa[ch_subj], ch_T, ch_D_L,
            R_L, req_L, gamma, epsilon, p_esc, sigma_motor, u_grid)

        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_heavy = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_heavy, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # ---- VIGOR: predict u* for the chosen cookie ----
        u_star, _ = compute_fitness_grid(
            omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
            vig_R, vig_req, gamma, epsilon, p_esc, sigma_motor, u_grid)
        numpyro.deterministic('rate_pred', u_star)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(u_star, sigma_v),
                           obs=vig_rate)

    return model


# ============================================================
# V2: Joint model + baseline vigor intercept per cookie type
# ============================================================

def make_joint_v2(N_S, N_choice, N_vigor):
    """V2: V1 + cookie intercept in vigor (like M5).
    The EU model may not perfectly predict the mean rate per cookie type.
    """
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_rate, vig_cookie):

        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        eps_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
        epsilon = numpyro.deterministic('epsilon', jnp.exp(eps_raw))
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sm_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sm_raw), 0.01, 1.0)

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        mu_om = numpyro.sample('mu_om', dist.Normal(0.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(0.5))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-1.0, 1.0))
        sigma_kap = numpyro.sample('sigma_kap', dist.HalfNormal(0.5))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))

        omega = jnp.exp(mu_om + sigma_om * om_raw)
        kappa = jnp.exp(mu_kap + sigma_kap * kap_raw)
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)

        u_grid = jnp.linspace(0.1, 1.5, 30)

        # Choice
        R_H = jnp.full(N_choice, 5.0)
        R_L = jnp.full(N_choice, 1.0)
        req_H = jnp.full(N_choice, 0.9)
        req_L = jnp.full(N_choice, 0.4)

        _, V_H = compute_fitness_grid(
            omega[ch_subj], kappa[ch_subj], ch_T, ch_D_H,
            R_H, req_H, gamma, epsilon, p_esc, sigma_motor, u_grid)
        _, V_L = compute_fitness_grid(
            omega[ch_subj], kappa[ch_subj], ch_T, ch_D_L,
            R_L, req_L, gamma, epsilon, p_esc, sigma_motor, u_grid)

        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_heavy = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_heavy, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # Vigor with cookie intercept
        u_star, _ = compute_fitness_grid(
            omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
            vig_R, vig_req, gamma, epsilon, p_esc, sigma_motor, u_grid)
        rate_pred = u_star + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(rate_pred, sigma_v),
                           obs=vig_rate)

    return model


# ============================================================
# V3: Joint model + per-subject vigor intercept (baseline motor speed)
# ============================================================

def make_joint_v3(N_S, N_choice, N_vigor):
    """V3: V2 + per-subject vigor intercept.
    Some people just press faster in general, independent of optimization.
    This intercept does NOT enter choice (it's purely motor).
    """
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_rate, vig_cookie):

        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        eps_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
        epsilon = numpyro.deterministic('epsilon', jnp.exp(eps_raw))
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sm_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sm_raw), 0.01, 1.0)

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        mu_om = numpyro.sample('mu_om', dist.Normal(0.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(0.5))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-1.0, 1.0))
        sigma_kap = numpyro.sample('sigma_kap', dist.HalfNormal(0.5))
        mu_base = numpyro.sample('mu_base', dist.Normal(0.0, 0.3))
        sigma_base = numpyro.sample('sigma_base', dist.HalfNormal(0.2))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
            base_raw = numpyro.sample('base_raw', dist.Normal(0.0, 1.0))

        omega = jnp.exp(mu_om + sigma_om * om_raw)
        kappa = jnp.exp(mu_kap + sigma_kap * kap_raw)
        baseline = mu_base + sigma_base * base_raw  # additive vigor offset
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)
        numpyro.deterministic('baseline', baseline)

        u_grid = jnp.linspace(0.1, 1.5, 30)

        # Choice (no baseline — choice is purely from W comparison)
        R_H = jnp.full(N_choice, 5.0)
        R_L = jnp.full(N_choice, 1.0)
        req_H = jnp.full(N_choice, 0.9)
        req_L = jnp.full(N_choice, 0.4)

        _, V_H = compute_fitness_grid(
            omega[ch_subj], kappa[ch_subj], ch_T, ch_D_H,
            R_H, req_H, gamma, epsilon, p_esc, sigma_motor, u_grid)
        _, V_L = compute_fitness_grid(
            omega[ch_subj], kappa[ch_subj], ch_T, ch_D_L,
            R_L, req_L, gamma, epsilon, p_esc, sigma_motor, u_grid)

        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_heavy = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_heavy, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # Vigor: u* + baseline + cookie intercept
        u_star, _ = compute_fitness_grid(
            omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
            vig_R, vig_req, gamma, epsilon, p_esc, sigma_motor, u_grid)
        rate_pred = u_star + baseline[vig_subj] + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(rate_pred, sigma_v),
                           obs=vig_rate)

    return model


# ============================================================
# V4: Simpler survival function (exponential, no sigmoid escape)
# ============================================================

def compute_fitness_grid_exp(omega, kappa, T, D, R, req, gamma,
                             hazard_scale, u_grid):
    """W with exponential survival: S = exp(-hazard_scale · T^γ · D / max(u, 0.1))

    Higher pressing rate → less time exposed → higher survival.
    No sigmoid threshold — continuous benefit from pressing harder.
    """
    u_g = u_grid[None, :]
    T_w = jnp.power(T[:, None], gamma)
    # Time exposed ∝ D / u. Hazard ∝ T_w. Survival = exp(-hazard * time)
    time_exposed = D[:, None] / jnp.clip(u_g, 0.1, None)
    S = jnp.exp(-hazard_scale * T_w * time_exposed)

    W = (S * R[:, None]
         - (1.0 - S) * omega[:, None] * (R[:, None] + C_PENALTY)
         - kappa[:, None] * (u_g - req[:, None]) ** 2 * D[:, None])

    weights = jax.nn.softmax(W * 20.0, axis=1)
    u_star = jnp.sum(weights * u_g, axis=1)
    V_star = jnp.sum(weights * W, axis=1)
    return u_star, V_star


def make_joint_v4(N_S, N_choice, N_vigor):
    """V4: Exponential survival (no sigmoid escape)."""
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_rate, vig_cookie):

        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        hz_raw = numpyro.sample('hazard_raw', dist.Normal(-1.0, 1.0))
        hazard_scale = numpyro.deterministic('hazard_scale', jnp.exp(hz_raw))

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        mu_om = numpyro.sample('mu_om', dist.Normal(0.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(0.5))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-1.0, 1.0))
        sigma_kap = numpyro.sample('sigma_kap', dist.HalfNormal(0.5))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))

        omega = jnp.exp(mu_om + sigma_om * om_raw)
        kappa = jnp.exp(mu_kap + sigma_kap * kap_raw)
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)

        u_grid = jnp.linspace(0.1, 1.5, 30)

        R_H = jnp.full(N_choice, 5.0)
        R_L = jnp.full(N_choice, 1.0)
        req_H = jnp.full(N_choice, 0.9)
        req_L = jnp.full(N_choice, 0.4)

        _, V_H = compute_fitness_grid_exp(
            omega[ch_subj], kappa[ch_subj], ch_T, ch_D_H,
            R_H, req_H, gamma, hazard_scale, u_grid)
        _, V_L = compute_fitness_grid_exp(
            omega[ch_subj], kappa[ch_subj], ch_T, ch_D_L,
            R_L, req_L, gamma, hazard_scale, u_grid)

        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_heavy = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_heavy, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        u_star, _ = compute_fitness_grid_exp(
            omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
            vig_R, vig_req, gamma, hazard_scale, u_grid)
        rate_pred = u_star + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(rate_pred, sigma_v),
                           obs=vig_rate)

    return model


# ============================================================
# V5: Only ω per-subject, κ at population level
# ============================================================

def make_joint_v5(N_S, N_choice, N_vigor):
    """V5: Only ω varies per subject. κ fixed at population level.
    Tests whether individual κ is needed or if effort cost is universal.
    """
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_rate, vig_cookie):

        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        eps_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
        epsilon = numpyro.deterministic('epsilon', jnp.exp(eps_raw))
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sm_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sm_raw), 0.01, 1.0)

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        # κ is population-level only
        kap_raw = numpyro.sample('kap_pop_raw', dist.Normal(-1.0, 1.0))
        kappa_pop = jnp.exp(kap_raw)

        mu_om = numpyro.sample('mu_om', dist.Normal(0.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(0.5))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))

        omega = jnp.exp(mu_om + sigma_om * om_raw)
        numpyro.deterministic('omega', omega)

        u_grid = jnp.linspace(0.1, 1.5, 30)

        # Broadcast kappa_pop to all trials
        kappa_ch = jnp.full(N_choice, kappa_pop)
        kappa_vig = jnp.full(N_vigor, kappa_pop)

        R_H = jnp.full(N_choice, 5.0)
        R_L = jnp.full(N_choice, 1.0)
        req_H = jnp.full(N_choice, 0.9)
        req_L = jnp.full(N_choice, 0.4)

        _, V_H = compute_fitness_grid(
            omega[ch_subj], kappa_ch, ch_T, ch_D_H,
            R_H, req_H, gamma, epsilon, p_esc, sigma_motor, u_grid)
        _, V_L = compute_fitness_grid(
            omega[ch_subj], kappa_ch, ch_T, ch_D_L,
            R_L, req_L, gamma, epsilon, p_esc, sigma_motor, u_grid)

        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_heavy = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_heavy, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        u_star, _ = compute_fitness_grid(
            omega[vig_subj], kappa_vig, vig_T, vig_dist,
            vig_R, vig_req, gamma, epsilon, p_esc, sigma_motor, u_grid)
        rate_pred = u_star + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(rate_pred, sigma_v),
                           obs=vig_rate)

    return model


# ============================================================
# Fitting and evaluation (shared)
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

    best_loss = float('inf')
    best_params = None
    t0 = time.time()
    losses = []

    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        l = float(loss)
        losses.append(l)
        if l < best_loss and not np.isnan(l):
            best_loss = l
            best_params = svi.get_params(state)
        if (i + 1) % 5000 == 0:
            print(f"    {name} step {i+1}: loss={l:.1f} (best={best_loss:.1f})")

    elapsed = time.time() - t0
    print(f"    {name} done in {elapsed:.0f}s, best_loss={best_loss:.1f}")

    return {
        'name': name, 'best_loss': best_loss, 'best_params': best_params,
        'guide': guide, 'model_fn': model_fn, 'kwargs': kwargs, 'losses': losses,
    }


def evaluate_model(fit_result, data, n_samples=300, seed=44):
    guide = fit_result['guide']
    model_fn = fit_result['model_fn']
    params_fit = fit_result['best_params']
    kwargs = fit_result['kwargs']
    name = fit_result['name']

    # Get predictions
    return_sites = ['omega', 'kappa', 'rate_pred', 'gamma', 'tau_raw',
                    'b_cookie', 'baseline']
    return_sites = [s for s in return_sites]  # keep all, missing ones silently skipped

    pred = Predictive(model_fn, guide=guide, params=params_fit,
                      num_samples=n_samples, return_sites=return_sites)
    samples = pred(random.PRNGKey(seed), **kwargs)

    # Choice reconstruction
    # We need to reconstruct P(heavy) from the model's parameters
    # For the joint model, we need the full computation
    # Simpler: use the fact that we can get omega, kappa, and recompute
    omega_mean = np.array(samples['omega']).mean(0) if 'omega' in samples else None
    kappa_mean = np.array(samples['kappa']).mean(0) if 'kappa' in samples else None

    # For now, use the loss as main metric and compute vigor r²
    vig_rate_np = np.array(data['vig_rate'])
    if 'rate_pred' in samples:
        rp = np.array(samples['rate_pred']).mean(0)
        r_vigor, _ = pearsonr(rp, vig_rate_np)
    else:
        r_vigor = np.nan

    # Choice accuracy: reconstruct from parameters
    # This is model-specific — for now extract from loss
    # We'll compute proper accuracy below
    ch_choice_np = np.array(data['ch_choice'])

    result = {
        'vigor_r': r_vigor,
        'vigor_r2': r_vigor ** 2 if not np.isnan(r_vigor) else np.nan,
        'omega_mean': omega_mean,
        'kappa_mean': kappa_mean,
    }

    # Try to compute choice accuracy by running the model forward
    try:
        if omega_mean is not None:
            ch_subj = np.array(data['ch_subj'])
            ch_T = np.array(data['ch_T'])
            ch_D_H = np.array(data['ch_D_H'])

            gamma_val = float(np.array(samples['gamma']).mean())
            tau_val = float(np.exp(np.array(samples['tau_raw']).mean()))

            # Get S params
            if 'epsilon' in samples:
                eps_val = float(np.array(samples['epsilon']).mean())
                p_esc_val = float(expit(np.array(samples.get('p_esc_raw',
                                  np.zeros(1))).mean())) if 'p_esc_raw' in samples else 0.5
                sm_val = float(np.exp(np.array(samples.get('sigma_motor_raw',
                               np.zeros(1))).mean())) if 'sigma_motor_raw' in samples else 0.3
            else:
                eps_val, p_esc_val, sm_val = 0.3, 0.5, 0.3

            # Compute V_H, V_L for each choice trial
            u_grid = np.linspace(0.1, 1.5, 30)
            V_H_all = np.zeros(len(ch_subj))
            V_L_all = np.zeros(len(ch_subj))

            for idx in range(len(ch_subj)):
                s = ch_subj[idx]
                T_val = ch_T[idx]
                D_H_val = ch_D_H[idx]
                om_val = omega_mean[s]
                kap_val = kappa_mean[s] if kappa_mean is not None else 0.1

                for R, req, D, store, store_idx in [
                    (5.0, 0.9, D_H_val, V_H_all, idx),
                    (1.0, 0.4, 1.0, V_L_all, idx)
                ]:
                    T_w = T_val ** gamma_val
                    S_vals = ((1 - T_w)
                              + eps_val * T_w * p_esc_val
                              * expit((u_grid - req) / sm_val))
                    W_vals = (S_vals * R
                              - (1 - S_vals) * om_val * (R + C_PENALTY)
                              - kap_val * (u_grid - req) ** 2 * D)
                    store[store_idx] = W_vals.max()

            delta_V = V_H_all - V_L_all
            p_H = expit(np.clip(delta_V / tau_val, -20, 20))
            choice_pred = (p_H >= 0.5).astype(int)
            choice_acc = (choice_pred == ch_choice_np).mean()

            # Per-subject choice correlation
            ch_df = pd.DataFrame({'subj': ch_subj, 'choice': ch_choice_np, 'p_H': p_H})
            sc = ch_df.groupby('subj').agg(o=('choice', 'mean'), p=('p_H', 'mean'))
            try:
                r_choice, _ = pearsonr(sc['o'], sc['p'])
            except:
                r_choice = np.nan

            result['choice_acc'] = choice_acc
            result['choice_r'] = r_choice
            result['choice_r2'] = r_choice ** 2 if not np.isnan(r_choice) else np.nan
    except Exception as e:
        print(f"  Warning: choice evaluation failed: {e}")
        result['choice_acc'] = np.nan
        result['choice_r'] = np.nan
        result['choice_r2'] = np.nan

    return result


# ============================================================
# Main
# ============================================================

MODEL_SPECS = [
    ('V1', make_joint_v1, 2, 'Joint basic (ω,κ)'),
    ('V2', make_joint_v2, 2, 'Joint + cookie intercept'),
    ('V3', make_joint_v3, 3, 'Joint + per-subj baseline'),
    ('V4', make_joint_v4, 2, 'Joint exponential S'),
    ('V5', make_joint_v5, 1, 'Joint ω only (κ pop-level)'),
]

PARAM_COUNTS = {
    'V1': lambda N: 2*N + 10,   # om+kap raw + pop params
    'V2': lambda N: 2*N + 11,   # V1 + b_cookie
    'V3': lambda N: 3*N + 13,   # V2 + baseline raw + mu_base, sigma_base
    'V4': lambda N: 2*N + 8,    # fewer pop params (no eps, p_esc, sigma_motor)
    'V5': lambda N: N + 10,     # only omega per-subject
}


if __name__ == '__main__':
    t_start = time.time()

    print("=" * 70)
    print("JOINT OPTIMAL CONTROL: Bednekoff/Brown Unified Fitness Models")
    print("  W(u) = S(u)·R - (1-S(u))·ω·(R+C) - κ·(u-req)²·D")
    print("  Choice = argmax_j max_u W_j(u)")
    print("  Vigor = argmax_u W_chosen(u)")
    print("=" * 70)

    print("\nPreparing data...")
    data = prepare_data()
    N_S = data['N_S']

    results = []

    for name, make_fn, n_per_subj, description in MODEL_SPECS:
        print(f"\n{'='*50}")
        print(f"--- {name}: {description} ---")
        print(f"{'='*50}")
        model_fn = make_fn(N_S, data['N_choice'], data['N_vigor'])

        fit_result = fit_model(name, model_fn, data, n_steps=30000, lr=0.001)

        if fit_result['best_params'] is None:
            print(f"  {name} FAILED to converge")
            continue

        print("  Evaluating...")
        metrics = evaluate_model(fit_result, data)

        n_params = PARAM_COUNTS[name](N_S)
        n_obs = data['N_choice'] + data['N_vigor']
        bic = 2 * fit_result['best_loss'] + n_params * np.log(n_obs)

        row = {
            'Model': name, 'Description': description,
            'n_per_subj': n_per_subj, 'n_params': n_params,
            'ELBO': -fit_result['best_loss'], 'BIC': bic,
            'choice_acc': metrics.get('choice_acc', np.nan),
            'choice_r2': metrics.get('choice_r2', np.nan),
            'vigor_r2': metrics.get('vigor_r2', np.nan),
        }
        results.append(row)

        print(f"\n  ELBO = {-fit_result['best_loss']:.1f}, BIC = {bic:.0f}")
        print(f"  Choice: acc={metrics.get('choice_acc', 'N/A'):.3f}, "
              f"r²={metrics.get('choice_r2', 'N/A'):.3f}")
        print(f"  Vigor:  r²={metrics.get('vigor_r2', 'N/A'):.3f}")

        # Parameter summary
        if metrics.get('omega_mean') is not None:
            om = metrics['omega_mean']
            print(f"  ω: mean={om.mean():.3f}, SD={om.std():.3f}, "
                  f"range=[{om.min():.3f}, {om.max():.3f}]")
        if metrics.get('kappa_mean') is not None:
            kap = metrics['kappa_mean']
            print(f"  κ: mean={kap.mean():.3f}, SD={kap.std():.3f}, "
                  f"range=[{kap.min():.3f}, {kap.max():.3f}]")

    # Comparison table
    if results:
        df = pd.DataFrame(results)
        best_bic = df['BIC'].min()
        df['dBIC'] = df['BIC'] - best_bic

        print("\n" + "=" * 70)
        print("JOINT MODEL COMPARISON")
        print("=" * 70)
        print(df[['Model', 'Description', 'n_per_subj', 'ELBO', 'BIC',
                   'dBIC', 'choice_acc', 'choice_r2', 'vigor_r2']].to_string(index=False))

        # Compare to M5 benchmark
        print("\n--- M5 benchmark (from separate-equations model) ---")
        print("  Choice: acc=0.794, r²=0.952, Vigor: r²=0.496")
        print("  BIC = 16,191")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "joint_comparison.csv", index=False)
        print(f"\nSaved to {OUT_DIR / 'joint_comparison.csv'}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")
