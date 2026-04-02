"""
Optimal Control solver for the Effort Foraging Under Threat task.

Core idea: subjects maximize expected utility by choosing (1) which cookie to pursue
and (2) how hard to press. A single cost function with two parameters — c_effort
(effort cost sensitivity) and c_death (capture aversion) — jointly determines both.

The task has discrete speed tiers, so the optimal constant-rate policy is to press
at exactly a tier threshold. We evaluate EU at all 4 tiers for each cookie option,
pick the best tier per option, then softmax over the two options.

Survival function: Uses empirical S(T, D, tier) — the probability of escaping given
threat level, distance, and speed tier. This is precomputed from data rather than
derived from a Gaussian strike model, because the game's spatial dynamics (predator
approach speed, chase phase, proximity to safety) make analytical derivation fragile.

All functions are JAX-differentiable for use inside NumPyro SVI/MCMC.
"""

import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)


# ── Speed tier constants ─────────────────────────────────────────────────────
# Fraction of required rate at each tier threshold
TIER_THRESHOLDS = jnp.array([0.0, 0.25, 0.50, 1.00])
SPEED_FRACTIONS = jnp.array([0.0, 0.25, 0.50, 1.00])
N_TIERS = 4

# Required press rates as fraction of calMax
REQ_RATE_LIGHT = 0.4
REQ_RATE_HEAVY = 0.9


# ── Survival function ────────────────────────────────────────────────────────

def hyperbolic_survival(T, distance_level, lambda_param):
    """
    Survival probability using the hyperbolic form from the descriptive model.

    S = (1 - T) + T / (1 + lambda * D)

    where D is the distance level (1, 2, 3) and lambda controls distance sensitivity.
    This is independent of speed tier — it represents survival at FULL speed.
    At lower tiers, survival is worse (takes longer, more exposure).

    For the OC model, we adjust S by tier: slower tiers have proportionally
    less survival benefit. Specifically:
        S(tier) = (1 - T) + T * f(tier) / (1 + lambda * D)
    where f(tier) captures the survival advantage of faster movement.
    """
    return (1.0 - T) + T / (1.0 + lambda_param * distance_level)


def tier_survival(T, distance_level, lambda_param, tier_survival_fracs):
    """
    Survival probability at each speed tier.

    Parameters
    ----------
    T : (n_trials,)
    distance_level : (n_trials,) — 1, 2, or 3
    lambda_param : scalar — distance sensitivity
    tier_survival_fracs : (4,) — relative survival at each tier.
        tier_survival_fracs[3] = 1.0 (full speed = baseline).
        Lower tiers have lower fractions.

    Returns
    -------
    p_surv : (n_trials, 4) — survival probability at each tier
    """
    # Base survival at full speed
    # S_full = (1-T) + T / (1 + lambda * D)
    # The T/(1+lambda*D) term is the escape component (only matters on attack)
    # At lower tiers, the escape component is reduced
    escape_component = T[:, None] / (1.0 + lambda_param * distance_level[:, None])
    no_attack = (1.0 - T)[:, None]

    # Scale escape component by tier
    p_surv = no_attack + escape_component * tier_survival_fracs[None, :]

    return jnp.clip(p_surv, 1e-7, 1.0 - 1e-7)


def compute_tier_eu(
    c_effort,           # (n_trials,) — effort cost (per-subject, already indexed)
    c_death,            # (n_trials,) — capture aversion
    T,                  # (n_trials,) — threat probability
    distance_level,     # (n_trials,) — distance level (1, 2, 3)
    R,                  # (n_trials,) — reward
    C,                  # scalar — capture penalty
    req_rate,           # (n_trials,) — required press rate (calMax fraction)
    lambda_param,       # scalar — distance sensitivity
    tier_surv_fracs,    # (4,) — survival fraction at each tier
    t_scale,            # scalar — time scaling: t_arr_full ≈ t_scale * distance_level
):
    """
    Compute expected utility at each of the 4 speed tiers for a set of trials.

    Returns
    -------
    eu_all : (n_trials, 4) — EU at each tier
    u_all  : (n_trials, 4) — press rate (calMax fraction) at each tier
    p_surv : (n_trials, 4) — survival probability at each tier
    """
    # Press rate at each tier (in calMax units)
    u_all = req_rate[:, None] * TIER_THRESHOLDS[None, :]  # (n_trials, 4)

    # Survival at each tier
    p_surv = tier_survival(T, distance_level, lambda_param, tier_surv_fracs)

    # Arrival time at each tier (for effort cost calculation)
    # At full speed: t_arr = t_scale * distance_level
    # At tier j: t_arr = t_arr_full / speed_fraction (slower = takes longer)
    t_arr_full = t_scale * distance_level[:, None]  # (n_trials, 1)
    t_arr = jnp.where(
        SPEED_FRACTIONS[None, :] > 0,
        t_arr_full / SPEED_FRACTIONS[None, :],
        100.0  # large but finite for zero-speed tier
    )

    # Expected utility
    eu = (
        p_surv * R[:, None]
        - (1.0 - p_surv) * c_death[:, None] * (R[:, None] + C)
        - c_effort[:, None] * u_all ** 2 * t_arr
    )

    return eu, u_all, p_surv


def optimal_tier(
    c_effort, c_death, T, distance_level, R, C,
    req_rate, lambda_param, tier_surv_fracs, t_scale,
):
    """
    Find the optimal speed tier for each trial.

    Returns
    -------
    eu_star  : (n_trials,) — EU at optimal tier
    u_star   : (n_trials,) — optimal press rate (calMax fraction)
    p_star   : (n_trials,) — survival probability at optimal tier
    tier_idx : (n_trials,) — index of optimal tier (0-3)
    """
    eu_all, u_all, p_surv = compute_tier_eu(
        c_effort, c_death, T, distance_level, R, C,
        req_rate, lambda_param, tier_surv_fracs, t_scale,
    )

    tier_idx = jnp.argmax(eu_all, axis=1)
    n = T.shape[0]
    idx = jnp.arange(n)

    return eu_all[idx, tier_idx], u_all[idx, tier_idx], p_surv[idx, tier_idx], tier_idx


def soft_optimal_tier(
    c_effort, c_death, T, distance_level, R, C,
    req_rate, lambda_param, tier_surv_fracs, t_scale,
    tier_temp=0.1,
):
    """
    Soft (differentiable) version of optimal_tier using softmax over tiers.
    Returns expected values instead of argmax, which is better for gradient-based fitting.
    """
    eu_all, u_all, p_surv = compute_tier_eu(
        c_effort, c_death, T, distance_level, R, C,
        req_rate, lambda_param, tier_surv_fracs, t_scale,
    )

    # Softmax weights over tiers
    weights = jax.nn.softmax(eu_all / tier_temp, axis=1)  # (n_trials, 4)

    eu_star = jnp.sum(weights * eu_all, axis=1)
    u_star = jnp.sum(weights * u_all, axis=1)
    p_star = jnp.sum(weights * p_surv, axis=1)

    return eu_star, u_star, p_star


def choice_probability(
    c_effort, c_death, T,
    dist_level_H, dist_level_L,
    R_H, R_L, C,
    req_rate_H, req_rate_L,
    lambda_param, tier_surv_fracs, t_scale,
    tau,
):
    """
    Compute P(choose H) and optimal vigor for each option.

    Returns
    -------
    p_choose_H : (n_trials,)
    eu_H, eu_L : (n_trials,) — EU at optimal tier
    u_star_H, u_star_L : (n_trials,) — optimal press rate
    """
    eu_H, u_star_H, p_surv_H = soft_optimal_tier(
        c_effort, c_death, T, dist_level_H, R_H, C,
        req_rate_H, lambda_param, tier_surv_fracs, t_scale,
    )
    eu_L, u_star_L, p_surv_L = soft_optimal_tier(
        c_effort, c_death, T, dist_level_L, R_L, C,
        req_rate_L, lambda_param, tier_surv_fracs, t_scale,
    )

    logit = jnp.clip((eu_H - eu_L) / tau, -20.0, 20.0)
    p_choose_H = jax.nn.sigmoid(logit)

    return p_choose_H, eu_H, eu_L, u_star_H, u_star_L


# ── Data preparation ─────────────────────────────────────────────────────────

def prepare_trial_data(beh_df):
    """
    Convert behavior_rich DataFrame into JAX arrays for the OC model.
    Uses ALL trials (type 1, 5, 6).
    """
    df = beh_df.copy()

    # Subject indexing
    subjects = sorted(df['subj'].unique())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    subj_idx = jnp.array([subj_to_idx[s] for s in df['subj']])

    return {
        'subj_idx': subj_idx,
        'n_subjects': len(subjects),
        'subjects': subjects,
        'T': jnp.array(df['threat'].values),
        'dist_level_H': jnp.array(df['distance_H'].values, dtype=jnp.float64),
        'dist_level_L': jnp.ones(len(df)),  # light cookie always at distance 1
        'R_H': jnp.full(len(df), 5.0),
        'R_L': jnp.full(len(df), 1.0),
        'C': 5.0,
        'req_rate_H': jnp.full(len(df), REQ_RATE_HEAVY),
        'req_rate_L': jnp.full(len(df), REQ_RATE_LIGHT),
        'choice': jnp.array(df['choice'].values),
    }
