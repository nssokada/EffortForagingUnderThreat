"""
Hierarchical Bayesian Optimal Control model for choice in the
Effort Foraging Under Threat task.

Two subject-level parameters:
    c_effort  — effort cost sensitivity (log-normal)
    c_death   — capture aversion (log-normal)

Population-level parameters:
    tau            — choice temperature
    lambda_param   — distance sensitivity in survival function
    t_scale        — time scaling for effort cost (fixed)
    tier_surv_fracs — relative survival at each speed tier (fixed)
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoNormal
from jax import random

from .optimal_control import choice_probability, REQ_RATE_HEAVY, REQ_RATE_LIGHT

jax.config.update("jax_enable_x64", True)

# Fixed structural parameters (estimated from game mechanics)
# These could be freed later but fixing them avoids identifiability issues
DEFAULT_T_SCALE = 2.0
DEFAULT_TIER_SURV_FRACS = jnp.array([0.0, 0.05, 0.3, 1.0])


def make_oc_choice_model(n_subjects, n_trials, t_scale=None, tier_surv_fracs=None):
    """
    Factory that returns a NumPyro model function with n_subjects and n_trials
    baked in (required for JIT compilation with numpyro.plate).
    """
    _t_scale = t_scale if t_scale is not None else DEFAULT_T_SCALE
    _tier_surv = tier_surv_fracs if tier_surv_fracs is not None else DEFAULT_TIER_SURV_FRACS

    def model(subj_idx, T, dist_level_H, dist_level_L, R_H, R_L,
              req_rate_H, req_rate_L, choice=None, C=5.0):
        # ── Population priors ─────────────────────────────────────────
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 0.5))
        tau = numpyro.deterministic('tau', jnp.clip(jnp.exp(tau_raw), 0.01, 20.0))

        lambda_raw = numpyro.sample('lambda_raw', dist.Normal(0.0, 1.0))
        lambda_param = numpyro.deterministic(
            'lambda_param', jnp.clip(jnp.exp(lambda_raw), 0.01, 50.0)
        )

        # ── Subject-level (non-centered) ─────────────────────────────
        with numpyro.plate('subjects', n_subjects):
            ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))

        c_effort = numpyro.deterministic('c_effort', jnp.exp(mu_ce + sigma_ce * ce_raw))
        c_death = numpyro.deterministic('c_death', jnp.exp(mu_cd + sigma_cd * cd_raw))

        # ── Choice probabilities ──────────────────────────────────────
        p_H, eu_H, eu_L, u_H, u_L = choice_probability(
            c_effort[subj_idx], c_death[subj_idx], T,
            dist_level_H, dist_level_L, R_H, R_L, C,
            req_rate_H, req_rate_L,
            lambda_param, _tier_surv, _t_scale, tau,
        )

        numpyro.deterministic('p_H', p_H)
        numpyro.deterministic('u_star_H', u_H)
        numpyro.deterministic('u_star_L', u_L)

        # ── Likelihood ────────────────────────────────────────────────
        with numpyro.plate('trials', n_trials):
            numpyro.sample(
                'obs',
                dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                obs=choice,
            )

    return model


def fit_oc_svi(
    data_dict,
    n_steps=30000,
    lr=0.002,
    seed=42,
    guide_type='autonormal',
    print_every=5000,
    t_scale=None,
    tier_surv_fracs=None,
):
    """
    Fit the OC choice model using SVI.
    """
    n_subjects = data_dict['n_subjects']
    n_trials = len(data_dict['T'])

    model = make_oc_choice_model(
        n_subjects, n_trials,
        t_scale=t_scale,
        tier_surv_fracs=tier_surv_fracs,
    )

    model_kwargs = {
        'subj_idx': data_dict['subj_idx'],
        'T': data_dict['T'],
        'dist_level_H': data_dict['dist_level_H'],
        'dist_level_L': data_dict['dist_level_L'],
        'R_H': data_dict['R_H'],
        'R_L': data_dict['R_L'],
        'req_rate_H': data_dict['req_rate_H'],
        'req_rate_L': data_dict['req_rate_L'],
        'choice': data_dict['choice'],
        'C': data_dict['C'],
    }

    if guide_type == 'automvn':
        guide = AutoMultivariateNormal(model)
    else:
        guide = AutoNormal(model)

    optimizer = numpyro.optim.Adam(lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key = random.PRNGKey(seed)
    state = svi.init(rng_key, **model_kwargs)

    losses = []
    update_fn = jax.jit(svi.update)

    print(f"Fitting OC model (SVI, {n_steps} steps, lr={lr})")
    print(f"  {n_subjects} subjects, {n_trials} trials")

    for step in range(n_steps):
        state, loss = update_fn(state, **model_kwargs)
        losses.append(float(loss))

        if (step + 1) % print_every == 0:
            window = min(1000, len(losses))
            recent = sum(losses[-window:]) / window
            print(f"  Step {step+1:6d}: loss={losses[-1]:.1f}, avg_1k={recent:.1f}")

    params = svi.get_params(state)
    print(f"  Final ELBO: {-losses[-1]:.1f}")

    return {
        'params': params,
        'losses': losses,
        'guide': guide,
        'svi': svi,
        'state': state,
        'model': model,
        'model_kwargs': model_kwargs,
        'n_subjects': n_subjects,
        'subjects': data_dict['subjects'],
    }


def extract_params(fit_result, n_samples=2000, seed=43):
    """
    Extract subject-level and population parameters from SVI fit.
    """
    import pandas as pd

    guide = fit_result['guide']
    params = fit_result['params']
    model_kwargs = fit_result['model_kwargs']

    rng_key = random.PRNGKey(seed)
    predictive = Predictive(guide, params=params, num_samples=n_samples)
    samples = predictive(rng_key, **model_kwargs)

    # Population params
    pop = {}
    for name in ['tau', 'lambda_param', 'mu_ce', 'mu_cd', 'sigma_ce', 'sigma_cd']:
        if name in samples:
            pop[name] = {
                'mean': float(np.mean(samples[name])),
                'std': float(np.std(samples[name])),
            }

    # Subject params
    c_effort_samples = np.array(samples['c_effort'])  # (n_samples, n_subjects)
    c_death_samples = np.array(samples['c_death'])

    subjects = fit_result['subjects']
    records = []
    for i, subj in enumerate(subjects):
        records.append({
            'subj': subj,
            'c_effort_mean': float(np.mean(c_effort_samples[:, i])),
            'c_effort_std': float(np.std(c_effort_samples[:, i])),
            'c_death_mean': float(np.mean(c_death_samples[:, i])),
            'c_death_std': float(np.std(c_death_samples[:, i])),
        })

    return {
        'population': pop,
        'subject': pd.DataFrame(records),
        'samples': samples,
    }
