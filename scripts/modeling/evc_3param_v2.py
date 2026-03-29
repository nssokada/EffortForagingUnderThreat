"""
EVC 3-parameter model v2: k + beta + cd (no gamma, no epsilon)

Key insight: dropping gamma and epsilon from the choice equation makes
beta the SOLE carrier of threat information in choice. No confounding
with population-level survival function parameters.

Architecture:
    Per-subject (log-normal, non-centered):
        k (effort cost)       -- drives choice via effort-distance term
        beta (threat aversion) -- drives choice via threat term T
        cd (capture aversion)  -- drives vigor via survival incentive

    Population-level:
        tau         -- choice temperature
        p_esc       -- escape probability at full speed (vigor only)
        sigma_motor -- motor noise around speed threshold (vigor only)
        ce_vigor    -- LQR deviation motor cost for vigor
        sigma_v     -- vigor observation noise

    Choice equation (simple, no S):
        dEU = R_diff - k_i * effort(D) - beta_i * T
        P(heavy) = sigmoid(dEU / tau)

        where R_diff = 4 (heavy reward 5 minus light reward 1)
        effort(D) = 0.81 * D_H - 0.16 (LQR commitment cost differential)
        T = threat probability {0.1, 0.5, 0.9}

        k captures: "I avoid far cookies" (distance/effort gradient)
        beta captures: "I avoid cookies when threat is high" (threat gradient)
        T×D interaction emerges from sigmoid nonlinearity

    Vigor equation (S depends on press rate, no gamma):
        S(u) = (1 - T) + T * p_esc * sigmoid((u - req) / sigma_motor)
        EU(u) = S(u)*R - (1-S(u))*cd_i*(R+C) - ce_vigor*(u-req)^2*D
        u* = soft_argmax over 30-point grid
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
import ast
from scipy.stats import pearsonr

jax.config.update('jax_enable_x64', True)


def prepare_data(behavior_rich_path, psych_path=None):
    """Load and prepare data for the 3-param v2 model."""
    beh = pd.read_csv(behavior_rich_path)

    # Choice data: type=1 only
    choice_df = beh[beh['type'] == 1].copy()

    # Vigor data: ALL types (1, 5, 6)
    vigor_df = beh.copy()
    vigor_df['actual_dist'] = vigor_df['startDistance'].map({5: 1, 7: 2, 9: 3})
    vigor_df['actual_req'] = np.where(
        vigor_df['trialCookie_weight'] == 3.0, 0.9, 0.4)
    vigor_df['actual_R'] = np.where(
        vigor_df['trialCookie_weight'] == 3.0, 5.0, 1.0)
    vigor_df['is_heavy'] = (vigor_df['trialCookie_weight'] == 3.0).astype(int)

    # Compute median press rate
    rates = []
    for _, row in vigor_df.iterrows():
        try:
            pt = np.array(
                ast.literal_eval(row['alignedEffortRate']), dtype=float)
            ipis = np.diff(pt)
            ipis = ipis[ipis > 0.01]
            if len(ipis) >= 5:
                rates.append(
                    np.median((1.0 / ipis) / row['calibrationMax']))
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

    # Subject indexing
    subjects = sorted(
        set(choice_df['subj'].unique()) & set(vigor_df['subj'].unique()))
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)

    ch_subj = jnp.array([subj_to_idx[s] for s in choice_df['subj']])
    ch_T = jnp.array(choice_df['threat'].values)
    ch_dist_H = jnp.array(choice_df['distance_H'].values, dtype=jnp.float64)
    ch_choice = jnp.array(choice_df['choice'].values)

    vig_subj = jnp.array([subj_to_idx[s] for s in vigor_df['subj']])
    vig_T = jnp.array(vigor_df['threat'].values)
    vig_R = jnp.array(vigor_df['actual_R'].values)
    vig_req = jnp.array(vigor_df['actual_req'].values)
    vig_dist = jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64)
    vig_excess = jnp.array(vigor_df['excess_cc'].values)
    vig_offset = jnp.array(np.where(
        vigor_df['is_heavy'].values == 1, heavy_mean, light_mean))

    data = {
        'ch_subj': ch_subj, 'ch_T': ch_T, 'ch_dist_H': ch_dist_H,
        'ch_choice': ch_choice,
        'vig_subj': vig_subj, 'vig_T': vig_T, 'vig_R': vig_R,
        'vig_req': vig_req, 'vig_dist': vig_dist,
        'vig_excess': vig_excess, 'vig_offset': vig_offset,
        'subjects': subjects, 'N_S': N_S,
        'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
        'heavy_mean': heavy_mean, 'light_mean': light_mean,
        'vigor_df': vigor_df,
    }

    if psych_path is not None:
        data['psych'] = pd.read_csv(psych_path)

    return data


def make_model(N_S, N_choice, N_vigor):
    """3-param model: k + beta in choice (no S), cd in vigor."""

    def evc_3param_v2(ch_subj, ch_T, ch_dist_H, ch_choice,
                      vig_subj, vig_T, vig_R, vig_req, vig_dist,
                      vig_excess, vig_offset):
        # ── Population priors for hierarchical k, beta, cd ──
        mu_k = numpyro.sample('mu_k', dist.Normal(0.0, 1.0))
        mu_beta = numpyro.sample('mu_beta', dist.Normal(1.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_k = numpyro.sample('sigma_k', dist.HalfNormal(0.5))
        sigma_beta = numpyro.sample('sigma_beta', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        # ── Population params (no gamma, no epsilon) ──
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample(
            'sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)
        ce_vigor_raw = numpyro.sample(
            'ce_vigor_raw', dist.Normal(-3.0, 1.0))
        ce_vigor = numpyro.deterministic('ce_vigor', jnp.exp(ce_vigor_raw))

        # ── Subject-level (non-centered): k, beta, cd ──
        with numpyro.plate('subjects', N_S):
            k_raw = numpyro.sample('k_raw', dist.Normal(0.0, 1.0))
            beta_raw = numpyro.sample('beta_raw', dist.Normal(0.0, 1.0))
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))

        k = jnp.exp(mu_k + sigma_k * k_raw)
        beta = jnp.exp(mu_beta + sigma_beta * beta_raw)
        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        numpyro.deterministic('k', k)
        numpyro.deterministic('beta', beta)
        numpyro.deterministic('c_death', c_death)

        # ── CHOICE: simple, no survival function ──
        # dEU = R_diff - k * effort(D) - beta * T
        k_ch = k[ch_subj]
        beta_ch = beta[ch_subj]
        effort_cost = 0.81 * ch_dist_H - 0.16
        threat_cost = ch_T  # direct: T ∈ {0.1, 0.5, 0.9}

        delta_eu = 4.0 - k_ch * effort_cost - beta_ch * threat_cost
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample(
                'obs_choice',
                dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                obs=ch_choice)

        # ── VIGOR: S(u) with physical escape mechanism (no gamma) ──
        cd_v = c_death[vig_subj]
        # S(u) = (1-T) + T * p_esc * sigmoid((u-req)/sigma_motor)
        # No gamma: T enters linearly
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - vig_T[:, None])
               + vig_T[:, None] * p_esc
               * jax.nn.sigmoid(
                   (u_g - vig_req[:, None]) / sigma_motor))
        deviation = u_g - vig_req[:, None]
        eu_grid = (S_u * vig_R[:, None]
                   - (1.0 - S_u) * cd_v[:, None]
                   * (vig_R[:, None] + 5.0)
                   - ce_vigor * deviation ** 2 * vig_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - vig_req - vig_offset
        numpyro.deterministic('excess_pred', excess_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample(
                'obs_vigor',
                dist.Normal(excess_pred, sigma_v),
                obs=vig_excess)

    return evc_3param_v2


def fit(data, n_steps=40000, lr=0.001, seed=42, print_every=5000):
    """Fit the 3-param v2 model via SVI."""
    model = make_model(data['N_S'], data['N_choice'], data['N_vigor'])

    kwargs = {k: data[k] for k in [
        'ch_subj', 'ch_T', 'ch_dist_H', 'ch_choice',
        'vig_subj', 'vig_T', 'vig_R', 'vig_req', 'vig_dist',
        'vig_excess', 'vig_offset',
    ]}

    guide = AutoNormal(model)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    print(f"Fitting EVC 3-param v2 (SVI, {n_steps} steps, lr={lr})")
    print(f"  {data['N_S']} subjects, 3 per-subj params (k, beta, cd)")
    print(f"  Population: tau, p_esc, sigma_motor, ce_vigor, sigma_v")
    print(f"  NO gamma, NO epsilon")
    print(f"  {data['N_choice']} choice trials, {data['N_vigor']} vigor trials")

    losses = []
    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        losses.append(float(loss))
        if (i + 1) % print_every == 0:
            print(f"  Step {i + 1}: loss={loss:.1f}")

    params_fit = svi.get_params(state)
    return {
        'params': params_fit, 'losses': losses,
        'guide': guide, 'model': model, 'kwargs': kwargs,
        'data': data,
    }


def evaluate(fit_result, n_samples=500, seed=44):
    """Extract parameters and evaluate fit quality."""
    guide = fit_result['guide']
    model = fit_result['model']
    params_fit = fit_result['params']
    data = fit_result['data']
    kwargs = fit_result['kwargs']

    pred = Predictive(
        model, guide=guide, params=params_fit,
        num_samples=n_samples,
        return_sites=['k', 'beta', 'c_death', 'ce_vigor',
                      'excess_pred', 'tau_raw',
                      'p_esc_raw', 'sigma_motor_raw', 'sigma_v'])
    samples = pred(random.PRNGKey(seed), **kwargs)

    k_vals = np.array(samples['k']).mean(0)
    beta_vals = np.array(samples['beta']).mean(0)
    cd = np.array(samples['c_death']).mean(0)
    ce_vigor_val = float(np.array(samples['ce_vigor']).mean())
    tau_val = float(np.exp(np.array(samples['tau_raw']).mean()))
    p_esc_val = float(1.0 / (1.0 + np.exp(-np.array(samples['p_esc_raw']).mean())))
    sigma_motor_val = float(np.exp(np.array(samples['sigma_motor_raw']).mean()))
    sigma_v_val = float(np.array(samples['sigma_v']).mean())
    ep = np.array(samples['excess_pred']).mean(0)

    r_vigor, _ = pearsonr(ep, np.array(data['vig_excess']))

    # BIC
    # mu_k, sigma_k, mu_beta, sigma_beta, mu_cd, sigma_cd,
    # tau_raw, p_esc_raw, sigma_motor_raw, ce_vigor_raw, sigma_v = 11 pop params
    n_params = 3 * data['N_S'] + 11
    bic = (2 * fit_result['losses'][-1]
           + n_params * np.log(data['N_choice'] + data['N_vigor']))

    # Per-subject choice predictions
    from scipy.special import expit
    ch_subj_np = np.array(data['ch_subj'])
    ch_T_np = np.array(data['ch_T'])
    ch_dist_np = np.array(data['ch_dist_H'])
    ch_choice_np = np.array(data['ch_choice'])

    effort_cost = 0.81 * ch_dist_np - 0.16
    threat_cost = ch_T_np
    delta_eu = 4.0 - k_vals[ch_subj_np] * effort_cost - beta_vals[ch_subj_np] * threat_cost
    logit_ch = np.clip(delta_eu / tau_val, -20, 20)
    p_H = expit(logit_ch)

    choice_acc = ((p_H >= 0.5).astype(int) == ch_choice_np).mean()

    ch_df = pd.DataFrame({
        'subj': ch_subj_np, 'choice': ch_choice_np, 'p_H': p_H
    })
    subj_choice = ch_df.groupby('subj').agg(
        obs_pH=('choice', 'mean'), pred_pH=('p_H', 'mean')
    ).reset_index()
    r_choice_subj, _ = pearsonr(subj_choice['obs_pH'], subj_choice['pred_pH'])

    # Per-subject vigor r^2
    vig_subj_np = np.array(data['vig_subj'])
    vig_excess_np = np.array(data['vig_excess'])
    vig_df = pd.DataFrame({
        'subj': vig_subj_np, 'obs': vig_excess_np, 'pred': ep
    })
    subj_vigor = vig_df.groupby('subj').agg(
        obs_mean=('obs', 'mean'), pred_mean=('pred', 'mean')
    ).reset_index()
    r_vigor_subj, _ = pearsonr(subj_vigor['obs_mean'], subj_vigor['pred_mean'])

    # Condition-level predictions
    ch_full = pd.DataFrame({
        'subj': ch_subj_np, 'threat': ch_T_np, 'dist': ch_dist_np,
        'choice': ch_choice_np, 'p_H': p_H
    })
    cond_preds = ch_full.groupby(['threat', 'dist']).agg(
        obs=('choice', 'mean'), pred=('p_H', 'mean'), n=('choice', 'count')
    ).reset_index()

    print(f"\n{'=' * 60}")
    print(f"EVC 3-param v2 Results (k + beta + cd, NO gamma/epsilon)")
    print(f"{'=' * 60}")
    print(f"BIC: {bic:.0f}")
    print(f"Choice accuracy: {choice_acc:.3f}")
    print(f"Per-subject choice r: {r_choice_subj:.3f} (r^2={r_choice_subj**2:.3f})")
    print(f"Vigor r: {r_vigor:.3f} (r^2={r_vigor**2:.3f})")
    print(f"Per-subject vigor r: {r_vigor_subj:.3f} (r^2={r_vigor_subj**2:.3f})")
    print(f"\nPopulation params:")
    print(f"  ce_vigor={ce_vigor_val:.4f}, tau={tau_val:.3f}")
    print(f"  p_esc={p_esc_val:.3f}, sigma_motor={sigma_motor_val:.3f}, sigma_v={sigma_v_val:.3f}")
    print(f"\nPer-subject k (effort): median={np.median(k_vals):.3f}, mean={k_vals.mean():.3f}, "
          f"SD={k_vals.std():.3f}, range=[{k_vals.min():.3f}, {k_vals.max():.3f}]")
    print(f"Per-subject beta (threat): median={np.median(beta_vals):.3f}, mean={beta_vals.mean():.3f}, "
          f"SD={beta_vals.std():.3f}, range=[{beta_vals.min():.3f}, {beta_vals.max():.3f}]")
    print(f"Per-subject cd (vigor): median={np.median(cd):.3f}, mean={cd.mean():.3f}, "
          f"SD={cd.std():.3f}, range=[{cd.min():.3f}, {cd.max():.3f}]")

    lk, lb, lcd = np.log(k_vals), np.log(beta_vals), np.log(cd)
    r_kb, p_kb = pearsonr(lk, lb)
    r_kcd, p_kcd = pearsonr(lk, lcd)
    r_bcd, p_bcd = pearsonr(lb, lcd)
    print(f"\nLog correlations:")
    print(f"  k x beta = {r_kb:+.3f} (p={p_kb:.4f})")
    print(f"  k x cd   = {r_kcd:+.3f} (p={p_kcd:.4f})")
    print(f"  beta x cd = {r_bcd:+.3f} (p={p_bcd:.4f})")

    print(f"\nCondition-level predictions (choice):")
    print(f"  {'T':>4} {'D':>4} {'Obs':>6} {'Pred':>6} {'N':>4}")
    for _, row in cond_preds.iterrows():
        print(f"  {row['threat']:4.1f} {row['dist']:4.0f} "
              f"{row['obs']:6.3f} {row['pred']:6.3f} {row['n']:4.0f}")

    param_df = pd.DataFrame({
        'subj': data['subjects'],
        'k': k_vals, 'beta': beta_vals, 'c_death': cd,
    })

    pop_df = pd.DataFrame([{
        'ce_vigor': ce_vigor_val, 'tau': tau_val, 'p_esc': p_esc_val,
        'sigma_motor': sigma_motor_val, 'sigma_v': sigma_v_val,
    }])

    return {
        'param_df': param_df, 'pop_df': pop_df,
        'ce_vigor': ce_vigor_val, 'tau': tau_val,
        'p_esc': p_esc_val, 'sigma_motor': sigma_motor_val,
        'r_vigor': r_vigor, 'r_vigor_subj': r_vigor_subj,
        'r_choice_subj': r_choice_subj, 'choice_acc': choice_acc,
        'bic': bic, 'samples': samples,
        'k': k_vals, 'beta': beta_vals, 'cd': cd,
        'cond_preds': cond_preds,
    }


if __name__ == '__main__':
    import time
    t0 = time.time()

    DATA_DIR = ('data/exploratory_350/processed/'
                'stage5_filtered_data_20260320_191950')

    data = prepare_data(
        f'{DATA_DIR}/behavior_rich.csv',
        psych_path=f'{DATA_DIR}/psych.csv',
    )

    print(f"N_subjects={data['N_S']}")
    print(f"N_choice={data['N_choice']}, N_vigor={data['N_vigor']}")

    fit_result = fit(data, n_steps=40000, lr=0.001)
    result = evaluate(fit_result)

    # Save
    out_path = 'results/stats/oc_evc_3param_v2_params.csv'
    result['param_df'].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    pop_path = 'results/stats/oc_evc_3param_v2_population.csv'
    result['pop_df'].to_csv(pop_path, index=False)
    print(f"Saved: {pop_path}")

    cond_path = 'results/stats/oc_evc_3param_v2_conditions.csv'
    result['cond_preds'].to_csv(cond_path, index=False)
    print(f"Saved: {cond_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
