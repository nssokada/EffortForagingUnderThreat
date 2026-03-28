"""
DEFINITIVE EVC model for Effort Foraging Under Threat.
"2+2 model": 2 per-subject params + 2 key population params.

Architecture:
    Per-subject (log-normal, non-centered):
        ce (c_effort) -- effort cost, drives choice through distance gradient
        cd (c_death)  -- capture aversion, drives vigor through survival incentive

    Population-level:
        epsilon  -- effort efficacy (modulates S for both choice & vigor)
        gamma    -- probability weighting (T_w = T^gamma)
        ce_vigor -- LQR deviation motor cost for vigor
        tau      -- choice temperature
        p_esc    -- escape probability at full speed
        sigma_motor -- motor noise around speed threshold
        sigma_v  -- vigor observation noise

    Choice equation (cd ABSENT -- cancels mathematically):
        dEU = S * 4 - ce_i * (0.81*D_H - 0.16)
        P(heavy) = sigmoid(dEU / tau)
        where S = (1 - T^gamma) + epsilon * T^gamma * p_esc

    Vigor equation (cd drives pressing):
        EU(u) = S(u)*R - (1-S(u))*cd_i*(R+C) - ce_vigor*(u-req)^2*D
        u* = soft_argmax over 30-point grid

    Data used:
        Choice: 45 choice trials (type=1)
        Vigor:  ALL 81 trials (type=1,5,6 -- includes probes)
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
    """Load and prepare data for the 2+2 model.

    Returns separate choice and vigor data arrays.
    Probe trials use startDistance for distance (not distance_H).
    """
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

    # Compute median press rate for all trials
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

    # Cookie-type centering (using choice trial means)
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

    # Choice arrays
    ch_subj = jnp.array([subj_to_idx[s] for s in choice_df['subj']])
    ch_T = jnp.array(choice_df['threat'].values)
    ch_dist_H = jnp.array(
        choice_df['distance_H'].values, dtype=jnp.float64)
    ch_choice = jnp.array(choice_df['choice'].values)

    # Vigor arrays
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
    """Create the 2+2 NumPyro model (epsilon is population-level)."""

    def evc_2plus2(ch_subj, ch_T, ch_dist_H, ch_choice,
                   vig_subj, vig_T, vig_R, vig_req, vig_dist,
                   vig_excess, vig_offset):
        # ── Population priors for hierarchical ce, cd ──
        mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        # ── Population-level epsilon (NOT per-subject) ──
        eps_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
        epsilon = numpyro.deterministic('epsilon', jnp.exp(eps_raw))

        # ── Population-level gamma ──
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic(
            'gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))

        # ── Other population params ──
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample(
            'sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)

        # Population vigor effort cost
        ce_vigor_raw = numpyro.sample(
            'ce_vigor_raw', dist.Normal(-3.0, 1.0))
        ce_vigor = numpyro.deterministic('ce_vigor', jnp.exp(ce_vigor_raw))

        # ── Subject-level (non-centered): ce, cd only ──
        with numpyro.plate('subjects', N_S):
            ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))

        c_effort = jnp.exp(mu_ce + sigma_ce * ce_raw)
        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        numpyro.deterministic('c_effort', c_effort)
        numpyro.deterministic('c_death', c_death)

        # ── CHOICE: cd cancels, ce drives it ──
        ce_ch = c_effort[ch_subj]
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc

        delta_eu = S_ch * 4.0 - ce_ch * (0.81 * ch_dist_H - 0.16)
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample(
                'obs_choice',
                dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                obs=ch_choice)

        # ── VIGOR: cd drives survival incentive ──
        cd_v = c_death[vig_subj]
        T_w_v = jnp.power(vig_T, gamma)

        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T_w_v[:, None])
               + epsilon * T_w_v[:, None] * p_esc
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

    return evc_2plus2


def fit(data, n_steps=40000, lr=0.002, seed=42, print_every=5000):
    """Fit the 2+2 model via SVI."""
    model = make_model(data['N_S'], data['N_choice'], data['N_vigor'])

    kwargs = {k: data[k] for k in [
        'ch_subj', 'ch_T', 'ch_dist_H', 'ch_choice',
        'vig_subj', 'vig_T', 'vig_R', 'vig_req', 'vig_dist',
        'vig_excess', 'vig_offset',
    ]}

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    print(f"Fitting EVC 2+2 (SVI, {n_steps} steps, lr={lr})")
    print(f"  {data['N_S']} subjects, 2 per-subj params (ce, cd)")
    print(f"  Population: epsilon, gamma, ce_vigor, tau, p_esc, sigma_motor, sigma_v")
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
        return_sites=['c_effort', 'c_death', 'epsilon',
                      'gamma', 'ce_vigor', 'excess_pred', 'tau_raw',
                      'p_esc_raw', 'sigma_motor_raw', 'sigma_v'])
    samples = pred(random.PRNGKey(seed), **kwargs)

    ce = np.array(samples['c_effort']).mean(0)
    cd = np.array(samples['c_death']).mean(0)
    eps_val = float(np.array(samples['epsilon']).mean())
    gamma_val = float(np.array(samples['gamma']).mean())
    ce_vigor_val = float(np.array(samples['ce_vigor']).mean())
    tau_val = float(np.exp(np.array(samples['tau_raw']).mean()))
    p_esc_val = float(1.0 / (1.0 + np.exp(-np.array(samples['p_esc_raw']).mean())))
    sigma_motor_val = float(np.exp(np.array(samples['sigma_motor_raw']).mean()))
    sigma_v_val = float(np.array(samples['sigma_v']).mean())
    ep = np.array(samples['excess_pred']).mean(0)

    r_vigor, _ = pearsonr(ep, np.array(data['vig_excess']))

    # BIC: 2 per-subject + population params
    n_params = 2 * data['N_S'] + 9  # 2*N_S (ce,cd) + eps,gamma,ce_vigor,tau,p_esc,sigma_motor,sigma_v,mu_ce,mu_cd...
    # More precisely: mu_ce, sigma_ce, mu_cd, sigma_cd, eps_raw, gamma_raw, tau_raw, p_esc_raw, sigma_motor_raw, ce_vigor_raw, sigma_v + 2*N_S
    n_params = 2 * data['N_S'] + 11
    bic = (2 * fit_result['losses'][-1]
           + n_params * np.log(data['N_choice'] + data['N_vigor']))

    # Per-subject choice r^2
    from scipy.special import expit
    ch_subj_np = np.array(data['ch_subj'])
    ch_T_np = np.array(data['ch_T'])
    ch_dist_np = np.array(data['ch_dist_H'])
    ch_choice_np = np.array(data['ch_choice'])

    T_w = ch_T_np ** gamma_val
    S_ch = (1.0 - T_w) + eps_val * T_w * p_esc_val
    delta_eu = S_ch * 4.0 - ce[ch_subj_np] * (0.81 * ch_dist_np - 0.16)
    logit_ch = np.clip(delta_eu / tau_val, -20, 20)
    p_H = expit(logit_ch)

    # Choice accuracy
    choice_acc = ((p_H >= 0.5).astype(int) == ch_choice_np).mean()

    # Per-subject choice r^2
    import pandas as pd
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

    print(f"\n{'=' * 60}")
    print(f"EVC 2+2 Results (population epsilon)")
    print(f"{'=' * 60}")
    print(f"BIC: {bic:.0f}")
    print(f"Choice accuracy: {choice_acc:.3f}")
    print(f"Per-subject choice r: {r_choice_subj:.3f} (r^2={r_choice_subj**2:.3f})")
    print(f"Vigor r: {r_vigor:.3f} (r^2={r_vigor**2:.3f})")
    print(f"Per-subject vigor r: {r_vigor_subj:.3f} (r^2={r_vigor_subj**2:.3f})")
    print(f"\nPopulation params:")
    print(f"  epsilon={eps_val:.4f}, gamma={gamma_val:.3f}")
    print(f"  ce_vigor={ce_vigor_val:.4f}, tau={tau_val:.3f}")
    print(f"  p_esc={p_esc_val:.3f}, sigma_motor={sigma_motor_val:.3f}, sigma_v={sigma_v_val:.3f}")
    print(f"\nPer-subject ce: median={np.median(ce):.3f}, mean={ce.mean():.3f}, "
          f"SD={ce.std():.3f}, range=[{ce.min():.3f}, {ce.max():.3f}]")
    print(f"Per-subject cd: median={np.median(cd):.3f}, mean={cd.mean():.3f}, "
          f"SD={cd.std():.3f}, range=[{cd.min():.3f}, {cd.max():.3f}]")

    lce, lcd = np.log(ce), np.log(cd)
    r1, p1 = pearsonr(lce, lcd)
    print(f"\nLog correlations: ce x cd = {r1:+.3f} (p={p1:.4f})")

    param_df = pd.DataFrame({
        'subj': data['subjects'],
        'c_effort': ce, 'c_death': cd,
    })

    pop_df = pd.DataFrame([{
        'epsilon': eps_val, 'gamma': gamma_val, 'ce_vigor': ce_vigor_val,
        'tau': tau_val, 'p_esc': p_esc_val,
        'sigma_motor': sigma_motor_val, 'sigma_v': sigma_v_val,
    }])

    return {
        'param_df': param_df, 'pop_df': pop_df,
        'epsilon': eps_val, 'gamma': gamma_val,
        'ce_vigor': ce_vigor_val, 'tau': tau_val,
        'p_esc': p_esc_val, 'sigma_motor': sigma_motor_val,
        'r_vigor': r_vigor, 'r_vigor_subj': r_vigor_subj,
        'r_choice_subj': r_choice_subj, 'choice_acc': choice_acc,
        'bic': bic, 'samples': samples,
        'ce': ce, 'cd': cd,
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

    fit_result = fit(data, n_steps=40000, lr=0.002)
    result = evaluate(fit_result)

    # Save params
    out_path = 'results/stats/oc_evc_final_params.csv'
    result['param_df'].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    pop_path = 'results/stats/oc_evc_final_population.csv'
    result['pop_df'].to_csv(pop_path, index=False)
    print(f"Saved: {pop_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
