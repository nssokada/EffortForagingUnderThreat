"""
Branch B: Three-parameter model with upgraded vigor
- Choice: ΔEU = 4 - k × effort(D) - β × T (same as 3-param v2)
- Vigor: frac_full with Gaussian CDF survival + log-odds effort cost
  S(f, T, D) = (1-T) + T × P_escape(f, D)
  P_escape = p_floor + (1-p_floor) × (1 - Φ((arrival - μ_strike - buffer) / σ_strike))
  EU(f) = S(f,T,D)×R - (1-S(f,T,D))×cd×(R+C) - α×[log(f/(1-f))]²×D
  f* = soft_argmax over f grid
"""

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
from scipy.stats import pearsonr, norm as sp_norm
from scipy.special import expit
from pathlib import Path

jax.config.update('jax_enable_x64', True)

# Fixed survival function parameters (from Phase 1)
V_FULL = 1.450
REMAINING_FRAC = 0.900
BUFFER = 0.600
P_FLOOR = 0.090
D_GAME = {1: 5.0, 2: 7.0, 3: 9.0}
MU_STRIKE = {1: 2.460, 2: 3.478, 3: 4.765}
SD_STRIKE = {1: 0.574, 2: 0.896, 3: 1.121}
T_ENC = {1: 2.5, 2: 3.5, 3: 5.0}

# JAX arrays for the survival function (indexed by dist-1)
D_GAME_ARR = jnp.array([5.0, 7.0, 9.0])
MU_STRIKE_ARR = jnp.array([2.460, 3.478, 4.765])
SD_STRIKE_ARR = jnp.array([0.574, 0.896, 1.121])


def prepare_data(behavior_rich_path, psych_path=None):
    """Load data, compute frac_full for all trials."""
    beh = pd.read_csv(behavior_rich_path)
    beh['actual_req'] = np.where(beh['trialCookie_weight'] == 3.0, 0.9, 0.4)
    beh['actual_dist'] = beh['startDistance'].map({5: 1, 7: 2, 9: 3})
    beh['actual_R'] = np.where(beh['trialCookie_weight'] == 3.0, 5.0, 1.0)
    beh['is_heavy'] = (beh['trialCookie_weight'] == 3.0).astype(int)

    ff_list = []
    for _, row in beh.iterrows():
        try:
            pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
            ipis = np.diff(pt)
            ipis = ipis[ipis > 0.01]
            if len(ipis) < 5:
                ff_list.append(np.nan)
                continue
            rates = (1.0 / ipis) / row['calibrationMax']
            ff_list.append(np.mean(rates >= row['actual_req']))
        except:
            ff_list.append(np.nan)

    beh['frac_full'] = ff_list

    # Choice data
    choice_df = beh[beh['type'] == 1].copy()

    # Vigor data (all types)
    vigor_df = beh.dropna(subset=['frac_full']).copy()

    # Cookie-type centering
    choice_vigor = vigor_df[vigor_df['type'] == 1]
    heavy_ff_mean = choice_vigor[choice_vigor['is_heavy'] == 1]['frac_full'].mean()
    light_ff_mean = choice_vigor[choice_vigor['is_heavy'] == 0]['frac_full'].mean()
    vigor_df['frac_full_cc'] = vigor_df['frac_full'] - np.where(
        vigor_df['is_heavy'] == 1, heavy_ff_mean, light_ff_mean)

    # Shared subjects
    subjects = sorted(set(choice_df['subj'].unique()) & set(vigor_df['subj'].unique()))
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)
    choice_df = choice_df[choice_df['subj'].isin(subjects)]
    vigor_df = vigor_df[vigor_df['subj'].isin(subjects)]

    data = {
        'ch_subj': jnp.array([subj_to_idx[s] for s in choice_df['subj']]),
        'ch_T': jnp.array(choice_df['threat'].values),
        'ch_dist_H': jnp.array(choice_df['distance_H'].values, dtype=jnp.float64),
        'ch_choice': jnp.array(choice_df['choice'].values),
        'vig_subj': jnp.array([subj_to_idx[s] for s in vigor_df['subj']]),
        'vig_T': jnp.array(vigor_df['threat'].values),
        'vig_R': jnp.array(vigor_df['actual_R'].values),
        'vig_dist': jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64),
        'vig_ff': jnp.array(vigor_df['frac_full_cc'].values),
        'vig_ff_offset': jnp.array(np.where(
            vigor_df['is_heavy'].values == 1, heavy_ff_mean, light_ff_mean)),
        'subjects': subjects, 'N_S': N_S,
        'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
        'heavy_ff_mean': heavy_ff_mean, 'light_ff_mean': light_ff_mean,
    }
    if psych_path:
        data['psych'] = pd.read_csv(psych_path)
    return data


def make_model(N_S, N_choice, N_vigor):
    def branchB(ch_subj, ch_T, ch_dist_H, ch_choice,
                vig_subj, vig_T, vig_R, vig_dist, vig_ff, vig_ff_offset):

        # Hierarchical priors
        mu_k = numpyro.sample('mu_k', dist.Normal(0.0, 1.0))
        mu_beta = numpyro.sample('mu_beta', dist.Normal(1.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(2.0, 1.0))
        sigma_k = numpyro.sample('sigma_k', dist.HalfNormal(0.5))
        sigma_beta = numpyro.sample('sigma_beta', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)

        # Log-odds effort cost scaling
        alpha_raw = numpyro.sample('alpha_raw', dist.Normal(-2.0, 1.0))
        alpha = numpyro.deterministic('alpha', jnp.clip(jnp.exp(alpha_raw), 0.001, 50.0))

        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.3))

        # Subject-level
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

        # ═══════════════════════════════════════════════════════
        # CHOICE: same as 3-param v2
        # ═══════════════════════════════════════════════════════
        k_ch = k[ch_subj]
        beta_ch = beta[ch_subj]
        effort_cost = 0.81 * ch_dist_H - 0.16
        delta_eu = 4.0 - k_ch * effort_cost - beta_ch * ch_T
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                obs=ch_choice)

        # ═══════════════════════════════════════════════════════
        # VIGOR: frac_full with Gaussian CDF survival + log-odds cost
        # ═══════════════════════════════════════════════════════
        cd_v = c_death[vig_subj]

        # f grid
        f_grid = jnp.linspace(0.02, 0.98, 30)
        f_g = f_grid[None, :]  # (1, 30)

        # Survival: S(f, T, D) = (1-T) + T × P_escape(f, D)
        # P_escape = p_floor + (1-p_floor) × (1 - Φ((arrival - mu - buffer) / sigma))
        d_game_v = jnp.where(vig_dist == 1, 5.0, jnp.where(vig_dist == 2, 7.0, 9.0))
        mu_s = jnp.where(vig_dist == 1, MU_STRIKE_ARR[0],
               jnp.where(vig_dist == 2, MU_STRIKE_ARR[1], MU_STRIKE_ARR[2]))
        sd_s = jnp.where(vig_dist == 1, SD_STRIKE_ARR[0],
               jnp.where(vig_dist == 2, SD_STRIKE_ARR[1], SD_STRIKE_ARR[2]))

        remaining = d_game_v * REMAINING_FRAC  # (N,)
        eff_speed = V_FULL * (0.5 + 0.5 * f_g)  # (1, 30)
        arrival = remaining[:, None] / eff_speed  # (N, 30)
        z_strike = (arrival - (mu_s[:, None] + BUFFER)) / sd_s[:, None]
        p_gaussian = 1.0 - jax.scipy.stats.norm.cdf(z_strike)
        p_escape = P_FLOOR + (1.0 - P_FLOOR) * p_gaussian  # (N, 30)
        S_f = (1.0 - vig_T[:, None]) + vig_T[:, None] * p_escape  # (N, 30)

        # Log-odds effort cost: g(f) = [log(f/(1-f))]² × req² × D
        logit_f = jnp.log(f_g / (1.0 - f_g))  # (1, 30)
        vig_req = jnp.where(vig_R == 5.0, 0.9, 0.4)  # req from cookie type
        effort_f = alpha * logit_f**2 * (vig_req[:, None]**2) * vig_dist[:, None]  # (N, 30)

        # EU(f) = S×R - (1-S)×cd×(R+C) - effort
        eu_grid = (S_f * vig_R[:, None]
                   - (1.0 - S_f) * cd_v[:, None] * (vig_R[:, None] + 5.0)
                   - effort_f)

        # Softmax → optimal f*
        weights = jax.nn.softmax(eu_grid * 5.0, axis=1)
        f_star = jnp.sum(weights * f_g, axis=1)

        f_pred = f_star - vig_ff_offset
        numpyro.deterministic('f_pred', f_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor',
                dist.Normal(f_pred, sigma_v),
                obs=vig_ff)

    return branchB


def fit(data, n_steps=40000, lr=0.0005, seed=42):
    model = make_model(data['N_S'], data['N_choice'], data['N_vigor'])
    kwargs = {k: data[k] for k in [
        'ch_subj', 'ch_T', 'ch_dist_H', 'ch_choice',
        'vig_subj', 'vig_T', 'vig_R', 'vig_dist', 'vig_ff', 'vig_ff_offset',
    ]}

    guide = AutoNormal(model)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    print(f"Fitting Branch B (3-param + frac_full + Gaussian CDF survival)")
    print(f"  {data['N_S']} subjects, {data['N_choice']} choice, {data['N_vigor']} vigor trials")

    best_loss = float('inf')
    best_params = None
    best_step = 0

    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        loss_val = float(loss)
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = svi.get_params(state)
            best_step = i + 1
        if (i + 1) % 5000 == 0:
            print(f"  Step {i+1}: loss={loss_val:.1f} (best={best_loss:.1f} at step {best_step})")

    print(f"  Using best params from step {best_step} (loss={best_loss:.1f})")
    return {'params': best_params, 'guide': guide, 'model': model, 'kwargs': kwargs, 'data': data, 'loss': best_loss, 'best_step': best_step}


def evaluate(fit_result, n_samples=300, seed=44):
    guide = fit_result['guide']
    model = fit_result['model']
    params_fit = fit_result['params']
    data = fit_result['data']
    kwargs = fit_result['kwargs']

    pred = Predictive(model, guide=guide, params=params_fit, num_samples=n_samples,
                      return_sites=['k', 'beta', 'c_death', 'alpha', 'f_pred', 'tau_raw', 'sigma_v'])
    samples = pred(random.PRNGKey(seed), **kwargs)

    k_vals = np.array(samples['k']).mean(0)
    beta_vals = np.array(samples['beta']).mean(0)
    cd_vals = np.array(samples['c_death']).mean(0)
    alpha_val = float(np.array(samples['alpha']).mean())
    tau_val = float(np.exp(np.array(samples['tau_raw']).mean()))
    sigma_v_val = float(np.array(samples['sigma_v']).mean())
    f_pred = np.array(samples['f_pred']).mean(0)

    # Vigor fit
    f_obs = np.array(data['vig_ff'])
    r_vigor, _ = pearsonr(f_pred, f_obs)

    vig_subj_np = np.array(data['vig_subj'])
    vdf = pd.DataFrame({'subj': vig_subj_np, 'obs': f_obs, 'pred': f_pred})
    sv = vdf.groupby('subj').agg(o=('obs', 'mean'), p=('pred', 'mean')).reset_index()
    r_vigor_subj, _ = pearsonr(sv['o'], sv['p'])

    # Choice fit
    ch_subj_np = np.array(data['ch_subj'])
    ch_T_np = np.array(data['ch_T'])
    ch_dist_np = np.array(data['ch_dist_H'])
    ch_choice_np = np.array(data['ch_choice'])

    effort = 0.81 * ch_dist_np - 0.16
    deu = 4.0 - k_vals[ch_subj_np] * effort - beta_vals[ch_subj_np] * ch_T_np
    p_H = expit(np.clip(deu / tau_val, -20, 20))
    choice_acc = ((p_H >= 0.5).astype(int) == ch_choice_np).mean()

    cdf = pd.DataFrame({'subj': ch_subj_np, 'choice': ch_choice_np, 'p_H': p_H})
    sc = cdf.groupby('subj').agg(o=('choice', 'mean'), p=('p_H', 'mean')).reset_index()
    r_choice_subj, _ = pearsonr(sc['o'], sc['p'])

    print(f"\n{'=' * 60}")
    print(f"BRANCH B RESULTS")
    print(f"{'=' * 60}")
    print(f"Choice accuracy: {choice_acc:.3f}")
    print(f"Per-subject choice r: {r_choice_subj:.3f} (r²={r_choice_subj**2:.3f})")
    print(f"Trial-level vigor r: {r_vigor:.3f} (r²={r_vigor**2:.3f})")
    print(f"Per-subject vigor r: {r_vigor_subj:.3f} (r²={r_vigor_subj**2:.3f})")
    print(f"\nPopulation: α={alpha_val:.4f}, τ={tau_val:.3f}, σ_v={sigma_v_val:.3f}")
    print(f"\nPer-subject:")
    print(f"  k: median={np.median(k_vals):.3f}, range=[{k_vals.min():.3f}, {k_vals.max():.3f}]")
    print(f"  β: median={np.median(beta_vals):.3f}, range=[{beta_vals.min():.3f}, {beta_vals.max():.3f}]")
    print(f"  cd: median={np.median(cd_vals):.3f}, range=[{cd_vals.min():.3f}, {cd_vals.max():.3f}]")

    lk, lb, lcd = np.log(k_vals), np.log(beta_vals), np.log(cd_vals)
    r_kb, _ = pearsonr(lk, lb)
    r_kcd, _ = pearsonr(lk, lcd)
    r_bcd, _ = pearsonr(lb, lcd)
    print(f"\n  k×β: r={r_kb:.3f}, k×cd: r={r_kcd:.3f}, β×cd: r={r_bcd:.3f}")

    param_df = pd.DataFrame({'subj': data['subjects'], 'k': k_vals, 'beta': beta_vals, 'c_death': cd_vals})
    return {
        'param_df': param_df, 'k': k_vals, 'beta': beta_vals, 'cd': cd_vals,
        'alpha': alpha_val, 'tau': tau_val, 'sigma_v': sigma_v_val,
        'r_choice_subj': r_choice_subj, 'r_vigor': r_vigor, 'r_vigor_subj': r_vigor_subj,
        'choice_acc': choice_acc,
    }


if __name__ == '__main__':
    import time
    t0 = time.time()

    DATA_DIR = 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
    data = prepare_data(f'{DATA_DIR}/behavior_rich.csv', f'{DATA_DIR}/psych.csv')
    print(f"N_subjects={data['N_S']}, N_choice={data['N_choice']}, N_vigor={data['N_vigor']}")

    fit_result = fit(data)
    result = evaluate(fit_result)

    result['param_df'].to_csv('results/stats/branchB_params.csv', index=False)
    pd.DataFrame([{
        'alpha': result['alpha'], 'tau': result['tau'], 'sigma_v': result['sigma_v'],
        'v_full': V_FULL, 'remaining_frac': REMAINING_FRAC, 'buffer': BUFFER, 'p_floor': P_FLOOR,
    }]).to_csv('results/stats/branchB_population.csv', index=False)

    print(f"\n{'=' * 60}")
    print(f"COMPARISON")
    print(f"{'=' * 60}")
    print(f"                    3-param v2 (orig)    Branch B (frac_full)")
    print(f"Choice r²:              0.981               {result['r_choice_subj']**2:.3f}")
    print(f"Vigor r² (trial):       0.424               {result['r_vigor']**2:.3f}")
    print(f"Vigor r² (subj):        0.669               {result['r_vigor_subj']**2:.3f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
