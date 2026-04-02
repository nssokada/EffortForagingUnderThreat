"""
EVC + Gamma model for the Effort Foraging Under Threat task.

4-parameter model:
    Subject-level (log-normal, non-centered):
        c_effort  — effort cost sensitivity
        c_death   — capture aversion
        epsilon   — effort efficacy (how much pressing improves survival)

    Population-level:
        gamma     — probability weighting exponent (T_w = T^gamma)

Key innovation: gamma compresses objective threat probabilities,
matching observed vigor-threat calibration. Without gamma, the model
overpredicts threat modulation of vigor by 2-4×.

Results (exploratory N=293):
    BIC = 23,007 | Excess r² = 0.504 | gamma = 0.283
    Threat compression: T=0.1→0.52, T=0.5→0.82, T=0.9→0.97
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


# ── Data preparation ─────────────────────────────────────────────────────────

def prepare_data(behavior_rich_path, psych_path=None):
    """Load and prepare data for the EVC+gamma model."""
    beh = pd.read_csv(behavior_rich_path)
    beh_c = beh[beh['type'] == 1].copy()

    # Compute median press rate per trial
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

    # Cookie-type centering
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

    if psych_path is not None:
        data['psych'] = pd.read_csv(psych_path)

    return data


# ── Model definition ─────────────────────────────────────────────────────────

def make_model(N_S):
    """Create the EVC+gamma NumPyro model with N_S subjects baked in."""

    def oc_evc_gamma(subj_idx, T, dist_H, choice=None, excess_cc=None,
                     chosen_R=None, chosen_req=None, chosen_dist=None,
                     chosen_offset=None):
        # ── Population priors ────────────────────────────────────────────
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

        # Individual epsilon (efficacy)
        mu_eps = numpyro.sample('mu_eps', dist.Normal(-0.5, 0.5))
        sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.3))

        # Population gamma (probability weighting): T_w = T^gamma
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))

        # ── Subject-level (non-centered) ─────────────────────────────────
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

        # ── Weighted threat ──────────────────────────────────────────────
        T_w = jnp.power(T, gamma)

        # ── Choice model ─────────────────────────────────────────────────
        # Binary survival: full-speed press vs stop
        S_full = (1.0 - T_w) + eps_i * T_w * p_esc
        S_stop = 1.0 - T_w

        # Heavy option
        eu_H_full = S_full * 5 - (1 - S_full) * cd_i * 10 - ce_i * 0.81 * dist_H
        eu_H_stop = S_stop * 5 - (1 - S_stop) * cd_i * 10
        eu_H = jnp.maximum(eu_H_full, eu_H_stop)

        # Light option
        eu_L_full = S_full * 1 - (1 - S_full) * cd_i * 6 - ce_i * 0.16
        eu_L_stop = S_stop * 1 - (1 - S_stop) * cd_i * 6
        eu_L = jnp.maximum(eu_L_full, eu_L_stop)

        logit = jnp.clip((eu_H - eu_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        # ── Vigor model ──────────────────────────────────────────────────
        # Continuous EU optimization over 30-point grid
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

        # ── Joint likelihood ─────────────────────────────────────────────
        N_T = subj_idx.shape[0]
        with numpyro.plate('trials', N_T):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                           obs=choice)
            numpyro.sample('obs_vigor',
                           dist.Normal(excess_pred, sigma_v),
                           obs=excess_cc)

    return oc_evc_gamma


# ── Fitting ──────────────────────────────────────────────────────────────────

def fit(data, n_steps=35000, lr=0.002, seed=42, print_every=5000):
    """Fit the EVC+gamma model via SVI."""
    model = make_model(data['N_S'])

    kwargs = {k: data[k] for k in [
        'subj_idx', 'T', 'dist_H', 'choice', 'excess_cc',
        'chosen_R', 'chosen_req', 'chosen_dist', 'chosen_offset',
    ]}

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    print(f"Fitting EVC+gamma (SVI, {n_steps} steps, lr={lr})")
    print(f"  {data['N_S']} subjects, {data['N_T']} trials")

    losses = []
    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        losses.append(float(loss))
        if (i + 1) % print_every == 0:
            print(f"  Step {i+1}: loss={loss:.1f}")

    params_fit = svi.get_params(state)

    return {
        'params': params_fit,
        'losses': losses,
        'guide': guide,
        'model': model,
        'kwargs': kwargs,
        'data': data,
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(fit_result, n_samples=500, seed=44):
    """Extract parameters and evaluate fit quality."""
    guide = fit_result['guide']
    params_fit = fit_result['params']
    data = fit_result['data']

    # Sample from posterior
    obs_kwargs = {k: v for k, v in fit_result['kwargs'].items()
                  if k not in ['choice', 'excess_cc']}
    pred = Predictive(fit_result['model'], guide=guide,
                      params=params_fit, num_samples=n_samples)
    samples = pred(random.PRNGKey(seed), **obs_kwargs)

    # Extract parameters
    ce = np.array(samples['c_effort']).mean(0)
    cd = np.array(samples['c_death']).mean(0)
    eps = np.array(samples['epsilon']).mean(0)
    gamma_val = float(np.array(samples['gamma']).mean())
    ep = np.array(samples['excess_pred']).mean(0)
    eo = np.array(data['excess_cc'])
    ch = np.array(data['choice'])
    T_np = np.array(data['T'])

    # Correlations
    r_all, _ = pearsonr(ep, eo)
    r_ce_cd, _ = pearsonr(ce, cd)
    r_ce_eps, _ = pearsonr(ce, eps)
    r_cd_eps, _ = pearsonr(cd, eps)

    # BIC
    n_params = 3 * data['N_S'] + 13  # 3 subject + ~13 population
    bic = 2 * fit_result['losses'][-1] + n_params * np.log(data['N_T'])

    print(f"\n{'='*60}")
    print(f"EVC + gamma (probability weighting)")
    print(f"{'='*60}")
    print(f"Loss: {fit_result['losses'][-1]:.1f}, BIC: {bic:.1f}")
    print(f"Excess r={r_all:.3f}, r²={r_all**2:.3f}")
    print(f"GAMMA = {gamma_val:.3f}")
    print(f"  T=0.1 → T_w={0.1**gamma_val:.3f}")
    print(f"  T=0.5 → T_w={0.5**gamma_val:.3f}")
    print(f"  T=0.9 → T_w={0.9**gamma_val:.3f}")
    print(f"  Objective range: 0.8, "
          f"Weighted range: {0.9**gamma_val - 0.1**gamma_val:.3f}")
    print()
    print(f"c_effort: mean={ce.mean():.4f} [{ce.min():.4f}, {ce.max():.4f}]")
    print(f"c_death:  mean={cd.mean():.3f} [{cd.min():.3f}, {cd.max():.3f}]")
    print(f"epsilon:  mean={eps.mean():.3f} [{eps.min():.3f}, {eps.max():.3f}], "
          f"sd={eps.std():.4f}")
    print(f"ce×cd:  r={r_ce_cd:.3f}")
    print(f"ce×eps: r={r_ce_eps:.3f}")
    print(f"cd×eps: r={r_cd_eps:.3f}")

    # Per cookie type
    for c_val, label in [(1, 'Heavy'), (0, 'Light')]:
        m = ch == c_val
        r2, _ = pearsonr(ep[m], eo[m])
        print(f"  {label}: r={r2:.3f}")

    # Vigor by threat|choice
    print(f"\nBy threat|choice:")
    for c_val, label in [(1, 'Heavy'), (0, 'Light')]:
        for t in [0.1, 0.5, 0.9]:
            m = (ch == c_val) & np.isclose(T_np, t)
            if m.sum() > 0:
                print(f"  {label} T={t}: pred={ep[m].mean():.4f}, "
                      f"obs={eo[m].mean():.4f}")

    # Threat modulation ratio
    pred_range_h = (ep[(ch == 1) & np.isclose(T_np, 0.9)].mean()
                    - ep[(ch == 1) & np.isclose(T_np, 0.1)].mean())
    obs_range_h = (eo[(ch == 1) & np.isclose(T_np, 0.9)].mean()
                   - eo[(ch == 1) & np.isclose(T_np, 0.1)].mean())
    pred_range_l = (ep[(ch == 0) & np.isclose(T_np, 0.9)].mean()
                    - ep[(ch == 0) & np.isclose(T_np, 0.1)].mean())
    obs_range_l = (eo[(ch == 0) & np.isclose(T_np, 0.9)].mean()
                   - eo[(ch == 0) & np.isclose(T_np, 0.1)].mean())
    print(f"\nThreat modulation (T=0.9 − T=0.1):")
    print(f"  Heavy: pred={pred_range_h:.4f}, obs={obs_range_h:.4f}, "
          f"ratio={pred_range_h/obs_range_h:.2f}")
    print(f"  Light: pred={pred_range_l:.4f}, obs={obs_range_l:.4f}, "
          f"ratio={pred_range_l/obs_range_l:.2f}")

    # Clinical correlations
    if 'psych' in data:
        psych = data['psych']
        param_df = pd.DataFrame({
            'subj': data['subjects'], 'c_effort': ce,
            'c_death': cd, 'epsilon': eps,
        })
        merged = param_df.merge(psych, on='subj')
        psych_cols = [c for c in psych.columns
                      if c not in ['subj', 'participantID']
                      and not c.endswith('_RT')]

        for param in ['c_effort', 'c_death', 'epsilon']:
            print(f"\n{param} vs clinical:")
            results = []
            for col in psych_cols:
                r, p = pearsonr(merged[param], merged[col])
                results.append((col, r, p))
            results.sort(key=lambda x: x[2])
            for col, r, p in results[:5]:
                sig = ('***' if p < .001 else '**' if p < .01
                       else '*' if p < .05 else '')
                print(f"  {col:25s}: r={r:+.3f}, p={p:.4f} {sig}")

    # Save params
    param_df = pd.DataFrame({
        'subj': data['subjects'],
        'c_effort': ce, 'c_death': cd, 'epsilon': eps,
    })

    return {
        'param_df': param_df,
        'gamma': gamma_val,
        'r_excess': r_all,
        'r2_excess': r_all ** 2,
        'bic': bic,
        'samples': samples,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_DIR = 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950'

    data = prepare_data(
        f'{DATA_DIR}/behavior_rich.csv',
        psych_path=f'{DATA_DIR}/psych.csv',
    )

    print(f"Cookie-type centering:")
    print(f"  Heavy mean excess: {data['heavy_mean']:.4f}")
    print(f"  Light mean excess: {data['light_mean']:.4f}")
    print(f"  N_subjects={data['N_S']}, N_trials={data['N_T']}")

    fit_result = fit(data, n_steps=35000, lr=0.002)
    result = evaluate(fit_result)

    out_path = 'results/stats/oc_evc_gamma_params.csv'
    result['param_df'].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
