"""
Definitive EVC-LQR model for Effort Foraging Under Threat.

Architecture:
    Per-subject (log-normal, non-centered):
        c_death   — capture aversion / survival incentive
        epsilon   — effort efficacy (how much pressing improves survival)

    Population-level:
        c_effort  — effort cost (commitment cost for choice, deviation cost for vigor)
        gamma     — probability weighting (T_w = T^gamma)
        tau       — choice temperature
        p_esc     — escape probability at full speed
        sigma_motor — motor noise around speed threshold
        sigma_v   — vigor observation noise

    Effort cost functional form (LQR-inspired):
        Choice:  ce × req² × D  (commitment cost — total effort of engaging this option)
        Vigor:   ce × (u - req)² × D  (deviation cost — cost of pressing above/below setpoint)

    At the committed rate (u = req), vigor effort = 0. The cost of pressing
    at the required rate is already paid at the choice stage. Only deviations
    from the setpoint incur additional motor cost.

    This resolves the choice-vigor conflict: choice wants ce ~ 0.5 (to explain
    distance gradients), vigor wants ce ~ 0.003 (to avoid suppressing press rates).
    The LQR formulation naturally produces large choice costs (req² × D ~ 0.8-2.4)
    and tiny vigor costs ((u-req)² × D ~ 0.01-0.05) from the SAME ce parameter.

Results (exploratory N=293):
    BIC = 22,836 (best of all specifications tested)
    Vigor r² = 0.502
    gamma = 0.315 (probability compression)

    Compared to original (ce × u² × D everywhere):
        BIC improvement: -152
        Same vigor r², same parameter count
        Theoretically grounded in LQR optimal control theory
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
    """Load and prepare data for the EVC-LQR model."""
    beh = pd.read_csv(behavior_rich_path)
    beh_c = beh[beh['type'] == 1].copy()

    # Compute median press rate per trial
    rates = []
    for _, row in beh_c.iterrows():
        try:
            press_times = np.array(
                ast.literal_eval(row['alignedEffortRate']), dtype=float
            )
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
    beh_c['req_rate'] = np.where(
        beh_c['trialCookie_weight'] == 3.0, 0.9, 0.4
    )
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
    """Create the EVC-LQR NumPyro model."""

    def evc_lqr(subj_idx, T, dist_H, choice=None, excess_cc=None,
                chosen_R=None, chosen_req=None, chosen_dist=None,
                chosen_offset=None):
        # ── Population priors ────────────────────────────────────────
        # Effort cost (population-level)
        mu_ce_raw = numpyro.sample('mu_ce_raw', dist.Normal(0.0, 1.0))
        c_effort = numpyro.deterministic(
            'c_effort', jnp.clip(jnp.exp(mu_ce_raw), 1e-6, 100.0)
        )

        # Death aversion (per-subject)
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        # Efficacy (per-subject)
        mu_eps = numpyro.sample('mu_eps', dist.Normal(-0.5, 0.5))
        sigma_eps = numpyro.sample('sigma_eps', dist.HalfNormal(0.3))

        # Choice temperature
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)

        # Escape probability
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)

        # Vigor noise
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))

        # Motor noise
        sigma_motor_raw = numpyro.sample(
            'sigma_motor_raw', dist.Normal(-1.0, 0.5)
        )
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)

        # Probability weighting
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic(
            'gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0)
        )

        # ── Subject-level (non-centered) ─────────────────────────────
        with numpyro.plate('subjects', N_S):
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))
            eps_raw = numpyro.sample('eps_raw', dist.Normal(0.0, 1.0))

        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        epsilon = jnp.exp(mu_eps + sigma_eps * eps_raw)

        numpyro.deterministic('c_death', c_death)
        numpyro.deterministic('epsilon', epsilon)

        cd_i = c_death[subj_idx]
        eps_i = epsilon[subj_idx]

        # ── Weighted threat ──────────────────────────────────────────
        T_w = jnp.power(T, gamma)

        # ── Choice: commitment cost = ce × req² × D ─────────────────
        S_full = (1.0 - T_w) + eps_i * T_w * p_esc
        S_stop = 1.0 - T_w

        # Heavy option (req=0.9, distance varies)
        eu_H_full = (S_full * 5.0
                     - (1.0 - S_full) * cd_i * 10.0
                     - c_effort * 0.81 * dist_H)
        eu_H_stop = (S_stop * 5.0
                     - (1.0 - S_stop) * cd_i * 10.0)
        eu_H = jnp.maximum(eu_H_full, eu_H_stop)

        # Light option (req=0.4, distance=1)
        eu_L_full = (S_full * 1.0
                     - (1.0 - S_full) * cd_i * 6.0
                     - c_effort * 0.16)
        eu_L_stop = (S_stop * 1.0
                     - (1.0 - S_stop) * cd_i * 6.0)
        eu_L = jnp.maximum(eu_L_full, eu_L_stop)

        logit = jnp.clip((eu_H - eu_L) / tau, -20.0, 20.0)
        p_H = jax.nn.sigmoid(logit)

        # ── Vigor: deviation cost = ce × (u - req)² × D ─────────────
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]

        S_u = ((1.0 - T_w[:, None])
               + eps_i[:, None] * T_w[:, None] * p_esc
               * jax.nn.sigmoid(
                   (u_g - chosen_req[:, None]) / sigma_motor
               ))

        # LQR deviation cost
        deviation = u_g - chosen_req[:, None]
        effort_vigor = c_effort * deviation ** 2 * chosen_dist[:, None]

        eu_grid = (S_u * chosen_R[:, None]
                   - (1.0 - S_u) * cd_i[:, None]
                   * (chosen_R[:, None] + 5.0)
                   - effort_vigor)

        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - chosen_req - chosen_offset

        numpyro.deterministic('excess_pred', excess_pred)

        # ── Joint likelihood ─────────────────────────────────────────
        N_T = subj_idx.shape[0]
        with numpyro.plate('trials', N_T):
            numpyro.sample(
                'obs_choice',
                dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                obs=choice,
            )
            numpyro.sample(
                'obs_vigor',
                dist.Normal(excess_pred, sigma_v),
                obs=excess_cc,
            )

    return evc_lqr


# ── Fitting ──────────────────────────────────────────────────────────────────

def fit(data, n_steps=40000, lr=0.002, seed=42, print_every=5000):
    """Fit the EVC-LQR model via SVI."""
    model = make_model(data['N_S'])

    kwargs = {k: data[k] for k in [
        'subj_idx', 'T', 'dist_H', 'choice', 'excess_cc',
        'chosen_R', 'chosen_req', 'chosen_dist', 'chosen_offset',
    ]}

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    print(f"Fitting EVC-LQR (SVI, {n_steps} steps, lr={lr})")
    print(f"  {data['N_S']} subjects, {data['N_T']} trials")
    print(f"  2 per-subject params (c_death, epsilon)")
    print(f"  Population: c_effort, gamma, tau, p_esc, sigma_motor, sigma_v")

    losses = []
    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        losses.append(float(loss))
        if (i + 1) % print_every == 0:
            print(f"  Step {i + 1}: loss={loss:.1f}")

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

    obs_kwargs = {k: v for k, v in fit_result['kwargs'].items()
                  if k not in ['choice', 'excess_cc']}
    pred = Predictive(fit_result['model'], guide=guide,
                      params=params_fit, num_samples=n_samples)
    samples = pred(random.PRNGKey(seed), **obs_kwargs)

    # Extract parameters
    cd = np.array(samples['c_death']).mean(0)
    eps = np.array(samples['epsilon']).mean(0)
    ce = float(np.array(samples['c_effort']).mean())
    gamma_val = float(np.array(samples['gamma']).mean())
    ep = np.array(samples['excess_pred']).mean(0)
    eo = np.array(data['excess_cc'])
    ch = np.array(data['choice'])
    T_np = np.array(data['T'])

    # Vigor correlation
    r_vigor, _ = pearsonr(ep, eo)

    # BIC
    n_params = 2 * data['N_S'] + 10  # 2 subject + ~10 population
    bic = 2 * fit_result['losses'][-1] + n_params * np.log(data['N_T'])

    print(f"\n{'=' * 60}")
    print(f"EVC-LQR Model Results")
    print(f"{'=' * 60}")
    print(f"BIC: {bic:.0f}")
    print(f"Vigor: r={r_vigor:.3f}, r²={r_vigor**2:.3f}")
    print(f"c_effort (pop): {ce:.4f}")
    print(f"gamma: {gamma_val:.3f}")
    print(f"c_death:  median={np.median(cd):.3f}, mean={cd.mean():.3f}, "
          f"range=[{cd.min():.3f}, {cd.max():.3f}]")
    print(f"epsilon:  median={np.median(eps):.3f}, mean={eps.mean():.3f}, "
          f"range=[{eps.min():.3f}, {eps.max():.3f}]")

    # Log-space correlation
    lcd, leps = np.log(cd), np.log(eps)
    r_cd_eps, p_cd_eps = pearsonr(lcd, leps)
    print(f"log(cd)×log(eps): r={r_cd_eps:+.3f} (p={p_cd_eps:.4f})")

    # Vigor by condition
    print(f"\nVigor by threat|choice:")
    for c_val, label in [(1, 'Heavy'), (0, 'Light')]:
        for t in [0.1, 0.5, 0.9]:
            m = (ch == c_val) & np.isclose(T_np, t)
            if m.sum() > 0:
                print(f"  {label} T={t}: pred={ep[m].mean():.4f}, "
                      f"obs={eo[m].mean():.4f}")

    # Clinical correlations
    if 'psych' in data:
        psych = data['psych']
        param_df = pd.DataFrame({
            'subj': data['subjects'],
            'c_death': cd, 'epsilon': eps,
        })
        merged = param_df.merge(psych, on='subj')
        psych_cols = [c for c in psych.columns
                      if c not in ['subj', 'participantID']
                      and not c.endswith('_RT')]

        print(f"\nClinical correlations (log space, p<0.1):")
        for param in ['c_death', 'epsilon']:
            for col in psych_cols:
                r, p = pearsonr(np.log(merged[param]), merged[col])
                if p < 0.1:
                    sig = ('***' if p < .001 else '**' if p < .01
                           else '*' if p < .05 else '†')
                    print(f"  log({param:8s}) → {col:20s}: "
                          f"r={r:+.3f}, p={p:.4f} {sig}")

    # Save parameters
    param_df = pd.DataFrame({
        'subj': data['subjects'],
        'c_death': cd, 'epsilon': eps,
    })

    return {
        'param_df': param_df,
        'c_effort': ce,
        'gamma': gamma_val,
        'r_excess': r_vigor,
        'r2_excess': r_vigor ** 2,
        'bic': bic,
        'samples': samples,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_DIR = ('data/exploratory_350/processed/'
                'stage5_filtered_data_20260320_191950')

    data = prepare_data(
        f'{DATA_DIR}/behavior_rich.csv',
        psych_path=f'{DATA_DIR}/psych.csv',
    )

    print(f"Cookie-type centering:")
    print(f"  Heavy mean excess: {data['heavy_mean']:.4f}")
    print(f"  Light mean excess: {data['light_mean']:.4f}")
    print(f"  N_subjects={data['N_S']}, N_trials={data['N_T']}")

    fit_result = fit(data, n_steps=40000, lr=0.002)
    result = evaluate(fit_result)

    out_path = 'results/stats/oc_evc_lqr_final_params.csv'
    result['param_df'].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Also save population params
    pop_path = 'results/stats/oc_evc_lqr_final_population.csv'
    pd.DataFrame([{
        'c_effort': result['c_effort'],
        'gamma': result['gamma'],
        'bic': result['bic'],
        'vigor_r2': result['r2_excess'],
    }]).to_csv(pop_path, index=False)
    print(f"Saved: {pop_path}")
