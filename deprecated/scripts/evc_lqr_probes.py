"""
EVC-LQR model with probe trials for vigor.

Key change from evc_lqr_final.py:
    - Choice likelihood: type=1 trials only (N~13k)
    - Vigor likelihood: ALL trials (type=1,5,6) (N~23k)
    - c_effort is now per-subject (probe vigor at different distances constrains it)
    - 3 per-subject params: c_effort, c_death, epsilon

Probe trial handling:
    - Distance from startDistance: 5->1, 7->2, 9->3 (NOT distance_H which is wrong for probes)
    - req_rate: 0.9 for heavy (trialCookie_weight=3.0), 0.4 for light (weight=1.0)
    - reward: 5 for heavy, 1 for light
    - Cookie-type centering uses heavy_mean/light_mean from choice trials only
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

def compute_median_rate(row):
    """Compute median normalized press rate for a trial."""
    try:
        press_times = np.array(
            ast.literal_eval(row['alignedEffortRate']), dtype=float
        )
    except Exception:
        return np.nan
    ipis = np.diff(press_times)
    ipis = ipis[ipis > 0.01]
    if len(ipis) < 5:
        return np.nan
    return np.median((1.0 / ipis) / row['calibrationMax'])


def prepare_data(behavior_rich_path, psych_path=None):
    """Load and prepare data for the EVC-LQR probes model."""
    beh = pd.read_csv(behavior_rich_path)

    # ── Choice trials (type=1) ────────────────────────────────────
    beh_c = beh[beh['type'] == 1].copy()
    beh_c['median_rate'] = [compute_median_rate(row) for _, row in beh_c.iterrows()]
    beh_c['req_rate'] = np.where(beh_c['trialCookie_weight'] == 3.0, 0.9, 0.4)
    beh_c['excess'] = beh_c['median_rate'] - beh_c['req_rate']
    beh_c = beh_c.dropna(subset=['excess']).copy()

    # Cookie-type centering (from choice trials only)
    heavy_mean = beh_c[beh_c['trialCookie_weight'] == 3.0]['excess'].mean()
    light_mean = beh_c[beh_c['trialCookie_weight'] == 1.0]['excess'].mean()
    beh_c['excess_cc'] = beh_c['excess'] - np.where(
        beh_c['trialCookie_weight'] == 3.0, heavy_mean, light_mean
    )

    # ── ALL trials (type=1,5,6) for vigor ─────────────────────────
    beh_all = beh[beh['type'].isin([1, 5, 6])].copy()
    beh_all['median_rate'] = [compute_median_rate(row) for _, row in beh_all.iterrows()]

    # For probes: distance from startDistance, req from weight
    # Map startDistance: 5->1, 7->2, 9->3
    dist_map = {5: 1, 7: 2, 9: 3}
    beh_all['probe_dist'] = beh_all['startDistance'].map(dist_map)

    # For choice trials, distance depends on actual choice
    # For probes, distance is from startDistance
    beh_all['vigor_dist'] = np.where(
        beh_all['type'] == 1,
        np.where(beh_all['choice'] == 1, beh_all['distance_H'], 1.0),
        beh_all['probe_dist'].values.astype(float)
    )

    # req_rate and reward based on the cookie being transported
    # For choice trials: choice=1 -> heavy, choice=0 -> light
    # For probe trials: trialCookie_weight determines it
    beh_all['vigor_req'] = np.where(
        beh_all['trialCookie_weight'] == 3.0,
        0.9, 0.4
    )
    # But for choice trials where choice=0, it's light regardless
    # Actually for all trials: if choice=1, they're transporting heavy (req=0.9)
    # if choice=0, they're transporting light (req=0.4)
    # But for probes, choice is forced: choice=1 means heavy, choice=0 means light
    # So we can just use choice uniformly:
    beh_all['vigor_req'] = np.where(beh_all['choice'] == 1, 0.9, 0.4)
    beh_all['vigor_R'] = np.where(beh_all['choice'] == 1, 5.0, 1.0)

    beh_all['excess'] = beh_all['median_rate'] - beh_all['vigor_req']
    beh_all = beh_all.dropna(subset=['excess']).copy()

    # Apply same cookie-type centering offsets from choice trials
    beh_all['vigor_offset'] = np.where(
        beh_all['choice'] == 1, heavy_mean, light_mean
    )
    beh_all['excess_cc'] = beh_all['excess'] - beh_all['vigor_offset']

    # ── Build subject mapping ─────────────────────────────────────
    # Use subjects that appear in BOTH choice and vigor
    choice_subjs = set(beh_c['subj'].unique())
    vigor_subjs = set(beh_all['subj'].unique())
    common_subjs = sorted(choice_subjs & vigor_subjs)
    subj_to_idx = {s: i for i, s in enumerate(common_subjs)}
    N_S = len(common_subjs)

    # Filter to common subjects
    beh_c = beh_c[beh_c['subj'].isin(common_subjs)].copy()
    beh_all = beh_all[beh_all['subj'].isin(common_subjs)].copy()

    N_choice = len(beh_c)
    N_vigor = len(beh_all)

    print(f"Data summary:")
    print(f"  Subjects: {N_S}")
    print(f"  Choice trials (type=1): {N_choice}")
    print(f"  Vigor trials (type=1,5,6): {N_vigor}")
    print(f"    Type 1: {(beh_all['type']==1).sum()}")
    print(f"    Type 5: {(beh_all['type']==5).sum()}")
    print(f"    Type 6: {(beh_all['type']==6).sum()}")
    print(f"  Cookie-type centering (from choice trials):")
    print(f"    Heavy mean excess: {heavy_mean:.4f}")
    print(f"    Light mean excess: {light_mean:.4f}")

    # ── Choice arrays ─────────────────────────────────────────────
    c_subj_idx = jnp.array([subj_to_idx[s] for s in beh_c['subj']])
    c_T = jnp.array(beh_c['threat'].values)
    c_dist_H = jnp.array(beh_c['distance_H'].values, dtype=jnp.float64)
    c_choice = jnp.array(beh_c['choice'].values)

    # ── Vigor arrays (ALL trials) ─────────────────────────────────
    v_subj_idx = jnp.array([subj_to_idx[s] for s in beh_all['subj']])
    v_T = jnp.array(beh_all['threat'].values)
    v_dist = jnp.array(beh_all['vigor_dist'].values, dtype=jnp.float64)
    v_req = jnp.array(beh_all['vigor_req'].values)
    v_R = jnp.array(beh_all['vigor_R'].values)
    v_offset = jnp.array(beh_all['vigor_offset'].values)
    v_excess_cc = jnp.array(beh_all['excess_cc'].values)

    data = {
        # Choice data
        'c_subj_idx': c_subj_idx, 'c_T': c_T, 'c_dist_H': c_dist_H,
        'c_choice': c_choice, 'N_choice': N_choice,
        # Vigor data
        'v_subj_idx': v_subj_idx, 'v_T': v_T, 'v_dist': v_dist,
        'v_req': v_req, 'v_R': v_R, 'v_offset': v_offset,
        'v_excess_cc': v_excess_cc, 'N_vigor': N_vigor,
        # Shared
        'subjects': common_subjs, 'N_S': N_S,
        'heavy_mean': heavy_mean, 'light_mean': light_mean,
        # DataFrames for evaluation
        'beh_c': beh_c, 'beh_all': beh_all,
    }

    if psych_path is not None:
        data['psych'] = pd.read_csv(psych_path)

    return data


# ── Model definition ─────────────────────────────────────────────────────────

def make_model(N_S):
    """Create the EVC-LQR probes NumPyro model.

    3 per-subject params: c_effort, c_death, epsilon
    Population: gamma, tau, p_esc, sigma_motor, sigma_v
    """

    def evc_lqr_probes(
        c_subj_idx, c_T, c_dist_H, c_choice,
        v_subj_idx, v_T, v_dist, v_req, v_R, v_offset,
        v_excess_cc=None,
    ):
        # ── Population priors ────────────────────────────────────────
        # c_effort (per-subject now)
        mu_ce = numpyro.sample('mu_ce', dist.Normal(-2.0, 1.0))
        sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))

        # c_death (per-subject)
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        # epsilon (per-subject)
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
            ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))
            eps_raw = numpyro.sample('eps_raw', dist.Normal(0.0, 1.0))

        c_effort = jnp.exp(mu_ce + sigma_ce * ce_raw)
        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        epsilon = jnp.exp(mu_eps + sigma_eps * eps_raw)

        numpyro.deterministic('c_effort', c_effort)
        numpyro.deterministic('c_death', c_death)
        numpyro.deterministic('epsilon', epsilon)

        # ══════════════════════════════════════════════════════════════
        # CHOICE LIKELIHOOD (type=1 trials only)
        # ══════════════════════════════════════════════════════════════
        ce_c = c_effort[c_subj_idx]
        cd_c = c_death[c_subj_idx]
        eps_c = epsilon[c_subj_idx]

        T_w_c = jnp.power(c_T, gamma)

        # Survival: S = (1-T_w) + eps * T_w * p_esc  (no distance in choice survival)
        S_full_c = (1.0 - T_w_c) + eps_c * T_w_c * p_esc
        S_stop_c = 1.0 - T_w_c

        # Heavy option (req=0.9, distance varies)
        eu_H_full = (S_full_c * 5.0
                     - (1.0 - S_full_c) * cd_c * 10.0
                     - ce_c * 0.81 * c_dist_H)
        eu_H_stop = (S_stop_c * 5.0
                     - (1.0 - S_stop_c) * cd_c * 10.0)
        eu_H = jnp.maximum(eu_H_full, eu_H_stop)

        # Light option (req=0.4, distance=1)
        eu_L_full = (S_full_c * 1.0
                     - (1.0 - S_full_c) * cd_c * 6.0
                     - ce_c * 0.16)
        eu_L_stop = (S_stop_c * 1.0
                     - (1.0 - S_stop_c) * cd_c * 6.0)
        eu_L = jnp.maximum(eu_L_full, eu_L_stop)

        logit = jnp.clip((eu_H - eu_L) / tau, -20.0, 20.0)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', c_subj_idx.shape[0]):
            numpyro.sample(
                'obs_choice',
                dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                obs=c_choice,
            )

        # ══════════════════════════════════════════════════════════════
        # VIGOR LIKELIHOOD (ALL trials: type=1,5,6)
        # ══════════════════════════════════════════════════════════════
        ce_v = c_effort[v_subj_idx]
        cd_v = c_death[v_subj_idx]
        eps_v = epsilon[v_subj_idx]

        T_w_v = jnp.power(v_T, gamma)

        # Vigor EU grid: S(u)*R - (1-S(u))*cd*(R+C) - ce*(u-req)^2*D
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]

        S_u = ((1.0 - T_w_v[:, None])
               + eps_v[:, None] * T_w_v[:, None] * p_esc
               * jax.nn.sigmoid(
                   (u_g - v_req[:, None]) / sigma_motor
               ))

        # LQR deviation cost
        deviation = u_g - v_req[:, None]
        effort_vigor = ce_v[:, None] * deviation ** 2 * v_dist[:, None]

        eu_grid = (S_u * v_R[:, None]
                   - (1.0 - S_u) * cd_v[:, None]
                   * (v_R[:, None] + 5.0)
                   - effort_vigor)

        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - v_req - v_offset

        numpyro.deterministic('excess_pred', excess_pred)

        with numpyro.plate('vigor_trials', v_subj_idx.shape[0]):
            numpyro.sample(
                'obs_vigor',
                dist.Normal(excess_pred, sigma_v),
                obs=v_excess_cc,
            )

    return evc_lqr_probes


# ── Fitting ──────────────────────────────────────────────────────────────────

def fit(data, n_steps=40000, lr=0.002, seed=42, print_every=5000):
    """Fit the EVC-LQR probes model via SVI."""
    model = make_model(data['N_S'])

    kwargs = {
        'c_subj_idx': data['c_subj_idx'],
        'c_T': data['c_T'],
        'c_dist_H': data['c_dist_H'],
        'c_choice': data['c_choice'],
        'v_subj_idx': data['v_subj_idx'],
        'v_T': data['v_T'],
        'v_dist': data['v_dist'],
        'v_req': data['v_req'],
        'v_R': data['v_R'],
        'v_offset': data['v_offset'],
        'v_excess_cc': data['v_excess_cc'],
    }

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    print(f"\nFitting EVC-LQR Probes (SVI, {n_steps} steps, lr={lr})")
    print(f"  {data['N_S']} subjects")
    print(f"  Choice trials: {data['N_choice']}")
    print(f"  Vigor trials: {data['N_vigor']}")
    print(f"  3 per-subject params (c_effort, c_death, epsilon)")
    print(f"  Population: gamma, tau, p_esc, sigma_motor, sigma_v")

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

    # Pass all kwargs but set observations to None so model generates them
    obs_kwargs = dict(fit_result['kwargs'])
    obs_kwargs['c_choice'] = None
    obs_kwargs['v_excess_cc'] = None
    pred = Predictive(fit_result['model'], guide=guide,
                      params=params_fit, num_samples=n_samples)
    samples = pred(random.PRNGKey(seed), **obs_kwargs)

    # Extract parameters
    ce = np.array(samples['c_effort']).mean(0)
    cd = np.array(samples['c_death']).mean(0)
    eps = np.array(samples['epsilon']).mean(0)
    gamma_val = float(np.array(samples['gamma']).mean())
    ep = np.array(samples['excess_pred']).mean(0)
    eo = np.array(data['v_excess_cc'])

    # ── Overall vigor r² ──────────────────────────────────────────
    r_vigor, _ = pearsonr(ep, eo)
    print(f"\n{'=' * 60}")
    print(f"EVC-LQR Probes Model Results")
    print(f"{'=' * 60}")

    # ── BIC ───────────────────────────────────────────────────────
    n_params = 3 * data['N_S'] + 10  # 3 subject + ~10 population
    N_obs = data['N_choice'] + data['N_vigor']
    bic = 2 * fit_result['losses'][-1] + n_params * np.log(N_obs)
    print(f"BIC: {bic:.0f}")
    print(f"  n_params={n_params}, N_obs={N_obs}")
    print(f"  Final loss: {fit_result['losses'][-1]:.1f}")
    print(f"Vigor overall: r={r_vigor:.3f}, r²={r_vigor**2:.3f}")

    # ── Per-subject choice r² ─────────────────────────────────────
    # Get choice predictions from samples
    obs_choice = np.array(samples['obs_choice']).mean(0)  # mean predicted probs
    actual_choice = np.array(data['c_choice'])
    r_choice, _ = pearsonr(obs_choice, actual_choice)
    # Accuracy
    pred_choice = (obs_choice > 0.5).astype(int)
    acc = (pred_choice == actual_choice).mean()
    print(f"Choice: r={r_choice:.3f}, r²={r_choice**2:.3f}, accuracy={acc:.1%}")

    # ── Vigor r² by trial type ────────────────────────────────────
    beh_all = data['beh_all']
    trial_types = beh_all['type'].values

    for ttype, label in [(1, 'Choice (type=1)'), ([5, 6], 'Probe (type=5,6)')]:
        if isinstance(ttype, list):
            mask = np.isin(trial_types, ttype)
        else:
            mask = trial_types == ttype
        if mask.sum() > 0:
            r, _ = pearsonr(ep[mask], eo[mask])
            print(f"  Vigor {label}: r={r:.3f}, r²={r**2:.3f} (N={mask.sum()})")

    # ── Parameter distributions ───────────────────────────────────
    print(f"\nParameter distributions:")
    print(f"c_effort: median={np.median(ce):.4f}, mean={ce.mean():.4f}, "
          f"range=[{ce.min():.4f}, {ce.max():.4f}]")
    print(f"c_death:  median={np.median(cd):.4f}, mean={cd.mean():.4f}, "
          f"range=[{cd.min():.4f}, {cd.max():.4f}]")
    print(f"epsilon:  median={np.median(eps):.4f}, mean={eps.mean():.4f}, "
          f"range=[{eps.min():.4f}, {eps.max():.4f}]")
    print(f"gamma (pop): {gamma_val:.3f}")

    # Log-space distributions
    lce, lcd, leps = np.log(ce), np.log(cd), np.log(eps)
    print(f"\nLog-space:")
    print(f"  log(ce): mean={lce.mean():.3f}, sd={lce.std():.3f}")
    print(f"  log(cd): mean={lcd.mean():.3f}, sd={lcd.std():.3f}")
    print(f"  log(eps): mean={leps.mean():.3f}, sd={leps.std():.3f}")

    # ── Log-space correlations ────────────────────────────────────
    print(f"\nLog-space correlations:")
    for (p1, n1, v1), (p2, n2, v2) in [
        (('ce', 'c_effort', lce), ('cd', 'c_death', lcd)),
        (('ce', 'c_effort', lce), ('eps', 'epsilon', leps)),
        (('cd', 'c_death', lcd), ('eps', 'epsilon', leps)),
    ]:
        r, p = pearsonr(v1, v2)
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
        print(f"  log({n1}) x log({n2}): r={r:+.3f} (p={p:.4f}) {sig}")

    # ── Vigor by condition ────────────────────────────────────────
    print(f"\nVigor by threat|choice (all trials):")
    v_choice = beh_all['choice'].values
    v_threat = np.array(data['v_T'])
    for c_val, label in [(1, 'Heavy'), (0, 'Light')]:
        for t in [0.1, 0.5, 0.9]:
            m = (v_choice == c_val) & np.isclose(v_threat, t)
            if m.sum() > 0:
                print(f"  {label} T={t}: pred={ep[m].mean():.4f}, "
                      f"obs={eo[m].mean():.4f} (N={m.sum()})")

    # ── Choice by distance predictions ────────────────────────────
    print(f"\nChoice by distance (type=1 only):")
    c_dist_H = np.array(data['c_dist_H'])
    for d in [1, 2, 3]:
        m = c_dist_H == d
        if m.sum() > 0:
            print(f"  D={d}: pred={obs_choice[m].mean():.3f}, "
                  f"obs={actual_choice[m].mean():.3f} (N={m.sum()})")

    # ── Vigor by distance (probe trials only — unbiased) ──────────
    print(f"\nVigor by distance (PROBE trials only — unbiased):")
    probe_mask = np.isin(trial_types, [5, 6])
    v_dist_np = np.array(data['v_dist'])
    for d in [1, 2, 3]:
        m = probe_mask & (v_dist_np == d)
        if m.sum() > 0:
            print(f"  D={d}: pred={ep[m].mean():.4f}, "
                  f"obs={eo[m].mean():.4f} (N={m.sum()})")

    # ── Clinical correlations ─────────────────────────────────────
    if 'psych' in data:
        psych = data['psych']
        param_df = pd.DataFrame({
            'subj': data['subjects'],
            'c_effort': ce, 'c_death': cd, 'epsilon': eps,
        })
        merged = param_df.merge(psych, on='subj')
        psych_cols = [c for c in psych.columns
                      if c not in ['subj', 'participantID']
                      and not c.endswith('_RT')]

        print(f"\nClinical correlations (log space):")
        for param in ['c_effort', 'c_death', 'epsilon']:
            print(f"  {param}:")
            for col in psych_cols:
                try:
                    r, p = pearsonr(np.log(merged[param]), merged[col])
                    if p < 0.1:
                        sig = ('***' if p < .001 else '**' if p < .01
                               else '*' if p < .05 else '+')
                        print(f"    -> {col:20s}: r={r:+.3f}, p={p:.4f} {sig}")
                except Exception:
                    pass

    # ── Comparison to previous models ─────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Model Comparison")
    print(f"{'=' * 60}")
    print(f"  EVC-LQR 2-param (pop ce):  BIC=20,368")
    print(f"  Original 3-param:          BIC=22,988")
    print(f"  THIS MODEL (probes):       BIC={bic:.0f}")
    print(f"    Vigor r² overall:        {r_vigor**2:.3f}")
    print(f"    Choice accuracy:         {acc:.1%}")

    # ── Save parameters ───────────────────────────────────────────
    param_df = pd.DataFrame({
        'subj': data['subjects'],
        'c_effort': ce, 'c_death': cd, 'epsilon': eps,
    })

    return {
        'param_df': param_df,
        'gamma': gamma_val,
        'r_vigor': r_vigor,
        'r2_vigor': r_vigor ** 2,
        'r_choice': r_choice,
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

    fit_result = fit(data, n_steps=40000, lr=0.002)
    result = evaluate(fit_result)

    out_path = 'results/stats/oc_evc_lqr_probes_params.csv'
    result['param_df'].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Also save population params
    pop_path = 'results/stats/oc_evc_lqr_probes_population.csv'
    pd.DataFrame([{
        'gamma': result['gamma'],
        'bic': result['bic'],
        'vigor_r2': result['r2_vigor'],
    }]).to_csv(pop_path, index=False)
    print(f"Saved: {pop_path}")
