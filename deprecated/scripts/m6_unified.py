"""
M6: Unified foraging optimization with per-subject threat perception (γ_i) and capture cost (ω_i).

Inspired by Bednekoff (2007) and Yoon & Shadmehr (2018): the forager jointly
selects patch and vigor to maximize fitness. Individual differences arise from:
  - γ_i: threat perception (how T is transformed into subjective danger)
  - ω_i: capture cost (how much the penalty of being caught weighs)

Key difference from M5: choice integrates anticipated vigor. For each cookie option,
the model computes the optimal vigor u* and resulting fitness V(cookie). The forager
then chooses the cookie with higher V. This makes ω visible to choice through the
anticipated survival benefit of vigorous pressing.

Architecture:
    Per-subject (log-normal, non-centered):
        γ_i  — threat perception (T_perceived = T^γ_i)
        ω_i  — capture cost (scales penalty in EU)

    Population-level:
        ε        — effort efficacy (escape probability scaling)
        p_esc    — base escape probability at full speed
        σ_motor  — motor noise around speed threshold
        c_vig    — quadratic motor cost
        σ_v      — vigor observation noise
        τ        — choice temperature

    Fitness for a given (cookie, u):
        S_i(u) = (1 - T^γ_i) + T^γ_i × ε × p_esc × σ((u - req) / σ_motor)
        W(cookie, u) = S_i(u) × R - (1 - S_i(u)) × ω_i × (R + C) - c_vig × (u - req)² × D

    Choice (joint optimization):
        For each cookie: u*(cookie) = soft_argmax_u W(cookie, u)
                         V(cookie) = W(cookie, u*(cookie))
        P(heavy) = σ((V(heavy) - V(light)) / τ)

    Vigor:
        u* = soft_argmax_u W(chosen_cookie, u)
        observed_vigor ~ Normal(u* - req, σ_v)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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

# Import data loading from comparison script
import importlib.util
spec = importlib.util.spec_from_file_location(
    'mc5', os.path.join(os.path.dirname(__file__), 'model_comparison_5models.py'))
mc5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc5)

OUT_DIR = Path("results/stats/avoidance_activation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_V_cookie(gamma_subj, omega_subj, T, R, req, D,
                     epsilon, p_esc, sigma_motor, ce_vigor):
    """Compute V(cookie) = max_u W(cookie, u) for one set of trials.

    Returns V (the fitness value) and u_star (the optimal vigor).
    All inputs are trial-level arrays except epsilon etc which are scalars.
    gamma_subj and omega_subj are per-trial (already indexed by subject).
    """
    u_grid = jnp.linspace(0.1, 1.5, 30)
    u_g = u_grid[None, :]  # (1, 30)

    # S(u) for each trial × each u on the grid
    T_w = jnp.power(T[:, None], gamma_subj[:, None])  # (N, 1) ^ (N, 1) = (N, 1)
    S_u = ((1.0 - T_w)
           + T_w * epsilon * p_esc
           * jax.nn.sigmoid((u_g - req[:, None]) / sigma_motor))  # (N, 30)

    # W(cookie, u) for each trial × each u
    deviation = u_g - req[:, None]
    W_grid = (S_u * R[:, None]
              - (1.0 - S_u) * omega_subj[:, None] * (R[:, None] + 5.0)
              - ce_vigor * deviation ** 2 * D[:, None])  # (N, 30)

    # Soft argmax
    weights = jax.nn.softmax(W_grid * 10.0, axis=1)  # (N, 30)
    u_star = jnp.sum(weights * u_g, axis=1)  # (N,)
    V = jnp.sum(weights * W_grid, axis=1)  # (N,) expected value at soft-optimal u

    return V, u_star


def make_model_m6(N_S, N_choice, N_vigor):
    """M6: Unified optimization with per-subject γ and ω."""

    def model(ch_subj, ch_T, ch_dist_H, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_rate, vig_cookie):

        # ── Population params ──
        epsilon_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
        epsilon = numpyro.deterministic('epsilon', jnp.exp(epsilon_raw))

        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)

        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)

        ce_vigor_raw = numpyro.sample('ce_vigor_raw', dist.Normal(-3.0, 1.0))
        ce_vigor = numpyro.deterministic('ce_vigor', jnp.exp(ce_vigor_raw))

        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))

        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 50.0)

        # ── Per-subject: γ_i (threat perception) and ω_i (capture cost) ──
        mu_gamma = numpyro.sample('mu_gamma', dist.Normal(-1.0, 1.0))
        sigma_gamma = numpyro.sample('sigma_gamma', dist.HalfNormal(0.5))
        mu_omega = numpyro.sample('mu_omega', dist.Normal(0.0, 1.0))
        sigma_omega = numpyro.sample('sigma_omega', dist.HalfNormal(0.5))

        with numpyro.plate('subjects', N_S):
            gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 1.0))
            omega_raw = numpyro.sample('omega_raw', dist.Normal(0.0, 1.0))

        # γ_i: positive, typically < 1 (compression). Use exp then clip.
        gamma_i = jnp.clip(jnp.exp(mu_gamma + sigma_gamma * gamma_raw), 0.05, 3.0)
        omega_i = jnp.exp(mu_omega + sigma_omega * omega_raw)
        numpyro.deterministic('gamma_i', gamma_i)
        numpyro.deterministic('omega_i', omega_i)

        # ════════════════════════════════════════════
        # CHOICE: Joint optimization over both cookies
        # ════════════════════════════════════════════

        gamma_ch = gamma_i[ch_subj]
        omega_ch = omega_i[ch_subj]

        # Heavy cookie: R=5, req=0.9, D=distance_H
        R_heavy = jnp.full_like(ch_T, 5.0)
        req_heavy = jnp.full_like(ch_T, 0.9)
        D_heavy = ch_dist_H.astype(jnp.float64)

        V_heavy, _ = compute_V_cookie(
            gamma_ch, omega_ch, ch_T, R_heavy, req_heavy, D_heavy,
            epsilon, p_esc, sigma_motor, ce_vigor)

        # Light cookie: R=1, req=0.4, D=1
        R_light = jnp.full_like(ch_T, 1.0)
        req_light = jnp.full_like(ch_T, 0.4)
        D_light = jnp.full_like(ch_T, 1.0)

        V_light, _ = compute_V_cookie(
            gamma_ch, omega_ch, ch_T, R_light, req_light, D_light,
            epsilon, p_esc, sigma_motor, ce_vigor)

        # Choice = which cookie has higher fitness
        delta_V = V_heavy - V_light
        logit_choice = jnp.clip(delta_V / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit_choice)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                           dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)),
                           obs=ch_choice)

        # ════════════════════════════════════════════
        # VIGOR: Optimal u for the chosen cookie
        # ════════════════════════════════════════════

        gamma_v = gamma_i[vig_subj]
        omega_v = omega_i[vig_subj]

        _, u_star = compute_V_cookie(
            gamma_v, omega_v, vig_T, vig_R, vig_req, vig_dist,
            epsilon, p_esc, sigma_motor, ce_vigor)

        # Cookie intercept (heavy vs light baseline difference)
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))
        rate_pred = u_star + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor',
                           dist.Normal(rate_pred, sigma_v),
                           obs=vig_rate)

    return model


def fit_m6(data, n_steps=40000, lr=0.001, seed=42):
    """Fit M6 and extract parameters."""
    N_S = data['N_S']
    model_fn = make_model_m6(N_S, data['N_choice'], data['N_vigor'])
    kwargs = {k: data[k] for k in mc5.KWARGS_KEYS}

    guide = AutoNormal(model_fn)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
    svi = SVI(model_fn, guide, optimizer, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    best_loss = float('inf')
    best_params = None
    best_step = 0

    print(f"Fitting M6 ({n_steps} steps)...")
    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        l = float(loss)
        if l < best_loss:
            best_loss = l
            best_params = svi.get_params(state)
            best_step = i + 1
        if (i + 1) % 10000 == 0:
            print(f"  Step {i+1}: loss={l:.1f} (best={best_loss:.1f} @ {best_step})")

    print(f"  Done. Best loss={best_loss:.1f} at step {best_step}")

    # Extract parameters
    pred = Predictive(model_fn, guide=guide, params=best_params,
                      num_samples=500,
                      return_sites=['gamma_i', 'omega_i', 'epsilon',
                                    'ce_vigor', 'tau_raw', 'p_esc_raw',
                                    'sigma_motor_raw', 'sigma_v', 'rate_pred'])
    samples = pred(random.PRNGKey(seed + 1), **kwargs)

    gamma_vals = np.array(samples['gamma_i']).mean(0)
    omega_vals = np.array(samples['omega_i']).mean(0)
    eps_val = float(np.array(samples['epsilon']).mean())
    ce_vig_val = float(np.array(samples['ce_vigor']).mean())
    tau_val = float(np.exp(np.array(samples['tau_raw']).mean()))
    p_esc_val = float(expit(np.array(samples['p_esc_raw']).mean()))
    sigma_motor_val = float(np.exp(np.array(samples['sigma_motor_raw']).mean()))
    sigma_v_val = float(np.array(samples['sigma_v']).mean())

    # Save params
    param_df = pd.DataFrame({
        'subj': data['subjects'],
        'gamma_i': gamma_vals,
        'omega_i': omega_vals,
        'log_gamma': np.log(gamma_vals),
        'log_omega': np.log(omega_vals),
    })
    param_df['log_gamma_z'] = (param_df['log_gamma'] - param_df['log_gamma'].mean()) / param_df['log_gamma'].std()
    param_df['log_omega_z'] = (param_df['log_omega'] - param_df['log_omega'].mean()) / param_df['log_omega'].std()
    param_df.to_csv(OUT_DIR / "m6_params.csv", index=False)

    pop_df = pd.DataFrame([{
        'epsilon': eps_val, 'ce_vigor': ce_vig_val, 'tau': tau_val,
        'p_esc': p_esc_val, 'sigma_motor': sigma_motor_val, 'sigma_v': sigma_v_val,
        'best_loss': best_loss,
    }])
    pop_df.to_csv(OUT_DIR / "m6_population.csv", index=False)

    # Evaluate
    print(f"\n  γ_i: mean={gamma_vals.mean():.3f}, SD={gamma_vals.std():.3f}, "
          f"range=[{gamma_vals.min():.3f}, {gamma_vals.max():.3f}]")
    print(f"  ω_i: median={np.median(omega_vals):.3f}, "
          f"range=[{omega_vals.min():.3f}, {omega_vals.max():.3f}]")

    r_go, p_go = pearsonr(np.log(gamma_vals), np.log(omega_vals))
    print(f"  r(log γ, log ω) = {r_go:.3f}, p = {p_go:.4f}")
    print(f"  ε = {eps_val:.3f}, τ = {tau_val:.3f}")

    # Vigor r
    rp = np.array(samples['rate_pred']).mean(0)
    r_vig, _ = pearsonr(rp, np.array(data['vig_rate']))
    print(f"  Vigor r = {r_vig:.3f}")

    # Choice accuracy (reconstruct from V)
    # Simpler: just use the model's P(heavy) from the Predictive
    # Re-run Predictive with obs_choice as a return site won't work,
    # so recompute manually
    ch_subj_np = np.array(data['ch_subj'])
    ch_T_np = np.array(data['ch_T'])
    ch_dist_np = np.array(data['ch_dist_H'])
    ch_choice_np = np.array(data['ch_choice'])

    # Compute V_heavy and V_light per trial
    def compute_V_np(gamma, omega, T, R, req, D):
        u_grid = np.linspace(0.1, 1.5, 30)
        T_w = T[:, None] ** gamma[:, None]
        S_u = (1 - T_w) + T_w * eps_val * p_esc_val * expit((u_grid[None, :] - req[:, None]) / sigma_motor_val)
        dev = u_grid[None, :] - req[:, None]
        W_grid = S_u * R[:, None] - (1 - S_u) * omega[:, None] * (R[:, None] + 5) - ce_vig_val * dev**2 * D[:, None]
        w = np.exp(W_grid * 10 - np.max(W_grid * 10, axis=1, keepdims=True))
        w = w / w.sum(axis=1, keepdims=True)
        V = np.sum(w * W_grid, axis=1)
        return V

    g_ch = gamma_vals[ch_subj_np]
    o_ch = omega_vals[ch_subj_np]

    V_h = compute_V_np(g_ch, o_ch, ch_T_np, np.full_like(ch_T_np, 5.0),
                        np.full_like(ch_T_np, 0.9), ch_dist_np.astype(float))
    V_l = compute_V_np(g_ch, o_ch, ch_T_np, np.full_like(ch_T_np, 1.0),
                        np.full_like(ch_T_np, 0.4), np.ones_like(ch_T_np))

    p_H = expit(np.clip((V_h - V_l) / tau_val, -20, 20))
    acc = ((p_H >= 0.5).astype(int) == ch_choice_np).mean()
    print(f"  Choice accuracy = {acc:.3f}")

    # Per-subject choice r
    ch_df = pd.DataFrame({'subj': ch_subj_np, 'choice': ch_choice_np, 'p_H': p_H})
    sc = ch_df.groupby('subj').agg(o=('choice', 'mean'), p=('p_H', 'mean'))
    try:
        r_ch, _ = pearsonr(sc['o'], sc['p'])
    except Exception:
        r_ch = np.nan
    print(f"  Choice r = {r_ch:.3f}")

    # BIC
    n_params = 2 * N_S + 10  # gamma_raw, omega_raw per subj + population params
    n_obs = data['N_choice'] + data['N_vigor']
    bic = 2 * best_loss + n_params * np.log(n_obs)
    print(f"  BIC = {bic:.0f}")

    # Compare to M5
    try:
        m5_bic = pd.read_csv(OUT_DIR.parent / "model_comparison_5models/comparison_table.csv")
        m5_bic = m5_bic.loc[m5_bic['Model'] == 'M5', 'BIC'].values[0]
        print(f"  M5 BIC = {m5_bic:.0f}")
        print(f"  ΔBIC (M6 - M5) = {bic - m5_bic:+.0f}")
    except Exception:
        pass

    # Does γ_i predict both choice and vigor?
    from scipy.stats import zscore
    beh = pd.read_csv("data/exploratory_350/processed/stage5_filtered_data_20260320_191950/behavior.csv")
    beh = beh[~beh['subj'].isin([154, 197, 208])]
    subj_choice = beh.groupby('subj')['choice'].mean()

    vigor_data = pd.read_csv("data/exploratory_350/processed/stage5_filtered_data_20260320_191950/trial_vigor.csv")
    vigor_data = vigor_data.dropna(subset=['median_rate'])
    subj_vigor = vigor_data.groupby('subj')['median_rate'].mean()

    common = param_df['subj'].values
    sc_map = {s: subj_choice.get(s, np.nan) for s in common}
    sv_map = {s: subj_vigor.get(s, np.nan) for s in common}
    param_df['p_heavy'] = [sc_map[s] for s in common]
    param_df['mean_vigor'] = [sv_map[s] for s in common]

    valid = param_df.dropna(subset=['p_heavy', 'mean_vigor'])

    print("\n  γ_i → behavior (THE COUPLING TEST):")
    r_gc, p_gc = pearsonr(valid['log_gamma_z'], valid['p_heavy'])
    r_gv, p_gv = pearsonr(valid['log_gamma_z'], valid['mean_vigor'])
    print(f"    r(log γ, P(heavy))    = {r_gc:+.3f}, p = {p_gc:.4f}")
    print(f"    r(log γ, mean vigor)  = {r_gv:+.3f}, p = {p_gv:.4f}")
    print(f"    (Unified model predicts: positive for P(heavy), negative for vigor)")

    r_oc, p_oc = pearsonr(valid['log_omega_z'], valid['p_heavy'])
    r_ov, p_ov = pearsonr(valid['log_omega_z'], valid['mean_vigor'])
    print(f"\n  ω_i → behavior:")
    print(f"    r(log ω, P(heavy))    = {r_oc:+.3f}, p = {p_oc:.4f}")
    print(f"    r(log ω, mean vigor)  = {r_ov:+.3f}, p = {p_ov:.4f}")

    return {
        'param_df': param_df, 'best_loss': best_loss, 'bic': bic,
        'acc': acc, 'r_choice': r_ch, 'r_vigor': r_vig,
        'gamma_sd': gamma_vals.std(), 'gamma_mean': gamma_vals.mean(),
    }


if __name__ == '__main__':
    t0 = time.time()

    print("=" * 70)
    print("M6: UNIFIED FORAGING OPTIMIZATION (γ_i + ω_i)")
    print("=" * 70)

    print("\nPreparing data...")
    data = mc5.prepare_data()
    print(f"  {data['N_S']} subjects, {data['N_choice']} choice, {data['N_vigor']} vigor")

    result = fit_m6(data, n_steps=40000, lr=0.001)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Done in {elapsed/60:.1f} min")

    # Key verdicts
    print(f"\nKEY QUESTIONS:")
    print(f"  1. Does γ_i have per-subject variance?  SD = {result['gamma_sd']:.3f}")
    if result['gamma_sd'] > 0.05:
        print(f"     YES — meaningful individual differences in threat perception")
    else:
        print(f"     NO — collapses to population constant, unified model doesn't work")

    print(f"  2. Does M6 beat M5?  ΔBIC = {result['bic'] - 16336:+.0f}")
    print(f"  3. Choice accuracy: {result['acc']:.3f} (M5 was 0.793)")
    print(f"  4. Vigor r: {result['r_vigor']:.3f} (M5 was 0.704)")
    print(f"{'='*70}")
