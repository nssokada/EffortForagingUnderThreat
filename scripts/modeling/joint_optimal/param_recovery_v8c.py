"""
Parameter recovery for V8c (joint optimal control model).

Strategy:
1. Use the fitted population parameters (γ, h, τ, σ_v, b_cookie)
2. Generate 3 sets of 50 synthetic subjects with known ω, κ, baseline
   - Low ω / low κ
   - Medium ω / medium κ
   - High ω / high κ
3. Simulate choice + vigor data from the generative model
4. Fit V8c to the simulated data
5. Correlate recovered vs true parameters
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
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit
from pathlib import Path

OUT_DIR = Path("results/stats/joint_optimal")
C_PENALTY = 5.0

# Population parameters from V8c fit
POP_PARAMS = {
    'gamma': 0.86,
    'hazard': 0.832,
    'tau': 2.01,
    'sigma_v': 0.12,  # approximate from fit
    'b_cookie': -0.03,
}

# Task design constants
THREATS = [0.1, 0.5, 0.9]
DIST_H = [1, 2, 3]
N_TRIALS_PER_CELL = 5  # per threat × distance combination = 9 cells × 5 = 45 trials


def exp_survival(u, T, D, gamma, hazard):
    T_w = T ** gamma
    return np.exp(-hazard * T_w * D / np.clip(u, 0.1, None))


def compute_W(u, omega, kappa, T, D, R, req, gamma, hazard):
    S = exp_survival(u, T, D, gamma, hazard)
    return S * R - (1 - S) * omega * (R + C_PENALTY) - kappa * (u - req)**2 * D


def find_ustar(omega, kappa, T, D, R, req, gamma, hazard):
    """Find optimal u* via grid search."""
    u_grid = np.linspace(0.1, 1.5, 100)
    W_vals = np.array([compute_W(u, omega, kappa, T, D, R, req, gamma, hazard)
                       for u in u_grid])
    best_idx = np.argmax(W_vals)
    return u_grid[best_idx], W_vals[best_idx]


def simulate_data(true_omega, true_kappa, true_baseline, seed=0):
    """Simulate choice + vigor data for a set of subjects."""
    rng = np.random.RandomState(seed)
    N_S = len(true_omega)
    gamma = POP_PARAMS['gamma']
    hazard = POP_PARAMS['hazard']
    tau = POP_PARAMS['tau']
    sigma_v = POP_PARAMS['sigma_v']
    b_cookie = POP_PARAMS['b_cookie']

    choice_records = []
    vigor_records = []

    for s in range(N_S):
        omega = true_omega[s]
        kappa = true_kappa[s]
        base = true_baseline[s]

        for T in THREATS:
            for D_H in DIST_H:
                for trial in range(N_TRIALS_PER_CELL):
                    # Compute V_H and V_L
                    u_H, V_H = find_ustar(omega, kappa, T, D_H, 5.0, 0.9, gamma, hazard)
                    u_L, V_L = find_ustar(omega, kappa, T, 1.0, 1.0, 0.4, gamma, hazard)

                    # Choice
                    p_heavy = expit((V_H - V_L) / tau)
                    chose_heavy = rng.rand() < p_heavy

                    choice_records.append({
                        'subj': s, 'threat': T, 'distance_H': D_H,
                        'choice': int(chose_heavy), 'p_heavy': p_heavy,
                    })

                    # Vigor on chosen cookie
                    if chose_heavy:
                        u_star = u_H
                        R, req, D = 5.0, 0.9, D_H
                        is_heavy = 1
                    else:
                        u_star = u_L
                        R, req, D = 1.0, 0.4, 1.0
                        is_heavy = 0

                    rate = u_star + base + b_cookie * is_heavy + rng.randn() * sigma_v

                    vigor_records.append({
                        'subj': s, 'threat': T, 'actual_dist': D,
                        'actual_R': R, 'actual_req': req,
                        'median_rate': rate, 'is_heavy': is_heavy,
                    })

    choice_df = pd.DataFrame(choice_records)
    vigor_df = pd.DataFrame(vigor_records)
    return choice_df, vigor_df


def prepare_sim_data(choice_df, vigor_df):
    """Convert simulated data to JAX arrays for fitting."""
    N_S = choice_df['subj'].nunique()
    return {
        'ch_subj': jnp.array(choice_df['subj'].values),
        'ch_T': jnp.array(choice_df['threat'].values),
        'ch_D_H': jnp.array(choice_df['distance_H'].values, dtype=jnp.float64),
        'ch_D_L': jnp.ones(len(choice_df)),
        'ch_choice': jnp.array(choice_df['choice'].values),
        'vig_subj': jnp.array(vigor_df['subj'].values),
        'vig_T': jnp.array(vigor_df['threat'].values),
        'vig_R': jnp.array(vigor_df['actual_R'].values),
        'vig_req': jnp.array(vigor_df['actual_req'].values),
        'vig_dist': jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64),
        'vig_rate': jnp.array(vigor_df['median_rate'].values),
        'vig_cookie': jnp.array(vigor_df['is_heavy'].values, dtype=jnp.float64),
        'N_S': N_S, 'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
    }


# ============================================================
# V8c model (same as round3)
# ============================================================

def exp_survival_jax(u, T, D, gamma, hazard):
    T_w = jnp.power(T, gamma)
    return jnp.exp(-hazard * T_w * D / jnp.clip(u, 0.1, None))


def vigor_eu_exp(omega, kappa, T, D, R, req, gamma, hazard, u_grid):
    u_g = u_grid[None, :]
    S = exp_survival_jax(u_g, T[:, None], D[:, None], gamma, hazard)
    W = (S * R[:, None]
         - (1.0 - S) * omega[:, None] * (R[:, None] + C_PENALTY)
         - kappa[:, None] * (u_g - req[:, None]) ** 2 * D[:, None])
    weights = jax.nn.softmax(W * 20.0, axis=1)
    u_star = jnp.sum(weights * u_g, axis=1)
    V_star = jnp.sum(weights * W, axis=1)
    return u_star, V_star


def make_v8c(N_S, N_ch, N_vig):
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist, vig_rate, vig_cookie):
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        hz_raw = numpyro.sample('hazard_raw', dist.Normal(-1.0, 1.0))
        hazard = numpyro.deterministic('hazard', jnp.exp(hz_raw))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 50.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        mu_om = numpyro.sample('mu_om', dist.Normal(1.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(1.0))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-2.0, 1.0))
        sigma_kap = numpyro.sample('sigma_kap', dist.HalfNormal(0.5))
        mu_base = numpyro.sample('mu_base', dist.Normal(0.0, 0.3))
        sigma_base = numpyro.sample('sigma_base', dist.HalfNormal(0.2))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
            base_raw = numpyro.sample('base_raw', dist.Normal(0.0, 1.0))
        omega = jnp.exp(mu_om + sigma_om * om_raw)
        kappa = jnp.exp(mu_kap + sigma_kap * kap_raw)
        baseline = mu_base + sigma_base * base_raw
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)
        numpyro.deterministic('baseline', baseline)

        u_grid = jnp.linspace(0.1, 1.5, 30)

        _, V_H = vigor_eu_exp(omega[ch_subj], kappa[ch_subj], ch_T, ch_D_H,
                              jnp.full(N_ch, 5.0), jnp.full(N_ch, 0.9),
                              gamma, hazard, u_grid)
        _, V_L = vigor_eu_exp(omega[ch_subj], kappa[ch_subj], ch_T, ch_D_L,
                              jnp.full(N_ch, 1.0), jnp.full(N_ch, 0.4),
                              gamma, hazard, u_grid)
        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice', N_ch):
            numpyro.sample('obs_ch', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        u_star, _ = vigor_eu_exp(omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
                                 vig_R, vig_req, gamma, hazard, u_grid)
        rate_pred = u_star + baseline[vig_subj] + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)
        with numpyro.plate('vigor', N_vig):
            numpyro.sample('obs_vig', dist.Normal(rate_pred, sigma_v), obs=vig_rate)
    return model


KWARGS_KEYS = ['ch_subj', 'ch_T', 'ch_D_H', 'ch_D_L', 'ch_choice',
               'vig_subj', 'vig_T', 'vig_R', 'vig_req', 'vig_dist',
               'vig_rate', 'vig_cookie']


def fit_and_recover(true_omega, true_kappa, true_baseline, label, seed=0):
    """Simulate, fit, and check recovery."""
    N_S = len(true_omega)
    print(f"\n  Simulating {label} (N={N_S})...")
    choice_df, vigor_df = simulate_data(true_omega, true_kappa, true_baseline, seed=seed)
    data = prepare_sim_data(choice_df, vigor_df)

    print(f"    {data['N_choice']} choice, {data['N_vigor']} vigor trials")
    print(f"    Choice base rate (heavy): {choice_df['choice'].mean():.3f}")

    # Fit
    kwargs = {k: data[k] for k in KWARGS_KEYS}
    model_fn = make_v8c(N_S, data['N_choice'], data['N_vigor'])
    guide = AutoNormal(model_fn)
    opt = numpyro.optim.ClippedAdam(step_size=0.001, clip_norm=10.0)
    svi = SVI(model_fn, guide, opt, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed + 100), **kwargs)
    update = jax.jit(svi.update)

    best_loss, best_params = float('inf'), None
    for i in range(25000):
        state, loss = update(state, **kwargs)
        l = float(loss)
        if l < best_loss and not np.isnan(l):
            best_loss = l; best_params = svi.get_params(state)
        if (i+1) % 10000 == 0:
            print(f"    step {i+1}: loss={l:.1f} (best={best_loss:.1f})")

    # Extract recovered params
    pred = Predictive(model_fn, guide=guide, params=best_params,
                      num_samples=200, return_sites=['omega', 'kappa', 'baseline', 'gamma', 'hazard'])
    samples = pred(random.PRNGKey(seed + 200), **kwargs)

    rec_omega = np.array(samples['omega']).mean(0)
    rec_kappa = np.array(samples['kappa']).mean(0)
    rec_baseline = np.array(samples['baseline']).mean(0)
    rec_gamma = float(np.array(samples['gamma']).mean())
    rec_hazard = float(np.array(samples['hazard']).mean())

    # Correlations
    r_om, p_om = pearsonr(true_omega, rec_omega)
    r_kap, p_kap = pearsonr(true_kappa, rec_kappa)
    r_base, p_base = pearsonr(true_baseline, rec_baseline)
    rho_om, _ = spearmanr(true_omega, rec_omega)
    rho_kap, _ = spearmanr(true_kappa, rec_kappa)

    print(f"\n  {label} Recovery Results:")
    print(f"    ω:  r = {r_om:.3f} (p={p_om:.2e}), ρ = {rho_om:.3f}")
    print(f"    κ:  r = {r_kap:.3f} (p={p_kap:.2e}), ρ = {rho_kap:.3f}")
    print(f"    baseline: r = {r_base:.3f} (p={p_base:.2e})")
    print(f"    γ: true={POP_PARAMS['gamma']:.2f}, recovered={rec_gamma:.2f}")
    print(f"    h: true={POP_PARAMS['hazard']:.3f}, recovered={rec_hazard:.3f}")

    # Check bias
    print(f"\n    ω bias: true mean={true_omega.mean():.3f}, rec mean={rec_omega.mean():.3f}")
    print(f"    κ bias: true mean={true_kappa.mean():.4f}, rec mean={rec_kappa.mean():.4f}")
    print(f"    base bias: true mean={true_baseline.mean():.3f}, rec mean={rec_baseline.mean():.3f}")

    return {
        'label': label, 'r_omega': r_om, 'r_kappa': r_kap, 'r_baseline': r_base,
        'rho_omega': rho_om, 'rho_kappa': rho_kap,
        'gamma_true': POP_PARAMS['gamma'], 'gamma_rec': rec_gamma,
        'hazard_true': POP_PARAMS['hazard'], 'hazard_rec': rec_hazard,
        'true_omega': true_omega, 'rec_omega': rec_omega,
        'true_kappa': true_kappa, 'rec_kappa': rec_kappa,
        'true_baseline': true_baseline, 'rec_baseline': rec_baseline,
    }


if __name__ == '__main__':
    t0 = time.time()
    print("=" * 70)
    print("PARAMETER RECOVERY: V8c Joint Optimal Control Model")
    print("=" * 70)

    rng = np.random.RandomState(42)
    all_results = []

    # Scenario 1: Parameters drawn from fitted distribution
    N = 50
    print("\n" + "=" * 50)
    print("Scenario 1: Parameters from fitted distribution")
    print("=" * 50)
    true_om = np.exp(rng.randn(N) * 0.5 + 0.8)  # ~mean 2.2, spread
    true_kap = np.exp(rng.randn(N) * 0.5 - 3.0)  # ~mean 0.05, spread
    true_base = rng.randn(N) * 0.1  # small baseline variation
    res1 = fit_and_recover(true_om, true_kap, true_base, "Fitted dist", seed=1)
    all_results.append(res1)

    # Scenario 2: Wide parameter spread (stress test)
    print("\n" + "=" * 50)
    print("Scenario 2: Wide spread (stress test)")
    print("=" * 50)
    true_om = np.exp(rng.randn(N) * 1.0 + 0.5)  # wider spread
    true_kap = np.exp(rng.randn(N) * 1.0 - 2.0)  # wider spread
    true_base = rng.randn(N) * 0.15
    res2 = fit_and_recover(true_om, true_kap, true_base, "Wide spread", seed=2)
    all_results.append(res2)

    # Scenario 3: Correlated ω and κ (test if model can recover independent when correlated)
    print("\n" + "=" * 50)
    print("Scenario 3: Correlated ω-κ")
    print("=" * 50)
    z1 = rng.randn(N)
    z2 = 0.5 * z1 + 0.866 * rng.randn(N)  # r ≈ 0.5
    true_om = np.exp(z1 * 0.5 + 0.8)
    true_kap = np.exp(z2 * 0.5 - 3.0)
    true_base = rng.randn(N) * 0.1
    print(f"  True ω-κ correlation: {pearsonr(true_om, true_kap)[0]:.3f}")
    res3 = fit_and_recover(true_om, true_kap, true_base, "Correlated", seed=3)
    all_results.append(res3)

    # Summary
    print("\n" + "=" * 70)
    print("PARAMETER RECOVERY SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<20} {'r(ω)':<10} {'r(κ)':<10} {'r(base)':<10} {'γ rec':<10} {'h rec':<10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['label']:<20} {r['r_omega']:<10.3f} {r['r_kappa']:<10.3f} "
              f"{r['r_baseline']:<10.3f} {r['gamma_rec']:<10.2f} {r['hazard_rec']:<10.3f}")

    print(f"\nTrue population: γ={POP_PARAMS['gamma']:.2f}, h={POP_PARAMS['hazard']:.3f}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        'scenario': r['label'], 'r_omega': r['r_omega'], 'r_kappa': r['r_kappa'],
        'r_baseline': r['r_baseline'], 'rho_omega': r['rho_omega'],
        'rho_kappa': r['rho_kappa'],
        'gamma_rec': r['gamma_rec'], 'hazard_rec': r['hazard_rec'],
    } for r in all_results])
    summary.to_csv(OUT_DIR / "param_recovery_v8c.csv", index=False)
    print(f"Saved to {OUT_DIR / 'param_recovery_v8c.csv'}")
