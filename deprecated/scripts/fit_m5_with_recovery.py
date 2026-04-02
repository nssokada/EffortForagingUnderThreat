"""
Fit the M5 (λ-ω) avoidance-activation model, save per-subject parameters,
and run parameter recovery (3 datasets × 50 subjects).

Outputs:
    results/stats/avoidance_activation/m5_params.csv        — per-subject λ, ω
    results/stats/avoidance_activation/m5_population.csv    — population params
    results/stats/avoidance_activation/m5_recovery.csv      — recovery correlations
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

# Import data loading and M5 model from the comparison script
import importlib.util
spec = importlib.util.spec_from_file_location(
    'mc5', os.path.join(os.path.dirname(__file__), 'model_comparison_5models.py'))
mc5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc5)

OUT_DIR = Path("results/stats/avoidance_activation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Part 1: Fit M5 on real data and save parameters
# ============================================================

def fit_and_save_m5(data, n_steps=40000, lr=0.001, seed=42):
    """Fit M5, extract per-subject λ and ω, save to CSV."""
    N_S = data['N_S']
    model_fn = mc5.make_model_m5(N_S, data['N_choice'], data['N_vigor'])
    kwargs = {k: data[k] for k in mc5.KWARGS_KEYS}

    guide = AutoNormal(model_fn)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
    svi = SVI(model_fn, guide, optimizer, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    best_loss = float('inf')
    best_params = None
    best_step = 0

    print(f"Fitting M5 ({n_steps} steps, lr={lr})...")
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
                      return_sites=['lam', 'omega', 'epsilon', 'gamma',
                                    'ce_vigor', 'tau_raw', 'p_esc_raw',
                                    'sigma_motor_raw', 'sigma_v', 'excess_pred'])
    samples = pred(random.PRNGKey(seed + 1), **kwargs)

    lam = np.array(samples['lam']).mean(0)
    omega = np.array(samples['omega']).mean(0)
    eps_val = float(np.array(samples['epsilon']).mean())
    gamma_val = float(np.array(samples['gamma']).mean())
    ce_vigor_val = float(np.array(samples['ce_vigor']).mean())
    tau_val = float(np.exp(np.array(samples['tau_raw']).mean()))
    p_esc_val = float(expit(np.array(samples['p_esc_raw']).mean()))
    sigma_motor_val = float(np.exp(np.array(samples['sigma_motor_raw']).mean()))
    sigma_v_val = float(np.array(samples['sigma_v']).mean())

    # Per-subject params
    param_df = pd.DataFrame({
        'subj': data['subjects'],
        'lam': lam,
        'omega': omega,
        'log_lam': np.log(lam),
        'log_omega': np.log(omega),
    })
    param_df['log_lam_z'] = (param_df['log_lam'] - param_df['log_lam'].mean()) / param_df['log_lam'].std()
    param_df['log_omega_z'] = (param_df['log_omega'] - param_df['log_omega'].mean()) / param_df['log_omega'].std()

    param_df.to_csv(OUT_DIR / "m5_params.csv", index=False)
    print(f"  Saved {len(param_df)} subjects to {OUT_DIR / 'm5_params.csv'}")

    # Population params
    pop_df = pd.DataFrame([{
        'epsilon': eps_val, 'gamma': gamma_val, 'ce_vigor': ce_vigor_val,
        'tau': tau_val, 'p_esc': p_esc_val,
        'sigma_motor': sigma_motor_val, 'sigma_v': sigma_v_val,
        'best_loss': best_loss,
    }])
    pop_df.to_csv(OUT_DIR / "m5_population.csv", index=False)
    print(f"  Saved population params to {OUT_DIR / 'm5_population.csv'}")

    # Fit quality
    r_lam_omega, _ = pearsonr(param_df['log_lam'], param_df['log_omega'])
    print(f"\n  λ: median={np.median(lam):.3f}, range=[{lam.min():.3f}, {lam.max():.3f}]")
    print(f"  ω: median={np.median(omega):.3f}, range=[{omega.min():.3f}, {omega.max():.3f}]")
    print(f"  r(log λ, log ω) = {r_lam_omega:.3f}")
    print(f"  γ = {gamma_val:.3f}, ε = {eps_val:.3f}, τ = {tau_val:.3f}")

    # Vigor r
    ep = np.array(samples['excess_pred']).mean(0)
    r_vig, _ = pearsonr(ep, np.array(data['vig_excess']))
    print(f"  Vigor r = {r_vig:.3f}")

    # Choice accuracy
    ch_subj_np = np.array(data['ch_subj'])
    ch_T_np = np.array(data['ch_T'])
    ch_dist_np = np.array(data['ch_dist_H'])
    ch_choice_np = np.array(data['ch_choice'])
    T_w = ch_T_np ** gamma_val
    S = (1 - T_w) + eps_val * T_w * p_esc_val
    effort = 0.81 * ch_dist_np - 0.16
    delta_eu = S * 4.0 - lam[ch_subj_np] * effort
    p_H = expit(np.clip(delta_eu / tau_val, -20, 20))
    acc = ((p_H >= 0.5).astype(int) == ch_choice_np).mean()
    print(f"  Choice accuracy = {acc:.3f}")

    return {
        'param_df': param_df, 'pop_params': pop_df.iloc[0].to_dict(),
        'best_loss': best_loss, 'guide': guide, 'best_params': best_params,
    }


# ============================================================
# Part 2: Parameter recovery
# ============================================================

def run_recovery(pop_params, n_datasets=5, n_subj=50, n_steps=30000, lr=0.001):
    """Simulate datasets from M5 and refit to test recovery."""
    gamma_val = pop_params['gamma']
    eps_val = pop_params['epsilon']
    tau_val = pop_params['tau']
    p_esc_val = pop_params['p_esc']

    all_lt, all_lr = [], []  # lambda true/recovered
    all_ot, all_or = [], []  # omega true/recovered

    for ds in range(n_datasets):
        print(f"\n  Recovery dataset {ds+1}/{n_datasets}")
        np.random.seed(ds * 100)

        # Draw true params from empirical distribution
        # Use the fitted M5 population means
        mu_lam = pop_params.get('mu_lam', np.log(0.5))
        sig_lam = pop_params.get('sig_lam', 0.7)
        mu_om = pop_params.get('mu_om', np.log(5.0))
        sig_om = pop_params.get('sig_om', 0.5)

        lam_true = np.exp(mu_lam + sig_lam * np.random.randn(n_subj))
        omega_true = np.exp(mu_om + sig_om * np.random.randn(n_subj))

        # Simulate choice trials (45 per subject: 3 blocks × 15 trials)
        ch_records = []
        for s in range(n_subj):
            for _ in range(3):
                for T in [0.1, 0.5, 0.9]:
                    for di, D in enumerate([1, 2, 3]):
                        T_w = T ** gamma_val
                        S = (1 - T_w) + eps_val * T_w * p_esc_val
                        effort = 0.81 * D - 0.16
                        deu = S * 4.0 - lam_true[s] * effort
                        ch = int(np.random.random() < expit(deu / tau_val))
                        ch_records.append({'subj': s, 'threat': T, 'distance_H': D, 'choice': ch})
                # 6 extra random trials
                for _ in range(6):
                    T = [0.1, 0.5, 0.9][np.random.randint(3)]
                    di = np.random.randint(3)
                    D = di + 1
                    T_w = T ** gamma_val
                    S = (1 - T_w) + eps_val * T_w * p_esc_val
                    effort = 0.81 * D - 0.16
                    deu = S * 4.0 - lam_true[s] * effort
                    ch = int(np.random.random() < expit(deu / tau_val))
                    ch_records.append({'subj': s, 'threat': T, 'distance_H': D, 'choice': ch})

        ch_df = pd.DataFrame(ch_records)

        # Simulate vigor trials (81 per subject)
        sigma_motor = pop_params.get('sigma_motor', 0.3)
        ce_vigor = pop_params.get('ce_vigor', 0.003)
        sigma_v = pop_params.get('sigma_v', 0.2)

        vig_records = []
        for s in range(n_subj):
            for _ in range(81):
                T = [0.1, 0.5, 0.9][np.random.randint(3)]
                D = np.random.randint(1, 4)
                is_heavy = int(np.random.random() < 0.5)
                R = 5.0 if is_heavy else 1.0
                req = 0.9 if is_heavy else 0.4

                # Optimal vigor from grid
                u_grid = np.linspace(0.1, 1.5, 30)
                T_w = T ** gamma_val
                S_u = (1 - T_w) + eps_val * T_w * p_esc_val * expit((u_grid - req) / sigma_motor)
                eu = S_u * R - (1 - S_u) * omega_true[s] * (R + 5.0) - ce_vigor * (u_grid - req)**2 * D
                w = np.exp(eu * 10 - np.max(eu * 10))
                w /= w.sum()
                u_star = np.sum(w * u_grid)
                excess = u_star - req + np.random.normal(0, sigma_v)

                vig_records.append({
                    'subj': s, 'threat': T, 'actual_dist': D,
                    'actual_R': R, 'actual_req': req,
                    'is_heavy': is_heavy, 'excess': excess,
                })

        vig_df = pd.DataFrame(vig_records)

        # Cookie-type centering
        hm = vig_df.loc[vig_df['is_heavy'] == 1, 'excess'].mean()
        lm = vig_df.loc[vig_df['is_heavy'] == 0, 'excess'].mean()
        vig_df['excess_cc'] = vig_df['excess'] - np.where(vig_df['is_heavy'] == 1, hm, lm)

        # Prepare JAX arrays
        si = {s: i for i, s in enumerate(sorted(ch_df['subj'].unique()))}
        ch_subj = jnp.array([si[s] for s in ch_df['subj']])
        ch_T = jnp.array(ch_df['threat'].values)
        ch_dist_H = jnp.array(ch_df['distance_H'].values, dtype=jnp.float64)
        ch_choice = jnp.array(ch_df['choice'].values)
        N_choice = len(ch_df)

        vig_subj = jnp.array([si[s] for s in vig_df['subj']])
        vig_T = jnp.array(vig_df['threat'].values)
        vig_R = jnp.array(vig_df['actual_R'].values)
        vig_req = jnp.array(vig_df['actual_req'].values)
        vig_dist = jnp.array(vig_df['actual_dist'].values, dtype=jnp.float64)
        vig_excess = jnp.array(vig_df['excess_cc'].values)
        vig_offset = jnp.array(np.where(vig_df['is_heavy'].values == 1, hm, lm))
        N_vigor = len(vig_df)

        # Build and fit M5
        model_fn = mc5.make_model_m5(n_subj, N_choice, N_vigor)
        kwargs = {
            'ch_subj': ch_subj, 'ch_T': ch_T, 'ch_dist_H': ch_dist_H,
            'ch_choice': ch_choice,
            'vig_subj': vig_subj, 'vig_T': vig_T, 'vig_R': vig_R,
            'vig_req': vig_req, 'vig_dist': vig_dist,
            'vig_excess': vig_excess, 'vig_offset': vig_offset,
        }

        guide = AutoNormal(model_fn)
        svi = SVI(model_fn, guide, numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0),
                  Trace_ELBO())
        state = svi.init(random.PRNGKey(ds), **kwargs)
        update_fn = jax.jit(svi.update)

        best_loss = float('inf')
        best_p = None
        for i in range(n_steps):
            state, loss = update_fn(state, **kwargs)
            if float(loss) < best_loss:
                best_loss = float(loss)
                best_p = svi.get_params(state)

        # Extract recovered params
        pred = Predictive(model_fn, guide=guide, params=best_p,
                          num_samples=200, return_sites=['lam', 'omega'])
        samp = pred(random.PRNGKey(ds + 1), **kwargs)
        lam_rec = np.array(samp['lam']).mean(0)
        omega_rec = np.array(samp['omega']).mean(0)

        r_lam, _ = pearsonr(np.log(lam_true), np.log(lam_rec))
        r_om, _ = pearsonr(np.log(omega_true), np.log(omega_rec))
        r_cross_lo, _ = pearsonr(np.log(lam_true), np.log(omega_rec))
        r_cross_ol, _ = pearsonr(np.log(omega_true), np.log(lam_rec))
        print(f"    λ: r={r_lam:.3f}, ω: r={r_om:.3f}, "
              f"cross λ→ω={r_cross_lo:.3f}, ω→λ={r_cross_ol:.3f}")

        all_lt.extend(lam_true)
        all_lr.extend(lam_rec)
        all_ot.extend(omega_true)
        all_or.extend(omega_rec)

    # Overall
    all_lt = np.array(all_lt)
    all_lr = np.array(all_lr)
    all_ot = np.array(all_ot)
    all_or = np.array(all_or)

    r_lam_all, _ = pearsonr(np.log(all_lt), np.log(all_lr))
    r_om_all, _ = pearsonr(np.log(all_ot), np.log(all_or))
    r_cross_all, _ = pearsonr(np.log(all_lt), np.log(all_or))

    print(f"\n  OVERALL RECOVERY ({n_datasets}×{n_subj} = {len(all_lt)} subjects):")
    print(f"    λ: r = {r_lam_all:.3f}  {'PASS' if r_lam_all > 0.70 else 'FAIL'}")
    print(f"    ω: r = {r_om_all:.3f}  {'PASS' if r_om_all > 0.70 else 'FAIL'}")
    print(f"    cross λ→ω: r = {r_cross_all:.3f} (should be low)")

    # Save
    recovery_df = pd.DataFrame({
        'lam_true': all_lt, 'lam_rec': all_lr,
        'omega_true': all_ot, 'omega_rec': all_or,
    })
    recovery_df.to_csv(OUT_DIR / "m5_recovery.csv", index=False)

    summary = pd.DataFrame([{
        'r_lam': r_lam_all, 'r_omega': r_om_all,
        'r_cross_lam_omega': r_cross_all,
        'n_datasets': n_datasets, 'n_subj_per': n_subj,
        'pass_lam': r_lam_all > 0.70, 'pass_omega': r_om_all > 0.70,
    }])
    summary.to_csv(OUT_DIR / "m5_recovery_summary.csv", index=False)
    print(f"  Saved to {OUT_DIR / 'm5_recovery.csv'}")

    return r_lam_all, r_om_all


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    t0 = time.time()

    print("=" * 70)
    print("M5 FIT + PARAMETER RECOVERY")
    print("=" * 70)

    # Load data
    print("\nPreparing data...")
    data = mc5.prepare_data()
    print(f"  {data['N_S']} subjects, {data['N_choice']} choice, {data['N_vigor']} vigor")

    # Part 1: Fit M5 on real data
    print("\n" + "=" * 70)
    print("PART 1: FIT M5 ON REAL DATA")
    print("=" * 70)
    result = fit_and_save_m5(data, n_steps=40000, lr=0.001)

    # Part 2: Recovery
    print("\n" + "=" * 70)
    print("PART 2: PARAMETER RECOVERY (3 × 50 subjects)")
    print("=" * 70)

    # Use fitted population params for simulation
    pop = result['pop_params']
    param_df = result['param_df']
    # Add empirical distribution info
    pop['mu_lam'] = param_df['log_lam'].mean()
    pop['sig_lam'] = param_df['log_lam'].std()
    pop['mu_om'] = param_df['log_omega'].mean()
    pop['sig_om'] = param_df['log_omega'].std()

    r_lam, r_om = run_recovery(pop, n_datasets=5, n_subj=50, n_steps=30000)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"  λ recovery: r = {r_lam:.3f}")
    print(f"  ω recovery: r = {r_om:.3f}")
    print(f"  Outputs in {OUT_DIR}")
    print(f"{'='*70}")
