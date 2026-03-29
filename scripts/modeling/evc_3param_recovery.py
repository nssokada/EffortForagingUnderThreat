"""
Parameter recovery for the 3-param EVC model (k, beta, cd).

Simulates 3 datasets × 50 subjects × 81 trials from the fitted population
distribution, refits, and reports Pearson r in log space for k, beta, cd.

Success criteria: r > 0.70 for all three parameters.
Critical test: k and beta must be separable (both recover well AND
their cross-recovery is low — k_true shouldn't predict beta_recovered).
"""

import sys
import os
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
from scipy.stats import pearsonr
from scipy.special import expit

jax.config.update('jax_enable_x64', True)


def simulate_dataset(n_subj, n_trials_per_cond, pop_params, rng_key):
    """Simulate a dataset from the 3-param model."""
    eps = pop_params['epsilon']
    gamma = pop_params['gamma']
    tau = pop_params['tau']
    p_esc = pop_params['p_esc']
    sigma_motor = pop_params['sigma_motor']
    ce_vigor = pop_params['ce_vigor']
    sigma_v = pop_params['sigma_v']

    # Generate per-subject parameters from population distribution
    key1, key2, key3, key4 = random.split(rng_key, 4)

    # Log-normal: sample in log space then exponentiate
    log_k = pop_params['mu_k'] + pop_params['sigma_k'] * random.normal(key1, (n_subj,))
    log_beta = pop_params['mu_beta'] + pop_params['sigma_beta'] * random.normal(key2, (n_subj,))
    log_cd = pop_params['mu_cd'] + pop_params['sigma_cd'] * random.normal(key3, (n_subj,))

    k_true = np.exp(np.array(log_k))
    beta_true = np.exp(np.array(log_beta))
    cd_true = np.exp(np.array(log_cd))

    # Task conditions (3T x 3D x 3E = 27 conditions, but we do choice + probe)
    threats = [0.1, 0.5, 0.9]
    distances = [1, 2, 3]

    # Generate 45 choice trials + 36 probe trials = 81 total per subject
    # Choice trials: all 27 conditions + 18 repeats (like real task: 15/block × 3 blocks)
    # Probe trials: forced choice at various conditions

    records = []

    for s in range(n_subj):
        trial_idx = 0
        for block in range(3):
            # 15 choice trials per block
            for t_idx in range(3):
                for d_idx in range(3):
                    T = threats[t_idx]
                    D_H = distances[d_idx]

                    # Choice
                    T_w = T ** gamma
                    S = (1 - T_w) + eps * T_w * p_esc
                    effort_cost = 0.81 * D_H - 0.16
                    threat_cost = 1 - S
                    delta_eu = S * 4.0 - k_true[s] * effort_cost - beta_true[s] * threat_cost
                    p_heavy = float(expit(delta_eu / tau))
                    chose_heavy = int(np.random.random() < p_heavy)

                    # Vigor (for chosen cookie)
                    if chose_heavy:
                        R, req, dist_v = 5.0, 0.9, D_H
                    else:
                        R, req, dist_v = 1.0, 0.4, 1

                    # Optimal vigor from grid search
                    u_grid = np.linspace(0.1, 1.5, 30)
                    S_u = (1 - T_w) + eps * T_w * p_esc * expit((u_grid - req) / sigma_motor)
                    eu = S_u * R - (1 - S_u) * cd_true[s] * (R + 5.0) - ce_vigor * (u_grid - req)**2 * dist_v
                    weights = np.exp(eu * 10.0 - np.max(eu * 10.0))
                    weights /= weights.sum()
                    u_star = np.sum(weights * u_grid)
                    excess = u_star - req + np.random.normal(0, sigma_v)

                    records.append({
                        'subj': s, 'type': 1, 'trial': trial_idx,
                        'threat': T, 'distance_H': D_H, 'startDistance': {1:5, 2:7, 3:9}[D_H],
                        'choice': chose_heavy,
                        'trialCookie_weight': 3.0 if chose_heavy else 1.0,
                        'excess': excess, 'req': req, 'R': R, 'dist_v': dist_v,
                    })
                    trial_idx += 1

                    # Extra choice trials to get to 15/block
                    if t_idx == 0 and d_idx < 2:  # 6 extra per block
                        T2 = threats[np.random.randint(3)]
                        D2 = distances[np.random.randint(3)]
                        T_w2 = T2 ** gamma
                        S2 = (1 - T_w2) + eps * T_w2 * p_esc
                        effort2 = 0.81 * D2 - 0.16
                        threat2 = 1 - S2
                        deu2 = S2 * 4.0 - k_true[s] * effort2 - beta_true[s] * threat2
                        p2 = float(expit(deu2 / tau))
                        ch2 = int(np.random.random() < p2)
                        if ch2:
                            R2, req2, dv2 = 5.0, 0.9, D2
                        else:
                            R2, req2, dv2 = 1.0, 0.4, 1
                        u_grid2 = np.linspace(0.1, 1.5, 30)
                        S_u2 = (1 - T_w2) + eps * T_w2 * p_esc * expit((u_grid2 - req2) / sigma_motor)
                        eu2 = S_u2 * R2 - (1 - S_u2) * cd_true[s] * (R2 + 5.0) - ce_vigor * (u_grid2 - req2)**2 * dv2
                        w2 = np.exp(eu2 * 10 - np.max(eu2 * 10)); w2 /= w2.sum()
                        u2 = np.sum(w2 * u_grid2)
                        excess2 = u2 - req2 + np.random.normal(0, sigma_v)
                        records.append({
                            'subj': s, 'type': 1, 'trial': trial_idx,
                            'threat': T2, 'distance_H': D2, 'startDistance': {1:5, 2:7, 3:9}[D2],
                            'choice': ch2,
                            'trialCookie_weight': 3.0 if ch2 else 1.0,
                            'excess': excess2, 'req': req2, 'R': R2, 'dist_v': dv2,
                        })
                        trial_idx += 1

            # 12 probe trials per block (forced choice — random cookie assignment)
            for p_idx in range(12):
                T = threats[np.random.randint(3)]
                D = distances[np.random.randint(3)]
                is_heavy = int(np.random.random() < 0.5)
                if is_heavy:
                    R, req, dist_v = 5.0, 0.9, D
                else:
                    R, req, dist_v = 1.0, 0.4, 1

                T_w = T ** gamma
                u_grid = np.linspace(0.1, 1.5, 30)
                S_u = (1 - T_w) + eps * T_w * p_esc * expit((u_grid - req) / sigma_motor)
                eu = S_u * R - (1 - S_u) * cd_true[s] * (R + 5.0) - ce_vigor * (u_grid - req)**2 * dist_v
                weights = np.exp(eu * 10.0 - np.max(eu * 10.0))
                weights /= weights.sum()
                u_star = np.sum(weights * u_grid)
                excess = u_star - req + np.random.normal(0, sigma_v)

                records.append({
                    'subj': s, 'type': 5, 'trial': trial_idx,
                    'threat': T, 'distance_H': D, 'startDistance': {1:5, 2:7, 3:9}[D],
                    'choice': is_heavy,
                    'trialCookie_weight': 3.0 if is_heavy else 1.0,
                    'excess': excess, 'req': req, 'R': R, 'dist_v': dist_v,
                })
                trial_idx += 1

    df = pd.DataFrame(records)
    return df, k_true, beta_true, cd_true


def prepare_simulated_data(df):
    """Convert simulated dataframe to model input arrays."""
    choice_df = df[df['type'] == 1].copy()
    vigor_df = df.copy()

    # Cookie-type centering
    choice_vigor = vigor_df[vigor_df['type'] == 1]
    heavy_mask = choice_vigor['trialCookie_weight'] == 3.0
    heavy_mean = choice_vigor.loc[heavy_mask, 'excess'].mean()
    light_mean = choice_vigor.loc[~heavy_mask, 'excess'].mean()
    vigor_df['excess_cc'] = vigor_df['excess'] - np.where(
        vigor_df['trialCookie_weight'] == 3.0, heavy_mean, light_mean)

    subjects = sorted(df['subj'].unique())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)

    ch_subj = jnp.array([subj_to_idx[s] for s in choice_df['subj']])
    ch_T = jnp.array(choice_df['threat'].values)
    ch_dist_H = jnp.array(choice_df['distance_H'].values, dtype=jnp.float64)
    ch_choice = jnp.array(choice_df['choice'].values)

    vig_subj = jnp.array([subj_to_idx[s] for s in vigor_df['subj']])
    vig_T = jnp.array(vigor_df['threat'].values)
    vig_R = jnp.array(vigor_df['R'].values)
    vig_req = jnp.array(vigor_df['req'].values)
    vig_dist = jnp.array(vigor_df['dist_v'].values, dtype=jnp.float64)
    vig_excess = jnp.array(vigor_df['excess_cc'].values)
    vig_offset = jnp.array(np.where(
        vigor_df['trialCookie_weight'].values == 3.0, heavy_mean, light_mean))

    return {
        'ch_subj': ch_subj, 'ch_T': ch_T, 'ch_dist_H': ch_dist_H,
        'ch_choice': ch_choice,
        'vig_subj': vig_subj, 'vig_T': vig_T, 'vig_R': vig_R,
        'vig_req': vig_req, 'vig_dist': vig_dist,
        'vig_excess': vig_excess, 'vig_offset': vig_offset,
        'subjects': subjects, 'N_S': N_S,
        'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
        'heavy_mean': heavy_mean, 'light_mean': light_mean,
    }


def make_recovery_model(N_S, N_choice, N_vigor):
    """Same model as evc_3param but for recovery."""
    def evc_3param(ch_subj, ch_T, ch_dist_H, ch_choice,
                   vig_subj, vig_T, vig_R, vig_req, vig_dist,
                   vig_excess, vig_offset):
        mu_k = numpyro.sample('mu_k', dist.Normal(0.0, 1.0))
        mu_beta = numpyro.sample('mu_beta', dist.Normal(-1.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_k = numpyro.sample('sigma_k', dist.HalfNormal(0.5))
        sigma_beta = numpyro.sample('sigma_beta', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        eps_raw = numpyro.sample('eps_raw', dist.Normal(-1.0, 0.5))
        epsilon = numpyro.deterministic('epsilon', jnp.exp(eps_raw))
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)
        ce_vigor_raw = numpyro.sample('ce_vigor_raw', dist.Normal(-3.0, 1.0))
        ce_vigor = numpyro.deterministic('ce_vigor', jnp.exp(ce_vigor_raw))

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

        # Choice
        k_ch = k[ch_subj]
        beta_ch = beta[ch_subj]
        T_w_ch = jnp.power(ch_T, gamma)
        S_ch = (1.0 - T_w_ch) + epsilon * T_w_ch * p_esc
        effort_cost = 0.81 * ch_dist_H - 0.16
        threat_cost = 1.0 - S_ch
        delta_eu = S_ch * 4.0 - k_ch * effort_cost - beta_ch * threat_cost
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                obs=ch_choice)

        # Vigor
        cd_v = c_death[vig_subj]
        T_w_v = jnp.power(vig_T, gamma)
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - T_w_v[:, None])
               + epsilon * T_w_v[:, None] * p_esc
               * jax.nn.sigmoid((u_g - vig_req[:, None]) / sigma_motor))
        deviation = u_g - vig_req[:, None]
        eu_grid = (S_u * vig_R[:, None]
                   - (1.0 - S_u) * cd_v[:, None] * (vig_R[:, None] + 5.0)
                   - ce_vigor * deviation ** 2 * vig_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - vig_req - vig_offset

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v), obs=vig_excess)

    return evc_3param


def fit_recovery(data, n_steps=30000, lr=0.001, seed=42):
    """Fit model for recovery."""
    model = make_recovery_model(data['N_S'], data['N_choice'], data['N_vigor'])
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

    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)

    params_fit = svi.get_params(state)

    # Extract recovered parameters
    pred = Predictive(model, guide=guide, params=params_fit,
                      num_samples=200, return_sites=['k', 'beta', 'c_death'])
    samples = pred(random.PRNGKey(seed + 1), **kwargs)

    k_rec = np.array(samples['k']).mean(0)
    beta_rec = np.array(samples['beta']).mean(0)
    cd_rec = np.array(samples['c_death']).mean(0)

    return k_rec, beta_rec, cd_rec, float(loss)


if __name__ == '__main__':
    import time
    t0 = time.time()

    # Use fitted population parameters from the real data fit
    pop_params = {
        'mu_k': np.log(0.41),      # median k from fit
        'sigma_k': 0.8,             # spread in log space
        'mu_beta': np.log(0.17),    # median beta from fit
        'sigma_beta': 0.8,          # spread in log space
        'mu_cd': np.log(24.9),      # median cd from fit
        'sigma_cd': 1.0,            # spread in log space
        'epsilon': 0.149,
        'gamma': 0.258,
        'tau': 0.604,
        'p_esc': 0.016,
        'sigma_motor': 1.0,
        'ce_vigor': 0.003,
        'sigma_v': 0.229,
    }

    N_SUBJ = 50
    N_DATASETS = 3

    all_k_true, all_k_rec = [], []
    all_beta_true, all_beta_rec = [], []
    all_cd_true, all_cd_rec = [], []

    for ds in range(N_DATASETS):
        print(f"\n{'='*60}")
        print(f"Dataset {ds+1}/{N_DATASETS}")
        print(f"{'='*60}")

        # Simulate
        rng_key = random.PRNGKey(ds * 100)
        df_sim, k_true, beta_true, cd_true = simulate_dataset(
            N_SUBJ, 81, pop_params, rng_key)

        print(f"  Simulated: {len(df_sim)} trials, {N_SUBJ} subjects")
        print(f"  k_true: median={np.median(k_true):.3f}, range=[{k_true.min():.3f}, {k_true.max():.3f}]")
        print(f"  beta_true: median={np.median(beta_true):.3f}, range=[{beta_true.min():.3f}, {beta_true.max():.3f}]")
        print(f"  cd_true: median={np.median(cd_true):.3f}, range=[{cd_true.min():.3f}, {cd_true.max():.3f}]")

        # Prepare and fit
        data = prepare_simulated_data(df_sim)
        print(f"  Fitting: {data['N_choice']} choice, {data['N_vigor']} vigor trials")

        k_rec, beta_rec, cd_rec, final_loss = fit_recovery(data, n_steps=30000, lr=0.001, seed=ds)
        print(f"  Final loss: {final_loss:.1f}")

        # Correlations in log space
        r_k, p_k = pearsonr(np.log(k_true), np.log(k_rec))
        r_beta, p_beta = pearsonr(np.log(beta_true), np.log(beta_rec))
        r_cd, p_cd = pearsonr(np.log(cd_true), np.log(cd_rec))

        print(f"\n  Recovery (log space):")
        print(f"    k:    r={r_k:.3f} (p={p_k:.4f})")
        print(f"    beta: r={r_beta:.3f} (p={p_beta:.4f})")
        print(f"    cd:   r={r_cd:.3f} (p={p_cd:.4f})")

        # Cross-recovery (identifiability check)
        r_k_beta, _ = pearsonr(np.log(k_true), np.log(beta_rec))
        r_beta_k, _ = pearsonr(np.log(beta_true), np.log(k_rec))
        r_k_cd, _ = pearsonr(np.log(k_true), np.log(cd_rec))
        r_beta_cd, _ = pearsonr(np.log(beta_true), np.log(cd_rec))

        print(f"\n  Cross-recovery (should be LOW):")
        print(f"    k_true → beta_rec:  r={r_k_beta:.3f}")
        print(f"    beta_true → k_rec:  r={r_beta_k:.3f}")
        print(f"    k_true → cd_rec:    r={r_k_cd:.3f}")
        print(f"    beta_true → cd_rec: r={r_beta_cd:.3f}")

        all_k_true.extend(k_true)
        all_k_rec.extend(k_rec)
        all_beta_true.extend(beta_true)
        all_beta_rec.extend(beta_rec)
        all_cd_true.extend(cd_true)
        all_cd_rec.extend(cd_rec)

    # Overall
    all_k_true = np.array(all_k_true)
    all_k_rec = np.array(all_k_rec)
    all_beta_true = np.array(all_beta_true)
    all_beta_rec = np.array(all_beta_rec)
    all_cd_true = np.array(all_cd_true)
    all_cd_rec = np.array(all_cd_rec)

    r_k_all, _ = pearsonr(np.log(all_k_true), np.log(all_k_rec))
    r_beta_all, _ = pearsonr(np.log(all_beta_true), np.log(all_beta_rec))
    r_cd_all, _ = pearsonr(np.log(all_cd_true), np.log(all_cd_rec))

    # Cross-recovery overall
    r_kb_cross, _ = pearsonr(np.log(all_k_true), np.log(all_beta_rec))
    r_bk_cross, _ = pearsonr(np.log(all_beta_true), np.log(all_k_rec))

    print(f"\n{'='*60}")
    print(f"OVERALL RECOVERY ({N_DATASETS} datasets × {N_SUBJ} subjects = {len(all_k_true)} total)")
    print(f"{'='*60}")
    print(f"  k:    r={r_k_all:.3f}")
    print(f"  beta: r={r_beta_all:.3f}")
    print(f"  cd:   r={r_cd_all:.3f}")
    print(f"\n  Cross-recovery:")
    print(f"    k_true → beta_rec:  r={r_kb_cross:.3f}")
    print(f"    beta_true → k_rec:  r={r_bk_cross:.3f}")

    PASS = r_k_all > 0.70 and r_beta_all > 0.70 and r_cd_all > 0.70
    print(f"\n  {'PASS' if PASS else 'FAIL'}: all r > 0.70? k={r_k_all>.7}, beta={r_beta_all>.7}, cd={r_cd_all>.7}")

    # Save results
    recovery_df = pd.DataFrame({
        'k_true': all_k_true, 'k_rec': all_k_rec,
        'beta_true': all_beta_true, 'beta_rec': all_beta_rec,
        'cd_true': all_cd_true, 'cd_rec': all_cd_rec,
    })
    out_path = 'results/stats/oc_evc_3param_recovery.csv'
    recovery_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    summary_df = pd.DataFrame([{
        'r_k': r_k_all, 'r_beta': r_beta_all, 'r_cd': r_cd_all,
        'cross_k_beta': r_kb_cross, 'cross_beta_k': r_bk_cross,
        'n_datasets': N_DATASETS, 'n_subj_per': N_SUBJ,
        'pass': PASS,
    }])
    summary_path = 'results/stats/oc_evc_3param_recovery_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
