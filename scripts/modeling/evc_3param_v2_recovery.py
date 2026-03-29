"""
Parameter recovery for EVC 3-param v2 (k + beta + cd, no gamma/epsilon).
3 datasets × 50 subjects.
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
from scipy.stats import pearsonr
from scipy.special import expit

jax.config.update('jax_enable_x64', True)


def simulate_dataset(n_subj, pop_params, rng_key):
    """Simulate from the v2 model (no gamma, no epsilon in choice)."""
    tau = pop_params['tau']
    p_esc = pop_params['p_esc']
    sigma_motor = pop_params['sigma_motor']
    ce_vigor = pop_params['ce_vigor']
    sigma_v = pop_params['sigma_v']

    key1, key2, key3 = random.split(rng_key, 3)
    log_k = pop_params['mu_k'] + pop_params['sigma_k'] * random.normal(key1, (n_subj,))
    log_beta = pop_params['mu_beta'] + pop_params['sigma_beta'] * random.normal(key2, (n_subj,))
    log_cd = pop_params['mu_cd'] + pop_params['sigma_cd'] * random.normal(key3, (n_subj,))
    k_true = np.exp(np.array(log_k))
    beta_true = np.exp(np.array(log_beta))
    cd_true = np.exp(np.array(log_cd))

    threats = [0.1, 0.5, 0.9]
    distances = [1, 2, 3]
    records = []

    for s in range(n_subj):
        trial_idx = 0
        for block in range(3):
            # 15 choice trials per block
            for t_idx in range(3):
                for d_idx in range(3):
                    T = threats[t_idx]
                    D_H = distances[d_idx]

                    # Choice: dEU = 4 - k*effort - beta*T
                    effort_cost = 0.81 * D_H - 0.16
                    delta_eu = 4.0 - k_true[s] * effort_cost - beta_true[s] * T
                    p_heavy = float(expit(delta_eu / tau))
                    chose_heavy = int(np.random.random() < p_heavy)

                    if chose_heavy:
                        R, req, dist_v = 5.0, 0.9, D_H
                    else:
                        R, req, dist_v = 1.0, 0.4, 1

                    # Vigor
                    u_grid = np.linspace(0.1, 1.5, 30)
                    S_u = (1 - T) + T * p_esc * expit((u_grid - req) / sigma_motor)
                    eu = S_u * R - (1 - S_u) * cd_true[s] * (R + 5.0) - ce_vigor * (u_grid - req)**2 * dist_v
                    w = np.exp(eu * 10 - np.max(eu * 10)); w /= w.sum()
                    u_star = np.sum(w * u_grid)
                    excess = u_star - req + np.random.normal(0, sigma_v)

                    records.append({
                        'subj': s, 'type': 1, 'trial': trial_idx,
                        'threat': T, 'distance_H': D_H,
                        'choice': chose_heavy,
                        'trialCookie_weight': 3.0 if chose_heavy else 1.0,
                        'excess': excess, 'req': req, 'R': R, 'dist_v': dist_v,
                    })
                    trial_idx += 1

                    # Extra choice trials
                    if t_idx == 0 and d_idx < 2:
                        T2 = threats[np.random.randint(3)]
                        D2 = distances[np.random.randint(3)]
                        ef2 = 0.81 * D2 - 0.16
                        deu2 = 4.0 - k_true[s] * ef2 - beta_true[s] * T2
                        p2 = float(expit(deu2 / tau))
                        ch2 = int(np.random.random() < p2)
                        R2, req2, dv2 = (5.0, 0.9, D2) if ch2 else (1.0, 0.4, 1)
                        u_g2 = np.linspace(0.1, 1.5, 30)
                        S_u2 = (1-T2) + T2*p_esc*expit((u_g2-req2)/sigma_motor)
                        eu2 = S_u2*R2 - (1-S_u2)*cd_true[s]*(R2+5.0) - ce_vigor*(u_g2-req2)**2*dv2
                        w2 = np.exp(eu2*10-np.max(eu2*10)); w2 /= w2.sum()
                        u2 = np.sum(w2*u_g2)
                        excess2 = u2 - req2 + np.random.normal(0, sigma_v)
                        records.append({
                            'subj': s, 'type': 1, 'trial': trial_idx,
                            'threat': T2, 'distance_H': D2,
                            'choice': ch2,
                            'trialCookie_weight': 3.0 if ch2 else 1.0,
                            'excess': excess2, 'req': req2, 'R': R2, 'dist_v': dv2,
                        })
                        trial_idx += 1

            # 12 probe trials per block
            for _ in range(12):
                T = threats[np.random.randint(3)]
                D = distances[np.random.randint(3)]
                is_heavy = int(np.random.random() < 0.5)
                R, req, dist_v = (5.0, 0.9, D) if is_heavy else (1.0, 0.4, 1)
                u_grid = np.linspace(0.1, 1.5, 30)
                S_u = (1-T) + T*p_esc*expit((u_grid-req)/sigma_motor)
                eu = S_u*R - (1-S_u)*cd_true[s]*(R+5.0) - ce_vigor*(u_grid-req)**2*dist_v
                w = np.exp(eu*10-np.max(eu*10)); w /= w.sum()
                u_star = np.sum(w*u_grid)
                excess = u_star - req + np.random.normal(0, sigma_v)
                records.append({
                    'subj': s, 'type': 5, 'trial': trial_idx,
                    'threat': T, 'distance_H': D,
                    'choice': is_heavy,
                    'trialCookie_weight': 3.0 if is_heavy else 1.0,
                    'excess': excess, 'req': req, 'R': R, 'dist_v': dist_v,
                })
                trial_idx += 1

    return pd.DataFrame(records), k_true, beta_true, cd_true


def prepare_simulated_data(df):
    """Convert sim dataframe to model arrays."""
    choice_df = df[df['type'] == 1].copy()
    vigor_df = df.copy()
    choice_vigor = vigor_df[vigor_df['type'] == 1]
    heavy_mask = choice_vigor['trialCookie_weight'] == 3.0
    heavy_mean = choice_vigor.loc[heavy_mask, 'excess'].mean()
    light_mean = choice_vigor.loc[~heavy_mask, 'excess'].mean()
    vigor_df['excess_cc'] = vigor_df['excess'] - np.where(
        vigor_df['trialCookie_weight'] == 3.0, heavy_mean, light_mean)

    subjects = sorted(df['subj'].unique())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)

    return {
        'ch_subj': jnp.array([subj_to_idx[s] for s in choice_df['subj']]),
        'ch_T': jnp.array(choice_df['threat'].values),
        'ch_dist_H': jnp.array(choice_df['distance_H'].values, dtype=jnp.float64),
        'ch_choice': jnp.array(choice_df['choice'].values),
        'vig_subj': jnp.array([subj_to_idx[s] for s in vigor_df['subj']]),
        'vig_T': jnp.array(vigor_df['threat'].values),
        'vig_R': jnp.array(vigor_df['R'].values),
        'vig_req': jnp.array(vigor_df['req'].values),
        'vig_dist': jnp.array(vigor_df['dist_v'].values, dtype=jnp.float64),
        'vig_excess': jnp.array(vigor_df['excess_cc'].values),
        'vig_offset': jnp.array(np.where(
            vigor_df['trialCookie_weight'].values == 3.0, heavy_mean, light_mean)),
        'subjects': subjects, 'N_S': N_S,
        'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
    }


def make_recovery_model(N_S, N_choice, N_vigor):
    """Same as v2 model."""
    def model(ch_subj, ch_T, ch_dist_H, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_excess, vig_offset):
        mu_k = numpyro.sample('mu_k', dist.Normal(0.0, 1.0))
        mu_beta = numpyro.sample('mu_beta', dist.Normal(1.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_k = numpyro.sample('sigma_k', dist.HalfNormal(0.5))
        sigma_beta = numpyro.sample('sigma_beta', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
        p_esc = jax.nn.sigmoid(p_esc_raw)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 1.0)
        ce_vigor_raw = numpyro.sample('ce_vigor_raw', dist.Normal(-3.0, 1.0))
        ce_vigor = jnp.exp(ce_vigor_raw)

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
        k_ch = k[ch_subj]; beta_ch = beta[ch_subj]
        effort_cost = 0.81 * ch_dist_H - 0.16
        delta_eu = 4.0 - k_ch * effort_cost - beta_ch * ch_T
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)), obs=ch_choice)

        # Vigor
        cd_v = c_death[vig_subj]
        u_grid = jnp.linspace(0.1, 1.5, 30)
        u_g = u_grid[None, :]
        S_u = ((1.0 - vig_T[:, None])
               + vig_T[:, None] * p_esc
               * jax.nn.sigmoid((u_g - vig_req[:, None]) / sigma_motor))
        deviation = u_g - vig_req[:, None]
        eu_grid = (S_u * vig_R[:, None]
                   - (1.0 - S_u) * cd_v[:, None] * (vig_R[:, None] + 5.0)
                   - ce_vigor * deviation**2 * vig_dist[:, None])
        weights = jax.nn.softmax(eu_grid * 10.0, axis=1)
        u_star = jnp.sum(weights * u_g, axis=1)
        excess_pred = u_star - vig_req - vig_offset
        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(excess_pred, sigma_v), obs=vig_excess)

    return model


def fit_recovery(data, n_steps=30000, lr=0.001, seed=42):
    """Fit for recovery."""
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
    pred = Predictive(model, guide=guide, params=params_fit,
                      num_samples=200, return_sites=['k', 'beta', 'c_death'])
    samples = pred(random.PRNGKey(seed + 1), **kwargs)

    return (np.array(samples['k']).mean(0),
            np.array(samples['beta']).mean(0),
            np.array(samples['c_death']).mean(0),
            float(loss))


if __name__ == '__main__':
    import time
    t0 = time.time()

    # Population params from v2 fit
    pop_params = {
        'mu_k': np.log(1.48),
        'sigma_k': 0.5,
        'mu_beta': np.log(4.30),
        'sigma_beta': 0.5,
        'mu_cd': np.log(30.9),
        'sigma_cd': 1.0,
        'tau': 1.06,
        'p_esc': 0.002,
        'sigma_motor': 0.82,
        'ce_vigor': 0.0027,
        'sigma_v': 0.241,
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

        rng_key = random.PRNGKey(ds * 100)
        df_sim, k_true, beta_true, cd_true = simulate_dataset(N_SUBJ, pop_params, rng_key)
        print(f"  Simulated: {len(df_sim)} trials, {N_SUBJ} subjects")
        print(f"  k_true: median={np.median(k_true):.3f}")
        print(f"  beta_true: median={np.median(beta_true):.3f}")
        print(f"  cd_true: median={np.median(cd_true):.3f}")

        data = prepare_simulated_data(df_sim)
        print(f"  Fitting: {data['N_choice']} choice, {data['N_vigor']} vigor")

        k_rec, beta_rec, cd_rec, final_loss = fit_recovery(data, n_steps=30000, seed=ds)
        print(f"  Final loss: {final_loss:.1f}")

        r_k, p_k = pearsonr(np.log(k_true), np.log(k_rec))
        r_beta, p_beta = pearsonr(np.log(beta_true), np.log(beta_rec))
        r_cd, p_cd = pearsonr(np.log(cd_true), np.log(cd_rec))
        print(f"\n  Recovery: k r={r_k:.3f}, beta r={r_beta:.3f}, cd r={r_cd:.3f}")

        r_kb_cross, _ = pearsonr(np.log(k_true), np.log(beta_rec))
        r_bk_cross, _ = pearsonr(np.log(beta_true), np.log(k_rec))
        print(f"  Cross: k→beta_rec r={r_kb_cross:.3f}, beta→k_rec r={r_bk_cross:.3f}")

        all_k_true.extend(k_true); all_k_rec.extend(k_rec)
        all_beta_true.extend(beta_true); all_beta_rec.extend(beta_rec)
        all_cd_true.extend(cd_true); all_cd_rec.extend(cd_rec)

    all_k_true, all_k_rec = np.array(all_k_true), np.array(all_k_rec)
    all_beta_true, all_beta_rec = np.array(all_beta_true), np.array(all_beta_rec)
    all_cd_true, all_cd_rec = np.array(all_cd_true), np.array(all_cd_rec)

    r_k_all, _ = pearsonr(np.log(all_k_true), np.log(all_k_rec))
    r_beta_all, _ = pearsonr(np.log(all_beta_true), np.log(all_beta_rec))
    r_cd_all, _ = pearsonr(np.log(all_cd_true), np.log(all_cd_rec))
    r_kb_cross, _ = pearsonr(np.log(all_k_true), np.log(all_beta_rec))
    r_bk_cross, _ = pearsonr(np.log(all_beta_true), np.log(all_k_rec))

    print(f"\n{'='*60}")
    print(f"OVERALL ({N_DATASETS} × {N_SUBJ} = {len(all_k_true)})")
    print(f"{'='*60}")
    print(f"  k:    r={r_k_all:.3f}")
    print(f"  beta: r={r_beta_all:.3f}")
    print(f"  cd:   r={r_cd_all:.3f}")
    print(f"  Cross: k→beta r={r_kb_cross:.3f}, beta→k r={r_bk_cross:.3f}")
    PASS = r_k_all > 0.70 and r_beta_all > 0.70 and r_cd_all > 0.70
    print(f"\n  {'PASS' if PASS else 'FAIL'}: all r > 0.70?")

    pd.DataFrame({
        'k_true': all_k_true, 'k_rec': all_k_rec,
        'beta_true': all_beta_true, 'beta_rec': all_beta_rec,
        'cd_true': all_cd_true, 'cd_rec': all_cd_rec,
    }).to_csv('results/stats/oc_evc_3param_v2_recovery.csv', index=False)

    pd.DataFrame([{
        'r_k': r_k_all, 'r_beta': r_beta_all, 'r_cd': r_cd_all,
        'cross_k_beta': r_kb_cross, 'cross_beta_k': r_bk_cross,
        'pass': PASS,
    }]).to_csv('results/stats/oc_evc_3param_v2_recovery_summary.csv', index=False)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
