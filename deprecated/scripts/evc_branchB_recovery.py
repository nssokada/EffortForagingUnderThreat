"""
Parameter recovery for Branch B: k + β + cd with frac_full vigor.
3 datasets × 50 subjects.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
from scipy.special import expit
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from jax import random

jax.config.update('jax_enable_x64', True)

# Fixed survival params
V_FULL = 1.450; REM_FRAC = 0.900; BUFFER = 0.600; P_FLOOR = 0.090
D_GAME = {1: 5.0, 2: 7.0, 3: 9.0}
MU_STRIKE = {1: 2.460, 2: 3.478, 3: 4.765}
SD_STRIKE = {1: 0.574, 2: 0.896, 3: 1.121}
D_GAME_ARR = jnp.array([5.0, 7.0, 9.0])
MU_STRIKE_ARR = jnp.array([2.460, 3.478, 4.765])
SD_STRIKE_ARR = jnp.array([0.574, 0.896, 1.121])

# Population params from the fit
ALPHA = 19.53
TAU = 1.06
SIGMA_V = 0.208


def P_escape_np(f, D):
    rem = D_GAME[D] * REM_FRAC
    spd = V_FULL * (0.5 + 0.5 * f)
    arr = rem / spd
    z = (arr - (MU_STRIKE[D] + BUFFER)) / SD_STRIKE[D]
    return P_FLOOR + (1 - P_FLOOR) * (1 - norm.cdf(z))


def S_np(f, T, D):
    return (1 - T) + T * P_escape_np(f, D)


def log_odds_cost(f):
    fc = np.clip(f, 0.02, 0.98)
    return np.log(fc / (1 - fc))**2


def simulate_dataset(n_subj, pop_params, rng_seed):
    np.random.seed(rng_seed)
    k_true = np.exp(pop_params['mu_k'] + pop_params['sigma_k'] * np.random.randn(n_subj))
    beta_true = np.exp(pop_params['mu_beta'] + pop_params['sigma_beta'] * np.random.randn(n_subj))
    cd_true = np.exp(pop_params['mu_cd'] + pop_params['sigma_cd'] * np.random.randn(n_subj))

    threats = [0.1, 0.5, 0.9]
    distances = [1, 2, 3]
    f_grid = np.linspace(0.02, 0.98, 100)
    records = []

    for s in range(n_subj):
        trial_idx = 0
        for block in range(3):
            # 15 choice trials
            for t_idx in range(3):
                for d_idx in range(3):
                    T = threats[t_idx]; D_H = distances[d_idx]
                    effort = 0.81 * D_H - 0.16
                    deu = 4.0 - k_true[s] * effort - beta_true[s] * T
                    p_heavy = float(expit(deu / TAU))
                    chose_heavy = int(np.random.random() < p_heavy)

                    R = 5.0 if chose_heavy else 1.0
                    req = 0.9 if chose_heavy else 0.4
                    D_v = D_H if chose_heavy else 1

                    eu = np.array([S_np(f, T, D_v) * R - (1 - S_np(f, T, D_v)) * cd_true[s] * (R + 5)
                                   - k_true[s] * ALPHA * log_odds_cost(f) * req**2 * D_v for f in f_grid])
                    f_star = f_grid[np.argmax(eu)]
                    f_obs = np.clip(f_star + np.random.normal(0, SIGMA_V), 0.01, 0.99)

                    records.append({'subj': s, 'type': 1, 'threat': T, 'distance_H': D_H,
                                   'choice': chose_heavy, 'frac_full': f_obs, 'R': R, 'req': req, 'dist_v': D_v,
                                   'trialCookie_weight': 3.0 if chose_heavy else 1.0})
                    trial_idx += 1

                    if t_idx == 0 and d_idx < 2:
                        T2 = threats[np.random.randint(3)]; D2 = distances[np.random.randint(3)]
                        ef2 = 0.81 * D2 - 0.16
                        deu2 = 4.0 - k_true[s] * ef2 - beta_true[s] * T2
                        ch2 = int(np.random.random() < expit(deu2 / TAU))
                        R2 = 5.0 if ch2 else 1.0; req2 = 0.9 if ch2 else 0.4; dv2 = D2 if ch2 else 1
                        eu2 = np.array([S_np(f, T2, dv2)*R2-(1-S_np(f, T2, dv2))*cd_true[s]*(R2+5)
                                       -k_true[s]*ALPHA*log_odds_cost(f)*req2**2*dv2 for f in f_grid])
                        f2 = np.clip(f_grid[np.argmax(eu2)] + np.random.normal(0, SIGMA_V), 0.01, 0.99)
                        records.append({'subj': s, 'type': 1, 'threat': T2, 'distance_H': D2,
                                       'choice': ch2, 'frac_full': f2, 'R': R2, 'req': req2, 'dist_v': dv2,
                                       'trialCookie_weight': 3.0 if ch2 else 1.0})
                        trial_idx += 1

            # 12 probe trials
            for _ in range(12):
                T = threats[np.random.randint(3)]; D = distances[np.random.randint(3)]
                is_heavy = int(np.random.random() < 0.5)
                R = 5.0 if is_heavy else 1.0; req = 0.9 if is_heavy else 0.4; dv = D if is_heavy else 1
                eu = np.array([S_np(f, T, dv)*R-(1-S_np(f, T, dv))*cd_true[s]*(R+5)
                              -k_true[s]*ALPHA*log_odds_cost(f)*req**2*dv for f in f_grid])
                f_obs = np.clip(f_grid[np.argmax(eu)] + np.random.normal(0, SIGMA_V), 0.01, 0.99)
                records.append({'subj': s, 'type': 5, 'threat': T, 'distance_H': D,
                               'choice': is_heavy, 'frac_full': f_obs, 'R': R, 'req': req, 'dist_v': dv,
                               'trialCookie_weight': 3.0 if is_heavy else 1.0})
                trial_idx += 1

    return pd.DataFrame(records), k_true, beta_true, cd_true


def prepare_sim(df):
    choice_df = df[df['type'] == 1].copy()
    vigor_df = df.copy()
    choice_vigor = vigor_df[vigor_df['type'] == 1]
    hm = choice_vigor[choice_vigor['trialCookie_weight'] == 3.0]['frac_full'].mean()
    lm = choice_vigor[choice_vigor['trialCookie_weight'] != 3.0]['frac_full'].mean()
    vigor_df['ff_cc'] = vigor_df['frac_full'] - np.where(vigor_df['trialCookie_weight'] == 3.0, hm, lm)

    subjects = sorted(df['subj'].unique())
    si = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)

    return {
        'ch_subj': jnp.array([si[s] for s in choice_df['subj']]),
        'ch_T': jnp.array(choice_df['threat'].values),
        'ch_dist_H': jnp.array(choice_df['distance_H'].values, dtype=jnp.float64),
        'ch_choice': jnp.array(choice_df['choice'].values),
        'vig_subj': jnp.array([si[s] for s in vigor_df['subj']]),
        'vig_T': jnp.array(vigor_df['threat'].values),
        'vig_R': jnp.array(vigor_df['R'].values),
        'vig_dist': jnp.array(vigor_df['dist_v'].values, dtype=jnp.float64),
        'vig_ff': jnp.array(vigor_df['ff_cc'].values),
        'vig_ff_offset': jnp.array(np.where(vigor_df['trialCookie_weight'].values == 3.0, hm, lm)),
        'subjects': subjects, 'N_S': N_S,
        'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
    }


def make_recovery_model(N_S, N_choice, N_vigor):
    def model(ch_subj, ch_T, ch_dist_H, ch_choice,
              vig_subj, vig_T, vig_R, vig_dist, vig_ff, vig_ff_offset):
        mu_k = numpyro.sample('mu_k', dist.Normal(0.0, 1.0))
        mu_beta = numpyro.sample('mu_beta', dist.Normal(1.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(2.0, 1.0))
        sigma_k = numpyro.sample('sigma_k', dist.HalfNormal(0.5))
        sigma_beta = numpyro.sample('sigma_beta', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        alpha_raw = numpyro.sample('alpha_raw', dist.Normal(-2.0, 1.0))
        alpha = jnp.clip(jnp.exp(alpha_raw), 0.001, 50.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.3))

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
        effort_cost = 0.81 * ch_dist_H - 0.16
        delta_eu = 4.0 - k[ch_subj] * effort_cost - beta[ch_subj] * ch_T
        p_H = jax.nn.sigmoid(jnp.clip(delta_eu / tau, -20, 20))
        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)), obs=ch_choice)

        # Vigor
        cd_v = c_death[vig_subj]; k_v = k[vig_subj]
        f_grid = jnp.linspace(0.02, 0.98, 30)
        f_g = f_grid[None, :]

        d_game_v = jnp.where(vig_dist == 1, 5.0, jnp.where(vig_dist == 2, 7.0, 9.0))
        mu_s = jnp.where(vig_dist == 1, 2.460, jnp.where(vig_dist == 2, 3.478, 4.765))
        sd_s = jnp.where(vig_dist == 1, 0.574, jnp.where(vig_dist == 2, 0.896, 1.121))

        remaining = d_game_v * 0.9
        eff_speed = 1.45 * (0.5 + 0.5 * f_g)
        arrival = remaining[:, None] / eff_speed
        z_strike = (arrival - (mu_s[:, None] + 0.6)) / sd_s[:, None]
        p_escape = 0.09 + 0.91 * (1.0 - jax.scipy.stats.norm.cdf(z_strike))
        S_f = (1.0 - vig_T[:, None]) + vig_T[:, None] * p_escape

        logit_f = jnp.log(f_g / (1.0 - f_g))
        vig_req = jnp.where(vig_R == 5.0, 0.9, 0.4)
        effort_f = alpha * logit_f**2 * (vig_req[:, None]**2) * vig_dist[:, None]

        eu_grid = S_f * vig_R[:, None] - (1.0 - S_f) * cd_v[:, None] * (vig_R[:, None] + 5.0) - k_v[:, None] * effort_f
        weights = jax.nn.softmax(eu_grid * 5.0, axis=1)
        f_star = jnp.sum(weights * f_g, axis=1)
        f_pred = f_star - vig_ff_offset

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(f_pred, sigma_v), obs=vig_ff)

    return model


def fit_recovery(data, n_steps=20000, lr=0.0005, seed=42):
    model = make_recovery_model(data['N_S'], data['N_choice'], data['N_vigor'])
    kwargs = {k: data[k] for k in ['ch_subj','ch_T','ch_dist_H','ch_choice',
                                     'vig_subj','vig_T','vig_R','vig_dist','vig_ff','vig_ff_offset']}
    guide = AutoNormal(model)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    best_loss = float('inf'); best_params = None
    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_params = svi.get_params(state)

    pred = Predictive(model, guide=guide, params=best_params, num_samples=200,
                      return_sites=['k', 'beta', 'c_death'])
    samples = pred(random.PRNGKey(seed + 1), **kwargs)
    return (np.array(samples['k']).mean(0),
            np.array(samples['beta']).mean(0),
            np.array(samples['c_death']).mean(0),
            best_loss)


if __name__ == '__main__':
    import time
    t0 = time.time()

    pop_params = {
        'mu_k': np.log(1.47), 'sigma_k': 0.5,
        'mu_beta': np.log(4.27), 'sigma_beta': 0.5,
        'mu_cd': np.log(141.0), 'sigma_cd': 1.0,
    }

    N_SUBJ = 50; N_DS = 3
    all_k_t, all_k_r = [], []
    all_b_t, all_b_r = [], []
    all_c_t, all_c_r = [], []

    for ds in range(N_DS):
        print(f"\n{'='*60}\nDataset {ds+1}/{N_DS}\n{'='*60}")
        df, kt, bt, ct = simulate_dataset(N_SUBJ, pop_params, ds * 100)
        print(f"  Simulated: {len(df)} trials")
        data = prepare_sim(df)
        print(f"  Fitting: {data['N_choice']} choice, {data['N_vigor']} vigor")

        kr, br, cr, loss = fit_recovery(data, n_steps=20000, seed=ds)
        print(f"  Loss: {loss:.1f}")

        rk, _ = pearsonr(np.log(kt), np.log(kr))
        rb, _ = pearsonr(np.log(bt), np.log(br))
        rc, _ = pearsonr(np.log(ct), np.log(cr))
        rkb, _ = pearsonr(np.log(kt), np.log(br))
        rbk, _ = pearsonr(np.log(bt), np.log(kr))
        rkc, _ = pearsonr(np.log(kt), np.log(cr))
        rbc, _ = pearsonr(np.log(bt), np.log(cr))

        print(f"  Recovery: k={rk:.3f}, β={rb:.3f}, cd={rc:.3f}")
        print(f"  Cross: k→β={rkb:.3f}, β→k={rbk:.3f}, k→cd={rkc:.3f}, β→cd={rbc:.3f}")

        all_k_t.extend(kt); all_k_r.extend(kr)
        all_b_t.extend(bt); all_b_r.extend(br)
        all_c_t.extend(ct); all_c_r.extend(cr)

    ak_t, ak_r = np.array(all_k_t), np.array(all_k_r)
    ab_t, ab_r = np.array(all_b_t), np.array(all_b_r)
    ac_t, ac_r = np.array(all_c_t), np.array(all_c_r)

    rk_all, _ = pearsonr(np.log(ak_t), np.log(ak_r))
    rb_all, _ = pearsonr(np.log(ab_t), np.log(ab_r))
    rc_all, _ = pearsonr(np.log(ac_t), np.log(ac_r))
    rkb_all, _ = pearsonr(np.log(ak_t), np.log(ab_r))
    rbk_all, _ = pearsonr(np.log(ab_t), np.log(ak_r))

    print(f"\n{'='*60}")
    print(f"OVERALL ({N_DS} × {N_SUBJ} = {len(ak_t)})")
    print(f"{'='*60}")
    print(f"  k:    r={rk_all:.3f}")
    print(f"  β:    r={rb_all:.3f}")
    print(f"  cd:   r={rc_all:.3f}")
    print(f"  Cross: k→β={rkb_all:.3f}, β→k={rbk_all:.3f}")

    PASS = rk_all > 0.70 and rb_all > 0.70 and rc_all > 0.70
    print(f"\n  {'PASS' if PASS else 'FAIL'}: all r > 0.70?")

    pd.DataFrame({
        'k_true': ak_t, 'k_rec': ak_r,
        'beta_true': ab_t, 'beta_rec': ab_r,
        'cd_true': ac_t, 'cd_rec': ac_r,
    }).to_csv('results/stats/branchB_recovery.csv', index=False)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
