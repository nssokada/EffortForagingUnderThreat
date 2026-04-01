"""
Round 3: Refine V8 (the winner from round 2).

V8 achieved: choice acc=0.755, r²=0.796, vigor r²=0.737
M5 benchmark: choice acc=0.794, r²=0.952, vigor r²=0.496

V8's architecture: ExpS(D/u) + reference-u choice + per-subject baseline + ω,κ per-subject

Refinements to try:
  V8a: V8 with 50k steps + lower lr (better convergence)
  V8b: V8 + κ enters choice too (effort cost difference between cookies)
  V8c: V8 with choice at u=u* (grid search) instead of u=req (test if scale mismatch is solved)
  V8d: V8b + 2x choice weight
  V8e: V8b + higher softmax temp for finer u* resolution
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
from scipy.stats import pearsonr
from scipy.special import expit
from pathlib import Path

EXCLUDE = [154, 197, 208]
DATA_DIR = Path("data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("results/stats/joint_optimal")
C_PENALTY = 5.0


def prepare_data():
    beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)]
    choice_df = beh[beh['type'] == 1].copy()
    vigor_df = pd.read_csv(DATA_DIR / "trial_vigor.csv")
    vigor_df = vigor_df[~vigor_df['subj'].isin(EXCLUDE)].dropna(subset=['median_rate']).copy()
    subjects = sorted(set(choice_df['subj'].unique()) & set(vigor_df['subj'].unique()))
    si = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)
    data = {
        'ch_subj': jnp.array([si[s] for s in choice_df['subj']]),
        'ch_T': jnp.array(choice_df['threat'].values),
        'ch_D_H': jnp.array(choice_df['distance_H'].values, dtype=jnp.float64),
        'ch_D_L': jnp.ones(len(choice_df)),
        'ch_choice': jnp.array(choice_df['choice'].values),
        'vig_subj': jnp.array([si[s] for s in vigor_df['subj']]),
        'vig_T': jnp.array(vigor_df['threat'].values),
        'vig_R': jnp.array(vigor_df['actual_R'].values),
        'vig_req': jnp.array(vigor_df['actual_req'].values),
        'vig_dist': jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64),
        'vig_rate': jnp.array(vigor_df['median_rate'].values),
        'vig_cookie': jnp.array(vigor_df['is_heavy'].values, dtype=jnp.float64),
        'subjects': subjects, 'N_S': N_S,
        'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
    }
    print(f"  {N_S} subjects, {data['N_choice']} choice, {data['N_vigor']} vigor")
    return data


def exp_survival(u, T, D, gamma, hazard):
    T_w = jnp.power(T, gamma)
    return jnp.exp(-hazard * T_w * D / jnp.clip(u, 0.1, None))


def vigor_eu_exp(omega, kappa, T, D, R, req, gamma, hazard, u_grid, temp=20.0):
    u_g = u_grid[None, :]
    S = exp_survival(u_g, T[:, None], D[:, None], gamma, hazard)
    W = (S * R[:, None]
         - (1.0 - S) * omega[:, None] * (R[:, None] + C_PENALTY)
         - kappa[:, None] * (u_g - req[:, None]) ** 2 * D[:, None])
    weights = jax.nn.softmax(W * temp, axis=1)
    u_star = jnp.sum(weights * u_g, axis=1)
    V_star = jnp.sum(weights * W, axis=1)
    return u_star, V_star


KWARGS_KEYS = ['ch_subj', 'ch_T', 'ch_D_H', 'ch_D_L', 'ch_choice',
               'vig_subj', 'vig_T', 'vig_R', 'vig_req', 'vig_dist',
               'vig_rate', 'vig_cookie']


# ============================================================
# V8b: V8 + κ enters choice (effort cost difference)
# ============================================================

def make_v8b(N_S, N_ch, N_vig):
    """V8 but choice includes effort cost difference between cookies.
    V_H = S_H·R_H - (1-S_H)·ω·(R_H+C) - κ·effort_H
    V_L = S_L·R_L - (1-S_L)·ω·(R_L+C) - κ·effort_L

    effort_j = req_j² · D_j (total quadratic cost at reference u=req)
    """
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

        u_grid = jnp.linspace(0.1, 1.5, 30)

        # Choice: W at reference u, INCLUDING effort cost from κ
        S_H = exp_survival(jnp.full(N_ch, 0.9), ch_T, ch_D_H, gamma, hazard)
        S_L = exp_survival(jnp.full(N_ch, 0.4), ch_T, ch_D_L, gamma, hazard)
        # Effort: total pressing cost req²×D (proportional to energy expended)
        effort_H = 0.9 ** 2 * ch_D_H  # = 0.81 * D_H
        effort_L = 0.4 ** 2 * ch_D_L  # = 0.16 * 1 = 0.16
        V_H = S_H * 5.0 - (1.0 - S_H) * omega[ch_subj] * 10.0 - kappa[ch_subj] * effort_H
        V_L = S_L * 1.0 - (1.0 - S_L) * omega[ch_subj] * 6.0 - kappa[ch_subj] * effort_L

        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice', N_ch):
            numpyro.sample('obs_ch', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        # Vigor: full EU optimization
        u_star, _ = vigor_eu_exp(omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
                                 vig_R, vig_req, gamma, hazard, u_grid)
        rate_pred = u_star + baseline[vig_subj] + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)
        with numpyro.plate('vigor', N_vig):
            numpyro.sample('obs_vig', dist.Normal(rate_pred, sigma_v), obs=vig_rate)
    return model


# ============================================================
# V8c: V8b + grid-search choice (test if scale mismatch is solved with ExpS)
# ============================================================

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

        u_grid = jnp.linspace(0.1, 1.5, 30)

        # Choice via GRID SEARCH on W (same as vigor)
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


# ============================================================
# V8d: V8b + 2x choice weight
# ============================================================

def make_v8d(N_S, N_ch, N_vig):
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

        u_grid = jnp.linspace(0.1, 1.5, 30)

        # Choice with effort cost, 2x weight
        S_H = exp_survival(jnp.full(N_ch, 0.9), ch_T, ch_D_H, gamma, hazard)
        S_L = exp_survival(jnp.full(N_ch, 0.4), ch_T, ch_D_L, gamma, hazard)
        effort_H = 0.81 * ch_D_H
        effort_L = 0.16 * ch_D_L
        V_H = S_H * 5.0 - (1.0 - S_H) * omega[ch_subj] * 10.0 - kappa[ch_subj] * effort_H
        V_L = S_L * 1.0 - (1.0 - S_L) * omega[ch_subj] * 6.0 - kappa[ch_subj] * effort_L
        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('ch1', N_ch):
            numpyro.sample('obs_ch1', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)
        with numpyro.plate('ch2', N_ch):
            numpyro.sample('obs_ch2', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        u_star, _ = vigor_eu_exp(omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
                                 vig_R, vig_req, gamma, hazard, u_grid)
        rate_pred = u_star + baseline[vig_subj] + b_cookie * vig_cookie
        numpyro.deterministic('rate_pred', rate_pred)
        with numpyro.plate('vigor', N_vig):
            numpyro.sample('obs_vig', dist.Normal(rate_pred, sigma_v), obs=vig_rate)
    return model


# ============================================================
# Fit + Evaluate
# ============================================================

def fit_model(name, model_fn, data, n_steps=30000, lr=0.001, seed=42):
    kwargs = {k: data[k] for k in KWARGS_KEYS}
    guide = AutoNormal(model_fn)
    opt = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
    svi = SVI(model_fn, guide, opt, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update = jax.jit(svi.update)
    best_loss, best_params = float('inf'), None
    t0 = time.time()
    for i in range(n_steps):
        state, loss = update(state, **kwargs)
        l = float(loss)
        if l < best_loss and not np.isnan(l):
            best_loss = l; best_params = svi.get_params(state)
        if (i+1) % 10000 == 0:
            print(f"    {name} step {i+1}: loss={l:.1f} (best={best_loss:.1f})")
    print(f"    {name} done in {time.time()-t0:.0f}s, best={best_loss:.1f}")
    return {'name': name, 'best_loss': best_loss, 'best_params': best_params,
            'guide': guide, 'model_fn': model_fn, 'kwargs': kwargs}


def evaluate(fit, data, n_samples=300):
    sites = ['omega', 'kappa', 'rate_pred', 'gamma', 'tau_raw', 'hazard', 'b_cookie']
    pred = Predictive(fit['model_fn'], guide=fit['guide'], params=fit['best_params'],
                      num_samples=n_samples, return_sites=sites)
    samples = pred(random.PRNGKey(44), **fit['kwargs'])

    rp = np.array(samples['rate_pred']).mean(0)
    r_vig = pearsonr(rp, np.array(data['vig_rate']))[0]

    omega = np.array(samples['omega']).mean(0)
    kappa = np.array(samples.get('kappa', np.zeros((1, data['N_S'])))).mean(0)
    gamma_v = float(np.array(samples['gamma']).mean())
    hazard_v = float(np.array(samples['hazard']).mean())
    tau_v = float(np.exp(np.array(samples['tau_raw']).mean()))

    ch_s = np.array(data['ch_subj']); ch_T = np.array(data['ch_T'])
    ch_D = np.array(data['ch_D_H']); ch_c = np.array(data['ch_choice'])

    S_H = np.exp(-hazard_v * ch_T**gamma_v * ch_D / 0.9)
    S_L = np.exp(-hazard_v * ch_T**gamma_v * 1.0 / 0.4)
    V_H = S_H * 5.0 - (1-S_H) * omega[ch_s] * 10.0 - kappa[ch_s] * 0.81 * ch_D
    V_L = S_L * 1.0 - (1-S_L) * omega[ch_s] * 6.0 - kappa[ch_s] * 0.16
    p_H = expit(np.clip((V_H - V_L) / tau_v, -20, 20))
    acc = ((p_H >= 0.5).astype(int) == ch_c).mean()
    ch_df = pd.DataFrame({'s': ch_s, 'c': ch_c, 'p': p_H})
    sc = ch_df.groupby('s').agg(o=('c','mean'), p=('p','mean'))
    try: r_ch = pearsonr(sc['o'], sc['p'])[0]
    except: r_ch = np.nan

    return {'acc': acc, 'ch_r2': r_ch**2 if not np.isnan(r_ch) else np.nan,
            'vig_r2': r_vig**2, 'omega': omega, 'kappa': kappa,
            'gamma': gamma_v, 'hazard': hazard_v, 'tau': tau_v}


# ============================================================
# Main
# ============================================================

MODELS = [
    ('V8b', make_v8b, 3, 'V8 + κ in choice', 30000, 0.001),
    ('V8c', make_v8c, 3, 'V8 + grid-search choice', 30000, 0.001),
    ('V8d', make_v8d, 3, 'V8b + 2x choice', 30000, 0.001),
    ('V8b_50k', make_v8b, 3, 'V8b 50k steps', 50000, 0.0005),
]


if __name__ == '__main__':
    t0 = time.time()
    print("=" * 70)
    print("ROUND 3: Refining V8 architecture")
    print("=" * 70)
    data = prepare_data()
    N_S = data['N_S']
    results = []

    for name, make_fn, nps, desc, steps, lr in MODELS:
        print(f"\n{'='*50}\n--- {name}: {desc} ---\n{'='*50}")
        model_fn = make_fn(N_S, data['N_choice'], data['N_vigor'])
        fit = fit_model(name, model_fn, data, n_steps=steps, lr=lr)
        if fit['best_params'] is None:
            print(f"  FAILED"); continue
        m = evaluate(fit, data)
        n_p = 3*N_S + 10
        n_obs = data['N_choice'] + data['N_vigor']
        bic = 2*fit['best_loss'] + n_p*np.log(n_obs)
        results.append({'Model': name, 'Desc': desc, 'ELBO': -fit['best_loss'],
                        'BIC': bic, 'acc': m['acc'], 'ch_r2': m['ch_r2'],
                        'vig_r2': m['vig_r2']})
        print(f"  ELBO={-fit['best_loss']:.1f} BIC={bic:.0f}")
        print(f"  Choice: acc={m['acc']:.3f} r²={m['ch_r2']:.3f}")
        print(f"  Vigor: r²={m['vig_r2']:.3f}")
        print(f"  γ={m['gamma']:.2f} h={m['hazard']:.3f} τ={m['tau']:.2f}")
        print(f"  ω: mean={m['omega'].mean():.2f} SD={m['omega'].std():.2f}")
        print(f"  κ: mean={m['kappa'].mean():.3f} SD={m['kappa'].std():.3f}")
        # ω-κ correlation
        r_ok, p_ok = pearsonr(m['omega'], m['kappa'])
        print(f"  ω-κ r={r_ok:.3f} (p={p_ok:.4f})")

    if results:
        df = pd.DataFrame(results)
        best = df['BIC'].min(); df['dBIC'] = df['BIC'] - best
        print("\n" + "="*70 + "\nROUND 3 COMPARISON\n" + "="*70)
        print(df.to_string(index=False))
        print("\nBenchmarks: V8=acc 0.755 ch_r² 0.796 vig_r² 0.737 | M5=acc 0.794 ch_r² 0.952 vig_r² 0.496")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "joint_round3.csv", index=False)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
