#!/usr/bin/env python3
"""
Integrated model: M3 with individual κ + cd from vigor.

Choice: ΔV = 5·exp(-κ_i·p·T_H) - exp(-κ_i·p·T_L) - λ_i·effort(D)
        P(heavy) = sigmoid(ΔV / τ)

Vigor:  EU(f) = S(f,T,D)×R - (1-S(f,T,D))×cd_i×(R+C) - ce_vigor·g(f)·D
        f* = soft_argmax

Three per-subject: λ (effort), κ (survival sensitivity), cd (capture aversion)
Population: τ (temperature), ce_vigor, σ_v
"""

import sys, time, warnings
warnings.filterwarnings("ignore")

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
from scipy.special import expit
from pathlib import Path

jax.config.update('jax_enable_x64', True)

DATA_DIR = Path("data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("results/stats/model_comparison_v2")


def load_data():
    beh = pd.read_csv(DATA_DIR / "behavior.csv")
    beh_rich = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)

    exclude = [154, 197, 208]
    beh = beh[~beh['subj'].isin(exclude)].copy()
    beh_rich = beh_rich[~beh_rich['subj'].isin(exclude)].copy()

    beh['T_H'] = beh['distance_H'].map({1: 5.0, 2: 7.0, 3: 9.0})
    beh['effort_reqT'] = beh['effort_H'] * beh['T_H'] - 0.4 * 5.0

    # Vigor: compute frac_full for all trials
    beh_rich['actual_req'] = np.where(beh_rich['trialCookie_weight'] == 3.0, 0.9, 0.4)
    beh_rich['actual_dist'] = beh_rich['startDistance'].map({5: 1, 7: 2, 9: 3})
    beh_rich['actual_R'] = np.where(beh_rich['trialCookie_weight'] == 3.0, 5.0, 1.0)
    beh_rich['is_heavy'] = (beh_rich['trialCookie_weight'] == 3.0).astype(int)
    beh_rich['T_H_vig'] = beh_rich['actual_dist'].map({1: 5.0, 2: 7.0, 3: 9.0})

    ff_list = []
    for _, row in beh_rich.iterrows():
        try:
            pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
            ipis = np.diff(pt); ipis = ipis[ipis > 0.01]
            if len(ipis) < 5: ff_list.append(np.nan); continue
            rates = (1.0 / ipis) / row['calibrationMax']
            ff_list.append(np.mean(rates >= row['actual_req']))
        except: ff_list.append(np.nan)
    beh_rich['frac_full'] = ff_list

    vigor_df = beh_rich.dropna(subset=['frac_full']).copy()
    choice_vigor = vigor_df[vigor_df['type'] == 1]
    hm = choice_vigor[choice_vigor['is_heavy'] == 1]['frac_full'].mean()
    lm = choice_vigor[choice_vigor['is_heavy'] == 0]['frac_full'].mean()
    vigor_df['ff_cc'] = vigor_df['frac_full'] - np.where(vigor_df['is_heavy'] == 1, hm, lm)

    subjects = sorted(set(beh['subj'].unique()) & set(vigor_df['subj'].unique()))
    si = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)
    beh = beh[beh['subj'].isin(subjects)]
    vigor_df = vigor_df[vigor_df['subj'].isin(subjects)]

    return {
        'ch_subj': jnp.array([si[s] for s in beh['subj']]),
        'ch_choice': jnp.array(beh['choice'].values),
        'p': jnp.array(beh['threat'].values),
        'T_H': jnp.array(beh['T_H'].values),
        'effort': jnp.array(beh['effort_reqT'].values),
        'vig_subj': jnp.array([si[s] for s in vigor_df['subj']]),
        'vig_T': jnp.array(vigor_df['threat'].values),
        'vig_R': jnp.array(vigor_df['actual_R'].values),
        'vig_dist': jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64),
        'vig_T_dur': jnp.array(vigor_df['T_H_vig'].values),
        'vig_req': jnp.array(vigor_df['actual_req'].values),
        'vig_ff': jnp.array(vigor_df['ff_cc'].values),
        'vig_ff_offset': jnp.array(np.where(vigor_df['is_heavy'].values == 1, hm, lm)),
        'subjects': subjects, 'N_S': N_S,
        'N_choice': len(beh), 'N_vigor': len(vigor_df),
        'beh': beh, 'hm': hm, 'lm': lm,
    }


def make_integrated_model(N_S, N_choice, N_vigor):
    def model(ch_subj, ch_choice, p, T_H, effort,
              vig_subj, vig_T, vig_R, vig_dist, vig_T_dur, vig_req,
              vig_ff, vig_ff_offset):

        # Hierarchical priors
        mu_lam = numpyro.sample('mu_lam', dist.Normal(0.0, 1.0))
        sig_lam = numpyro.sample('sig_lam', dist.HalfNormal(0.5))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-2.0, 1.0))
        sig_kap = numpyro.sample('sig_kap', dist.HalfNormal(0.5))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(2.0, 1.0))
        sig_cd = numpyro.sample('sig_cd', dist.HalfNormal(0.5))

        # Population temperature
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)

        # Vigor effort cost (population)
        ce_vig_raw = numpyro.sample('ce_vig_raw', dist.Normal(-2.0, 1.0))
        ce_vig = numpyro.deterministic('ce_vig', jnp.clip(jnp.exp(ce_vig_raw), 0.001, 50.0))
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.3))

        with numpyro.plate('subjects', N_S):
            lam_raw = numpyro.sample('lam_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))

        lam = jnp.exp(mu_lam + sig_lam * lam_raw)
        kap = jnp.exp(mu_kap + sig_kap * kap_raw)
        cd = jnp.exp(mu_cd + sig_cd * cd_raw)
        numpyro.deterministic('lam', lam)
        numpyro.deterministic('kap', kap)
        numpyro.deterministic('cd', cd)

        # ═══════════════════════════════════════════════════
        # CHOICE: survival function with individual κ
        # ═══════════════════════════════════════════════════
        T_L = 5.0
        S_H = jnp.exp(-kap[ch_subj] * p * T_H)
        S_L = jnp.exp(-kap[ch_subj] * p * T_L)
        dv = S_H * 5.0 - S_L * 1.0 - lam[ch_subj] * effort
        logit = jnp.clip(dv / tau, -20, 20)
        p_heavy = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                dist.Bernoulli(probs=jnp.clip(p_heavy, 1e-6, 1-1e-6)),
                obs=ch_choice)

        # ═══════════════════════════════════════════════════
        # VIGOR: frac_full with simple survival + log-odds cost
        # ═══════════════════════════════════════════════════
        cd_v = cd[vig_subj]
        f_grid = jnp.linspace(0.02, 0.98, 30)
        f_g = f_grid[None, :]

        # Simple survival for vigor: S(f) = exp(-p·T·(1-f))
        # Higher f → lower effective exposure → higher survival
        S_f = jnp.exp(-vig_T[:, None] * vig_T_dur[:, None] * (1.0 - f_g) * 0.1)

        logit_f = jnp.log(f_g / (1.0 - f_g))
        effort_f = ce_vig * logit_f**2 * (vig_req[:, None]**2) * vig_dist[:, None]

        eu_grid = (S_f * vig_R[:, None]
                   - (1.0 - S_f) * cd_v[:, None] * (vig_R[:, None] + 5.0)
                   - effort_f)
        weights = jax.nn.softmax(eu_grid * 5.0, axis=1)
        f_star = jnp.sum(weights * f_g, axis=1)
        f_pred = f_star - vig_ff_offset
        numpyro.deterministic('f_pred', f_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor', dist.Normal(f_pred, sigma_v), obs=vig_ff)

    return model


def fit_model(data, n_steps=40000, lr=0.0005, seed=42):
    model = make_integrated_model(data['N_S'], data['N_choice'], data['N_vigor'])
    kwargs = {k: data[k] for k in [
        'ch_subj', 'ch_choice', 'p', 'T_H', 'effort',
        'vig_subj', 'vig_T', 'vig_R', 'vig_dist', 'vig_T_dur', 'vig_req',
        'vig_ff', 'vig_ff_offset']}

    guide = AutoNormal(model)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    best_loss = float('inf'); best_params = None; best_step = 0
    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_params = svi.get_params(state)
            best_step = i + 1
        if (i+1) % 10000 == 0:
            print(f"  Step {i+1}: loss={float(loss):.1f} (best={best_loss:.1f} @ {best_step})")

    print(f"  Best at step {best_step}: loss={best_loss:.1f}")
    return {'params': best_params, 'guide': guide, 'model': model,
            'kwargs': kwargs, 'data': data, 'loss': best_loss, 'best_step': best_step}


def evaluate(fit_result, n_samples=300, seed=44):
    guide = fit_result['guide']; model = fit_result['model']
    params_fit = fit_result['params']; data = fit_result['data']
    kwargs = fit_result['kwargs']

    pred = Predictive(model, guide=guide, params=params_fit, num_samples=n_samples,
                      return_sites=['lam', 'kap', 'cd', 'ce_vig', 'f_pred', 'tau_raw', 'sigma_v'])
    samples = pred(random.PRNGKey(seed), **kwargs)

    lam = np.array(samples['lam']).mean(0)
    kap = np.array(samples['kap']).mean(0)
    cd = np.array(samples['cd']).mean(0)
    tau = float(np.exp(np.array(samples['tau_raw']).mean()))
    ce_vig = float(np.array(samples['ce_vig']).mean())
    sigma_v = float(np.array(samples['sigma_v']).mean())
    f_pred = np.array(samples['f_pred']).mean(0)

    # Choice evaluation
    ch_subj = np.array(data['ch_subj']); p_np = np.array(data['p'])
    T_H = np.array(data['T_H']); effort = np.array(data['effort'])
    choice = np.array(data['ch_choice'])

    S_H = np.exp(-kap[ch_subj] * p_np * T_H)
    S_L = np.exp(-kap[ch_subj] * p_np * 5.0)
    dv = S_H * 5.0 - S_L * 1.0 - lam[ch_subj] * effort
    p_heavy = expit(np.clip(dv / tau, -20, 20))
    acc = ((p_heavy >= 0.5).astype(int) == choice).mean()

    cdf = pd.DataFrame({'subj': ch_subj, 'choice': choice, 'p_H': p_heavy})
    sc = cdf.groupby('subj').agg(o=('choice', 'mean'), p=('p_H', 'mean')).reset_index()
    r_ch, _ = pearsonr(sc['o'], sc['p'])

    # Vigor evaluation
    f_obs = np.array(data['vig_ff'])
    r_vig, _ = pearsonr(f_pred, f_obs)
    vs = np.array(data['vig_subj'])
    vdf = pd.DataFrame({'subj': vs, 'obs': f_obs, 'pred': f_pred})
    sv = vdf.groupby('subj').agg(o=('obs', 'mean'), p=('pred', 'mean')).reset_index()
    r_vig_subj, _ = pearsonr(sv['o'], sv['p'])

    # BIC
    n_params = 3 * data['N_S'] + 9  # 3 per-subj + pop params
    bic = 2 * fit_result['loss'] + n_params * np.log(data['N_choice'] + data['N_vigor'])

    # Correlations
    ll, lk, lcd = np.log(lam), np.log(kap), np.log(cd)
    r_lk, p_lk = pearsonr(ll, lk)
    r_lcd, _ = pearsonr(ll, lcd)
    r_kcd, _ = pearsonr(lk, lcd)

    print(f"\n{'='*60}")
    print(f"INTEGRATED MODEL RESULTS")
    print(f"{'='*60}")
    print(f"Choice accuracy: {acc:.3f}")
    print(f"Per-subject choice r: {r_ch:.3f} (r²={r_ch**2:.3f})")
    print(f"Trial-level vigor r: {r_vig:.3f} (r²={r_vig**2:.3f})")
    print(f"Per-subject vigor r: {r_vig_subj:.3f} (r²={r_vig_subj**2:.3f})")
    print(f"BIC_approx: {bic:.0f}")
    print(f"\nPopulation: τ={tau:.3f}, ce_vig={ce_vig:.4f}, σ_v={sigma_v:.3f}")
    print(f"\nPer-subject:")
    print(f"  λ: median={np.median(lam):.3f}, range=[{lam.min():.3f}, {lam.max():.3f}]")
    print(f"  κ: median={np.median(kap):.4f}, range=[{kap.min():.4f}, {kap.max():.4f}]")
    print(f"  cd: median={np.median(cd):.1f}, range=[{cd.min():.1f}, {cd.max():.1f}]")
    print(f"\nCorrelations:")
    print(f"  λ × κ: r={r_lk:.3f} (p={p_lk:.4f})")
    print(f"  λ × cd: r={r_lcd:.3f}")
    print(f"  κ × cd: r={r_kcd:.3f}")

    # PPC
    beh = data['beh'].copy()
    beh['p_pred'] = p_heavy
    print(f"\nPosterior predictive check:")
    print(f"{'':>12} {'D=1':>16} {'D=2':>16} {'D=3':>16}")
    for T in [0.1, 0.5, 0.9]:
        row = f"T={T:.0%}  "
        for D in [1, 2, 3]:
            sub = beh[(beh['threat'].round(1) == T) & (beh['distance_H'] == D)]
            obs = sub['choice'].mean(); pred_v = sub['p_pred'].mean()
            flag = " *" if abs(pred_v - obs) > 0.10 else ""
            row += f"  {obs:.3f}/{pred_v:.3f}{flag}"
        print(row)

    # Comparison benchmarks
    print(f"\n{'='*60}")
    print(f"COMPARISON TO OTHER MODELS")
    print(f"{'='*60}")
    print(f"              Choice r²  Vigor r²(subj)  BIC_approx")
    print(f"  Integrated:   {r_ch**2:.3f}       {r_vig_subj**2:.3f}       {bic:.0f}")
    print(f"  M3 (fixed κ): 0.898       —              18,502 (choice-only)")
    print(f"  M2 (linear):  0.989       —              20,274 (choice-only)")
    print(f"  3-param v2:   0.981       0.669          — (diff vigor var)")

    param_df = pd.DataFrame({'subj': data['subjects'], 'lam': lam, 'kap': kap, 'cd': cd})
    return {
        'param_df': param_df, 'lam': lam, 'kap': kap, 'cd': cd,
        'tau': tau, 'ce_vig': ce_vig, 'sigma_v': sigma_v,
        'r_choice': r_ch, 'r_vigor': r_vig, 'r_vigor_subj': r_vig_subj,
        'accuracy': acc, 'bic': bic,
        'r_lam_kap': r_lk, 'p_lam_kap': p_lk,
    }


if __name__ == '__main__':
    t0 = time.time()
    data = load_data()
    print(f"N={data['N_S']} subjects, {data['N_choice']} choice, {data['N_vigor']} vigor")

    print("\nFitting integrated model (λ + κ + cd)...")
    fit_result = fit_model(data)
    result = evaluate(fit_result)

    result['param_df'].to_csv(OUT_DIR / 'integrated_params.csv', index=False)
    print(f"\nSaved params to {OUT_DIR / 'integrated_params.csv'}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
