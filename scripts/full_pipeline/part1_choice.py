#!/usr/bin/env python3
"""Part 1: Computational Model of Patch Selection (Steps 1.1–1.7)"""

import sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *

import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from jax import random


def fit_3param(data, n_steps=40000, lr=0.001, seed=42):
    """Step 1.1: Fit 3-param v2 with req·T effort."""
    subjects = sorted(data['subj'].unique())
    si = {s: i for i, s in enumerate(subjects)}
    N_S = len(subjects)

    ch_subj = jnp.array([si[s] for s in data['subj']])
    ch_choice = jnp.array(data['choice'].values)
    p = jnp.array(data['threat'].values)
    effort = jnp.array(data['effort_reqT'].values)
    N = len(data)

    def model(ch_subj, ch_choice, p, effort):
        mu_k = numpyro.sample('mu_k', dist.Normal(0.0, 1.0))
        mu_b = numpyro.sample('mu_b', dist.Normal(0.0, 1.0))
        sig_k = numpyro.sample('sig_k', dist.HalfNormal(0.5))
        sig_b = numpyro.sample('sig_b', dist.HalfNormal(0.5))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
        with numpyro.plate('subj', N_S):
            k_raw = numpyro.sample('k_raw', dist.Normal(0.0, 1.0))
            b_raw = numpyro.sample('b_raw', dist.Normal(0.0, 1.0))
        k = jnp.exp(mu_k + sig_k * k_raw)
        beta = jnp.exp(mu_b + sig_b * b_raw)
        numpyro.deterministic('k', k)
        numpyro.deterministic('beta', beta)
        dv = 4.0 - k[ch_subj] * effort - beta[ch_subj] * p
        logit = jnp.clip(dv / tau, -20, 20)
        pH = jax.nn.sigmoid(logit)
        with numpyro.plate('trials', N):
            numpyro.sample('obs', dist.Bernoulli(probs=jnp.clip(pH, 1e-6, 1-1e-6)), obs=ch_choice)

    kwargs = dict(ch_subj=ch_subj, ch_choice=ch_choice, p=p, effort=effort)
    guide = AutoNormal(model)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kwargs)
    update_fn = jax.jit(svi.update)

    best_loss = float('inf'); best_params = None; best_step = 0
    for i in range(n_steps):
        state, loss = update_fn(state, **kwargs)
        if float(loss) < best_loss:
            best_loss = float(loss); best_params = svi.get_params(state); best_step = i + 1
        if (i + 1) % 10000 == 0:
            print(f"    Step {i+1}: loss={float(loss):.1f} (best={best_loss:.1f} @ {best_step})")

    pred = Predictive(model, guide=guide, params=best_params, num_samples=300,
                      return_sites=['k', 'beta', 'tau_raw'])
    samples = pred(random.PRNGKey(seed + 1), **kwargs)
    k_vals = np.array(samples['k']).mean(0)
    beta_vals = np.array(samples['beta']).mean(0)
    tau_val = float(np.exp(np.array(samples['tau_raw']).mean()))

    # Evaluate
    ch_subj_np = np.array(ch_subj)
    dv = 4.0 - k_vals[ch_subj_np] * np.array(effort) - beta_vals[ch_subj_np] * np.array(p)
    p_heavy = expit(np.clip(dv / tau_val, -20, 20))
    acc = ((p_heavy >= 0.5).astype(int) == np.array(ch_choice)).mean()
    cdf = pd.DataFrame({'subj': ch_subj_np, 'choice': np.array(ch_choice), 'p_H': p_heavy})
    sc = cdf.groupby('subj').agg(o=('choice', 'mean'), p=('p_H', 'mean')).reset_index()
    r_ch, _ = pearsonr(sc['o'], sc['p'])

    n_params = 2 * N_S + 5
    bic = 2 * best_loss + n_params * np.log(N)

    param_df = pd.DataFrame({'subj': subjects, 'k': k_vals, 'beta': beta_vals})
    return {
        'param_df': param_df, 'k': k_vals, 'beta': beta_vals, 'tau': tau_val,
        'subjects': subjects, 'N_S': N_S,
        'accuracy': acc, 'r_choice': r_ch, 'r2_choice': r_ch**2,
        'bic': bic, 'best_loss': best_loss,
    }


def run_part1():
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PART 1: COMPUTATIONAL MODEL OF PATCH SELECTION")
    print("=" * 70)

    data = load_choice_data()
    N = len(data); N_S = data['subj'].nunique()
    print(f"\nN={N_S} subjects, {N} trials")

    # ── Step 1.1: Fit ──
    print("\n--- Step 1.1: Fitting 3-param v2 (req·T effort) ---")
    result = fit_3param(data)
    param_df = result['param_df']
    save_df(param_df, 'part1_params')

    print(f"\n  Accuracy: {result['accuracy']:.3f}")
    print(f"  Per-subject choice r²: {result['r2_choice']:.3f}")
    print(f"  BIC: {result['bic']:.0f}")
    print(f"  τ: {result['tau']:.3f}")

    if result['r2_choice'] < 0.90:
        return {'STOP': f"Choice r²={result['r2_choice']:.3f} < 0.90"}

    # ── Step 1.2: Orthogonality ──
    print("\n--- Step 1.2: Orthogonality ---")
    param_df['log_k'] = np.log(param_df['k'])
    param_df['log_beta'] = np.log(param_df['beta'])
    r_kb, p_kb = pearsonr(param_df['log_k'], param_df['log_beta'])
    print(f"  r(k, β) = {r_kb:.3f}, p = {p_kb:.4f}")

    # ── Step 1.3: Recovery (3×50 for speed) ──
    print("\n--- Step 1.3: Parameter recovery (3×50) ---")
    from scipy.special import expit as _expit
    all_kt, all_kr, all_bt, all_br = [], [], [], []
    pop = {'mu_k': np.log(np.median(result['k'])), 'sig_k': 0.5,
           'mu_b': np.log(np.median(result['beta'])), 'sig_b': 0.5,
           'tau': result['tau']}
    for ds in range(3):
        np.random.seed(ds * 100)
        n_s = 50
        kt = np.exp(pop['mu_k'] + pop['sig_k'] * np.random.randn(n_s))
        bt = np.exp(pop['mu_b'] + pop['sig_b'] * np.random.randn(n_s))
        # Simulate
        recs = []
        for s in range(n_s):
            for _ in range(3):
                for ti, T in enumerate([0.1, 0.5, 0.9]):
                    for di, D in enumerate([1, 2, 3]):
                        req_h = [0.6, 0.8, 1.0][di]; T_h = [5, 7, 9][di]
                        eff = req_h * T_h - 0.4 * 5
                        dv = 4 - kt[s] * eff - bt[s] * T
                        ch = int(np.random.random() < _expit(dv / pop['tau']))
                        recs.append({'subj': s, 'threat': T, 'distance_H': D + 1,
                                    'effort_H': req_h, 'effort_reqT': eff, 'choice': ch})
                for _ in range(6):
                    T = [0.1, 0.5, 0.9][np.random.randint(3)]
                    D = np.random.randint(1, 4); di = D - 1
                    req_h = [0.6, 0.8, 1.0][di]; T_h = [5, 7, 9][di]
                    eff = req_h * T_h - 0.4 * 5
                    dv = 4 - kt[s] * eff - bt[s] * T
                    ch = int(np.random.random() < _expit(dv / pop['tau']))
                    recs.append({'subj': s, 'threat': T, 'distance_H': D,
                                'effort_H': req_h, 'effort_reqT': eff, 'choice': ch})
        sim_df = pd.DataFrame(recs)
        sim_df['T_round'] = sim_df['threat'].round(1)
        # Fit
        si = {s: i for i, s in enumerate(sorted(sim_df['subj'].unique()))}
        cs = jnp.array([si[s] for s in sim_df['subj']])
        cc = jnp.array(sim_df['choice'].values)
        pp = jnp.array(sim_df['threat'].values)
        ee = jnp.array(sim_df['effort_reqT'].values)
        N_sim = len(sim_df)

        def sim_model(cs, cc, pp, ee):
            mk = numpyro.sample('mk', dist.Normal(0, 1))
            mb = numpyro.sample('mb', dist.Normal(0, 1))
            sk = numpyro.sample('sk', dist.HalfNormal(0.5))
            sb = numpyro.sample('sb', dist.HalfNormal(0.5))
            tr = numpyro.sample('tr', dist.Normal(0, 1))
            tau = jnp.clip(jnp.exp(tr), 0.01, 20)
            with numpyro.plate('s', n_s):
                kr = numpyro.sample('kr', dist.Normal(0, 1))
                br = numpyro.sample('br', dist.Normal(0, 1))
            k = jnp.exp(mk + sk * kr); b = jnp.exp(mb + sb * br)
            numpyro.deterministic('k', k); numpyro.deterministic('b', b)
            dv = 4 - k[cs] * ee - b[cs] * pp
            pH = jax.nn.sigmoid(jnp.clip(dv / tau, -20, 20))
            with numpyro.plate('t', N_sim):
                numpyro.sample('o', dist.Bernoulli(probs=jnp.clip(pH, 1e-6, 1-1e-6)), obs=cc)

        g = AutoNormal(sim_model)
        svi = SVI(sim_model, g, numpyro.optim.ClippedAdam(step_size=0.001, clip_norm=10), Trace_ELBO())
        st = svi.init(random.PRNGKey(ds), cs=cs, cc=cc, pp=pp, ee=ee)
        uf = jax.jit(svi.update)
        bl = float('inf'); bp = None
        for i in range(20000):
            st, loss = uf(st, cs=cs, cc=cc, pp=pp, ee=ee)
            if float(loss) < bl: bl = float(loss); bp = svi.get_params(st)
        pred = Predictive(sim_model, guide=g, params=bp, num_samples=200, return_sites=['k', 'b'])
        samp = pred(random.PRNGKey(ds + 1), cs=cs, cc=cc, pp=pp, ee=ee)
        kr = np.array(samp['k']).mean(0); br = np.array(samp['b']).mean(0)
        rk, _ = pearsonr(np.log(kt), np.log(kr))
        rb, _ = pearsonr(np.log(bt), np.log(br))
        rkb, _ = pearsonr(np.log(kt), np.log(br))
        rbk, _ = pearsonr(np.log(bt), np.log(kr))
        print(f"  DS{ds+1}: k={rk:.3f}, β={rb:.3f}, cross k→β={rkb:.3f}, β→k={rbk:.3f}")
        all_kt.extend(kt); all_kr.extend(kr); all_bt.extend(bt); all_br.extend(br)

    rk_all, _ = pearsonr(np.log(np.array(all_kt)), np.log(np.array(all_kr)))
    rb_all, _ = pearsonr(np.log(np.array(all_bt)), np.log(np.array(all_br)))
    rkb_all, _ = pearsonr(np.log(np.array(all_kt)), np.log(np.array(all_br)))
    print(f"  Overall: k={rk_all:.3f}, β={rb_all:.3f}, cross={rkb_all:.3f}")

    if rk_all < 0.70 or rb_all < 0.70:
        return {'STOP': f"Recovery failed: k={rk_all:.3f}, β={rb_all:.3f}"}

    # ── Step 1.4: Triple dissociation ──
    print("\n--- Step 1.4: Triple dissociation ---")
    # Need cd — load from existing 3-param v2 fit (cd comes from vigor, not refitted here)
    v2_params = pd.read_csv("results/stats/oc_evc_3param_v2_params.csv")
    v2_params = v2_params[~v2_params['subj'].isin(EXCLUDE)]
    param_df = param_df.merge(v2_params[['subj', 'c_death']], on='subj', how='left')
    param_df['log_cd'] = np.log(param_df['c_death'])
    for c in ['log_k', 'log_beta', 'log_cd']:
        param_df[f'{c}_z'] = (param_df[c] - param_df[c].mean()) / param_df[c].std()
    save_df(param_df, 'part1_params_full')

    # Compute behavioral outcomes
    beh = data.copy()
    # Overcautious rate needs optimal policy — compute simple version
    # Heavy optimal when EV_heavy > EV_light using S=exp(-p*T)
    beh['S_H'] = np.exp(-beh['threat'] * beh['T_H'])
    beh['S_L'] = np.exp(-beh['threat'] * 5.0)
    beh['EV_H'] = beh['S_H'] * 5 - (1 - beh['S_H']) * 5 - beh['effort_reqT']
    beh['EV_L'] = beh['S_L'] * 1 - (1 - beh['S_L']) * 5 - (0.4 * 5)
    beh['optimal'] = (beh['EV_H'] > beh['EV_L']).astype(int)
    beh['overcautious'] = ((beh['choice'] == 0) & (beh['optimal'] == 1)).astype(int)

    subj_oc = beh.groupby('subj').agg(
        overcautious_rate=('overcautious', 'mean'),
        heavy_rate=('choice', 'mean'),
    ).reset_index()
    # Threat slope
    slopes = []
    for s, sdf in beh.groupby('subj'):
        if len(sdf) > 10:
            from scipy.stats import linregress
            sl, _, _, _, _ = linregress(sdf['threat'], sdf['choice'])
            slopes.append({'subj': s, 'threat_slope': sl})
    slopes_df = pd.DataFrame(slopes)

    # frac_full from existing data
    beh_rich = load_all_trials()
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
    subj_ff = beh_rich.dropna(subset=['frac_full']).groupby('subj')['frac_full'].mean().reset_index()

    merged = param_df.merge(subj_oc, on='subj').merge(slopes_df, on='subj').merge(subj_ff, on='subj')

    for label, dv, expected in [
        ('A: Overcaution', 'overcautious_rate', 'k'),
        ('B: Threat slope', 'threat_slope', 'β'),
        ('C: Frac full', 'frac_full', 'cd'),
    ]:
        X = sm.add_constant(merged[['log_k_z', 'log_beta_z', 'log_cd_z']].values)
        y = merged[dv].values
        ols = sm.OLS(y, X).fit()
        print(f"\n  Regression {label} (R²={ols.rsquared:.3f}):")
        for i, nm in enumerate(['Intercept', 'k_z', 'β_z', 'cd_z']):
            sig = "***" if ols.pvalues[i] < .001 else ("**" if ols.pvalues[i] < .01 else ("*" if ols.pvalues[i] < .05 else ""))
            print(f"    {nm:<10}: β={ols.params[i]:>7.3f}, t={ols.tvalues[i]:>6.3f}, p={ols.pvalues[i]:.4f} {sig}")

    # ── Step 1.5: Integration test ──
    print("\n--- Step 1.5: Choice integration test ---")
    ch = data.merge(param_df[['subj', 'log_k_z', 'log_beta_z']], on='subj')
    m15 = smf.mixedlm("choice ~ log_k_z * log_beta_z + threat + distance_H + trial_number + current_score",
                       ch, groups=ch["subj"]).fit(reml=False)
    print_table("P(heavy) ~ k_z * β_z + threat + distance + trial + score + (1|subj)", m15)

    # ── Step 1.6: PPC ──
    print("\n--- Step 1.6: Posterior predictive check ---")
    beh_pred = data.copy()
    si_map = {s: i for i, s in enumerate(result['subjects'])}
    beh_pred['p_pred'] = expit(np.clip(
        (4.0 - result['k'][[si_map[s] for s in beh_pred['subj']]] * beh_pred['effort_reqT'].values
         - result['beta'][[si_map[s] for s in beh_pred['subj']]] * beh_pred['threat'].values) / result['tau'],
        -20, 20))
    print(f"\n  {'':>12} {'D=1':>16} {'D=2':>16} {'D=3':>16}")
    for T in [0.1, 0.5, 0.9]:
        row = f"  T={T:.0%}  "
        for D in [1, 2, 3]:
            sub = beh_pred[(beh_pred['T_round'] == T) & (beh_pred['distance_H'] == D)]
            obs = sub['choice'].mean(); pred_v = sub['p_pred'].mean()
            flag = " *" if abs(pred_v - obs) > 0.10 else ""
            row += f"  {obs:.3f}/{pred_v:.3f}{flag}"
        print(row)

    # ── Step 1.7: M3 robustness ──
    print("\n--- Step 1.7: M3 robustness check ---")
    # M3 already fitted in model_comparison_v2
    m3_comp = pd.read_csv("results/stats/model_comparison_v2/model_comparison.csv")
    m3_bic = m3_comp[m3_comp['Model'] == 'M3']['BIC_approx'].values[0]
    m2_bic = m3_comp[m3_comp['Model'] == 'M2']['BIC_approx'].values[0]
    print(f"  M3 BIC: {m3_bic:.0f}")
    print(f"  M2 (≈3-param) BIC: {m2_bic:.0f}")
    print(f"  ΔBIC (M3 - M2): {m3_bic - m2_bic:.0f}")
    print(f"  M3 wins by {m2_bic - m3_bic:.0f} BIC points")

    elapsed = (time.time() - t0) / 60
    print(f"\n  Part 1 complete ({elapsed:.1f} min)")

    return {
        'param_df': param_df, 'result': result, 'merged': merged,
        'data': data, 'beh_rich': beh_rich,
    }


if __name__ == '__main__':
    p1 = run_part1()
    if p1.get('STOP'):
        print(f"\n*** STOPPED: {p1['STOP']} ***")
