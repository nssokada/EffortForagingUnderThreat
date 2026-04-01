"""
Joint Optimal Control — Final Version

KEY CHANGES from V8c:
1. Anticipatory epoch vigor (strategic pressing, before encounter)
2. Linear effort cost: κ·u·D (not quadratic deviation from req)
3. NO free baseline — κ must explain pressing level directly
4. Exponential S(u, T, D) = exp(-h·T^γ·D/u)

W(u) = S(u)·R - (1-S(u))·ω·(R+C) - κ·u·D

Per-subject: ω (capture cost), κ (effort cost / motor intensity)
Both enter BOTH choice and vigor through the same W function.

Variants:
  F1: Basic (ω, κ per subject, no baseline)
  F2: F1 + cookie intercept (small residual cookie offset)
  F3: F1 with quadratic cost κ·u²·D for comparison
"""

import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from jax import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit
from pathlib import Path

EXCLUDE = [154, 197, 208]; C_PEN = 5.0
DATA_DIR = Path("data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("results/stats/joint_optimal")

KWARGS_KEYS = ['cs','cT','cDH','cDL','cc',
               'vs','vT','vR','vq','vD','vr','vc']


def prepare_data():
    """Load choice + ANTICIPATORY EPOCH vigor."""
    # Choice data
    beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)]
    cdf = beh[beh['type'] == 1].copy()

    # Anticipatory vigor from precomputed metrics
    metrics = pd.read_csv("results/stats/vigor_analysis/vigor_metrics.csv")
    metrics = metrics[~metrics['subj'].isin(EXCLUDE)]
    antic = metrics[metrics['epoch'] == 'anticipatory'].dropna(subset=['norm_rate']).copy()

    # Add distance_H for choice trials
    antic = antic.merge(beh[['subj','trial','distance_H','choice']].drop_duplicates(),
                        on=['subj','trial'], how='left')

    # Vigor data: all trials with anticipatory vigor
    vdf = antic.copy()
    # Map cookie to R, req
    vdf['actual_R'] = np.where(vdf['cookie'] == 1, 5.0, 1.0)
    vdf['actual_req'] = np.where(vdf['cookie'] == 1, 0.9, 0.4)

    subjects = sorted(set(cdf['subj'].unique()) & set(vdf['subj'].unique()))
    si = {s: i for i, s in enumerate(subjects)}
    NS = len(subjects)
    NC = len(cdf); NV = len(vdf)

    data = {
        'cs': jnp.array([si[s] for s in cdf['subj']]),
        'cT': jnp.array(cdf['threat'].values),
        'cDH': jnp.array(cdf['distance_H'].values, dtype=jnp.float64),
        'cDL': jnp.ones(NC),
        'cc': jnp.array(cdf['choice'].values),
        'vs': jnp.array([si[s] for s in vdf['subj']]),
        'vT': jnp.array(vdf['T_round'].values),
        'vR': jnp.array(vdf['actual_R'].values),
        'vq': jnp.array(vdf['actual_req'].values),
        'vD': jnp.array(vdf['distance'].values, dtype=jnp.float64),
        'vr': jnp.array(vdf['norm_rate'].values),
        'vc': jnp.array(vdf['cookie'].values, dtype=jnp.float64),
        'subjects': subjects, 'N_S': NS, 'N_choice': NC, 'N_vigor': NV,
    }
    print(f"  {NS} subjects, {NC} choice, {NV} anticipatory vigor trials")
    print(f"  Vigor rate: mean={float(data['vr'].mean()):.4f}, SD={float(data['vr'].std()):.4f}")
    return data


# ============================================================
# EU with LINEAR effort cost: κ·u·D
# ============================================================

def eu_linear(om, kap, T, D, R, req, g, h, ug):
    """W(u) = S(u)·R - (1-S(u))·ω·(R+C) - κ·u·D"""
    u = ug[None, :]
    S = jnp.exp(-h * jnp.power(T[:, None], g) * D[:, None] / jnp.clip(u, 0.1, None))
    W = (S * R[:, None]
         - (1.0 - S) * om[:, None] * (R[:, None] + C_PEN)
         - kap[:, None] * u * D[:, None])
    w = jax.nn.softmax(W * 20.0, axis=1)
    return jnp.sum(w * u, 1), jnp.sum(w * W, 1)


# ============================================================
# EU with QUADRATIC total cost: κ·u²·D
# ============================================================

def eu_quadtotal(om, kap, T, D, R, req, g, h, ug):
    """W(u) = S(u)·R - (1-S(u))·ω·(R+C) - κ·u²·D"""
    u = ug[None, :]
    S = jnp.exp(-h * jnp.power(T[:, None], g) * D[:, None] / jnp.clip(u, 0.1, None))
    W = (S * R[:, None]
         - (1.0 - S) * om[:, None] * (R[:, None] + C_PEN)
         - kap[:, None] * u ** 2 * D[:, None])
    w = jax.nn.softmax(W * 20.0, axis=1)
    return jnp.sum(w * u, 1), jnp.sum(w * W, 1)


# ============================================================
# F1: Linear cost, no baseline
# ============================================================

def make_f1(NS, NC, NV):
    def model(cs, cT, cDH, cDL, cc, vs, vT, vR, vq, vD, vr, vc):
        gr = numpyro.sample('gr', dist.Normal(0, 0.5))
        g = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gr), 0.1, 3.0))
        hr = numpyro.sample('hr', dist.Normal(-1, 1))
        h = numpyro.deterministic('hazard', jnp.exp(hr))
        tr = numpyro.sample('tr', dist.Normal(0, 1))
        tau = jnp.clip(jnp.exp(tr), 0.01, 50.)
        sv = numpyro.sample('sv', dist.HalfNormal(0.5))

        mo = numpyro.sample('mo', dist.Normal(1, 1))
        so = numpyro.sample('so', dist.HalfNormal(1.))
        mk = numpyro.sample('mk', dist.Normal(-1, 1))
        sk = numpyro.sample('sk', dist.HalfNormal(0.5))

        with numpyro.plate('s', NS):
            or_ = numpyro.sample('or', dist.Normal(0, 1))
            kr_ = numpyro.sample('kr', dist.Normal(0, 1))
        om = jnp.exp(mo + so * or_)
        kap = jnp.exp(mk + sk * kr_)
        numpyro.deterministic('omega', om)
        numpyro.deterministic('kappa', kap)

        ug = jnp.linspace(0.1, 1.5, 40)  # finer grid

        # Choice: grid search both cookies
        _, VH = eu_linear(om[cs], kap[cs], cT, cDH,
                          jnp.full(NC, 5.), jnp.full(NC, .9), g, h, ug)
        _, VL = eu_linear(om[cs], kap[cs], cT, cDL,
                          jnp.full(NC, 1.), jnp.full(NC, .4), g, h, ug)
        pH = jax.nn.sigmoid(jnp.clip((VH - VL) / tau, -20, 20))
        with numpyro.plate('c', NC):
            numpyro.sample('oc', dist.Bernoulli(probs=jnp.clip(pH, 1e-6, 1-1e-6)), obs=cc)

        # Vigor: u* from same W, no baseline
        us, _ = eu_linear(om[vs], kap[vs], vT, vD, vR, vq, g, h, ug)
        numpyro.deterministic('rp', us)
        with numpyro.plate('v', NV):
            numpyro.sample('ov', dist.Normal(us, sv), obs=vr)
    return model


# ============================================================
# F2: F1 + cookie intercept
# ============================================================

def make_f2(NS, NC, NV):
    def model(cs, cT, cDH, cDL, cc, vs, vT, vR, vq, vD, vr, vc):
        gr = numpyro.sample('gr', dist.Normal(0, 0.5))
        g = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gr), 0.1, 3.0))
        hr = numpyro.sample('hr', dist.Normal(-1, 1))
        h = numpyro.deterministic('hazard', jnp.exp(hr))
        tr = numpyro.sample('tr', dist.Normal(0, 1))
        tau = jnp.clip(jnp.exp(tr), 0.01, 50.)
        sv = numpyro.sample('sv', dist.HalfNormal(0.5))
        bc = numpyro.sample('bc', dist.Normal(0, 0.5))

        mo = numpyro.sample('mo', dist.Normal(1, 1))
        so = numpyro.sample('so', dist.HalfNormal(1.))
        mk = numpyro.sample('mk', dist.Normal(-1, 1))
        sk = numpyro.sample('sk', dist.HalfNormal(0.5))

        with numpyro.plate('s', NS):
            or_ = numpyro.sample('or', dist.Normal(0, 1))
            kr_ = numpyro.sample('kr', dist.Normal(0, 1))
        om = jnp.exp(mo + so * or_)
        kap = jnp.exp(mk + sk * kr_)
        numpyro.deterministic('omega', om)
        numpyro.deterministic('kappa', kap)

        ug = jnp.linspace(0.1, 1.5, 40)

        _, VH = eu_linear(om[cs], kap[cs], cT, cDH,
                          jnp.full(NC, 5.), jnp.full(NC, .9), g, h, ug)
        _, VL = eu_linear(om[cs], kap[cs], cT, cDL,
                          jnp.full(NC, 1.), jnp.full(NC, .4), g, h, ug)
        pH = jax.nn.sigmoid(jnp.clip((VH - VL) / tau, -20, 20))
        with numpyro.plate('c', NC):
            numpyro.sample('oc', dist.Bernoulli(probs=jnp.clip(pH, 1e-6, 1-1e-6)), obs=cc)

        us, _ = eu_linear(om[vs], kap[vs], vT, vD, vR, vq, g, h, ug)
        rp = us + bc * vc
        numpyro.deterministic('rp', rp)
        with numpyro.plate('v', NV):
            numpyro.sample('ov', dist.Normal(rp, sv), obs=vr)
    return model


# ============================================================
# F3: Quadratic total cost (κ·u²·D), no baseline
# ============================================================

def make_f3(NS, NC, NV):
    def model(cs, cT, cDH, cDL, cc, vs, vT, vR, vq, vD, vr, vc):
        gr = numpyro.sample('gr', dist.Normal(0, 0.5))
        g = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gr), 0.1, 3.0))
        hr = numpyro.sample('hr', dist.Normal(-1, 1))
        h = numpyro.deterministic('hazard', jnp.exp(hr))
        tr = numpyro.sample('tr', dist.Normal(0, 1))
        tau = jnp.clip(jnp.exp(tr), 0.01, 50.)
        sv = numpyro.sample('sv', dist.HalfNormal(0.5))
        bc = numpyro.sample('bc', dist.Normal(0, 0.5))

        mo = numpyro.sample('mo', dist.Normal(1, 1))
        so = numpyro.sample('so', dist.HalfNormal(1.))
        mk = numpyro.sample('mk', dist.Normal(-1, 1))
        sk = numpyro.sample('sk', dist.HalfNormal(0.5))

        with numpyro.plate('s', NS):
            or_ = numpyro.sample('or', dist.Normal(0, 1))
            kr_ = numpyro.sample('kr', dist.Normal(0, 1))
        om = jnp.exp(mo + so * or_)
        kap = jnp.exp(mk + sk * kr_)
        numpyro.deterministic('omega', om)
        numpyro.deterministic('kappa', kap)

        ug = jnp.linspace(0.1, 1.5, 40)

        _, VH = eu_quadtotal(om[cs], kap[cs], cT, cDH,
                             jnp.full(NC, 5.), jnp.full(NC, .9), g, h, ug)
        _, VL = eu_quadtotal(om[cs], kap[cs], cT, cDL,
                             jnp.full(NC, 1.), jnp.full(NC, .4), g, h, ug)
        pH = jax.nn.sigmoid(jnp.clip((VH - VL) / tau, -20, 20))
        with numpyro.plate('c', NC):
            numpyro.sample('oc', dist.Bernoulli(probs=jnp.clip(pH, 1e-6, 1-1e-6)), obs=cc)

        us, _ = eu_quadtotal(om[vs], kap[vs], vT, vD, vR, vq, g, h, ug)
        rp = us + bc * vc
        numpyro.deterministic('rp', rp)
        with numpyro.plate('v', NV):
            numpyro.sample('ov', dist.Normal(rp, sv), obs=vr)
    return model


# ============================================================
# Fit + Evaluate
# ============================================================

def fit_model(name, model_fn, data, n_steps=35000, lr=0.001, seed=42):
    kw = {k: data[k] for k in KWARGS_KEYS}
    guide = AutoNormal(model_fn)
    opt = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.)
    svi = SVI(model_fn, guide, opt, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed), **kw)
    upd = jax.jit(svi.update)
    bl, bp = float('inf'), None
    t0 = time.time()
    for i in range(n_steps):
        state, loss = upd(state, **kw)
        l = float(loss)
        if l < bl and not np.isnan(l): bl = l; bp = svi.get_params(state)
        if (i+1) % 10000 == 0: print(f"    {name} step {i+1}: {l:.1f} (best={bl:.1f})")
    print(f"    {name} done in {time.time()-t0:.0f}s, best={bl:.1f}")
    return {'name': name, 'best_loss': bl, 'best_params': bp,
            'guide': guide, 'model_fn': model_fn, 'kwargs': kw}


def evaluate(fit, data, n_samples=300):
    sites = ['omega', 'kappa', 'rp', 'gamma', 'hazard', 'tr', 'bc', 'sv']
    samp = Predictive(fit['model_fn'], guide=fit['guide'], params=fit['best_params'],
                      num_samples=n_samples, return_sites=sites)(
        random.PRNGKey(44), **fit['kwargs'])

    # Vigor r²
    rp = np.array(samp['rp']).mean(0)
    vr = np.array(data['vr'])
    r_vig = pearsonr(rp, vr)[0]

    # Params
    omega = np.array(samp['omega']).mean(0)
    kappa = np.array(samp['kappa']).mean(0)
    gamma_v = float(np.array(samp['gamma']).mean())
    hazard_v = float(np.array(samp['hazard']).mean())
    tau_v = float(np.exp(np.array(samp['tr']).mean()))

    # Choice reconstruction
    cs = np.array(data['cs']); cT = np.array(data['cT'])
    cDH = np.array(data['cDH']); cc = np.array(data['cc'])

    # Reconstruct V_H - V_L analytically for the linear cost at u=reference
    # Use grid search like the model does
    ug = np.linspace(0.1, 1.5, 40)
    VH_all = np.zeros(len(cs)); VL_all = np.zeros(len(cs))
    for idx in range(len(cs)):
        s = cs[idx]; T = cT[idx]; DH = cDH[idx]
        om = omega[s]; kp = kappa[s]
        for R, req, D, store in [(5., .9, DH, VH_all), (1., .4, 1., VL_all)]:
            S = np.exp(-hazard_v * T**gamma_v * D / np.clip(ug, 0.1, None))
            W = S*R - (1-S)*om*(R+C_PEN) - kp*ug*D
            store[idx] = W.max()
    pH = expit(np.clip((VH_all - VL_all) / tau_v, -20, 20))
    acc = ((pH >= 0.5).astype(int) == cc).mean()
    ch_df = pd.DataFrame({'s': cs, 'c': cc, 'p': pH})
    sc = ch_df.groupby('s').agg(o=('c','mean'), p=('p','mean'))
    try: r_ch = pearsonr(sc['o'], sc['p'])[0]
    except: r_ch = np.nan

    # ω-κ independence
    r_ok = pearsonr(omega, kappa)[0]

    # Behavioral correlations
    subjects = data['subjects']
    vdf_rate = pd.DataFrame({'subj': np.array(data['vs']), 'rate': vr})
    sr = vdf_rate.groupby('subj')['rate'].mean()
    rates = np.array([sr.get(i, np.nan) for i in range(len(subjects))])
    choices = np.array([sc.loc[i, 'o'] if i in sc.index else np.nan for i in range(len(subjects))])
    v = ~np.isnan(rates) & ~np.isnan(choices)

    return {
        'acc': acc, 'ch_r2': r_ch**2 if not np.isnan(r_ch) else np.nan,
        'vig_r2': r_vig**2, 'omega': omega, 'kappa': kappa,
        'gamma': gamma_v, 'hazard': hazard_v, 'tau': tau_v,
        'r_ok': r_ok,
        'om_choice': pearsonr(omega[v], choices[v])[0] if v.sum() > 10 else np.nan,
        'om_vigor': pearsonr(omega[v], rates[v])[0] if v.sum() > 10 else np.nan,
        'kap_choice': pearsonr(kappa[v], choices[v])[0] if v.sum() > 10 else np.nan,
        'kap_vigor': pearsonr(kappa[v], rates[v])[0] if v.sum() > 10 else np.nan,
        'sigma_v': float(np.array(samp['sv']).mean()),
    }


# ============================================================
# Main
# ============================================================

MODELS = [
    ('F1', make_f1, 2, 'Linear κ·u·D, no baseline'),
    ('F2', make_f2, 2, 'F1 + cookie intercept'),
    ('F3', make_f3, 2, 'Quadratic κ·u²·D + cookie'),
]

if __name__ == '__main__':
    t0 = time.time()
    print("=" * 70)
    print("FINAL JOINT MODEL: Anticipatory vigor + linear effort cost")
    print("  W(u) = S(u)·R - (1-S(u))·ω·(R+C) - κ·u·D")
    print("  No baseline — κ must explain pressing level")
    print("=" * 70)
    data = prepare_data()
    NS = data['N_S']
    results = []

    for name, make_fn, nps, desc in MODELS:
        print(f"\n{'='*50}\n--- {name}: {desc} ---\n{'='*50}")
        model_fn = make_fn(NS, data['N_choice'], data['N_vigor'])
        fit = fit_model(name, model_fn, data, n_steps=35000)
        if fit['best_params'] is None: print("FAILED"); continue
        m = evaluate(fit, data)
        n_p = 2*NS + 8 + (1 if 'bc' in name or name != 'F1' else 0)
        bic = 2*fit['best_loss'] + n_p*np.log(data['N_choice']+data['N_vigor'])
        results.append({'Model': name, 'Desc': desc, 'ELBO': -fit['best_loss'],
                        'BIC': bic, 'acc': m['acc'], 'ch_r2': m['ch_r2'],
                        'vig_r2': m['vig_r2']})

        print(f"\n  ELBO={-fit['best_loss']:.1f} BIC={bic:.0f}")
        print(f"  Choice: acc={m['acc']:.3f} r²={m['ch_r2']:.3f}")
        print(f"  Vigor:  r²={m['vig_r2']:.3f} (σ_v={m['sigma_v']:.3f})")
        print(f"  γ={m['gamma']:.2f} h={m['hazard']:.3f} τ={m['tau']:.2f}")
        print(f"  ω: mean={m['omega'].mean():.3f} SD={m['omega'].std():.3f}")
        print(f"  κ: mean={m['kappa'].mean():.3f} SD={m['kappa'].std():.3f}")
        print(f"  ω↔κ: r={m['r_ok']:.3f}")
        print(f"\n  DOUBLE DISSOCIATION CHECK:")
        print(f"    ω → P(heavy):  r = {m['om_choice']:+.3f}")
        print(f"    ω → vigor:     r = {m['om_vigor']:+.3f}")
        print(f"    κ → P(heavy):  r = {m['kap_choice']:+.3f}")
        print(f"    κ → vigor:     r = {m['kap_vigor']:+.3f}")

    if results:
        df = pd.DataFrame(results)
        best = df['BIC'].min(); df['dBIC'] = df['BIC'] - best
        print(f"\n{'='*70}\nFINAL COMPARISON\n{'='*70}")
        print(df.to_string(index=False))
        print("\nM5 benchmark: acc=0.794 ch_r²=0.952 vig_r²=0.496")
        print("V8c benchmark: acc=0.742 ch_r²=0.885 vig_r²=0.735 (with baseline)")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "joint_final.csv", index=False)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
