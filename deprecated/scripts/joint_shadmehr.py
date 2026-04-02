"""
Shadmehr/Yoon Reward Rate Joint Model

Core idea: organisms maximize reward RATE, not total fitness.

ρ(u) = [S(u)·R - (1-S(u))·ω·(R+C)] / t(u) - κ·g(u)

where:
  t(u) = D/u + t_iti  (time per trial = travel time + inter-trial interval)
  g(u) = u²            (metabolic cost rate, quadratic in pressing speed)
  S(u,T,D) = exp(-h·T^γ·D/u)  (survival probability)

Pressing faster has DOUBLE benefit: better survival AND less time per trial.
This creates tighter vigor-choice coupling than Bednekoff's total fitness.

Choice: pick cookie with higher reward rate ρ_j(u*_j)
Vigor: u* = argmax_u ρ(u) for the chosen cookie

Per-subject: ω (capture cost), κ (metabolic cost)
Both enter the reward rate computation jointly.

Variants:
  S1: Basic Shadmehr (ω, κ per subject, anticipatory vigor)
  S2: S1 + cookie intercept
  S3: S1 + per-subject reward rate offset (ρ̄_i — subjective task richness)
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

KWARGS = ['cs','cT','cDH','cDL','cc','vs','vT','vR','vq','vD','vr','vc']


def prepare_data():
    beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)]
    cdf = beh[beh['type'] == 1].copy()

    # Anticipatory vigor
    metrics = pd.read_csv("results/stats/vigor_analysis/vigor_metrics.csv")
    metrics = metrics[~metrics['subj'].isin(EXCLUDE)]
    antic = metrics[metrics['epoch'] == 'anticipatory'].dropna(subset=['norm_rate']).copy()
    antic = antic.merge(beh[['subj','trial','distance_H','choice']].drop_duplicates(),
                        on=['subj','trial'], how='left')
    antic['actual_R'] = np.where(antic['cookie'] == 1, 5.0, 1.0)
    antic['actual_req'] = np.where(antic['cookie'] == 1, 0.9, 0.4)

    subjects = sorted(set(cdf['subj'].unique()) & set(antic['subj'].unique()))
    si = {s: i for i, s in enumerate(subjects)}
    NS = len(subjects); NC = len(cdf); NV = len(antic)

    data = {
        'cs': jnp.array([si[s] for s in cdf.subj]),
        'cT': jnp.array(cdf.threat.values),
        'cDH': jnp.array(cdf.distance_H.values, dtype=jnp.float64),
        'cDL': jnp.ones(NC),
        'cc': jnp.array(cdf.choice.values),
        'vs': jnp.array([si[s] for s in antic.subj]),
        'vT': jnp.array(antic.T_round.values),
        'vR': jnp.array(antic.actual_R.values),
        'vq': jnp.array(antic.actual_req.values),
        'vD': jnp.array(antic.distance.values, dtype=jnp.float64),
        'vr': jnp.array(antic.norm_rate.values),
        'vc': jnp.array(antic.cookie.values, dtype=jnp.float64),
        'subjects': subjects, 'N_S': NS, 'N_choice': NC, 'N_vigor': NV,
    }
    print(f"  {NS} subjects, {NC} choice, {NV} anticipatory vigor")
    return data


# ============================================================
# Reward rate optimization
# ============================================================

def reward_rate_grid(om, kap, T, D, R, req, gamma, hazard, t_iti, u_grid):
    """Compute reward rate ρ(u) and find optimal u*.

    ρ(u) = [S(u)·R - (1-S(u))·ω·(R+C)] / (D/u + t_iti) - κ·u²

    The first term is net reward divided by trial time.
    The second term is metabolic cost (quadratic in speed).

    Returns u_star, rho_star
    """
    u = u_grid[None, :]  # (1, n_grid)

    # Survival
    S = jnp.exp(-hazard * jnp.power(T[:, None], gamma) * D[:, None]
                / jnp.clip(u, 0.1, None))

    # Net reward per trial
    net_reward = S * R[:, None] - (1.0 - S) * om[:, None] * (R[:, None] + C_PEN)

    # Time per trial
    trial_time = D[:, None] / jnp.clip(u, 0.1, None) + t_iti

    # Reward rate = net_reward / trial_time - metabolic cost
    rho = net_reward / trial_time - kap[:, None] * u ** 2

    # Soft argmax
    weights = jax.nn.softmax(rho * 30.0, axis=1)
    u_star = jnp.sum(weights * u, axis=1)
    rho_star = jnp.sum(weights * rho, axis=1)

    return u_star, rho_star


# ============================================================
# S1: Basic Shadmehr (ω, κ per subject)
# ============================================================

def make_s1(NS, NC, NV):
    def model(cs, cT, cDH, cDL, cc, vs, vT, vR, vq, vD, vr, vc):
        gr = numpyro.sample('gr', dist.Normal(0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gr), 0.1, 3.0))
        hr = numpyro.sample('hr', dist.Normal(-1, 1))
        hazard = numpyro.deterministic('hazard', jnp.exp(hr))
        tr = numpyro.sample('tr', dist.Normal(0, 1))
        tau = jnp.clip(jnp.exp(tr), 0.01, 50.)
        sv = numpyro.sample('sv', dist.HalfNormal(0.5))
        # Inter-trial interval (seconds, learned from data)
        iti_raw = numpyro.sample('iti_raw', dist.Normal(0, 0.5))
        t_iti = jnp.clip(jnp.exp(iti_raw), 0.1, 10.0)

        mo = numpyro.sample('mo', dist.Normal(1, 1))
        so = numpyro.sample('so', dist.HalfNormal(1.))
        mk = numpyro.sample('mk', dist.Normal(-2, 1))
        sk = numpyro.sample('sk', dist.HalfNormal(0.5))

        with numpyro.plate('s', NS):
            or_ = numpyro.sample('or', dist.Normal(0, 1))
            kr_ = numpyro.sample('kr', dist.Normal(0, 1))
        omega = jnp.exp(mo + so * or_)
        kappa = jnp.exp(mk + sk * kr_)
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)

        ug = jnp.linspace(0.1, 1.5, 40)

        # Choice: compare reward rates
        _, rho_H = reward_rate_grid(omega[cs], kappa[cs], cT, cDH,
                                    jnp.full(NC, 5.), jnp.full(NC, .9),
                                    gamma, hazard, t_iti, ug)
        _, rho_L = reward_rate_grid(omega[cs], kappa[cs], cT, cDL,
                                    jnp.full(NC, 1.), jnp.full(NC, .4),
                                    gamma, hazard, t_iti, ug)
        pH = jax.nn.sigmoid(jnp.clip((rho_H - rho_L) / tau, -20, 20))
        with numpyro.plate('c', NC):
            numpyro.sample('oc', dist.Bernoulli(probs=jnp.clip(pH, 1e-6, 1-1e-6)), obs=cc)

        # Vigor: u* from same reward rate
        u_star, _ = reward_rate_grid(omega[vs], kappa[vs], vT, vD, vR, vq,
                                     gamma, hazard, t_iti, ug)
        numpyro.deterministic('rp', u_star)
        with numpyro.plate('v', NV):
            numpyro.sample('ov', dist.Normal(u_star, sv), obs=vr)
    return model


# ============================================================
# S2: S1 + cookie intercept
# ============================================================

def make_s2(NS, NC, NV):
    def model(cs, cT, cDH, cDL, cc, vs, vT, vR, vq, vD, vr, vc):
        gr = numpyro.sample('gr', dist.Normal(0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gr), 0.1, 3.0))
        hr = numpyro.sample('hr', dist.Normal(-1, 1))
        hazard = numpyro.deterministic('hazard', jnp.exp(hr))
        tr = numpyro.sample('tr', dist.Normal(0, 1))
        tau = jnp.clip(jnp.exp(tr), 0.01, 50.)
        sv = numpyro.sample('sv', dist.HalfNormal(0.5))
        bc = numpyro.sample('bc', dist.Normal(0, 0.5))
        iti_raw = numpyro.sample('iti_raw', dist.Normal(0, 0.5))
        t_iti = jnp.clip(jnp.exp(iti_raw), 0.1, 10.0)

        mo = numpyro.sample('mo', dist.Normal(1, 1))
        so = numpyro.sample('so', dist.HalfNormal(1.))
        mk = numpyro.sample('mk', dist.Normal(-2, 1))
        sk = numpyro.sample('sk', dist.HalfNormal(0.5))

        with numpyro.plate('s', NS):
            or_ = numpyro.sample('or', dist.Normal(0, 1))
            kr_ = numpyro.sample('kr', dist.Normal(0, 1))
        omega = jnp.exp(mo + so * or_)
        kappa = jnp.exp(mk + sk * kr_)
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)

        ug = jnp.linspace(0.1, 1.5, 40)

        _, rho_H = reward_rate_grid(omega[cs], kappa[cs], cT, cDH,
                                    jnp.full(NC, 5.), jnp.full(NC, .9),
                                    gamma, hazard, t_iti, ug)
        _, rho_L = reward_rate_grid(omega[cs], kappa[cs], cT, cDL,
                                    jnp.full(NC, 1.), jnp.full(NC, .4),
                                    gamma, hazard, t_iti, ug)
        pH = jax.nn.sigmoid(jnp.clip((rho_H - rho_L) / tau, -20, 20))
        with numpyro.plate('c', NC):
            numpyro.sample('oc', dist.Bernoulli(probs=jnp.clip(pH, 1e-6, 1-1e-6)), obs=cc)

        u_star, _ = reward_rate_grid(omega[vs], kappa[vs], vT, vD, vR, vq,
                                     gamma, hazard, t_iti, ug)
        rp = u_star + bc * vc
        numpyro.deterministic('rp', rp)
        with numpyro.plate('v', NV):
            numpyro.sample('ov', dist.Normal(rp, sv), obs=vr)
    return model


# ============================================================
# S3: S1 + per-subject reward rate offset (ρ̄_i)
# This is the Shadmehr "subjective environment richness" parameter.
# People who perceive the task as more rewarding press harder overall.
# Unlike baseline (additive shift on vigor), ρ̄ could enter choice too.
# ============================================================

def make_s3(NS, NC, NV):
    def model(cs, cT, cDH, cDL, cc, vs, vT, vR, vq, vD, vr, vc):
        gr = numpyro.sample('gr', dist.Normal(0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gr), 0.1, 3.0))
        hr = numpyro.sample('hr', dist.Normal(-1, 1))
        hazard = numpyro.deterministic('hazard', jnp.exp(hr))
        tr = numpyro.sample('tr', dist.Normal(0, 1))
        tau = jnp.clip(jnp.exp(tr), 0.01, 50.)
        sv = numpyro.sample('sv', dist.HalfNormal(0.5))
        bc = numpyro.sample('bc', dist.Normal(0, 0.5))
        iti_raw = numpyro.sample('iti_raw', dist.Normal(0, 0.5))
        t_iti = jnp.clip(jnp.exp(iti_raw), 0.1, 10.0)

        mo = numpyro.sample('mo', dist.Normal(1, 1))
        so = numpyro.sample('so', dist.HalfNormal(1.))
        mk = numpyro.sample('mk', dist.Normal(-2, 1))
        sk = numpyro.sample('sk', dist.HalfNormal(0.5))
        # Per-subject reward scaling (subjective task richness)
        mu_rho = numpyro.sample('mu_rho', dist.Normal(0, 0.5))
        sigma_rho = numpyro.sample('sigma_rho', dist.HalfNormal(0.3))

        with numpyro.plate('s', NS):
            or_ = numpyro.sample('or', dist.Normal(0, 1))
            kr_ = numpyro.sample('kr', dist.Normal(0, 1))
            rho_raw = numpyro.sample('rho_raw', dist.Normal(0, 1))
        omega = jnp.exp(mo + so * or_)
        kappa = jnp.exp(mk + sk * kr_)
        # rho_bar scales the reward: effective R_i = R * exp(rho_bar_i)
        rho_bar = jnp.exp(mu_rho + sigma_rho * rho_raw)
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)
        numpyro.deterministic('rho_bar', rho_bar)

        ug = jnp.linspace(0.1, 1.5, 40)

        # Choice: reward rate with per-subject reward scaling
        # R_eff = R * rho_bar_i — person who values task more sees higher rewards
        _, rho_H = reward_rate_grid(omega[cs], kappa[cs], cT, cDH,
                                    rho_bar[cs] * 5., jnp.full(NC, .9),
                                    gamma, hazard, t_iti, ug)
        _, rho_L = reward_rate_grid(omega[cs], kappa[cs], cT, cDL,
                                    rho_bar[cs] * 1., jnp.full(NC, .4),
                                    gamma, hazard, t_iti, ug)
        pH = jax.nn.sigmoid(jnp.clip((rho_H - rho_L) / tau, -20, 20))
        with numpyro.plate('c', NC):
            numpyro.sample('oc', dist.Bernoulli(probs=jnp.clip(pH, 1e-6, 1-1e-6)), obs=cc)

        # Vigor: reward rate with same scaling
        u_star, _ = reward_rate_grid(omega[vs], kappa[vs], vT, vD,
                                     rho_bar[vs] * vR, vq,
                                     gamma, hazard, t_iti, ug)
        rp = u_star + bc * vc
        numpyro.deterministic('rp', rp)
        with numpyro.plate('v', NV):
            numpyro.sample('ov', dist.Normal(rp, sv), obs=vr)
    return model


# ============================================================
# Fit + Evaluate
# ============================================================

def fit_model(name, model_fn, data, n_steps=35000, lr=0.001, seed=42):
    kw = {k: data[k] for k in KWARGS}
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
    sites = ['omega', 'kappa', 'rho_bar', 'rp', 'gamma', 'hazard', 'tr', 'bc', 'sv']
    samp = Predictive(fit['model_fn'], guide=fit['guide'], params=fit['best_params'],
                      num_samples=n_samples, return_sites=sites)(
        random.PRNGKey(44), **fit['kwargs'])

    rp = np.array(samp['rp']).mean(0)
    vr = np.array(data['vr'])
    r_vig = pearsonr(rp, vr)[0]

    omega = np.array(samp['omega']).mean(0)
    kappa = np.array(samp['kappa']).mean(0)
    rho_bar = np.array(samp['rho_bar']).mean(0) if 'rho_bar' in samp else None

    # Quick choice eval using the model's logic
    gamma_v = float(np.array(samp['gamma']).mean())
    hazard_v = float(np.array(samp['hazard']).mean())
    tau_v = float(np.exp(np.array(samp['tr']).mean()))
    t_iti_v = 1.0  # approximate

    cs = np.array(data['cs']); cT = np.array(data['cT'])
    cDH = np.array(data['cDH']); cc = np.array(data['cc'])

    ug = np.linspace(0.1, 1.5, 40)
    VH = np.zeros(len(cs)); VL = np.zeros(len(cs))
    for idx in range(len(cs)):
        s = cs[idx]; T = cT[idx]; DH = cDH[idx]
        om = omega[s]; kp = kappa[s]
        R_scale = rho_bar[s] if rho_bar is not None else 1.0
        for R, req, D, store in [(5.*R_scale, .9, DH, VH), (1.*R_scale, .4, 1., VL)]:
            S = np.exp(-hazard_v * T**gamma_v * D / np.clip(ug, 0.1, None))
            net = S*R - (1-S)*om*(R+C_PEN)
            t_trial = D / np.clip(ug, 0.1, None) + t_iti_v
            rho = net/t_trial - kp*ug**2
            store[idx] = rho.max()

    pH = expit(np.clip((VH-VL)/tau_v, -20, 20))
    acc = ((pH >= 0.5).astype(int) == cc).mean()
    ch_df = pd.DataFrame({'s': cs, 'c': cc, 'p': pH})
    sc = ch_df.groupby('s').agg(o=('c','mean'), p=('p','mean'))
    try: r_ch = pearsonr(sc['o'], sc['p'])[0]
    except: r_ch = np.nan

    # Correlations
    subjects = data['subjects']
    r_ok = pearsonr(omega, kappa)[0]

    # Behavioral
    import statsmodels.api as sm
    metrics = pd.read_csv("results/stats/vigor_analysis/vigor_metrics.csv")
    metrics = metrics[~metrics['subj'].isin(EXCLUDE)]
    antic = metrics[metrics['epoch']=='anticipatory'].dropna(subset=['norm_rate'])
    si = {s:i for i,s in enumerate(subjects)}

    # Threat-vigor slopes
    threat_slopes = {}
    mean_rates = {}
    for s, sdf in antic.groupby('subj'):
        if s not in si or len(sdf) < 10: continue
        X = sm.add_constant(sdf[['T_round','cookie']].values)
        y = sdf['norm_rate'].values
        v = np.isfinite(y)
        if v.sum() >= 10:
            ols = sm.OLS(y[v], X[v]).fit()
            threat_slopes[si[s]] = ols.params[1]
            mean_rates[si[s]] = y[v].mean()

    choice_rates = {}
    beh = pd.read_csv(DATA_DIR/"behavior_rich.csv", usecols=['subj','type','choice'], low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)]
    for s, rate in beh[beh['type']==1].groupby('subj')['choice'].mean().items():
        if s in si: choice_rates[si[s]] = rate

    shared = sorted(set(threat_slopes) & set(mean_rates) & set(choice_rates))
    om_a = np.array([omega[i] for i in shared])
    kap_a = np.array([kappa[i] for i in shared])
    rho_a = np.array([rho_bar[i] for i in shared]) if rho_bar is not None else None
    sl_a = np.array([threat_slopes[i] for i in shared])
    mr_a = np.array([mean_rates[i] for i in shared])
    ch_a = np.array([choice_rates[i] for i in shared])

    result = {
        'acc': acc, 'ch_r2': r_ch**2 if not np.isnan(r_ch) else np.nan,
        'vig_r2': r_vig**2, 'r_ok': r_ok,
        'gamma': gamma_v, 'hazard': hazard_v, 'tau': tau_v,
        'omega': omega, 'kappa': kappa,
        'om_choice': pearsonr(om_a, ch_a)[0],
        'om_vigor': pearsonr(om_a, mr_a)[0],
        'om_slope': pearsonr(om_a, sl_a)[0],
        'kap_choice': pearsonr(kap_a, ch_a)[0],
        'kap_vigor': pearsonr(kap_a, mr_a)[0],
        'kap_slope': pearsonr(kap_a, sl_a)[0],
    }
    if rho_a is not None:
        result['rho_choice'] = pearsonr(rho_a, ch_a)[0]
        result['rho_vigor'] = pearsonr(rho_a, mr_a)[0]
        result['rho_slope'] = pearsonr(rho_a, sl_a)[0]
        result['rho_bar'] = rho_bar

    return result


# ============================================================
# Main
# ============================================================

MODELS = [
    ('S1', make_s1, 2, 'Shadmehr basic (ω,κ)'),
    ('S2', make_s2, 2, 'S1 + cookie intercept'),
    ('S3', make_s3, 3, 'S2 + ρ̄ reward scaling'),
]

if __name__ == '__main__':
    t0 = time.time()
    print("=" * 70)
    print("SHADMEHR/YOON REWARD RATE MODEL")
    print("  ρ(u) = [S(u)·R - (1-S(u))·ω·(R+C)] / (D/u + t_iti) - κ·u²")
    print("  Pressing faster → double benefit: better S AND less time")
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
        n_p = nps * NS + 10
        bic = 2*fit['best_loss'] + n_p*np.log(data['N_choice']+data['N_vigor'])
        results.append({'Model': name, 'Desc': desc, 'ELBO': -fit['best_loss'],
                        'BIC': bic, 'acc': m['acc'], 'ch_r2': m['ch_r2'],
                        'vig_r2': m['vig_r2']})

        print(f"\n  ELBO={-fit['best_loss']:.1f} BIC={bic:.0f}")
        print(f"  Choice: acc={m['acc']:.3f} r²={m['ch_r2']:.3f}")
        print(f"  Vigor:  r²={m['vig_r2']:.3f}")
        print(f"  γ={m['gamma']:.2f} h={m['hazard']:.3f} τ={m['tau']:.2f}")
        print(f"  ω: mean={m['omega'].mean():.3f} SD={m['omega'].std():.3f}")
        print(f"  κ: mean={m['kappa'].mean():.4f} SD={m['kappa'].std():.4f}")
        print(f"  ω↔κ: r={m['r_ok']:.3f}")
        if 'rho_bar' in m and m.get('rho_bar') is not None:
            print(f"  ρ̄: mean={m['rho_bar'].mean():.3f} SD={m['rho_bar'].std():.3f}")

        print(f"\n  TRIPLE DISSOCIATION CHECK:")
        print(f"  {'Param':<6} {'→Choice':>10} {'→Mean Vigor':>13} {'→Threat Slope':>15}")
        print(f"  {'-'*46}")
        print(f"  {'ω':<6} {m['om_choice']:>+10.3f} {m['om_vigor']:>+13.3f} {m['om_slope']:>+15.3f}")
        print(f"  {'κ':<6} {m['kap_choice']:>+10.3f} {m['kap_vigor']:>+13.3f} {m['kap_slope']:>+15.3f}")
        if 'rho_choice' in m:
            print(f"  {'ρ̄':<6} {m['rho_choice']:>+10.3f} {m['rho_vigor']:>+13.3f} {m['rho_slope']:>+15.3f}")

    if results:
        df = pd.DataFrame(results)
        best = df['BIC'].min(); df['dBIC'] = df['BIC'] - best
        print(f"\n{'='*70}\nSHADMEHR COMPARISON\n{'='*70}")
        print(df.to_string(index=False))
        print("\nBenchmarks:")
        print("  M5: acc=0.794 ch_r²=0.952 vig_r²=0.496")
        print("  V8c: acc=0.742 ch_r²=0.885 vig_r²=0.735 (with baseline)")
        print("  F3:  acc=0.702 ch_r²=0.886 vig_r²=0.395 (no baseline)")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "joint_shadmehr.csv", index=False)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
