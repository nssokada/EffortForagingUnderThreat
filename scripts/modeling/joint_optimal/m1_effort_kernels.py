"""
M1 effort-discounting-form comparison — supplemental.

Question: in the effort-only null model M1, which functional form best describes
how reward is discounted by effort? This is the modern replacement for the
deprecated FET result_206 (exponential vs hyperbolic vs quadratic vs linear),
re-run inside the current M1 specification.

The effort/cost argument is held fixed at the theory-motivated energy term
    E = req^2 * D
(quadratic instantaneous press cost x duration -- exactly M1's cost term, and
consistent with M4's vigor deviation cost). Only the DISCOUNT FUNCTION that
trades reward against E is varied. All variants are M1 EXACTLY otherwise
(per-subject kappa, intercept-only null vigor, same priors, free tau), with
equal parameter counts -> clean dBIC.

    V_opt = f(R_opt, E_opt);   dV = V_H - V_L

  Linear (current M1) : V = R - kappa*E
  Quadratic           : V = R - kappa*E^2
  Hyperbolic          : V = R / (1 + kappa*E)
  Exponential         : V = R * exp(-kappa*E)

R_H = 5, R_L = 1; req_H = 0.9, req_L = 0.4; D_H = distance_H in {1,2,3}, D_L = 1.
=> E_H = 0.81 * D_H, E_L = 0.16.
"""

import sys, time, os, argparse
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
from scipy.stats import pearsonr
from scipy.special import expit
from pathlib import Path

MODEL_INPUT_DIR = Path("data/model_input")   # overridable via --model-input-dir
OUT_DIR = Path("results/stats/joint_optimal")
FIG_DIR = Path("results/figs/paper")
SUFFIX = ""                                   # set via --suffix (e.g. "_confirmatory")

REQ_H, REQ_L = 0.9, 0.4
R_H, R_L = 5.0, 1.0

KK = ['cs', 'cDH', 'cc', 'vs', 'vr', 'vc', 'vn']


def prepare_data():
    choice = pd.read_csv(MODEL_INPUT_DIR / "choice_trials.csv")
    vigor = pd.read_csv(MODEL_INPUT_DIR / "vigor_cell_means.csv")
    subj_map = pd.read_csv(MODEL_INPUT_DIR / "subject_mapping.csv")
    NS = len(subj_map)
    NC = len(choice)
    NV = len(vigor)
    data = {
        'cs': jnp.array(choice['subj_idx'].values),
        'cDH': jnp.array(choice['distance_H'].values, dtype=jnp.float64),
        'cc': jnp.array(choice['choice'].values),
        'vs': jnp.array(vigor['subj_idx'].values, dtype=jnp.int32),
        'vr': jnp.array(vigor['mean_rate'].values),
        'vc': jnp.array(vigor['is_heavy'].values, dtype=jnp.float64),
        'vn': jnp.array(vigor['n_trials'].values, dtype=jnp.float64),
        'N_S': NS, 'N_choice': NC, 'N_vigor': NV,
    }
    print(f"  {NS} subjects, {NC} choice, {NV} cell-mean vigor")
    return data


def null_vigor(vr, vc, vn, NV):
    """Intercept-only null vigor model — identical across all variants."""
    mu_vigor = numpyro.sample('mu_vigor', dist.Normal(0.8, 0.5))
    bc = numpyro.sample('bc', dist.Normal(0, 0.5))
    sv = numpyro.sample('sv', dist.HalfNormal(0.3))
    rp = mu_vigor + bc * vc
    numpyro.deterministic('rp', rp)
    with numpyro.plate('v', NV):
        numpyro.sample('ov', dist.Normal(rp, sv / jnp.sqrt(vn)), obs=vr)


def hier_kappa(NS):
    mk = numpyro.sample('mk', dist.Normal(-1, 1))
    sk = numpyro.sample('sk', dist.HalfNormal(.5))
    with numpyro.plate('s', NS):
        kr_ = numpyro.sample('kr', dist.Normal(0, 1))
    kap = jnp.exp(mk + sk * kr_)
    numpyro.deterministic('kappa', kap)
    return kap


def choice_likelihood(dV, tau, cc, NC):
    pH = jax.nn.sigmoid(jnp.clip(dV / tau, -20, 20))
    with numpyro.plate('c', NC):
        numpyro.sample('oc', dist.Bernoulli(probs=jnp.clip(pH, 1e-6, 1 - 1e-6)), obs=cc)


E_H_COEF = REQ_H ** 2   # E_H = req_H^2 * D_H = 0.81 * D_H
E_L = REQ_L ** 2        # E_L = req_L^2 * D_L = 0.16


def _value(R, E, kap, form):
    """Discounted value of an option given reward R, effort E, per-subject kappa."""
    if form == 'linear':
        return R - kap * E
    if form == 'quadratic':
        return R - kap * E ** 2
    if form == 'hyperbolic':
        return R / (1.0 + kap * E)
    if form == 'exponential':
        return R * jnp.exp(-kap * E)
    raise ValueError(form)


def make_discount(NS, NC, NV, form):
    """M1 with effort cost argument E = req^2 * D fixed; discount FUNCTION = `form`."""
    def model(cs, cDH, cc, vs, vr, vc, vn):
        tr = numpyro.sample('tr', dist.Normal(0, 1))
        tau = jnp.clip(jnp.exp(tr), .01, 50.)
        kap = hier_kappa(NS)
        E_H = E_H_COEF * cDH
        VH = _value(R_H, E_H, kap[cs], form)
        VL = _value(R_L, E_L, kap[cs], form)
        choice_likelihood(VH - VL, tau, cc, NC)
        null_vigor(vr, vc, vn, NV)
    return model


# ---- Axis A: effort EXPONENT on req (discount held linear) -----------------
# cost = kappa * req^p * D ; value = R - cost. Tests whether the press-rate
# requirement enters linearly (p=1, M4 choice total-demand form) or convexly
# (p=2, current M1), with p also estimated freely.

def make_exponent(NS, NC, NV, p_fixed=None):
    def model(cs, cDH, cc, vs, vr, vc, vn):
        tr = numpyro.sample('tr', dist.Normal(0, 1))
        tau = jnp.clip(jnp.exp(tr), .01, 50.)
        if p_fixed is None:
            pr = numpyro.sample('pr', dist.Normal(0.4, 0.5))
            p = numpyro.deterministic('p', jnp.clip(jnp.exp(pr), 0.1, 10.0))
        else:
            p = p_fixed
        kap = hier_kappa(NS)
        dV = (R_H - R_L) - kap[cs] * (jnp.power(REQ_H, p) * cDH - jnp.power(REQ_L, p))
        choice_likelihood(dV, tau, cc, NC)
        null_vigor(vr, vc, vn, NV)
    return model


def evaluate_exponent(fit, data, p_fixed=None, n_samples=300):
    kw = fit['kwargs']
    sites = ['kappa', 'rp', 'tr'] + (['p'] if p_fixed is None else [])
    pred = Predictive(fit['model_fn'], guide=fit['guide'], params=fit['best_params'],
                      num_samples=n_samples, return_sites=sites)
    samp = pred(random.PRNGKey(44), **kw)
    rp = np.array(samp['rp']).mean(0)
    r_vig = pearsonr(rp, np.array(data['vr']))[0]
    cs = np.array(data['cs']); cDH = np.array(data['cDH']); cc = np.array(data['cc'])
    tau_v = float(np.exp(np.array(samp['tr']).mean()))
    kap = np.array(samp['kappa']).mean(0)[cs]
    p_hat = float(np.array(samp['p']).mean()) if p_fixed is None else float(p_fixed)
    dV = (R_H - R_L) - kap * (REQ_H ** p_hat * cDH - REQ_L ** p_hat)
    pH = expit(np.clip(dV / tau_v, -20, 20))
    acc = ((pH >= 0.5).astype(int) == cc).mean()
    ch = pd.DataFrame({'s': cs, 'c': cc, 'p': pH}).groupby('s').agg(
        o=('c', 'mean'), p=('p', 'mean'))
    return {'choice_acc': acc, 'choice_r2': pearsonr(ch['o'], ch['p'])[0] ** 2,
            'vigor_r2': r_vig ** 2, 'p_hat': p_hat}


def fit_model(name, model_fn, data, n_steps=35000, lr=0.001, seed=42):
    kw = {k: data[k] for k in KK}
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
        if l < bl and not np.isnan(l):
            bl = l; bp = svi.get_params(state)
        if (i + 1) % 10000 == 0:
            print(f"    {name} step {i+1}: {l:.1f} (best={bl:.1f})")
    print(f"    {name} done in {time.time()-t0:.0f}s, best={bl:.1f}")
    return {'name': name, 'best_loss': bl, 'best_params': bp,
            'guide': guide, 'model_fn': model_fn, 'kwargs': kw}


def _value_np(R, E, kap, form):
    if form == 'linear':
        return R - kap * E
    if form == 'quadratic':
        return R - kap * E ** 2
    if form == 'hyperbolic':
        return R / (1.0 + kap * E)
    if form == 'exponential':
        return R * np.exp(-kap * E)
    raise ValueError(form)


def evaluate(fit, data, form, n_samples=300):
    kw = fit['kwargs']
    pred = Predictive(fit['model_fn'], guide=fit['guide'], params=fit['best_params'],
                      num_samples=n_samples, return_sites=['kappa', 'rp', 'tr'])
    samp = pred(random.PRNGKey(44), **kw)

    vr = np.array(data['vr'])
    rp = np.array(samp['rp']).mean(0)
    r_vig = pearsonr(rp, vr)[0]

    cs = np.array(data['cs']); cDH = np.array(data['cDH']); cc = np.array(data['cc'])
    tau_v = float(np.exp(np.array(samp['tr']).mean()))
    kap = np.array(samp['kappa']).mean(0)[cs]
    E_H = (REQ_H ** 2) * cDH
    E_L = REQ_L ** 2
    dV = _value_np(R_H, E_H, kap, form) - _value_np(R_L, E_L, kap, form)
    pH = expit(np.clip(dV / tau_v, -20, 20))

    acc = ((pH >= 0.5).astype(int) == cc).mean()
    ch_df = pd.DataFrame({'s': cs, 'c': cc, 'p': pH})
    sc = ch_df.groupby('s').agg(o=('c', 'mean'), p=('p', 'mean'))
    r_ch = pearsonr(sc['o'], sc['p'])[0]

    return {'choice_acc': acc, 'choice_r2': r_ch ** 2, 'vigor_r2': r_vig ** 2}


FORM_SPECS = [
    ('Linear (M1)',  'linear'),
    ('Quadratic',    'quadratic'),
    ('Hyperbolic',   'hyperbolic'),
    ('Exponential',  'exponential'),
]

EXP_SPECS = [
    ('Linear (p=1, M4-style)', 1.0),
    ('Quadratic (p=2, M1)',    2.0),
    ('Free power (p est.)',    None),
]


def run_exponent_sweep(data, n_obs):
    """Axis A: cost = kappa * req^p * D, discount linear. Vary p."""
    NS = data['N_S']
    print("\n" + "=" * 70)
    print("AXIS A — EFFORT EXPONENT ON req  (cost = kappa * req^p * D)")
    print("=" * 70)
    rows = []
    for name, p_fixed in EXP_SPECS:
        print(f"\n--- {name} ---")
        model_fn = make_exponent(NS, data['N_choice'], data['N_vigor'], p_fixed)
        fit = fit_model(name, model_fn, data)
        if fit['best_params'] is None:
            print(f"  {name} FAILED"); continue
        m = evaluate_exponent(fit, data, p_fixed)
        n_params = NS + 6 + (1 if p_fixed is None else 0)
        bic = 2 * fit['best_loss'] + n_params * np.log(n_obs)
        rows.append({'Model': name, 'p_hat': m['p_hat'], 'n_params': n_params,
                     'ELBO': -fit['best_loss'], 'BIC': bic,
                     'choice_acc': m['choice_acc'], 'choice_r2': m['choice_r2'],
                     'vigor_r2': m['vigor_r2']})
        print(f"  p_hat={m['p_hat']:.3f}  ELBO={-fit['best_loss']:.1f}  BIC={bic:.0f}"
              f"  choice_r2={m['choice_r2']:.3f}")
    df = pd.DataFrame(rows)
    df['dBIC'] = df['BIC'] - df['BIC'].min()
    print("\n" + df[['Model', 'p_hat', 'n_params', 'ELBO', 'BIC', 'dBIC',
                     'choice_r2']].to_string(index=False))
    out = OUT_DIR / f"m1_effort_exponent{SUFFIX}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    return df


def run_form_sweep(data, n_obs):
    """Axis B: E = req^2 * D fixed; vary the discount FUNCTION."""
    NS = data['N_S']
    print("\n" + "=" * 70)
    print("AXIS B — DISCOUNT FUNCTION  (E = req^2 * D fixed)")
    print("=" * 70)
    rows = []
    for name, form in FORM_SPECS:
        print(f"\n--- {name} ({form}) ---")
        model_fn = make_discount(NS, data['N_choice'], data['N_vigor'], form)
        fit = fit_model(name, model_fn, data)
        if fit['best_params'] is None:
            print(f"  {name} FAILED"); continue
        m = evaluate(fit, data, form)
        n_params = NS + 6   # equal across forms
        bic = 2 * fit['best_loss'] + n_params * np.log(n_obs)
        rows.append({'Model': name, 'form': form, 'n_params': n_params,
                     'ELBO': -fit['best_loss'], 'BIC': bic,
                     'choice_acc': m['choice_acc'], 'choice_r2': m['choice_r2'],
                     'vigor_r2': m['vigor_r2']})
        print(f"  ELBO={-fit['best_loss']:.1f}  BIC={bic:.0f}"
              f"  choice_r2={m['choice_r2']:.3f}")
    df = pd.DataFrame(rows)
    df['dBIC'] = df['BIC'] - df['BIC'].min()
    print("\n" + df[['Model', 'form', 'n_params', 'ELBO', 'BIC', 'dBIC',
                     'choice_acc', 'choice_r2']].to_string(index=False))
    out = OUT_DIR / f"m1_effort_kernels{SUFFIX}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    return df


def main():
    global MODEL_INPUT_DIR, SUFFIX
    parser = argparse.ArgumentParser(description="M1 effort-shape supplement")
    parser.add_argument("--figure-only", action="store_true",
                        help="Regenerate the figure from committed CSVs without recomputing fits")
    parser.add_argument("--model-input-dir", default=str(MODEL_INPUT_DIR),
                        help="Dir with choice_trials.csv / vigor_cell_means.csv / subject_mapping.csv")
    parser.add_argument("--suffix", default="",
                        help="Suffix for output CSVs/figure, e.g. _confirmatory")
    parsed_args = parser.parse_args()
    MODEL_INPUT_DIR = Path(parsed_args.model_input_dir)
    SUFFIX = parsed_args.suffix

    if parsed_args.figure_only:
        make_figure()
        return

    t_start = time.time()
    print("=" * 70)
    print("M1 EFFORT-SHAPE SUPPLEMENT")
    print("=" * 70)
    data = prepare_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_obs = data['N_choice'] + data['N_vigor']

    run_exponent_sweep(data, n_obs)
    run_form_sweep(data, n_obs)

    make_figure()
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")


def make_figure():
    """Axis-B-only supplemental figure, read from the committed CSV so it is
    reproducible without recomputing the SVI fits. Left: discount-form ΔBIC
    (the headline). Right: subject-level choice R² by form (0–1 axis so all
    bars are visible). Axis A (effort exponent) is reported in the text/table
    only and is intentionally not plotted."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df_form = pd.read_csv(OUT_DIR / f"m1_effort_kernels{SUFFIX}.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    colors = ['#4C72B0', '#C44E52', '#55A868', '#8172B3']
    labels = list(df_form['Model'])

    # Panel A: discount-function ΔBIC — the headline
    ax1.bar(labels, df_form['dBIC'], color=colors[:len(df_form)])
    ax1.set_ylabel('ΔBIC (vs best)')
    ax1.set_title('Effort-discounting form\n(E = req²·D fixed; lower = better)')
    ax1.axhline(0, color='k', lw=0.8)
    for i, v in enumerate(df_form['dBIC']):
        ax1.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=9)

    # Panel B: subject-level choice R² by form (full 0–1 axis)
    ax2.bar(labels, df_form['choice_r2'], color=colors[:len(df_form)])
    ax2.set_ylabel('choice R² (subject-level)')
    ax2.set_title('Choice fit by form')
    ax2.set_ylim(0, 1.0)
    for i, v in enumerate(df_form['choice_r2']):
        ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    for ax in (ax1, ax2):
        ax.tick_params(axis='x', labelrotation=12, labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f"fig_s_m1_effort_kernels{SUFFIX}.{ext}", dpi=150, bbox_inches='tight')
    print(f"Saved figure {FIG_DIR / f'fig_s_m1_effort_kernels{SUFFIX}.pdf'}")


if __name__ == '__main__':
    main()
