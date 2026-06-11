"""
Mediation analysis: Does the AMI_Total → log(ω) relationship work THROUGH
within-task anxiety/confidence states?

Mediation framework:
  X = AMI_Total (predictor)
  M = mediator (within-task affect variable)
  Y = log(ω)  [or log(κ) for sensitivity]

Models:
  c-path:   Y ~ X            (total effect)
  a-path:   M ~ X            (effect of X on mediator)
  b-path:   Y ~ X + M        (mediator effect controlling for X)
  c'-path:  Y ~ X + M        (direct effect of X controlling for M)
  Indirect: a × b           (Bayesian: compute on each posterior draw)

Mediators tested:
  - mean_anxiety   (trait-like task-state anxiety)
  - mean_confidence (task-state confidence)
  - anx_slope      (anxiety reactivity to threat — slope)
  - anx_calibration (anxiety-threat correlation per subject)

Reports: c, a, b, c', indirect effect (a×b), proportion mediated.
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO); sys.path.insert(0, str(REPO/'notebooks'/'analysis'))
from load_data import load_both

BKW = dict(draws=2000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)


def fit_t(formula, data):
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    return az.summary(fit, hdi_prob=0.95), fit


def surv(r):
    return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)


def mediation_test(df, X, M, Y, label=''):
    """Compute Bayesian mediation: X → M → Y."""
    sub = df[[X, M, Y]].dropna()
    if len(sub) < 30:
        print(f'  Skipping {label}: not enough data')
        return None

    # c-path (total effect of X on Y)
    s_c, fit_c = fit_t(f'{Y} ~ {X}', sub)
    c_samples = fit_c.posterior[X].values.flatten()

    # a-path (X on M)
    s_a, fit_a = fit_t(f'{M} ~ {X}', sub)
    a_samples = fit_a.posterior[X].values.flatten()

    # b-path and c'-path (joint: Y ~ X + M)
    s_b, fit_b = fit_t(f'{Y} ~ {X} + {M}', sub)
    cprime_samples = fit_b.posterior[X].values.flatten()
    b_samples = fit_b.posterior[M].values.flatten()

    # Indirect effect = a * b (per draw)
    # Need matching shapes; both have draws*chains
    if len(a_samples) != len(b_samples):
        n = min(len(a_samples), len(b_samples))
        a_samples = a_samples[:n]; b_samples = b_samples[:n]
    indirect_samples = a_samples * b_samples

    # Proportion mediated = indirect / (indirect + direct) — careful with signs
    # Use absolute values for proportion to handle suppression cases
    total_samples = indirect_samples + cprime_samples[:len(indirect_samples)]
    prop_med = np.where(np.abs(total_samples) > 0.001,
                         indirect_samples / total_samples, np.nan)

    def hdi95(x):
        return np.percentile(x[~np.isnan(x)], [2.5, 97.5])

    def stats(x):
        return float(x.mean()), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    c_mean, c_lo, c_hi = stats(c_samples)
    a_mean, a_lo, a_hi = stats(a_samples)
    b_mean, b_lo, b_hi = stats(b_samples)
    cp_mean, cp_lo, cp_hi = stats(cprime_samples)
    ind_mean, ind_lo, ind_hi = stats(indirect_samples)

    surv_c = (c_lo > 0) or (c_hi < 0)
    surv_a = (a_lo > 0) or (a_hi < 0)
    surv_b = (b_lo > 0) or (b_hi < 0)
    surv_cp = (cp_lo > 0) or (cp_hi < 0)
    surv_ind = (ind_lo > 0) or (ind_hi < 0)

    print(f'\n  --- {label} ---')
    print(f'  c  (total {X}→{Y}):           β={c_mean:+.3f}  HDI [{c_lo:+.3f}, {c_hi:+.3f}]  {"★" if surv_c else " "}')
    print(f'  a  ({X}→{M}):                 β={a_mean:+.3f}  HDI [{a_lo:+.3f}, {a_hi:+.3f}]  {"★" if surv_a else " "}')
    print(f'  b  ({M}→{Y} | {X}):           β={b_mean:+.3f}  HDI [{b_lo:+.3f}, {b_hi:+.3f}]  {"★" if surv_b else " "}')
    print(f'  c\' (direct {X}→{Y} | {M}):    β={cp_mean:+.3f}  HDI [{cp_lo:+.3f}, {cp_hi:+.3f}]  {"★" if surv_cp else " "}')
    print(f'  a×b (INDIRECT effect):       β={ind_mean:+.3f}  HDI [{ind_lo:+.3f}, {ind_hi:+.3f}]  {"★" if surv_ind else " "}')
    if surv_ind:
        # Proportion mediated only meaningful when indirect is significant
        # Use point estimate for clarity
        if abs(c_mean) > 0.01:
            pm_point = ind_mean / c_mean
            print(f'  Proportion mediated (point est.): {pm_point:+.1%}')

    # Verdict
    if surv_a and surv_b and surv_ind:
        if not surv_cp:
            verdict = 'FULL MEDIATION (indirect ★, direct n.s.)'
        else:
            verdict = 'PARTIAL MEDIATION (both direct and indirect ★)'
    elif surv_a and surv_b:
        verdict = 'mediator passes both a/b paths but indirect not significant'
    else:
        verdict = 'NO MEDIATION'
    print(f'  Verdict: {verdict}')

    return {
        'label': label, 'X': X, 'M': M, 'Y': Y, 'N': len(sub),
        'c_mean': c_mean, 'c_hdi': (c_lo, c_hi), 'c_surv': surv_c,
        'a_mean': a_mean, 'a_hdi': (a_lo, a_hi), 'a_surv': surv_a,
        'b_mean': b_mean, 'b_hdi': (b_lo, b_hi), 'b_surv': surv_b,
        'cprime_mean': cp_mean, 'cprime_hdi': (cp_lo, cp_hi), 'cprime_surv': surv_cp,
        'indirect_mean': ind_mean, 'indirect_hdi': (ind_lo, ind_hi), 'indirect_surv': surv_ind,
        'verdict': verdict,
    }


def main():
    exp, conf = load_both()
    rows = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa'])
        m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        if 'AMI_Total' not in m.columns:
            m['AMI_Total'] = m[['AMI_Social','AMI_Behavioural','AMI_Emotional']].sum(axis=1)
        rows.append(m)
    df = pd.concat(rows, ignore_index=True)
    print(f'N = {len(df)}')

    # Within-sample z
    cols = ['log_omega','log_kappa','AMI_Total',
            'mean_anxiety','mean_confidence','anx_slope','anx_calibration']
    for c in cols:
        if c not in df.columns: continue
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask, c]
            if x.std() > 0:
                df.loc[mask, f'{c}_z'] = (x-x.mean())/x.std()

    # Mediation tests
    print('\n' + '='*72)
    print('MEDIATION ANALYSIS: AMI_Total → mediator → log(ω)')
    print('='*72)

    results = []
    for mediator, label in [
        ('mean_anxiety_z',   'mean_anxiety as mediator'),
        ('mean_confidence_z', 'mean_confidence as mediator'),
        ('anx_slope_z',      'anx_slope (anxiety reactivity to threat) as mediator'),
        ('anx_calibration_z', 'anx_calibration (anx-threat correlation) as mediator'),
    ]:
        r = mediation_test(df, 'AMI_Total_z', mediator, 'log_omega_z', label=label)
        if r: results.append(r)

    # Also try log(κ) for completeness
    print('\n' + '='*72)
    print('SENSITIVITY: AMI_Total → mediator → log(κ)')
    print('='*72)
    for mediator, label in [
        ('mean_anxiety_z',   'mean_anxiety → log(κ)'),
        ('mean_confidence_z', 'mean_confidence → log(κ)'),
        ('anx_slope_z',      'anx_slope → log(κ)'),
    ]:
        r = mediation_test(df, 'AMI_Total_z', mediator, 'log_kappa_z', label=label)
        if r: results.append(r)

    # Save
    out = REPO/'results/stats/affect_analysis/ami_omega_affect_mediation.csv'
    flat_rows = []
    for r in results:
        flat_rows.append({
            'label': r['label'], 'X': r['X'], 'M': r['M'], 'Y': r['Y'], 'N': r['N'],
            'c_mean': r['c_mean'], 'c_hdi_lo': r['c_hdi'][0], 'c_hdi_hi': r['c_hdi'][1], 'c_surv': r['c_surv'],
            'a_mean': r['a_mean'], 'a_hdi_lo': r['a_hdi'][0], 'a_hdi_hi': r['a_hdi'][1], 'a_surv': r['a_surv'],
            'b_mean': r['b_mean'], 'b_hdi_lo': r['b_hdi'][0], 'b_hdi_hi': r['b_hdi'][1], 'b_surv': r['b_surv'],
            'cprime_mean': r['cprime_mean'], 'cprime_hdi_lo': r['cprime_hdi'][0], 'cprime_hdi_hi': r['cprime_hdi'][1],
            'cprime_surv': r['cprime_surv'],
            'indirect_mean': r['indirect_mean'], 'indirect_hdi_lo': r['indirect_hdi'][0],
            'indirect_hdi_hi': r['indirect_hdi'][1], 'indirect_surv': r['indirect_surv'],
            'verdict': r['verdict'],
        })
    pd.DataFrame(flat_rows).to_csv(out, index=False)
    print(f'\nSaved: {out}')

    # Summary
    print('\n' + '='*72)
    print('SUMMARY — mediated paths (indirect effect ★)')
    print('='*72)
    for r in results:
        if r['indirect_surv']:
            print(f"  {r['label']}: indirect β = {r['indirect_mean']:+.3f}  HDI [{r['indirect_hdi'][0]:+.3f}, {r['indirect_hdi'][1]:+.3f}]  → {r['verdict']}")


if __name__ == '__main__':
    main()
