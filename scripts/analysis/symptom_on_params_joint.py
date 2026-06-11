"""
Joint-regression framing (user's suggestion):
  symptom_z ~ log_omega_z + log_kappa_z + log_omega_z:log_kappa_z

This avoids forcing a particular balance metric. Instead it lets the data
reveal:
  - β(log_omega): does vigilance independently predict the symptom?
  - β(log_kappa): does mobilization independently predict it?
  - β(log_omega × log_kappa): does the COMBINATION matter beyond main effects?

If β(log_omega) ≈ -β(log_kappa) and interaction ≈ 0, the balance metric was
the right frame. If they don't cancel, balance was the wrong frame.

Runs four parameterizations for each symptom:
  (a) log-scale, additive only:    symptom ~ log_omega + log_kappa
  (b) log-scale, with interaction: symptom ~ log_omega * log_kappa
  (c) raw-scale, additive only:    symptom ~ omega + kappa   (sensitivity)
  (d) raw-scale, with interaction: symptom ~ omega * kappa   (sensitivity)

All N=571, within-sample z, Student-t likelihood.

Output: results/stats/affect_analysis/symptom_on_params_joint.csv
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

BKW = dict(draws=1000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)

SYMPTOMS = ['AMI_Social', 'DASS21_Stress', 'DASS21_Anxiety', 'DASS21_Depression']


def build():
    exp, conf = load_both()
    rows = []
    for sn, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d['master'].reset_index().rename(columns={'index': 'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega', 'kappa'])
        m = m[(m['omega'] > 0) & (m['kappa'] > 0)]
        m['log_omega'] = np.log(m['omega'])
        m['log_kappa'] = np.log(m['kappa'])
        rows.append(m)
    df = pd.concat(rows, ignore_index=True)

    # Within-sample z for the 4 params and every symptom
    cols = ['log_omega', 'log_kappa', 'omega', 'kappa'] + [s for s in SYMPTOMS if s in df.columns]
    for c in cols:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            df.loc[mask, f'{c}_z'] = (x - x.mean()) / x.std()

    return df


def fit_t(formula, data):
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={"log_likelihood": False})
    return az.summary(fit, hdi_prob=0.95)


def survives(row):
    return (row['hdi_2.5%'] > 0) or (row['hdi_97.5%'] < 0)


def fmt(row, name):
    sv = survives(row)
    return (f'    {name:<32} β={row["mean"]:+.3f}  '
            f'HDI [{row["hdi_2.5%"]:+.3f}, {row["hdi_97.5%"]:+.3f}]  {"★" if sv else ""}')


def run_one(df, symptom, kind, log_scale):
    """kind in {'additive', 'interaction'}, log_scale in {True, False}."""
    if log_scale:
        x1, x2 = 'log_omega_z', 'log_kappa_z'
    else:
        x1, x2 = 'omega_z', 'kappa_z'
    y = f'{symptom}_z'
    sub = df[[y, x1, x2]].dropna()
    if kind == 'additive':
        formula = f'{y} ~ {x1} + {x2}'
    else:
        formula = f'{y} ~ {x1} * {x2}'
    s = fit_t(formula, sub)
    return s, len(sub), formula


def main():
    df = build()
    print(f'N = {len(df)}  (no outlier filter)')

    rows = []

    for symptom in SYMPTOMS:
        if symptom not in df.columns:
            print(f'\n*** {symptom} not in data, skipping')
            continue
        print(f'\n{"="*72}\nSymptom (outcome): {symptom}\n{"="*72}')

        for log_scale, scale_lbl in [(True, 'log'), (False, 'raw')]:
            for kind in ['additive', 'interaction']:
                s, N, formula = run_one(df, symptom, kind, log_scale)
                print(f'\n  {symptom}_z ~ {formula.split("~")[1].strip()}  ({scale_lbl}-scale, N={N})')
                if log_scale:
                    main_terms = ['log_omega_z', 'log_kappa_z']
                    inter_term = 'log_omega_z:log_kappa_z'
                else:
                    main_terms = ['omega_z', 'kappa_z']
                    inter_term = 'omega_z:kappa_z'
                for t in main_terms:
                    if t in s.index:
                        r = s.loc[t]
                        print(fmt(r, t))
                        rows.append({'symptom': symptom, 'scale': scale_lbl, 'kind': kind,
                                     'term': t, 'N': N, 'mean': float(r['mean']),
                                     'sd': float(r['sd']),
                                     'hdi_lo': float(r['hdi_2.5%']),
                                     'hdi_hi': float(r['hdi_97.5%']),
                                     'survives': bool(survives(r))})
                if kind == 'interaction' and inter_term in s.index:
                    r = s.loc[inter_term]
                    print(fmt(r, inter_term))
                    rows.append({'symptom': symptom, 'scale': scale_lbl, 'kind': kind,
                                 'term': inter_term, 'N': N, 'mean': float(r['mean']),
                                 'sd': float(r['sd']),
                                 'hdi_lo': float(r['hdi_2.5%']),
                                 'hdi_hi': float(r['hdi_97.5%']),
                                 'survives': bool(survives(r))})

        # Test the balance-collapse hypothesis: are β(log_omega) and β(log_kappa) opposite sign?
        s_add, _, _ = run_one(df, symptom, 'additive', True)
        b_omega = s_add.loc['log_omega_z', 'mean']
        b_kappa = s_add.loc['log_kappa_z', 'mean']
        if abs(b_omega) < 0.01 and abs(b_kappa) < 0.01:
            verdict = 'both null'
        elif np.sign(b_omega) != np.sign(b_kappa):
            verdict = f'OPPOSITE signs → balance metric appropriate (β_ω={b_omega:+.3f}, β_κ={b_kappa:+.3f})'
        else:
            verdict = f'SAME signs → balance metric MISSES signal (β_ω={b_omega:+.3f}, β_κ={b_kappa:+.3f})'
        print(f'\n  >>> {symptom} verdict (log-scale additive): {verdict}')

    out = REPO / 'results/stats/affect_analysis/symptom_on_params_joint.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nSaved: {out}')

    # Summary
    print('\n' + '='*72)
    print('SUMMARY — surviving effects')
    print('='*72)
    surv = [r for r in rows if r['survives']]
    if not surv:
        print('  None.')
    for r in surv:
        print(f"  [{r['symptom']:<18} | {r['scale']}-scale {r['kind']:<11}] "
              f"{r['term']:<28} β={r['mean']:+.3f}  HDI [{r['hdi_lo']:+.3f}, {r['hdi_hi']:+.3f}]")


if __name__ == '__main__':
    main()
