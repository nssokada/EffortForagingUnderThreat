"""
Kitchen-sink ω and κ models using TOTAL scores (not subscales) for AMI and DASS.

Predictor set (7 scales):
  - AMI_Total (instead of Social + Behavioural + Emotional)
  - DASS21_Total (instead of Anxiety + Depression + Stress)
  - OASIS_Total, STICSA_Total, STAI_Trait_corrected, MFIS_Total, PHQ9_Total

Plus log(κ) or log(ω) as covariate for the other parameter.

Reports:
  1. Kitchen-sink totals ω model
  2. Kitchen-sink totals κ model
  3. Compare to AMI_Social univariate (does AMI_Total recover the AMI_Social signal?)
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


def fit_t(formula, data):
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    return az.summary(fit, hdi_prob=0.95)


def surv(r):
    return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)


def main():
    # Load params + clinical scales from master
    exp, conf = load_both()
    pm_rows = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa'])
        m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        keep_cols = ['subj','sample','log_omega','log_kappa',
                     'AMI_Total','AMI_Social',
                     'DASS21_Total',
                     'OASIS_Total','STICSA_Total','MFIS_Total','PHQ9_Total']
        # AMI_Total and DASS21_Total: compute if missing
        if 'AMI_Total' not in m.columns:
            ami_cols = ['AMI_Social','AMI_Behavioural','AMI_Emotional']
            m['AMI_Total'] = m[ami_cols].sum(axis=1)
        if 'DASS21_Total' not in m.columns:
            dass_cols = ['DASS21_Anxiety','DASS21_Stress','DASS21_Depression']
            m['DASS21_Total'] = m[dass_cols].sum(axis=1)
        pm_rows.append(m[keep_cols])
    params = pd.concat(pm_rows, ignore_index=True)

    # Load corrected STAI
    csta = []
    for sample in ['exp', 'con']:
        path = REPO/f'results/stats/affect_analysis/clinical_scores_corrected_{sample}.csv'
        if path.exists():
            csta.append(pd.read_csv(path))
    cstai = pd.concat(csta, ignore_index=True)
    cstai = cstai[['subj','sample','STAI_Trait_corrected']]
    cstai['sample'] = cstai['sample'].replace({'exp':'exploratory','con':'confirmatory'})

    df = params.merge(cstai, on=['subj','sample'], how='inner')
    print(f'N = {len(df)}')

    # Within-sample z all scales
    scales = ['log_omega','log_kappa',
              'AMI_Total','AMI_Social',
              'DASS21_Total',
              'OASIS_Total','STICSA_Total','STAI_Trait_corrected','MFIS_Total','PHQ9_Total']
    for c in scales:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            if x.std() > 0:
                df.loc[mask, f'{c}_z'] = (x - x.mean())/x.std()

    rows = []

    # ===== A. AMI_Total univariate (does it carry the AMI_Social signal?)
    print('\n--- A. AMI_Total vs AMI_Social — univariate on log(ω) ---')
    for scale in ['AMI_Total','AMI_Social']:
        sub = df[['log_omega_z',f'{scale}_z']].dropna()
        s = fit_t(f'log_omega_z ~ {scale}_z', sub)
        r = s.loc[f'{scale}_z']
        flag = '★' if surv(r) else ' '
        print(f'  {scale:<14} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
        rows.append({'analysis':'univariate', 'predictor':scale, 'outcome':'log_omega',
                     'mean':float(r['mean']), 'hdi_lo':float(r['hdi_2.5%']),
                     'hdi_hi':float(r['hdi_97.5%']), 'survives':bool(surv(r))})

    # ===== B. Kitchen-sink totals ω model
    print('\n--- B. Kitchen-sink TOTALS ω model: log(ω) ~ 7 totals + log(κ) ---')
    totals_z = ['AMI_Total_z','DASS21_Total_z','OASIS_Total_z','STICSA_Total_z',
                'STAI_Trait_corrected_z','MFIS_Total_z','PHQ9_Total_z']
    sub = df[['log_omega_z','log_kappa_z'] + totals_z].dropna()
    formula = 'log_omega_z ~ ' + ' + '.join(totals_z) + ' + log_kappa_z'
    s = fit_t(formula, sub)
    print(f'  N = {len(sub)}')
    for term in totals_z + ['log_kappa_z']:
        r = s.loc[term]
        flag = '★' if surv(r) else ' '
        print(f'  {term:<30} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
        rows.append({'analysis':'kitchen_sink_totals_omega', 'predictor':term, 'outcome':'log_omega',
                     'mean':float(r['mean']), 'hdi_lo':float(r['hdi_2.5%']),
                     'hdi_hi':float(r['hdi_97.5%']), 'survives':bool(surv(r))})

    # ===== C. Kitchen-sink totals κ model
    print('\n--- C. Kitchen-sink TOTALS κ model: log(κ) ~ 7 totals + log(ω) ---')
    formula = 'log_kappa_z ~ ' + ' + '.join(totals_z) + ' + log_omega_z'
    s = fit_t(formula, sub)
    for term in totals_z + ['log_omega_z']:
        r = s.loc[term]
        flag = '★' if surv(r) else ' '
        print(f'  {term:<30} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
        rows.append({'analysis':'kitchen_sink_totals_kappa', 'predictor':term, 'outcome':'log_kappa',
                     'mean':float(r['mean']), 'hdi_lo':float(r['hdi_2.5%']),
                     'hdi_hi':float(r['hdi_97.5%']), 'survives':bool(surv(r))})

    # ===== D. Sensitivity: AMI_Social as primary instead of AMI_Total in totals model
    print('\n--- D. Sensitivity: kitchen-sink with AMI_Social (not Total) + other totals ---')
    preds_alt = ['AMI_Social_z','DASS21_Total_z','OASIS_Total_z','STICSA_Total_z',
                 'STAI_Trait_corrected_z','MFIS_Total_z','PHQ9_Total_z']
    sub = df[['log_omega_z','log_kappa_z'] + preds_alt].dropna()
    formula = 'log_omega_z ~ ' + ' + '.join(preds_alt) + ' + log_kappa_z'
    s = fit_t(formula, sub)
    for term in preds_alt + ['log_kappa_z']:
        r = s.loc[term]
        flag = '★' if surv(r) else ' '
        print(f'  {term:<30} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
        rows.append({'analysis':'kitchen_sink_sensitivity', 'predictor':term, 'outcome':'log_omega',
                     'mean':float(r['mean']), 'hdi_lo':float(r['hdi_2.5%']),
                     'hdi_hi':float(r['hdi_97.5%']), 'survives':bool(surv(r))})

    out = REPO/'results/stats/affect_analysis/kitchen_sink_totals.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nSaved: {out}')

    # Summary
    print('\n' + '='*72)
    print('SUMMARY — surviving effects')
    print('='*72)
    for r in rows:
        if r['survives']:
            print(f"  [{r['analysis']:<28}] {r['predictor']:<32} → {r['outcome']}: "
                  f"β={r['mean']:+.3f}  HDI [{r['hdi_lo']:+.3f}, {r['hdi_hi']:+.3f}]")


if __name__ == '__main__':
    main()
