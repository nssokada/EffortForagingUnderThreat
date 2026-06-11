"""
Cross-sample replication of the ANX+DEP composite finding from §4.75.

Fits the same model (log_omega ~ AMI_Total + ANXDEP_composite + log_kappa)
in:
  - Exploratory only (N=290)
  - Confirmatory only (N=281)
  - Pooled (N=571) — for reference

Also runs a leave-STICSA-out sensitivity since STICSA pointed positive in the
mixed-direction kitchen-sink.
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

    csta = []
    for sample in ['exp','con']:
        p = REPO/f'results/stats/affect_analysis/clinical_scores_corrected_{sample}.csv'
        if p.exists():
            csta.append(pd.read_csv(p))
    cstai = pd.concat(csta, ignore_index=True)[['subj','sample','STAI_Trait_corrected']]
    cstai['sample'] = cstai['sample'].replace({'exp':'exploratory','con':'confirmatory'})
    df = df.merge(cstai, on=['subj','sample'], how='inner')

    # Within-sample z
    cols = ['log_omega','log_kappa','AMI_Total',
            'DASS21_Anxiety','DASS21_Depression','DASS21_Stress',
            'STAI_Trait_corrected','OASIS_Total','STICSA_Total','PHQ9_Total']
    for c in cols:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask,c]
            if x.std()>0:
                df.loc[mask,f'{c}_z'] = (x-x.mean())/x.std()

    anxdep_7 = ['DASS21_Anxiety_z','DASS21_Depression_z','DASS21_Stress_z',
                'STAI_Trait_corrected_z','OASIS_Total_z','STICSA_Total_z','PHQ9_Total_z']
    anxdep_6_no_sticsa = ['DASS21_Anxiety_z','DASS21_Depression_z','DASS21_Stress_z',
                          'STAI_Trait_corrected_z','OASIS_Total_z','PHQ9_Total_z']

    df['ANXDEP_7'] = df[anxdep_7].mean(axis=1)
    df['ANXDEP_6_no_sticsa'] = df[anxdep_6_no_sticsa].mean(axis=1)
    for c in ['ANXDEP_7','ANXDEP_6_no_sticsa']:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask, c]
            df.loc[mask, f'{c}_z'] = (x-x.mean())/x.std()

    def run(data, composite_name, label):
        sub = data[['log_omega_z','log_kappa_z','AMI_Total_z', f'{composite_name}_z']].dropna()
        formula = f'log_omega_z ~ AMI_Total_z + {composite_name}_z + log_kappa_z'
        m = bmb.Model(formula, data=sub, family='t')
        fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
        s = az.summary(fit, hdi_prob=0.95)
        print(f'\n  [{label}, N={len(sub)}, composite={composite_name}]')
        for term in ['AMI_Total_z', f'{composite_name}_z', 'log_kappa_z']:
            r = s.loc[term]
            samples = fit.posterior[term].values.flatten()
            p_dir = (samples > 0).mean() if r['mean'] > 0 else (samples < 0).mean()
            flag = '★' if (r['hdi_2.5%'] > 0 or r['hdi_97.5%'] < 0) else ' '
            print(f'    {term:<26} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(dir)={p_dir:.3f}  {flag}')
        return s

    print('='*72)
    print('CROSS-SAMPLE REPLICATION — 7-scale ANX+DEP composite')
    print('='*72)
    for sample, label in [('pooled','Pooled (N=571)'),
                          ('exploratory','Exploratory only'),
                          ('confirmatory','Confirmatory only')]:
        if sample == 'pooled':
            data = df
        else:
            data = df[df['sample']==sample]
        run(data, 'ANXDEP_7', label)

    print('\n' + '='*72)
    print('SENSITIVITY — drop STICSA (it pointed positive direction); 6-scale composite')
    print('='*72)
    for sample, label in [('pooled','Pooled'),
                          ('exploratory','Exploratory'),
                          ('confirmatory','Confirmatory')]:
        if sample == 'pooled':
            data = df
        else:
            data = df[df['sample']==sample]
        run(data, 'ANXDEP_6_no_sticsa', label)


if __name__ == '__main__':
    main()
