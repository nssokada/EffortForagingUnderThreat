"""
Two methodological refinements:

(A) WHY AMI_Social? Test if it's:
    - Unique (drops out when other AMI subscales are in the model)
    - The most sensitive proxy for a general apathy construct
    - Survives partialling EVERY other clinical scale
  Tests:
    1. AMI subscales side-by-side: AMI_Social vs Behavioural vs Emotional
    2. AMI_Social vs AMI_Total
    3. Full clinical kitchen-sink + AMI_Social uniqueness test
    4. Item-overlap test: does the F4 (non-social apathy EFA factor) compete?

(B) CONTROL FOR κ in multivariate models. The (ω, κ) parameters are
    structurally correlated in the joint posterior. A finding of X → log(ω)
    could partially be picking up X → log(κ) via that correlation.
    Re-run headline findings with log_kappa as covariate.

All N=571, Student-t robust likelihood, within-sample z.
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
    exp, conf = load_both()
    rows = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa']); m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        rows.append(m)
    df = pd.concat(rows, ignore_index=True)

    # within-sample z
    scales = ['log_omega', 'log_kappa',
              'AMI_Social', 'AMI_Behavioural', 'AMI_Emotional', 'AMI_Total',
              'DASS21_Anxiety', 'DASS21_Stress', 'DASS21_Depression',
              'OASIS_Total', 'STICSA_Total', 'STAI_Trait', 'MFIS_Total', 'PHQ9_Total']
    for c in scales:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            if x.std() > 0:
                df.loc[mask, f'{c}_z'] = (x - x.mean())/x.std()
    print(f'N = {len(df)}')

    # ==========================================================================
    # PHASE A — WHY AMI_Social?
    # ==========================================================================
    print('\n' + '='*72)
    print('PHASE A — WHY AMI_Social?')
    print('='*72)

    # A.1 AMI subscales side-by-side, univariate
    print('\n--- A.1 AMI subscales UNIVARIATE on log(ω) ---')
    print(f'  {"scale":<24} {"β":>10} {"95% HDI":>22}')
    for ami in ['AMI_Social', 'AMI_Behavioural', 'AMI_Emotional', 'AMI_Total']:
        sub = df[['log_omega_z', f'{ami}_z']].dropna()
        s = fit_t(f'log_omega_z ~ {ami}_z', sub)
        r = s.loc[f'{ami}_z']
        flag = '★' if surv(r) else ' '
        print(f'  {ami:<24} {r["mean"]:+10.3f} [{r["hdi_2.5%"]:+.3f},{r["hdi_97.5%"]:+.3f}]  {flag}')

    # A.2 AMI subscales JOINTLY (do AMI_Social and the others compete?)
    print('\n--- A.2 AMI subscales JOINTLY: ω ~ Social + Behavioural + Emotional ---')
    sub = df[['log_omega_z','AMI_Social_z','AMI_Behavioural_z','AMI_Emotional_z']].dropna()
    s = fit_t('log_omega_z ~ AMI_Social_z + AMI_Behavioural_z + AMI_Emotional_z', sub)
    for term in ['AMI_Social_z','AMI_Behavioural_z','AMI_Emotional_z']:
        r = s.loc[term]
        flag = '★' if surv(r) else ' '
        print(f'  {term:<24} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # A.3 AMI_Social uniqueness — control for EVERY other clinical scale
    print('\n--- A.3 AMI_Social UNIQUENESS: ω ~ AMI_Social + all other scales ---')
    others = ['AMI_Behavioural_z','AMI_Emotional_z',
              'DASS21_Anxiety_z','DASS21_Stress_z','DASS21_Depression_z',
              'OASIS_Total_z','STICSA_Total_z','STAI_Trait_z','MFIS_Total_z','PHQ9_Total_z']
    sub = df[['log_omega_z','AMI_Social_z'] + others].dropna()
    formula = 'log_omega_z ~ AMI_Social_z + ' + ' + '.join(others)
    s = fit_t(formula, sub)
    print(f'  AMI_Social (controlling everything else):')
    r = s.loc['AMI_Social_z']
    flag = '★' if surv(r) else ' '
    print(f'    β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # ==========================================================================
    # PHASE B — Control for κ in multivariate
    # ==========================================================================
    print('\n' + '='*72)
    print('PHASE B — Control for log(κ) in multivariate ω models')
    print('='*72)

    # B.1 AMI_Social effect with vs without log_κ control
    print('\n--- B.1 AMI_Social → log(ω) with vs without κ control ---')
    sub = df[['log_omega_z','log_kappa_z','AMI_Social_z']].dropna()
    s_no = fit_t('log_omega_z ~ AMI_Social_z', sub)
    s_yes = fit_t('log_omega_z ~ AMI_Social_z + log_kappa_z', sub)
    rno = s_no.loc['AMI_Social_z']; ryes = s_yes.loc['AMI_Social_z']
    print(f'  Without κ:  β={rno["mean"]:+.3f}  HDI [{rno["hdi_2.5%"]:+.3f}, {rno["hdi_97.5%"]:+.3f}]')
    print(f'  With κ:     β={ryes["mean"]:+.3f}  HDI [{ryes["hdi_2.5%"]:+.3f}, {ryes["hdi_97.5%"]:+.3f}]')
    rk = s_yes.loc['log_kappa_z']
    print(f'  β(log κ controlled): {rk["mean"]:+.3f}  HDI [{rk["hdi_2.5%"]:+.3f}, {rk["hdi_97.5%"]:+.3f}]')

    # B.2 OASIS → ω with both AMI_Social AND κ control
    print('\n--- B.2 OASIS → log(ω) with AMI_Social + κ control ---')
    sub = df[['log_omega_z','log_kappa_z','AMI_Social_z','OASIS_Total_z']].dropna()
    s = fit_t('log_omega_z ~ OASIS_Total_z + AMI_Social_z + log_kappa_z', sub)
    for term in ['OASIS_Total_z', 'AMI_Social_z', 'log_kappa_z']:
        r = s.loc[term]
        flag = '★' if surv(r) else ' '
        print(f'  {term:<20} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # B.3 Kitchen sink ω model with κ control
    print('\n--- B.3 Kitchen-sink: log(ω) ~ all clinical scales + log(κ) ---')
    all_scales = ['AMI_Social_z','AMI_Behavioural_z','AMI_Emotional_z',
                  'DASS21_Anxiety_z','DASS21_Stress_z','DASS21_Depression_z',
                  'OASIS_Total_z','STICSA_Total_z','STAI_Trait_z','MFIS_Total_z','PHQ9_Total_z']
    sub = df[['log_omega_z','log_kappa_z'] + all_scales].dropna()
    formula = 'log_omega_z ~ ' + ' + '.join(all_scales) + ' + log_kappa_z'
    s = fit_t(formula, sub)
    for term in all_scales + ['log_kappa_z']:
        r = s.loc[term]
        flag = '★' if surv(r) else ' '
        print(f'  {term:<24} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # B.4 Same for κ as outcome (just for completeness)
    print('\n--- B.4 Kitchen-sink: log(κ) ~ all clinical scales + log(ω) ---')
    formula = 'log_kappa_z ~ ' + ' + '.join(all_scales) + ' + log_omega_z'
    s = fit_t(formula, sub)
    for term in all_scales + ['log_omega_z']:
        r = s.loc[term]
        flag = '★' if surv(r) else ' '
        print(f'  {term:<24} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    print('\n' + '='*72)
    print('VERDICT')
    print('='*72)


if __name__ == '__main__':
    main()
