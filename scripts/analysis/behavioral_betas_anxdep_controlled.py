"""
One more honest check: does ANX+DEP have a behavioral threat-sensitivity effect
AFTER controlling for AMI? (Analogous to the §4.79 ω-axis suppression test.)

Tests:
  beta → AMI_Total + ANXDEP_composite (joint)
  Specifically: is there a residual ANXDEP effect on behavior once apathy is controlled?
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
import statsmodels.api as sm_mod
warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO); sys.path.insert(0, str(REPO/'notebooks'/'analysis'))
from load_data import load_both

BKW = dict(draws=2000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)


def per_subject_choice_betas(choice_df):
    rows = []
    for subj_idx, g in choice_df.groupby('subj_idx'):
        if len(g) < 10: continue
        X = pd.DataFrame({'threat': g['threat'].values, 'distance': g['distance_H'].values,
                          'T_x_D': g['threat'].values * g['distance_H'].values})
        X = sm_mod.add_constant(X)
        y = g['choice'].values
        try:
            mdl = sm_mod.OLS(y, X).fit()
            rows.append({'subj_idx': subj_idx,
                'beta_T_choice': mdl.params['threat'],
                'beta_TxD_choice': mdl.params['T_x_D']})
        except Exception: pass
    return pd.DataFrame(rows)


def per_subject_vigor_betas(vigor_df):
    rows = []
    for subj_idx, g in vigor_df.groupby('subj_idx'):
        if len(g) < 5: continue
        X = pd.DataFrame({'threat': g['T_round'].values, 'distance': g['actual_dist'].values})
        X = sm_mod.add_constant(X)
        y = g['mean_rate'].values
        try:
            mdl = sm_mod.WLS(y, X, weights=g['n_trials'].values).fit()
            rows.append({'subj_idx': subj_idx, 'beta_T_vigor': mdl.params['threat']})
        except Exception: pass
    return pd.DataFrame(rows)


def main():
    bx = pd.read_csv(REPO/'data/model_input_exploratory/choice_trials.csv')
    bv = pd.read_csv(REPO/'data/model_input_exploratory/vigor_cell_means.csv')
    sx = pd.read_csv(REPO/'data/model_input_exploratory/subject_mapping.csv')
    cx = per_subject_choice_betas(bx); vx = per_subject_vigor_betas(bv)
    bex = sx.merge(cx, on='subj_idx').merge(vx, on='subj_idx')
    bex['sample'] = 'exploratory'

    cx2 = pd.read_csv(REPO/'data/model_input_confirmatory/choice_trials.csv')
    vc2 = pd.read_csv(REPO/'data/model_input_confirmatory/vigor_cell_means.csv')
    sc2 = pd.read_csv(REPO/'data/model_input_confirmatory/subject_mapping.csv')
    cxc = per_subject_choice_betas(cx2); vxc = per_subject_vigor_betas(vc2)
    bec = sc2.merge(cxc, on='subj_idx').merge(vxc, on='subj_idx')
    bec['sample'] = 'confirmatory'

    all_betas = pd.concat([bex, bec], ignore_index=True)
    all_betas['threat_sens_composite'] = (all_betas['beta_T_choice'] + all_betas['beta_T_vigor'])/2  # raw, will z later

    # Clinical
    exp, conf = load_both()
    pm = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa'])
        m = m[(m['omega']>0)&(m['kappa']>0)]
        if 'AMI_Total' not in m.columns:
            m['AMI_Total'] = m[['AMI_Social','AMI_Behavioural','AMI_Emotional']].sum(axis=1)
        pm.append(m)
    params = pd.concat(pm, ignore_index=True)

    csta = []
    for sample in ['exp','con']:
        p = REPO/f'results/stats/affect_analysis/stai_fixed_{sample}.csv'
        if p.exists(): csta.append(pd.read_csv(p))
    cstai = pd.concat(csta)[['subj','sample','STAI_Trait_FIXED']]
    params = params.merge(cstai, on=['subj','sample'], how='left')
    df = all_betas.merge(params, on=['subj','sample'], how='inner')

    # Within-sample z
    cols = ['beta_T_choice','beta_T_vigor','beta_TxD_choice','threat_sens_composite',
            'AMI_Total','AMI_Social',
            'DASS21_Anxiety','DASS21_Depression','DASS21_Stress',
            'STAI_Trait_FIXED','OASIS_Total','STICSA_Total','PHQ9_Total']
    for c in cols:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask, c]
            if x.std()>0:
                df.loc[mask, f'{c}_z'] = (x-x.mean())/x.std()

    anxdep = ['DASS21_Anxiety_z','DASS21_Depression_z','DASS21_Stress_z',
              'STAI_Trait_FIXED_z','OASIS_Total_z','STICSA_Total_z','PHQ9_Total_z']
    df['ANXDEP_FIXED'] = df[anxdep].mean(axis=1)
    df['ANXDEP_FIXED_z'] = np.nan
    for s in df['sample'].unique():
        mask = df['sample']==s
        x = df.loc[mask,'ANXDEP_FIXED']
        df.loc[mask,'ANXDEP_FIXED_z'] = (x-x.mean())/x.std()
    print(f'N = {len(df)}')

    def fit_t(formula, data):
        m = bmb.Model(formula, data=data, family='t')
        fit = m.fit(**BKW, idata_kwargs={'log_likelihood':False})
        return az.summary(fit, hdi_prob=0.95), fit
    def surv(r): return (r['hdi_2.5%']>0) or (r['hdi_97.5%']<0)

    # ========================================================
    # Predicting each BEHAVIORAL beta from BOTH AMI + ANXDEP
    # If anxiety/depression has hidden behavioral signal, this should reveal it.
    # ========================================================
    print('\n' + '='*72)
    print('Each behavioral beta ~ AMI_Total + ANXDEP_composite (suppression test)')
    print('='*72)

    for beh in ['beta_T_choice','beta_T_vigor','threat_sens_composite','beta_TxD_choice']:
        sub = df[[f'{beh}_z','AMI_Total_z','ANXDEP_FIXED_z']].dropna()
        s, fit = fit_t(f'{beh}_z ~ AMI_Total_z + ANXDEP_FIXED_z', sub)
        print(f'\n  {beh}_z ~ AMI_Total + ANXDEP_composite (N={len(sub)}):')
        for term in ['AMI_Total_z','ANXDEP_FIXED_z']:
            r = s.loc[term]
            samples = fit.posterior[term].values.flatten()
            p_dir = (samples>0).mean() if r['mean']>0 else (samples<0).mean()
            flag = '★' if surv(r) else ' '
            print(f'    {term:<22} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(dir)={p_dir:.3f}  {flag}')

    # Also: each behavioral beta ~ all individual anxiety/depression scales (kitchen sink)
    print('\n' + '='*72)
    print('Behavioral beta ~ kitchen-sink (AMI + 7 anx/dep scales separately)')
    print('='*72)
    preds = ['AMI_Total_z','DASS21_Anxiety_z','DASS21_Depression_z','DASS21_Stress_z',
             'STAI_Trait_FIXED_z','OASIS_Total_z','STICSA_Total_z','PHQ9_Total_z']
    for beh in ['beta_T_choice','threat_sens_composite']:
        sub = df[[f'{beh}_z'] + preds].dropna()
        formula = f'{beh}_z ~ ' + ' + '.join(preds)
        s, fit = fit_t(formula, sub)
        print(f'\n  {beh}_z kitchen-sink (N={len(sub)}):')
        for term in preds:
            r = s.loc[term]
            samples = fit.posterior[term].values.flatten()
            p_dir = (samples>0).mean() if r['mean']>0 else (samples<0).mean()
            flag = '★' if surv(r) else ' '
            print(f'    {term:<22} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(dir)={p_dir:.3f}  {flag}')


if __name__ == '__main__':
    main()
