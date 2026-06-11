"""
Deep dive into behavioral-betas → mental health.

After §4.80 showed β_T_choice (-0.20★), β_TxD_choice (+0.12★), β_T_vigor (-0.11★)
predict AMI_Total, dig deeper:

  1. Each behavioral beta vs EVERY clinical scale individually
  2. AMI subscales (Social vs Behavioural vs Emotional) — where does threat sensitivity live?
  3. Per-sample replication of β_T_choice → AMI
  4. Clinical typology (Pure Apathy vs Pure Distress quadrants) — do behavioral betas
     show the same opposite-direction pattern?
  5. Composite behavioral threat-sensitivity measure (mean of beta_T_choice and beta_T_vigor)
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
        X = pd.DataFrame({
            'threat': g['threat'].values,
            'distance': g['distance_H'].values,
            'T_x_D': g['threat'].values * g['distance_H'].values,
        })
        X = sm_mod.add_constant(X)
        y = g['choice'].values
        try:
            mdl = sm_mod.OLS(y, X).fit()
            rows.append({'subj_idx': subj_idx,
                'beta_T_choice': mdl.params['threat'],
                'beta_D_choice': mdl.params['distance'],
                'beta_TxD_choice': mdl.params['T_x_D']})
        except Exception:
            pass
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
            rows.append({'subj_idx': subj_idx,
                'beta_T_vigor': mdl.params['threat'],
                'beta_D_vigor': mdl.params['distance']})
        except Exception:
            pass
    return pd.DataFrame(rows)


def build_subject_data(sample_name):
    if sample_name == 'exploratory':
        base = REPO/'data/model_input_exploratory'
    else:
        base = REPO/'data/model_input_confirmatory'
    if not base.exists() or not (base/'choice_trials.csv').exists():
        return None
    ct = pd.read_csv(base/'choice_trials.csv')
    vcm = pd.read_csv(base/'vigor_cell_means.csv')
    sm = pd.read_csv(base/'subject_mapping.csv')
    cb = per_subject_choice_betas(ct)
    vb = per_subject_vigor_betas(vcm)
    betas = sm.merge(cb, on='subj_idx', how='inner').merge(vb, on='subj_idx', how='inner')
    betas['sample'] = sample_name
    return betas


def fit_t(formula, data):
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    return az.summary(fit, hdi_prob=0.95), fit


def surv(r):
    return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)


def main():
    # Build behavioral betas for both samples
    bx = build_subject_data('exploratory')
    bc = build_subject_data('confirmatory')
    all_betas = pd.concat([bx, bc], ignore_index=True) if bc is not None else bx
    print(f'N betas: {len(all_betas)}')

    # Merge with clinical + (omega, kappa)
    exp, conf = load_both()
    pm_rows = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa']); m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        if 'AMI_Total' not in m.columns:
            m['AMI_Total'] = m[['AMI_Social','AMI_Behavioural','AMI_Emotional']].sum(axis=1)
        pm_rows.append(m)
    params = pd.concat(pm_rows, ignore_index=True)

    # FIXED STAI
    csta = []
    for sample in ['exp','con']:
        p = REPO/f'results/stats/affect_analysis/stai_fixed_{sample}.csv'
        if p.exists(): csta.append(pd.read_csv(p))
    cstai = pd.concat(csta)[['subj','sample','STAI_Trait_FIXED']]
    df_clin = params.merge(cstai, on=['subj','sample'], how='left')
    df = all_betas.merge(df_clin, on=['subj','sample'], how='inner')
    print(f'Final joined N = {len(df)}')

    # Within-sample z
    beta_cols = ['beta_T_choice','beta_D_choice','beta_TxD_choice','beta_T_vigor','beta_D_vigor']
    clinical_cols = ['AMI_Total','AMI_Social','AMI_Behavioural','AMI_Emotional',
                     'DASS21_Anxiety','DASS21_Depression','DASS21_Stress',
                     'STAI_Trait_FIXED','OASIS_Total','STICSA_Total','PHQ9_Total','MFIS_Total',
                     'log_omega','log_kappa']
    for c in beta_cols + clinical_cols:
        if c not in df.columns: continue
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask,c]
            if x.std()>0:
                df.loc[mask,f'{c}_z'] = (x-x.mean())/x.std()

    # ANX+DEP composite
    anxdep = ['DASS21_Anxiety_z','DASS21_Depression_z','DASS21_Stress_z',
              'STAI_Trait_FIXED_z','OASIS_Total_z','STICSA_Total_z','PHQ9_Total_z']
    df['ANXDEP_FIXED'] = df[anxdep].mean(axis=1)
    df['ANXDEP_FIXED_z'] = np.nan
    for s in df['sample'].unique():
        mask = df['sample']==s
        x = df.loc[mask,'ANXDEP_FIXED']
        df.loc[mask,'ANXDEP_FIXED_z'] = (x-x.mean())/x.std()

    # Composite behavioral threat-sensitivity
    df['threat_sens_composite'] = df[['beta_T_choice_z','beta_T_vigor_z']].mean(axis=1)
    df['threat_sens_composite_z'] = np.nan
    for s in df['sample'].unique():
        mask = df['sample']==s
        x = df.loc[mask,'threat_sens_composite']
        df.loc[mask,'threat_sens_composite_z'] = (x-x.mean())/x.std()

    # ============================================================
    # Test 1: Each behavioral beta against EVERY clinical scale
    # ============================================================
    print('\n' + '='*72)
    print('TEST 1: Behavioral betas vs ALL clinical scales (pooled N=571)')
    print('='*72)
    scales_to_test = ['AMI_Total','AMI_Social','AMI_Behavioural','AMI_Emotional',
                      'DASS21_Anxiety','DASS21_Depression','DASS21_Stress',
                      'STAI_Trait_FIXED','OASIS_Total','STICSA_Total','PHQ9_Total','MFIS_Total',
                      'ANXDEP_FIXED']
    print(f'\n  {"beta":<22} ' + '  '.join([f'{s[:11]:>12}' for s in scales_to_test]))
    matrix_rows = []
    for b in beta_cols + ['threat_sens_composite']:
        row = [f'  {b:<22}']
        rec = {'beta': b}
        for scale in scales_to_test:
            sub = df[[f'{scale}_z', f'{b}_z']].dropna()
            if len(sub) < 30:
                row.append(f'{"NA":>12}')
                continue
            s, fit = fit_t(f'{scale}_z ~ {b}_z', sub)
            r = s.loc[f'{b}_z']
            surv_flag = '★' if surv(r) else ' '
            row.append(f'{r["mean"]:+5.2f}{surv_flag:<2}{"":>4}')
            rec[scale] = (r['mean'], surv(r))
        print(''.join(row))
        matrix_rows.append(rec)

    # ============================================================
    # Test 2: AMI subscales — where does threat sensitivity live?
    # ============================================================
    print('\n' + '='*72)
    print('TEST 2: Which AMI subscale carries the threat-sensitivity signal?')
    print('='*72)
    for b in ['beta_T_choice', 'beta_T_vigor', 'beta_TxD_choice', 'threat_sens_composite']:
        print(f'\n  {b}_z → each AMI subscale (univariate):')
        for scale in ['AMI_Total','AMI_Social','AMI_Behavioural','AMI_Emotional']:
            sub = df[[f'{scale}_z', f'{b}_z']].dropna()
            s, fit = fit_t(f'{scale}_z ~ {b}_z', sub)
            r = s.loc[f'{b}_z']
            samples = fit.posterior[f'{b}_z'].values.flatten()
            p_dir = (samples > 0).mean() if r['mean'] > 0 else (samples < 0).mean()
            flag = '★' if surv(r) else ' '
            print(f'    {scale:<24} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(dir)={p_dir:.3f}  {flag}')

    # ============================================================
    # Test 3: Per-sample replication of β_T_choice → AMI_Total
    # ============================================================
    print('\n' + '='*72)
    print('TEST 3: Per-sample replication of key behavioral signatures → AMI_Total')
    print('='*72)
    for b in ['beta_T_choice','beta_TxD_choice','beta_T_vigor','threat_sens_composite']:
        print(f'\n  {b}_z → AMI_Total_z:')
        for sample in ['exploratory','confirmatory']:
            sub = df[df['sample']==sample][[f'AMI_Total_z', f'{b}_z']].dropna()
            s, fit = fit_t(f'AMI_Total_z ~ {b}_z', sub)
            r = s.loc[f'{b}_z']
            flag = '★' if surv(r) else ' '
            print(f'    [{sample:<14} N={len(sub)}] β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # ============================================================
    # Test 4: Clinical typology — Pure Apathy vs Pure Distress on behavioral betas
    # ============================================================
    print('\n' + '='*72)
    print('TEST 4: Clinical typology on behavioral betas')
    print('='*72)
    # Build quadrants on AMI_Total × ANX+DEP_FIXED
    df['AMI_hi'] = 0; df['ANXDEP_hi'] = 0
    for s in df['sample'].unique():
        mask = df['sample']==s
        df.loc[mask,'AMI_hi'] = (df.loc[mask,'AMI_Total'] > df.loc[mask,'AMI_Total'].median()).astype(int)
        df.loc[mask,'ANXDEP_hi'] = (df.loc[mask,'ANXDEP_FIXED'] > df.loc[mask,'ANXDEP_FIXED'].median()).astype(int)
    df['profile'] = df.apply(lambda r:
        '1_PureApathy' if (r['AMI_hi']==1 and r['ANXDEP_hi']==0)
        else '2_PureDistress' if (r['AMI_hi']==0 and r['ANXDEP_hi']==1)
        else '3_Comorbid' if (r['AMI_hi']==1 and r['ANXDEP_hi']==1)
        else '4_Healthy', axis=1)

    print('\n  Mean behavioral betas by profile:')
    g = df.groupby('profile')[[f'{b}_z' for b in beta_cols + ['threat_sens_composite']]].agg(['mean']).round(3)
    print(g.to_string())

    # PureApathy vs PureDistress contrast on each behavioral beta
    print('\n  PureApathy − PureDistress contrast on each behavioral beta:')
    sub_AB = df[df['profile'].isin(['1_PureApathy','2_PureDistress'])].copy()
    sub_AB['is_apathy'] = (sub_AB['profile']=='1_PureApathy').astype(int)
    for b in beta_cols + ['threat_sens_composite']:
        s, fit = fit_t(f'{b}_z ~ is_apathy', sub_AB)
        r = s.loc['is_apathy']
        flag = '★' if surv(r) else ' '
        print(f'    {b:<24} β(A−B)={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # ============================================================
    # Test 5: Threat-sensitivity composite → AMI (joint with other betas)
    # ============================================================
    print('\n' + '='*72)
    print('TEST 5: Threat-sensitivity composite vs individual T-betas')
    print('='*72)
    # Univariate threat_sens_composite
    sub = df[['AMI_Total_z','threat_sens_composite_z']].dropna()
    s, fit = fit_t('AMI_Total_z ~ threat_sens_composite_z', sub)
    r = s.loc['threat_sens_composite_z']
    flag = '★' if surv(r) else ' '
    print(f'\n  threat_sens_composite (mean of T_choice + T_vigor) → AMI_Total:')
    print(f'    β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # Both T-betas + interaction
    print(f'\n  β_T_choice + β_T_vigor + interaction → AMI_Total:')
    sub = df[['AMI_Total_z','beta_T_choice_z','beta_T_vigor_z']].dropna()
    s, fit = fit_t('AMI_Total_z ~ beta_T_choice_z * beta_T_vigor_z', sub)
    for term in ['beta_T_choice_z','beta_T_vigor_z','beta_T_choice_z:beta_T_vigor_z']:
        if term not in s.index: continue
        r = s.loc[term]
        flag = '★' if surv(r) else ' '
        print(f'    {term:<32} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # Save matrix
    out = REPO/'results/stats/affect_analysis/behavioral_betas_deep_dive.csv'
    pd.DataFrame(matrix_rows).to_csv(out, index=False)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
