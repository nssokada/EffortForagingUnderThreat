"""
Two extensions of §4.81:

PART A — Test behavioral betas against ALL previously-untested clinical scales:
   - DASS21_Total
   - MFIS_Physical, MFIS_Cognitive, MFIS_Psychosocial (MFIS subscales)
   - STAI_State (state anxiety — separate from trait)
   Verifies that the apathy-specificity finding holds across the full scale set.

PART B — Differential reactivity in choice vs vigor:
   Per-subject, compute:
   - diff_T = β_T_choice_z − β_T_vigor_z  (decision-level vs motor-level threat sensitivity)
   - sum_T = β_T_choice_z + β_T_vigor_z   (overall threat sensitivity magnitude)
   Test whether differential reactivity tells us anything beyond either beta alone.

PART C — Joint 2D profile in (β_T_choice, β_T_vigor):
   Median split → 4 behavioral types:
   - HighC_HighV: heightened on both
   - HighC_LowV: choice-deterred but vigor-unmoved
   - LowC_HighV: choice-unmoved but vigor-deterred
   - LowC_LowV: low threat-sensitivity on both
   Test clinical-scale differences across these behavioral profiles.
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
    base = REPO/f'data/model_input_{sample_name}'
    if not base.exists(): return None
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
    bx = build_subject_data('exploratory')
    bc = build_subject_data('confirmatory')
    all_betas = pd.concat([bx, bc], ignore_index=True) if bc is not None else bx
    print(f'N betas: {len(all_betas)}')

    exp, conf = load_both()
    pm_rows = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa']); m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
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

    # ALL clinical scales (including previously-untested ones)
    clinical_cols = [
        'AMI_Total','AMI_Social','AMI_Behavioural','AMI_Emotional',
        'DASS21_Anxiety','DASS21_Depression','DASS21_Stress','DASS21_Total',  # +Total
        'STAI_Trait_FIXED','STAI_State',                                       # +State
        'OASIS_Total','STICSA_Total','PHQ9_Total',
        'MFIS_Total','MFIS_Physical','MFIS_Cognitive','MFIS_Psychosocial',     # +subscales
        'log_omega','log_kappa',
    ]
    beta_cols = ['beta_T_choice','beta_D_choice','beta_TxD_choice','beta_T_vigor','beta_D_vigor']

    for c in beta_cols + clinical_cols:
        if c not in df.columns: continue
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask,c]
            if x.std()>0:
                df.loc[mask,f'{c}_z'] = (x-x.mean())/x.std()

    # Differential and composite measures
    df['diff_T']  = df['beta_T_choice_z'] - df['beta_T_vigor_z']
    df['sum_T']   = df['beta_T_choice_z'] + df['beta_T_vigor_z']
    df['diff_D']  = df['beta_D_choice_z'] - df['beta_D_vigor_z']
    df['threat_sens_composite'] = (df['beta_T_choice_z'] + df['beta_T_vigor_z']) / 2

    for c in ['diff_T','sum_T','diff_D','threat_sens_composite']:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask,c]
            if x.std()>0:
                df.loc[mask,f'{c}_z'] = (x-x.mean())/x.std()

    # ============================================================
    # PART A — Test against PREVIOUSLY UNTESTED scales
    # ============================================================
    print('\n' + '='*72)
    print('PART A — Behavioral betas vs PREVIOUSLY UNTESTED clinical scales (pooled N=571)')
    print('='*72)

    new_scales = ['DASS21_Total','STAI_State','MFIS_Physical','MFIS_Cognitive','MFIS_Psychosocial']
    print(f'\n  {"beta":<22} ' + '  '.join([f'{s[:13]:>14}' for s in new_scales]))
    for b in beta_cols + ['threat_sens_composite']:
        row = [f'  {b:<22}']
        for scale in new_scales:
            if scale not in df.columns:
                row.append(f'{"NA":>14}')
                continue
            sub = df[[f'{scale}_z', f'{b}_z']].dropna()
            if len(sub) < 30:
                row.append(f'{"NA":>14}')
                continue
            s, fit = fit_t(f'{scale}_z ~ {b}_z', sub)
            r = s.loc[f'{b}_z']
            surv_flag = '★' if surv(r) else ' '
            row.append(f'{r["mean"]:+5.2f}{surv_flag:<2}{"":>6}')
        print(''.join(row))

    # ============================================================
    # PART B — Differential reactivity
    # ============================================================
    print('\n' + '='*72)
    print('PART B — Differential reactivity: β_T_choice - β_T_vigor → clinical')
    print('='*72)

    print('\n  Distribution of diff_T (β_T_choice - β_T_vigor, in z-units):')
    print(df['diff_T'].describe().round(3).to_string())

    # Correlation between the two T-betas
    from scipy.stats import pearsonr
    r, p = pearsonr(df['beta_T_choice'], df['beta_T_vigor'])
    print(f'\n  Correlation β_T_choice vs β_T_vigor: r = {r:+.3f}, p = {p:.3g}')

    # Test diff_T against each clinical scale
    all_test_scales = ['AMI_Total','AMI_Social','AMI_Behavioural','AMI_Emotional',
                       'DASS21_Anxiety','DASS21_Depression','DASS21_Stress','DASS21_Total',
                       'STAI_Trait_FIXED','STAI_State','OASIS_Total','STICSA_Total',
                       'PHQ9_Total','MFIS_Total','MFIS_Physical','MFIS_Cognitive','MFIS_Psychosocial']

    print(f'\n  Univariate: diff_T → each clinical scale (= choice-vigor reactivity differential):')
    for scale in all_test_scales:
        if scale not in df.columns: continue
        sub = df[[f'{scale}_z','diff_T_z']].dropna()
        if len(sub) < 30: continue
        s, fit = fit_t(f'{scale}_z ~ diff_T_z', sub)
        r = s.loc['diff_T_z']
        flag = '★' if surv(r) else ' '
        samples = fit.posterior['diff_T_z'].values.flatten()
        p_dir = (samples>0).mean() if r['mean']>0 else (samples<0).mean()
        print(f'    {scale:<24} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(dir)={p_dir:.3f}  {flag}')

    # Joint model: clinical ~ β_T_choice + β_T_vigor + interaction
    print(f'\n  Joint model: clinical ~ β_T_choice + β_T_vigor + β_T_choice:β_T_vigor (for AMI subscales)')
    for scale in ['AMI_Total','AMI_Social','AMI_Behavioural','AMI_Emotional']:
        sub = df[[f'{scale}_z','beta_T_choice_z','beta_T_vigor_z']].dropna()
        if len(sub) < 30: continue
        s, fit = fit_t(f'{scale}_z ~ beta_T_choice_z * beta_T_vigor_z', sub)
        print(f'\n    {scale}:')
        for term in ['beta_T_choice_z','beta_T_vigor_z','beta_T_choice_z:beta_T_vigor_z']:
            if term not in s.index: continue
            r = s.loc[term]
            flag = '★' if surv(r) else ' '
            print(f'      {term:<34} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # ============================================================
    # PART C — 2D behavioral profile (β_T_choice, β_T_vigor)
    # ============================================================
    print('\n' + '='*72)
    print('PART C — Behavioral 2D quadrants in (β_T_choice, β_T_vigor)')
    print('='*72)

    df['Tc_hi'] = 0; df['Tv_hi'] = 0
    for s in df['sample'].unique():
        mask = df['sample']==s
        df.loc[mask,'Tc_hi'] = (df.loc[mask,'beta_T_choice'] > df.loc[mask,'beta_T_choice'].median()).astype(int)
        df.loc[mask,'Tv_hi'] = (df.loc[mask,'beta_T_vigor'] > df.loc[mask,'beta_T_vigor'].median()).astype(int)
    # Note: HIGHER beta_T_choice = LESS threat-deterred in choice (β_T_choice typically negative)
    #       HIGHER beta_T_vigor = MORE threat-driven vigor (typically positive)
    def label(r):
        # Above-median on both = "less threat-deterred choice + more threat-driven vigor"
        if r['Tc_hi']==1 and r['Tv_hi']==1: return 'B_HighTc_HighTv (less deterred + more activated)'
        if r['Tc_hi']==1 and r['Tv_hi']==0: return 'C_HighTc_LowTv (less deterred + less activated)'
        if r['Tc_hi']==0 and r['Tv_hi']==1: return 'D_LowTc_HighTv (more deterred + more activated)'
        return 'A_LowTc_LowTv (more deterred + less activated)'
    df['behav_profile'] = df.apply(label, axis=1)

    print('\nBehavioral profile cell sizes:')
    print(df['behav_profile'].value_counts().sort_index())

    print('\nMean clinical scales by behavioral profile (z-scored):')
    g = df.groupby('behav_profile')[
        [f'{s}_z' for s in ['AMI_Total','AMI_Social','AMI_Behavioural','AMI_Emotional','DASS21_Total','MFIS_Total']]
    ].mean()
    g.columns = [c.replace('_z','') for c in g.columns]
    print(g.round(3).to_string())

    # Bayesian ANOVA: AMI_Total ~ behav_profile
    print('\n  Bayesian: AMI_Total_z ~ behav_profile (contrasts vs first cell):')
    sub = df[['AMI_Total_z','behav_profile']].dropna()
    s, fit = fit_t('AMI_Total_z ~ behav_profile', sub)
    for term in s.index:
        if 'behav_profile' in term:
            r = s.loc[term]
            flag = '★' if surv(r) else ' '
            print(f'    {term:<55} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')


if __name__ == '__main__':
    main()
