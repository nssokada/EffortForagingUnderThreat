"""
Behavioral channel-modality analysis:
  Some subjects respond to threat/effort by changing CHOICES (high |β_choice|).
  Others respond by changing VIGOR (high |β_vigor|).
  Some respond strongly on both, some weakly on both.

Per-subject metrics (sign-agnostic — about modulation MAGNITUDE):
  choice_mod = sqrt(β_T_choice² + β_D_choice²)     — total choice modulation across T,D
  vigor_mod  = sqrt(β_T_vigor² + β_D_vigor²)        — total vigor modulation across T,D

Derived measures (within-sample z-scored):
  total_mod     = choice_mod_z + vigor_mod_z      — overall responsiveness
  balance       = choice_mod_z - vigor_mod_z      — channel preference (positive = more choice-modulated)

Test each against all clinical scales.

Also test: 2D quadrants in (choice_mod, vigor_mod) space — do "high-both" vs
"low-both" vs "choice-only" vs "vigor-only" subjects differ on mental health?
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
import statsmodels.api as sm_mod
import matplotlib.pyplot as plt
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
            rows.append({'subj_idx': subj_idx,
                'beta_T_vigor': mdl.params['threat'],
                'beta_D_vigor': mdl.params['distance']})
        except Exception: pass
    return pd.DataFrame(rows)


def build_data():
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

    # === MODALITY measures ===
    # Magnitude of choice modulation = sqrt(β_T_choice² + β_D_choice²)
    all_betas['choice_mod'] = np.sqrt(all_betas['beta_T_choice']**2 + all_betas['beta_D_choice']**2)
    # Magnitude of vigor modulation = sqrt(β_T_vigor² + β_D_vigor²)
    all_betas['vigor_mod']  = np.sqrt(all_betas['beta_T_vigor']**2 + all_betas['beta_D_vigor']**2)
    # Simpler alternatives (threat-only magnitudes):
    all_betas['choice_T_mod'] = np.abs(all_betas['beta_T_choice'])
    all_betas['vigor_T_mod']  = np.abs(all_betas['beta_T_vigor'])

    return all_betas


def fit_t(formula, data):
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    return az.summary(fit, hdi_prob=0.95), fit


def surv(r):
    return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)


def main():
    all_betas = build_data()
    print(f'N betas: {len(all_betas)}')

    # Clinical
    exp, conf = load_both()
    pm = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa'])
        m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        if 'AMI_Total' not in m.columns:
            m['AMI_Total'] = m[['AMI_Social','AMI_Behavioural','AMI_Emotional']].sum(axis=1)
        if 'DASS21_Total' not in m.columns:
            m['DASS21_Total'] = m[['DASS21_Anxiety','DASS21_Stress','DASS21_Depression']].sum(axis=1)
        pm.append(m)
    params = pd.concat(pm, ignore_index=True)

    csta = []
    for sample in ['exp','con']:
        p = REPO/f'results/stats/affect_analysis/stai_fixed_{sample}.csv'
        if p.exists(): csta.append(pd.read_csv(p))
    cstai = pd.concat(csta)[['subj','sample','STAI_Trait_FIXED']]
    params = params.merge(cstai, on=['subj','sample'], how='left')

    df = all_betas.merge(params, on=['subj','sample'], how='inner')
    print(f'Final N = {len(df)}')

    # Within-sample z everything
    cols_to_z = ['choice_mod','vigor_mod','choice_T_mod','vigor_T_mod',
                 'beta_T_choice','beta_T_vigor','beta_D_choice','beta_D_vigor',
                 'log_omega','log_kappa',
                 'AMI_Total','AMI_Social','AMI_Behavioural','AMI_Emotional',
                 'DASS21_Anxiety','DASS21_Depression','DASS21_Stress','DASS21_Total',
                 'STAI_Trait_FIXED','OASIS_Total','STICSA_Total','PHQ9_Total',
                 'MFIS_Total','MFIS_Physical','MFIS_Cognitive','MFIS_Psychosocial']
    for c in cols_to_z:
        if c not in df.columns: continue
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask,c]
            if x.std()>0:
                df.loc[mask,f'{c}_z'] = (x-x.mean())/x.std()

    # Composite measures (after z)
    df['total_mod'] = df['choice_mod_z'] + df['vigor_mod_z']
    df['channel_balance'] = df['choice_mod_z'] - df['vigor_mod_z']  # positive = more choice-modulated
    df['total_T_mod'] = df['choice_T_mod_z'] + df['vigor_T_mod_z']
    df['channel_T_balance'] = df['choice_T_mod_z'] - df['vigor_T_mod_z']

    for c in ['total_mod','channel_balance','total_T_mod','channel_T_balance']:
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

    # ====================================================================
    # Test 1: Channel modality measures → all clinical scales
    # ====================================================================
    print('\n' + '='*72)
    print('TEST 1: Channel modality measures vs clinical (pooled N=571)')
    print('='*72)

    test_scales = ['AMI_Total','AMI_Social','AMI_Behavioural','AMI_Emotional',
                   'DASS21_Anxiety','DASS21_Depression','DASS21_Stress','DASS21_Total',
                   'STAI_Trait_FIXED','OASIS_Total','STICSA_Total','PHQ9_Total',
                   'MFIS_Total','MFIS_Psychosocial','ANXDEP_FIXED']

    print(f'\n  {"modality":<22} ' + '  '.join([f'{s[:11]:>13}' for s in test_scales]))
    for mod in ['choice_mod','vigor_mod','total_mod','channel_balance',
                'choice_T_mod','vigor_T_mod','total_T_mod','channel_T_balance']:
        row = [f'  {mod:<22}']
        for scale in test_scales:
            sub = df[[f'{scale}_z', f'{mod}_z']].dropna()
            if len(sub) < 30:
                row.append(f'{"NA":>13}')
                continue
            s, fit = fit_t(f'{scale}_z ~ {mod}_z', sub)
            r = s.loc[f'{mod}_z']
            flag = '★' if surv(r) else ' '
            row.append(f'{r["mean"]:+.3f}{flag:<2}{"":>5}')
        print(''.join(row))

    # ====================================================================
    # Test 2: 2D quadrants in (choice_mod, vigor_mod) space
    # ====================================================================
    print('\n' + '='*72)
    print('TEST 2: 2D Behavioral profiles in (choice_mod, vigor_mod) space')
    print('='*72)

    df['choice_hi'] = 0; df['vigor_hi'] = 0
    for s in df['sample'].unique():
        mask = df['sample']==s
        df.loc[mask,'choice_hi'] = (df.loc[mask,'choice_mod'] > df.loc[mask,'choice_mod'].median()).astype(int)
        df.loc[mask,'vigor_hi'] = (df.loc[mask,'vigor_mod'] > df.loc[mask,'vigor_mod'].median()).astype(int)

    def label(r):
        if r['choice_hi']==1 and r['vigor_hi']==1: return '1_HighBoth (responsive)'
        if r['choice_hi']==1 and r['vigor_hi']==0: return '2_HighChoice_LowVigor'
        if r['choice_hi']==0 and r['vigor_hi']==1: return '3_LowChoice_HighVigor'
        return '4_LowBoth (unresponsive)'
    df['modality_profile'] = df.apply(label, axis=1)

    print('\nProfile cell sizes:')
    print(df['modality_profile'].value_counts().sort_index())

    print('\nMean clinical scales by behavioral modality profile (z-scored):')
    g = df.groupby('modality_profile').agg(
        N=('subj','count'),
        AMI_Total_z=('AMI_Total_z','mean'),
        AMI_Social_z=('AMI_Social_z','mean'),
        AMI_Behavioural_z=('AMI_Behavioural_z','mean'),
        AMI_Emotional_z=('AMI_Emotional_z','mean'),
        ANXDEP_z=('ANXDEP_FIXED_z','mean'),
        MFIS_Psych_z=('MFIS_Psychosocial_z','mean'),
    ).round(3)
    print(g.to_string())

    # Bayesian contrasts on AMI_Total
    print('\n  Bayesian contrasts on AMI_Total_z (vs HighBoth baseline):')
    sub = df[['AMI_Total_z','modality_profile']].dropna()
    s, fit = fit_t('AMI_Total_z ~ modality_profile', sub)
    for term in s.index:
        if 'modality_profile' in term:
            r = s.loc[term]
            flag = '★' if surv(r) else ' '
            print(f'    {term[:55]:<55} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # ====================================================================
    # Test 3: total_mod and channel_balance with AMI/ANXDEP simultaneously
    # ====================================================================
    print('\n' + '='*72)
    print('TEST 3: Joint model — clinical ~ total_mod + channel_balance')
    print('='*72)
    for scale in ['AMI_Total','AMI_Social','AMI_Behavioural','DASS21_Total','ANXDEP_FIXED']:
        sub = df[[f'{scale}_z','total_mod_z','channel_balance_z']].dropna()
        s, fit = fit_t(f'{scale}_z ~ total_mod_z + channel_balance_z', sub)
        print(f'\n  {scale}_z:')
        for term in ['total_mod_z','channel_balance_z']:
            r = s.loc[term]
            flag = '★' if surv(r) else ' '
            print(f'    {term:<22} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df['choice_mod'], df['vigor_mod'], c=df['AMI_Total'],
                    cmap='viridis', s=24, alpha=0.7, edgecolors='none')
    ax.axvline(df['choice_mod'].median(), color='red', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(df['vigor_mod'].median(), color='red', lw=0.8, ls='--', alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label('AMI_Total')
    ax.set_xlabel('Choice modulation magnitude  (√(β_T_choice² + β_D_choice²))')
    ax.set_ylabel('Vigor modulation magnitude  (√(β_T_vigor² + β_D_vigor²))')
    ax.set_title('Behavioral modality profile colored by AMI_Total')
    plt.tight_layout()
    out = REPO/'results/figs/affect_analysis/modality_profile_AMI.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved figure: {out}')


if __name__ == '__main__':
    main()
