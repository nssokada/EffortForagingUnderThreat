"""
Recover the §4.84 channel-modality findings using MODEL PARAMETERS (ω, κ).

Predicted mapping:
  total_mod (behavioral sum)       ↔ log(ω/κ)     [behavioral "responsiveness"]
  channel_balance (behavioral diff) ↔ log(ω·κ)    [behavioral "channel preference"]

Reasoning: choice_mod ∝ ω, vigor_mod ∝ 1/κ (lower κ = more vigor flexibility).
So choice_mod + vigor_mod ∝ log(ω) - log(κ) and choice_mod - vigor_mod ∝ log(ω) + log(κ).

Tests:
  1. Empirical correlation: total_mod vs log(ω/κ); channel_balance vs log(ω·κ)
  2. Replicate §4.84 joint regression: clinical ~ total_mod_model + channel_balance_model
     i.e., clinical ~ log(ω/κ) + log(ω·κ) — does the same pattern emerge?
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
import statsmodels.api as sm_mod
from scipy.stats import pearsonr
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
                'beta_D_choice': mdl.params['distance']})
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


def main():
    # Build behavioral betas
    rows = []
    for sample in ['exploratory','confirmatory']:
        base = REPO/f'data/model_input_{sample}'
        ct = pd.read_csv(base/'choice_trials.csv')
        vcm = pd.read_csv(base/'vigor_cell_means.csv')
        sm = pd.read_csv(base/'subject_mapping.csv')
        cb = per_subject_choice_betas(ct)
        vb = per_subject_vigor_betas(vcm)
        b = sm.merge(cb, on='subj_idx').merge(vb, on='subj_idx')
        b['sample'] = sample
        rows.append(b)
    all_betas = pd.concat(rows, ignore_index=True)

    # Behavioral modality measures
    all_betas['choice_mod'] = np.sqrt(all_betas['beta_T_choice']**2 + all_betas['beta_D_choice']**2)
    all_betas['vigor_mod']  = np.sqrt(all_betas['beta_T_vigor']**2 + all_betas['beta_D_vigor']**2)

    # Load params + clinical
    exp, conf = load_both()
    pm = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa'])
        m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        # MODEL ANALOGUES
        m['log_ratio'] = m['log_omega'] - m['log_kappa']   # ↔ total_mod
        m['log_sum']   = m['log_omega'] + m['log_kappa']   # ↔ channel_balance
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
    print(f'N = {len(df)}')

    # Within-sample z
    cols = ['choice_mod','vigor_mod','log_omega','log_kappa','log_ratio','log_sum',
            'AMI_Total','AMI_Social','AMI_Behavioural','AMI_Emotional',
            'DASS21_Stress','DASS21_Anxiety','DASS21_Depression',
            'STAI_Trait_FIXED','OASIS_Total','STICSA_Total','PHQ9_Total']
    for c in cols:
        if c not in df.columns: continue
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask, c]
            if x.std()>0:
                df.loc[mask, f'{c}_z'] = (x-x.mean())/x.std()

    df['total_mod_behav'] = df['choice_mod_z'] + df['vigor_mod_z']
    df['channel_balance_behav'] = df['choice_mod_z'] - df['vigor_mod_z']
    for c in ['total_mod_behav','channel_balance_behav']:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask, c]
            df.loc[mask, f'{c}_z'] = (x-x.mean())/x.std()

    def fit_t(formula, data):
        m = bmb.Model(formula, data=data, family='t')
        fit = m.fit(**BKW, idata_kwargs={'log_likelihood':False})
        return az.summary(fit, hdi_prob=0.95), fit
    def surv(r): return (r['hdi_2.5%']>0) or (r['hdi_97.5%']<0)

    # ========================================================
    # Test 1: Empirical correspondence
    # ========================================================
    print('\n' + '='*72)
    print('TEST 1: Empirical mapping behavioral ↔ model-parameter analogues')
    print('='*72)
    for behav, model in [('total_mod_behav','log_ratio'),
                         ('channel_balance_behav','log_sum'),
                         ('choice_mod','log_omega'),
                         ('vigor_mod','log_kappa')]:
        r, p = pearsonr(df[behav], df[model])
        print(f'  {behav:<24} vs {model:<14} r = {r:+.3f}  p = {p:.3g}')
    print('\n  Also: behavioral vs UNROTATED model parameters:')
    for behav in ['total_mod_behav','channel_balance_behav']:
        for mp in ['log_omega','log_kappa']:
            r, p = pearsonr(df[behav], df[mp])
            print(f'  {behav:<24} vs {mp:<14} r = {r:+.3f}  p = {p:.3g}')

    # ========================================================
    # Test 2: Joint regression with model analogues — replicate §4.84?
    # ========================================================
    print('\n' + '='*72)
    print('TEST 2: clinical ~ log(ω/κ) [≈ total_mod] + log(ω·κ) [≈ channel_balance]')
    print('='*72)
    print('  Compare to behavioral results from §4.84:')
    print('    AMI_Total: total_mod β=+0.111 ★, channel_balance β=+0.121 ★')
    print('    AMI_Social: total_mod β=+0.139 ★, channel_balance β=+0.115 ★')
    print('    AMI_Behavioural: total_mod n.s., channel_balance β=+0.159 ★')

    for scale in ['AMI_Total','AMI_Social','AMI_Behavioural','AMI_Emotional',
                  'DASS21_Stress','DASS21_Anxiety','DASS21_Depression']:
        sub = df[[f'{scale}_z','log_ratio_z','log_sum_z']].dropna()
        s, fit = fit_t(f'{scale}_z ~ log_ratio_z + log_sum_z', sub)
        print(f'\n  {scale}_z:')
        for term in ['log_ratio_z','log_sum_z']:
            r = s.loc[term]
            samples = fit.posterior[term].values.flatten()
            p_dir = (samples>0).mean() if r['mean']>0 else (samples<0).mean()
            flag = '★' if surv(r) else ' '
            print(f'    {term:<22} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(dir)={p_dir:.3f}  {flag}')

    # ========================================================
    # Test 3: Direct comparison — behavioral vs model in same model
    # ========================================================
    print('\n' + '='*72)
    print('TEST 3: clinical ~ behavioral + model (do both survive?)')
    print('='*72)
    for scale in ['AMI_Total','AMI_Social','AMI_Behavioural']:
        sub = df[[f'{scale}_z','total_mod_behav_z','channel_balance_behav_z','log_ratio_z','log_sum_z']].dropna()
        s, fit = fit_t(f'{scale}_z ~ total_mod_behav_z + channel_balance_behav_z + log_ratio_z + log_sum_z', sub)
        print(f'\n  {scale}_z:')
        for term in ['total_mod_behav_z','channel_balance_behav_z','log_ratio_z','log_sum_z']:
            r = s.loc[term]
            flag = '★' if surv(r) else ' '
            print(f'    {term:<28} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')


if __name__ == '__main__':
    main()
