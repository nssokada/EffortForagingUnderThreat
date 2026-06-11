"""
Fix STAI direction (PC1-sign approach gave the wrong overall direction)
and re-run all headline analyses with the properly-signed STAI.

Fix: after PC1-sign item-level reverse-coding, check correlation of the summed
STAI with DASS21_Anxiety (a known-direction anxiety scale). If negative, flip
the entire scale (subtract from max).

Then re-run:
  - AMI_Total + ANX+DEP composite kitchen-sink
  - Quadrant typology (AMI × ANX+DEP)
  - Pure Apathy vs Pure Distress contrast
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO); sys.path.insert(0, str(REPO/'notebooks'/'analysis'))
from load_data import load_both

ITEM_PATHS = {
    'exploratory':  REPO/'data/exploratory_350/processed/stage4_mental_health_20260403_133425/mental_health_items_wide.csv',
    'confirmatory': REPO/'data/confirmatory_350/processed/stage4_mental_health_20260403_142413/mental_health_items_wide.csv',
}
BKW = dict(draws=2000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)


def main():
    # Load items
    rows = []
    for sample, p in ITEM_PATHS.items():
        df = pd.read_csv(p); df['sample'] = sample
        rows.append(df)
    items = pd.concat(rows, ignore_index=True)
    for sample, _ in ITEM_PATHS.items():
        sd = REPO/f'data/{sample}_350/processed'
        cand = list(sd.glob('stage5_filtered_data_*/subject_mapping.csv'))
        if cand:
            mp = pd.read_csv(cand[0])
            mask = items['sample'] == sample
            items.loc[mask, 'subj'] = items.loc[mask, 'participantID'].map(
                dict(zip(mp['participantID'], mp['subj'])))

    # === STAI PROPER FIX ===
    stai_cols = [c for c in items.columns if c.startswith('STAI_item_')]
    sub = items[stai_cols].dropna()
    X = (sub.values - sub.values.mean(0)) / sub.values.std(0, ddof=1)
    pc1_loadings = PCA(n_components=1).fit(X).components_[0]
    neg_idx = np.where(pc1_loadings < 0)[0]
    max_val = sub.values.max()
    rev_cols = [stai_cols[i] for i in neg_idx]
    items[rev_cols] = max_val - items[rev_cols]
    items['STAI_pc1_sum'] = items[stai_cols].sum(axis=1)

    # Get DASS_Anxiety (need from psych.csv)
    psych_rows = []
    for sample, _ in ITEM_PATHS.items():
        sd = REPO/f'data/{sample}_350/processed'
        cand = list(sd.glob('stage5_filtered_data_*/psych.csv'))
        if cand:
            ps = pd.read_csv(cand[0])
            ps['sample'] = sample
            psych_rows.append(ps[['subj','sample','DASS21_Anxiety']])
    psych = pd.concat(psych_rows, ignore_index=True)
    merged = items.merge(psych, on=['subj','sample'], how='inner')

    # Anchor STAI direction to DASS_Anxiety (which we know measures anxiety positively)
    r_anchor, _ = pearsonr(merged['STAI_pc1_sum'], merged['DASS21_Anxiety'])
    print(f'STAI_pc1_sum vs DASS21_Anxiety: r = {r_anchor:+.3f}')
    if r_anchor < 0:
        print('  ⚠ NEGATIVE — flipping entire STAI scale')
        # max possible STAI sum = max_val * 20 items
        max_total = max_val * len(stai_cols)
        items['STAI_Trait_FIXED'] = max_total - items['STAI_pc1_sum']
    else:
        print('  ✓ POSITIVE — STAI already correctly signed')
        items['STAI_Trait_FIXED'] = items['STAI_pc1_sum']

    # Verify the fix
    merged = items.merge(psych, on=['subj','sample'], how='inner')
    for sc in ['DASS21_Anxiety']:
        r, _ = pearsonr(merged['STAI_Trait_FIXED'], merged[sc])
        print(f'  Post-fix: STAI_Trait_FIXED vs {sc}: r = {r:+.3f}')

    # Save corrected STAI
    out_rows = items[['subj','participantID','sample','STAI_Trait_FIXED']]
    for sample in ['exp','con']:
        sample_full = 'exploratory' if sample == 'exp' else 'confirmatory'
        sd = out_rows[out_rows['sample']==sample_full]
        sd.to_csv(REPO/f'results/stats/affect_analysis/stai_fixed_{sample}.csv', index=False)
    print(f'\n  Saved fixed STAI: results/stats/affect_analysis/stai_fixed_*.csv')

    # ============================================================
    # Now build the full analysis dataset
    # ============================================================
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

    df = params.merge(items[['subj','sample','STAI_Trait_FIXED']], on=['subj','sample'], how='inner')
    print(f'\nN = {len(df)}')

    # Verify the fix made it through
    print('\n--- STAI direction check after merge (should all be POSITIVE now) ---')
    for sc in ['DASS21_Anxiety','OASIS_Total','STICSA_Total']:
        r,_ = pearsonr(df['STAI_Trait_FIXED'], df[sc])
        verdict = '✓' if r > 0 else '✗ STILL WRONG'
        print(f'  STAI_Trait_FIXED vs {sc}: r = {r:+.3f}  {verdict}')
    r,_ = pearsonr(df['STAI_Trait_FIXED'], df['AMI_Total'])
    print(f'  STAI_Trait_FIXED vs AMI_Total: r = {r:+.3f}  (apathy-anxiety; positive expected)')

    # Within-sample z
    cols = ['log_omega','log_kappa','AMI_Total','AMI_Social',
            'DASS21_Anxiety','DASS21_Depression','DASS21_Stress',
            'STAI_Trait_FIXED','OASIS_Total','STICSA_Total','PHQ9_Total']
    for c in cols:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample']==s
            x = df.loc[mask,c]
            if x.std()>0:
                df.loc[mask,f'{c}_z'] = (x-x.mean())/x.std()

    anxdep = ['DASS21_Anxiety_z','DASS21_Depression_z','DASS21_Stress_z',
              'STAI_Trait_FIXED_z','OASIS_Total_z','STICSA_Total_z','PHQ9_Total_z']
    df['ANXDEP_FIXED'] = df[anxdep].mean(axis=1)
    df['ANXDEP_FIXED_z'] = np.nan
    for s in df['sample'].unique():
        mask = df['sample']==s
        x = df.loc[mask,'ANXDEP_FIXED']
        df.loc[mask,'ANXDEP_FIXED_z'] = (x-x.mean())/x.std()

    # ============================================================
    # Re-run main analyses with FIXED STAI
    # ============================================================
    def fit_t(formula, data):
        m = bmb.Model(formula, data=data, family='t')
        fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
        return az.summary(fit, hdi_prob=0.95), fit

    def surv(r):
        return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)

    print('\n' + '='*72)
    print('A. AMI_Total + ANX+DEP_FIXED + κ (pooled N=571, the headline)')
    print('='*72)
    sub = df[['log_omega_z','log_kappa_z','AMI_Total_z','ANXDEP_FIXED_z']].dropna()
    s, fit = fit_t('log_omega_z ~ AMI_Total_z + ANXDEP_FIXED_z + log_kappa_z', sub)
    for t in ['AMI_Total_z','ANXDEP_FIXED_z','log_kappa_z']:
        r = s.loc[t]; samples = fit.posterior[t].values.flatten()
        p_dir = (samples>0).mean() if r['mean']>0 else (samples<0).mean()
        flag = '★' if surv(r) else ' '
        print(f'  {t:<22} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(dir)={p_dir:.3f}  {flag}')

    print('\nCompare to broken-STAI version (§4.75):')
    print('  AMI_Total_z          β=+0.134 ★')
    print('  ANXDEP_composite_z   β=-0.092 ★ (broken STAI averaged in opposite direction)')

    # Per-sample
    print('\n' + '='*72)
    print('B. Per-sample replication with FIXED STAI')
    print('='*72)
    for sample in ['exploratory', 'confirmatory']:
        sub = df[df['sample']==sample][['log_omega_z','log_kappa_z','AMI_Total_z','ANXDEP_FIXED_z']].dropna()
        s, fit = fit_t('log_omega_z ~ AMI_Total_z + ANXDEP_FIXED_z + log_kappa_z', sub)
        print(f'\n  [{sample}, N={len(sub)}]')
        for t in ['AMI_Total_z','ANXDEP_FIXED_z']:
            r = s.loc[t]; samples = fit.posterior[t].values.flatten()
            p_dir = (samples>0).mean() if r['mean']>0 else (samples<0).mean()
            flag = '★' if surv(r) else ' '
            print(f'    {t:<22} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(dir)={p_dir:.3f}  {flag}')

    # Quadrants with FIXED STAI
    print('\n' + '='*72)
    print('C. Quadrant typology with FIXED STAI')
    print('='*72)
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
    print('Profile cell sizes:')
    print(df['profile'].value_counts().sort_index())
    print('\nMean log(ω)_z by profile:')
    g = df.groupby('profile').agg(n=('subj','count'),
        log_omega_z_mean=('log_omega_z','mean'),
        log_omega_z_std=('log_omega_z','std'))
    g['SE'] = g['log_omega_z_std']/np.sqrt(g['n'])
    print(g[['n','log_omega_z_mean','SE']].round(3).to_string())

    # PureApathy vs PureDistress contrast
    sub_AB = df[df['profile'].isin(['1_PureApathy','2_PureDistress'])].copy()
    sub_AB['is_apathy'] = (sub_AB['profile']=='1_PureApathy').astype(int)
    s, fit = fit_t('log_omega_z ~ is_apathy + log_kappa_z', sub_AB)
    r = s.loc['is_apathy']
    samples = fit.posterior['is_apathy'].values.flatten()
    p_pos = (samples>0).mean()
    flag = '★' if surv(r) else ' '
    print(f'\n  KEY CONTRAST: PureApathy − PureDistress: β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  P(β>0)={p_pos:.3f}  {flag}')


if __name__ == '__main__':
    main()
