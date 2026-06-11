"""
Attempts to recover the F3 → both-parameters finding using raw-scale specifications:

1. Multivariate kitchen-sink: log_ω/log_κ ~ all clinical scales (Student-t).
   Suppression effects (like DASS_Stress in §4.63) might emerge.

2. Joint with AMI_Social control: log_ω/log_κ ~ anxiety_scale + AMI_Social.
   Adjusting for apathy might reveal hidden anxiety signal.

3. Focused F3-item subset: take the 8 STAI items that loaded |≥0.6| on F3,
   sum them as a "refined trait NA" subscale, test against parameters.

4. Anxiety PC1 (the actual rotation-free anxiety axis): take PC1 of corrected
   STAI + DASS_Anx + OASIS + STICSA item sums. Test against parameters.

5. Sample-split robustness: does the effect (if any) appear in BOTH exp and conf?
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO); sys.path.insert(0, str(REPO/'notebooks'/'analysis'))
from load_data import load_both

ITEM_PATHS = {
    'exploratory':  REPO/'data/exploratory_350/processed/stage4_mental_health_20260403_133425/mental_health_items_wide.csv',
    'confirmatory': REPO/'data/confirmatory_350/processed/stage4_mental_health_20260403_142413/mental_health_items_wide.csv',
}
BKW = dict(draws=1000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)


def fit_t(formula, data):
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    return az.summary(fit, hdi_prob=0.95)


def surv(r):
    return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)


def main():
    # Load items + reverse-code STAI
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

    # PC1-sign reverse-code STAI
    stai_cols = [c for c in items.columns if c.startswith('STAI_item_')]
    sub = items[stai_cols].dropna()
    X = (sub.values - sub.values.mean(0)) / sub.values.std(0, ddof=1)
    pc1_loadings = PCA(n_components=1).fit(X).components_[0]
    neg_idx = np.where(pc1_loadings < 0)[0]
    max_val = sub.values.max()
    rev_cols = [stai_cols[i] for i in neg_idx]
    items[rev_cols] = max_val - items[rev_cols]

    items['STAI_Trait_corrected'] = items[stai_cols].sum(axis=1)

    # Compute F3-focused subset (top-loading STAI items on 3-factor F3)
    # From prior output: STAI items 0, 1, 4, 7, 10, 14, 15, 19 had |loading|≥0.63 on F3
    f3_subset = [f'STAI_item_{i}' for i in [0, 1, 4, 7, 10, 14, 15, 19]]
    items['STAI_F3_subset'] = items[f3_subset].sum(axis=1)

    # Load params
    exp, conf = load_both()
    pm = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa']); m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        pm.append(m[['subj','sample','omega','kappa','log_omega','log_kappa',
                     'DASS21_Anxiety', 'DASS21_Stress', 'DASS21_Depression',
                     'OASIS_Total', 'STICSA_Total', 'AMI_Social',
                     'AMI_Behavioural', 'AMI_Emotional', 'MFIS_Total']])
    params = pd.concat(pm, ignore_index=True)

    df = items.merge(params, on=['subj','sample'], how='inner')
    print(f'N = {len(df)}')

    # Within-sample z
    scales = ['STAI_Trait_corrected', 'STAI_F3_subset',
              'DASS21_Anxiety', 'DASS21_Stress', 'DASS21_Depression',
              'OASIS_Total', 'STICSA_Total', 'AMI_Social',
              'AMI_Behavioural', 'AMI_Emotional', 'MFIS_Total',
              'log_omega', 'log_kappa']
    for c in scales:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            if x.std() > 0:
                df.loc[mask, f'{c}_z'] = (x - x.mean())/x.std()

    # ===== Attempt 1: Multivariate kitchen-sink =====
    print('\n' + '='*72)
    print('Attempt 1: Multivariate kitchen-sink (all raw scales)')
    print('='*72)
    predictors = ['STAI_Trait_corrected_z', 'DASS21_Anxiety_z',
                  'DASS21_Stress_z', 'DASS21_Depression_z',
                  'OASIS_Total_z', 'STICSA_Total_z',
                  'AMI_Social_z', 'AMI_Behavioural_z', 'AMI_Emotional_z',
                  'MFIS_Total_z']
    formula_o = 'log_omega_z ~ ' + ' + '.join(predictors)
    formula_k = 'log_kappa_z ~ ' + ' + '.join(predictors)
    sub = df[['log_omega_z','log_kappa_z'] + predictors].dropna()
    s_o = fit_t(formula_o, sub)
    s_k = fit_t(formula_k, sub)
    print(f'  {"predictor":<26} {"β(log ω)":>16} {"β(log κ)":>16}')
    for p in predictors:
        ro = s_o.loc[p]; rk = s_k.loc[p]
        fo = '★' if surv(ro) else ' '
        fk = '★' if surv(rk) else ' '
        print(f'  {p:<26} {ro["mean"]:+8.3f} [{ro["hdi_2.5%"]:+.2f},{ro["hdi_97.5%"]:+.2f}]{fo}  '
              f'{rk["mean"]:+8.3f} [{rk["hdi_2.5%"]:+.2f},{rk["hdi_97.5%"]:+.2f}]{fk}')

    # ===== Attempt 2: Joint anxiety + apathy control =====
    print('\n' + '='*72)
    print('Attempt 2: Each anxiety scale + AMI_Social control')
    print('='*72)
    for anx in ['STAI_Trait_corrected_z', 'DASS21_Anxiety_z',
                'OASIS_Total_z', 'STICSA_Total_z']:
        sub2 = df[['log_omega_z','log_kappa_z', anx, 'AMI_Social_z']].dropna()
        s_o = fit_t(f'log_omega_z ~ {anx} + AMI_Social_z', sub2)
        s_k = fit_t(f'log_kappa_z ~ {anx} + AMI_Social_z', sub2)
        ro = s_o.loc[anx]; rk = s_k.loc[anx]
        fo = '★' if surv(ro) else ' '
        fk = '★' if surv(rk) else ' '
        print(f'  {anx:<26} → log_ω: β={ro["mean"]:+.3f} [{ro["hdi_2.5%"]:+.3f}, {ro["hdi_97.5%"]:+.3f}] {fo} | '
              f'log_κ: β={rk["mean"]:+.3f} [{rk["hdi_2.5%"]:+.3f}, {rk["hdi_97.5%"]:+.3f}] {fk}')

    # ===== Attempt 3: F3-focused STAI subset =====
    print('\n' + '='*72)
    print('Attempt 3: Top-F3-loading STAI subset (items 0,1,4,7,10,14,15,19)')
    print('='*72)
    sub = df[['log_omega_z','log_kappa_z','STAI_F3_subset_z']].dropna()
    s_o = fit_t('log_omega_z ~ STAI_F3_subset_z', sub)
    s_k = fit_t('log_kappa_z ~ STAI_F3_subset_z', sub)
    ro = s_o.loc['STAI_F3_subset_z']; rk = s_k.loc['STAI_F3_subset_z']
    fo = '★' if surv(ro) else ' '; fk = '★' if surv(rk) else ' '
    print(f'  STAI_F3_subset → log_ω: β={ro["mean"]:+.3f} [{ro["hdi_2.5%"]:+.3f}, {ro["hdi_97.5%"]:+.3f}] {fo}')
    print(f'  STAI_F3_subset → log_κ: β={rk["mean"]:+.3f} [{rk["hdi_2.5%"]:+.3f}, {rk["hdi_97.5%"]:+.3f}] {fk}')

    # ===== Attempt 4: PC1 of anxiety sums (rotation-free anxiety axis) =====
    print('\n' + '='*72)
    print('Attempt 4: PC1 of (STAI_corr + DASS_Anx + OASIS + STICSA) sums')
    print('='*72)
    anx_subscales = df[['STAI_Trait_corrected_z','DASS21_Anxiety_z','OASIS_Total_z','STICSA_Total_z']].dropna()
    pc = PCA(n_components=1).fit(anx_subscales.values)
    anx_pc1 = pc.transform(anx_subscales.values).flatten()
    print(f'  PC1 explained variance: {pc.explained_variance_ratio_[0]*100:.1f}%')
    print(f'  PC1 loadings: {dict(zip(anx_subscales.columns, pc.components_[0].round(2)))}')
    df_pc = df.loc[anx_subscales.index].copy()
    df_pc['ANX_PC1'] = anx_pc1
    df_pc['ANX_PC1_z'] = np.nan
    for s in df_pc['sample'].unique():
        mask = df_pc['sample'] == s
        x = df_pc.loc[mask, 'ANX_PC1']
        df_pc.loc[mask, 'ANX_PC1_z'] = (x - x.mean())/x.std()
    sub = df_pc[['log_omega_z','log_kappa_z','ANX_PC1_z']].dropna()
    s_o = fit_t('log_omega_z ~ ANX_PC1_z', sub)
    s_k = fit_t('log_kappa_z ~ ANX_PC1_z', sub)
    ro = s_o.loc['ANX_PC1_z']; rk = s_k.loc['ANX_PC1_z']
    fo = '★' if surv(ro) else ' '; fk = '★' if surv(rk) else ' '
    print(f'  ANX_PC1 → log_ω: β={ro["mean"]:+.3f} [{ro["hdi_2.5%"]:+.3f}, {ro["hdi_97.5%"]:+.3f}] {fo}')
    print(f'  ANX_PC1 → log_κ: β={rk["mean"]:+.3f} [{rk["hdi_2.5%"]:+.3f}, {rk["hdi_97.5%"]:+.3f}] {fk}')

    # ===== Attempt 5: Sample-split robustness for STAI_corrected =====
    print('\n' + '='*72)
    print('Attempt 5: Sample-split — STAI_corrected → params in EACH sample alone')
    print('='*72)
    for s in ['exploratory', 'confirmatory']:
        ss = df[df['sample']==s].copy()
        # re-z within this sample only
        for c in ['STAI_Trait_corrected','log_omega','log_kappa']:
            ss[f'{c}_z'] = (ss[c] - ss[c].mean())/ss[c].std()
        sub = ss[['log_omega_z','log_kappa_z','STAI_Trait_corrected_z']].dropna()
        s_o = fit_t('log_omega_z ~ STAI_Trait_corrected_z', sub)
        s_k = fit_t('log_kappa_z ~ STAI_Trait_corrected_z', sub)
        ro = s_o.loc['STAI_Trait_corrected_z']; rk = s_k.loc['STAI_Trait_corrected_z']
        fo = '★' if surv(ro) else ' '; fk = '★' if surv(rk) else ' '
        print(f'  {s:<15} (N={len(sub)}): '
              f'log_ω β={ro["mean"]:+.3f} [{ro["hdi_2.5%"]:+.3f}, {ro["hdi_97.5%"]:+.3f}] {fo}  '
              f'log_κ β={rk["mean"]:+.3f} [{rk["hdi_2.5%"]:+.3f}, {rk["hdi_97.5%"]:+.3f}] {fk}')

    print('\n' + '='*72)
    print('VERDICT')
    print('='*72)
    print('  Any anxiety scale showing parallel effect on both ω and κ that we could report?')
    print('  → Check the table above. If no ★ on both for any anxiety scale,')
    print('    the F3 finding is methodologically irrecoverable from raw data.')


if __name__ == '__main__':
    main()
