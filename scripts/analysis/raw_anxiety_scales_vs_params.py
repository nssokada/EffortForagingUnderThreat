"""
Sanity check: does the trait-anxiety → BOTH parameters finding (F3 in EFA)
appear when using raw clinical scales?

Tests each anxiety/depression scale (with corrected STAI reverse-coding) on
log_omega and log_kappa separately. The F3 finding predicts that STAI_Trait
(corrected) and possibly DASS_Anx, OASIS, STICSA should ALL show parallel
positive coefficients on both ω and κ.
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


def main():
    # Load item-level data
    rows = []
    for sample, p in ITEM_PATHS.items():
        df = pd.read_csv(p); df['sample'] = sample
        rows.append(df)
    items = pd.concat(rows, ignore_index=True)

    # Subject mapping
    for sample, _ in ITEM_PATHS.items():
        sd = REPO/f'data/{sample}_350/processed'
        cand = list(sd.glob('stage5_filtered_data_*/subject_mapping.csv'))
        if cand:
            mp = pd.read_csv(cand[0])
            mask = items['sample'] == sample
            items.loc[mask, 'subj'] = items.loc[mask, 'participantID'].map(
                dict(zip(mp['participantID'], mp['subj'])))

    # Apply PC1-sign reverse-coding to STAI (same as EFA pipeline)
    qcols = [c for c in items.columns if c.startswith('STAI_item_')]
    sub = items[qcols].dropna()
    X = (sub.values - sub.values.mean(0)) / sub.values.std(0, ddof=1)
    pc1_loadings = PCA(n_components=1).fit(X).components_[0]
    neg_idx = np.where(pc1_loadings < 0)[0]
    max_val = sub.values.max()
    rev_cols = [qcols[i] for i in neg_idx]
    print(f'Reverse-keying {len(neg_idx)}/{len(qcols)} STAI items: '
          f'{[int(c.split("_")[-1]) for c in rev_cols]}')
    items_corr = items.copy()
    items_corr[rev_cols] = max_val - items_corr[rev_cols]

    # Compute scale sums (corrected and original for comparison)
    items_corr['STAI_Trait_corrected'] = items_corr[qcols].sum(axis=1)
    items['STAI_Trait_original'] = items[qcols].sum(axis=1)

    # Load model params
    exp, conf = load_both()
    pm = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa']); m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        pm.append(m[['subj','sample','omega','kappa','log_omega','log_kappa',
                     'DASS21_Anxiety', 'DASS21_Stress', 'DASS21_Depression',
                     'OASIS_Total', 'STICSA_Total', 'AMI_Social']])
    params = pd.concat(pm, ignore_index=True)

    # Join
    df = items_corr.merge(params, on=['subj','sample'], how='inner')
    df['STAI_Trait_original'] = items[['participantID','sample']].merge(
        items[['STAI_Trait_original','participantID','sample']],
        on=['participantID','sample']).set_index(df.index)['STAI_Trait_original'] if False else df.index.map(
            dict(zip(items.index, items['STAI_Trait_original'])))
    # Simpler: just compute again
    df['STAI_Trait_original'] = df[qcols].apply(
        lambda r: items[items['participantID']==df.loc[r.name,'participantID']][qcols].sum(axis=1).iloc[0]
        if r.name in df.index else np.nan, axis=1) if False else None
    df = df.drop(columns=['STAI_Trait_original'])
    # Add original sum cleanly
    orig_sums = items.set_index('participantID')[qcols].sum(axis=1).to_dict()
    df['STAI_Trait_original'] = df['participantID'].map(orig_sums)

    print(f'\nN with both items + params = {len(df)}')

    # Within-sample z-score everything
    scales = ['STAI_Trait_corrected', 'STAI_Trait_original',
              'DASS21_Anxiety', 'DASS21_Stress', 'DASS21_Depression',
              'OASIS_Total', 'STICSA_Total', 'AMI_Social',
              'log_omega', 'log_kappa']
    for c in scales:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            if x.std() > 0:
                df.loc[mask, f'{c}_z'] = (x - x.mean())/x.std()

    # Run univariate Student-t Bayesian regressions
    def fit_t(formula, data):
        m = bmb.Model(formula, data=data, family='t')
        fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
        return az.summary(fit, hdi_prob=0.95)

    def surv(r):
        return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)

    print('\n' + '='*72)
    print('Raw scales vs log(ω) and log(κ), univariate Student-t')
    print('='*72)
    print(f'{"scale":<26} {"β(log_ω)":>16}  {"β(log_κ)":>16}')

    results = []
    for sc in ['STAI_Trait_corrected', 'STAI_Trait_original',
               'DASS21_Anxiety', 'DASS21_Stress', 'DASS21_Depression',
               'OASIS_Total', 'STICSA_Total', 'AMI_Social']:
        sub = df[[f'log_omega_z', f'log_kappa_z', f'{sc}_z']].dropna()
        s_o = fit_t(f'log_omega_z ~ {sc}_z', sub)
        s_k = fit_t(f'log_kappa_z ~ {sc}_z', sub)
        ro = s_o.loc[f'{sc}_z']; rk = s_k.loc[f'{sc}_z']
        fo = '★' if surv(ro) else ' '
        fk = '★' if surv(rk) else ' '
        print(f'{sc:<26} {ro["mean"]:+8.3f} [{ro["hdi_2.5%"]:+.2f},{ro["hdi_97.5%"]:+.2f}]{fo}  '
              f'{rk["mean"]:+8.3f} [{rk["hdi_2.5%"]:+.2f},{rk["hdi_97.5%"]:+.2f}]{fk}')
        results.append({
            'scale': sc, 'N': len(sub),
            'beta_omega': float(ro['mean']),
            'hdi_lo_omega': float(ro['hdi_2.5%']),
            'hdi_hi_omega': float(ro['hdi_97.5%']),
            'surv_omega': bool(surv(ro)),
            'beta_kappa': float(rk['mean']),
            'hdi_lo_kappa': float(rk['hdi_2.5%']),
            'hdi_hi_kappa': float(rk['hdi_97.5%']),
            'surv_kappa': bool(surv(rk)),
        })

    # Sanity check: F3 reproduction (composite of corrected STAI + DASS_Anx + OASIS + STICSA)
    print('\n' + '='*72)
    print('Anxiety COMPOSITE (z-mean of corrected STAI + DASS_Anx + OASIS + STICSA)')
    print('='*72)
    df['ANX_composite'] = df[['STAI_Trait_corrected_z', 'DASS21_Anxiety_z',
                              'OASIS_Total_z', 'STICSA_Total_z']].mean(axis=1)
    # Re-z within sample
    df['ANX_composite_z'] = np.nan
    for s in df['sample'].unique():
        mask = df['sample'] == s
        x = df.loc[mask, 'ANX_composite']
        df.loc[mask, 'ANX_composite_z'] = (x - x.mean())/x.std()

    sub = df[['log_omega_z','log_kappa_z','ANX_composite_z']].dropna()
    s_o = fit_t('log_omega_z ~ ANX_composite_z', sub)
    s_k = fit_t('log_kappa_z ~ ANX_composite_z', sub)
    ro = s_o.loc['ANX_composite_z']; rk = s_k.loc['ANX_composite_z']
    fo = '★' if surv(ro) else ' '
    fk = '★' if surv(rk) else ' '
    print(f'  ANX_composite (4 scales) → log_ω:  β={ro["mean"]:+.3f} [{ro["hdi_2.5%"]:+.3f}, {ro["hdi_97.5%"]:+.3f}] {fo}')
    print(f'  ANX_composite (4 scales) → log_κ:  β={rk["mean"]:+.3f} [{rk["hdi_2.5%"]:+.3f}, {rk["hdi_97.5%"]:+.3f}] {fk}')
    results.append({
        'scale': 'ANX_composite', 'N': len(sub),
        'beta_omega': float(ro['mean']), 'hdi_lo_omega': float(ro['hdi_2.5%']),
        'hdi_hi_omega': float(ro['hdi_97.5%']), 'surv_omega': bool(surv(ro)),
        'beta_kappa': float(rk['mean']), 'hdi_lo_kappa': float(rk['hdi_2.5%']),
        'hdi_hi_kappa': float(rk['hdi_97.5%']), 'surv_kappa': bool(surv(rk)),
    })

    out = REPO / 'results/stats/affect_analysis/raw_anxiety_vs_params.csv'
    pd.DataFrame(results).to_csv(out, index=False)
    print(f'\nSaved: {out}')

    # Summary
    print('\n' + '='*72)
    print('VERDICT')
    print('='*72)
    parallel_signals = [r for r in results if r['surv_omega'] and r['surv_kappa']
                        and np.sign(r['beta_omega']) == np.sign(r['beta_kappa'])]
    if parallel_signals:
        print('  ✓ Parallel anxiety→both-parameters effect FOUND in raw scales:')
        for r in parallel_signals:
            print(f"    {r['scale']}: β(log_ω)={r['beta_omega']:+.3f}, β(log_κ)={r['beta_kappa']:+.3f}")
    else:
        print('  ✗ No raw scale shows the parallel both-parameters pattern.')
        print('  The F3 finding is then either:')
        print('   (a) Emergent — only the latent composite captures the effect')
        print('   (b) Spurious — a varimax-rotation artifact, not a real construct')


if __name__ == '__main__':
    main()
