"""
Clean rerun:
  1. Load item-level data, apply PC1-sign reverse-coding to STAI (11 items)
     and any AMI items that need it (1 item from prior audit).
  2. Recompute STAI_Trait from corrected items.
  3. Save corrected master psych dataframe (per-sample) for future use.
  4. Within-sample z-score every clinical scale in the corrected set.
  5. Re-run the headline analyses:
     - AMI_Social uniqueness (joint with all other clinical scales)
     - Kitchen-sink ω with log(κ) control
     - Kitchen-sink κ with log(ω) control
     - All using *corrected* STAI_Trait

Output files:
  results/stats/affect_analysis/clinical_scores_corrected_{exp,conf}.csv
  results/stats/affect_analysis/headline_corrected_results.csv
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
QUESTIONNAIRES = ['STAI', 'DASS21', 'MFIS', 'STICSA', 'AMI', 'OASIS', 'PHQ9']
BKW = dict(draws=1000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)


# -------------- Preprocessing --------------

def reverse_code_items(items_df, q):
    """PC1-sign reverse-coding. Returns updated items_df + which items were flipped."""
    qcols = [c for c in items_df.columns if c.startswith(q+'_item_')]
    if len(qcols) < 4:
        return items_df, []
    sub = items_df[qcols].dropna()
    if len(sub) < 10:
        return items_df, []
    cm_pre = sub.corr().values
    offdiag = cm_pre[np.triu_indices_from(cm_pre, k=1)]
    neg_pre = (offdiag < 0).mean()*100
    if neg_pre < 5:
        return items_df, []
    X = (sub.values - sub.values.mean(0)) / sub.values.std(0, ddof=1)
    loadings = PCA(n_components=1).fit(X).components_[0]
    neg_idx = np.where(loadings < 0)[0]
    if len(neg_idx) == 0:
        return items_df, []
    max_val = sub.values.max()
    rev_cols = [qcols[i] for i in neg_idx]
    items_df = items_df.copy()
    items_df[rev_cols] = max_val - items_df[rev_cols]
    return items_df, neg_idx.tolist()


def recompute_clinical_scores():
    """Reload items, reverse-code, save corrected per-sample scores."""
    rows = []
    flips_per_q = {}
    for sample, p in ITEM_PATHS.items():
        df = pd.read_csv(p)
        df['sample'] = sample
        # Map participantID → subj
        sd = REPO/f'data/{sample}_350/processed'
        cand = list(sd.glob('stage5_filtered_data_*/subject_mapping.csv'))
        if cand:
            mp = pd.read_csv(cand[0])
            df['subj'] = df['participantID'].map(dict(zip(mp['participantID'], mp['subj'])))
        rows.append(df)
    items = pd.concat(rows, ignore_index=True)

    # Apply PC1-sign reverse-coding per questionnaire on POOLED items
    # (within-sample reverse-coding could miss the structure)
    for q in QUESTIONNAIRES:
        items, flips = reverse_code_items(items, q)
        if flips:
            flips_per_q[q] = flips
            print(f'  Reverse-keyed {q}: {len(flips)} items {flips}')
        else:
            print(f'  {q}: no reverse-keying needed')

    # Recompute STAI_Trait (only correction we trust — AMI subscale scoring is
    # ambiguous across sources, so we keep the original psych.csv AMI subscales
    # and only update STAI).
    stai_cols = [c for c in items.columns if c.startswith('STAI_item_')]
    items['STAI_Trait_corrected'] = items[stai_cols].sum(axis=1)

    # Save corrected STAI per sample
    for sample, p in ITEM_PATHS.items():
        sample_data = items[items['sample'] == sample].copy()
        out_cols = ['subj', 'participantID', 'sample', 'STAI_Trait_corrected']
        out_path = REPO/f'results/stats/affect_analysis/clinical_scores_corrected_{sample[:3]}.csv'
        sample_data[out_cols].to_csv(out_path, index=False)
        print(f'  Saved: {out_path}')

    return items, flips_per_q


# -------------- Modeling --------------

def fit_t(formula, data):
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    return az.summary(fit, hdi_prob=0.95)


def surv(r):
    return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)


def main():
    print('='*72); print('STEP 1 — Reverse-code items & recompute corrected scores'); print('='*72)
    items, flips = recompute_clinical_scores()

    # Print correction summary
    print('\n  Correction summary:')
    print(f'    STAI: original 20 items → flipped {len(flips.get("STAI",[]))} items')
    print(f'    AMI:  original 18 items → flipped {len(flips.get("AMI",[]))} items')
    print(f'    Others: no corrections needed')

    # Compare original vs corrected STAI
    if 'STAI' in flips:
        orig_stai_cols = [c for c in items.columns if c.startswith('STAI_item_')]
        # Re-load original items (without reverse-coding) just for comparison
        orig_rows = []
        for sample, p in ITEM_PATHS.items():
            d = pd.read_csv(p)
            d['STAI_orig_sum'] = d[orig_stai_cols].sum(axis=1)
            d['sample'] = sample
            orig_rows.append(d[['participantID', 'sample', 'STAI_orig_sum']])
        orig = pd.concat(orig_rows, ignore_index=True)
        merged = items.merge(orig, on=['participantID', 'sample'])
        r = np.corrcoef(merged['STAI_Trait_corrected'], merged['STAI_orig_sum'])[0,1]
        print(f'\n  Correlation between original (broken) and corrected STAI_Trait: r = {r:+.3f}')
        print(f'    (low correlation = the bug was serious; the two scales measure different things)')

    print('\n' + '='*72); print('STEP 2 — Load params + within-sample z everything'); print('='*72)
    exp, conf = load_both()
    pm_rows = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa'])
        m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        pm_rows.append(m[['subj','sample','log_omega','log_kappa',
                          'AMI_Social','AMI_Behavioural','AMI_Emotional','AMI_Total',
                          'DASS21_Anxiety', 'DASS21_Stress', 'DASS21_Depression',
                          'OASIS_Total', 'STICSA_Total', 'MFIS_Total', 'PHQ9_Total']])
    params = pd.concat(pm_rows, ignore_index=True)

    # Merge only the corrected STAI; keep original AMI subscales from psych.csv
    cscores = items[['subj','sample','STAI_Trait_corrected']]
    df = params.merge(cscores, on=['subj','sample'], how='inner')
    print(f'  N joined = {len(df)}')

    # Within-sample z-score all scales
    all_scales = ['log_omega', 'log_kappa',
                  'STAI_Trait_corrected',
                  'AMI_Social','AMI_Behavioural','AMI_Emotional','AMI_Total',
                  'DASS21_Anxiety', 'DASS21_Stress', 'DASS21_Depression',
                  'OASIS_Total', 'STICSA_Total', 'MFIS_Total', 'PHQ9_Total']
    for c in all_scales:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            if x.std() > 0:
                df.loc[mask, f'{c}_z'] = (x - x.mean())/x.std()
    print(f'  Within-sample z-scored {len(all_scales)} variables')

    results = []

    # --- AMI_Social univariate (sanity check; uses ORIGINAL psych.csv AMI_Social)
    print('\n' + '='*72); print('STEP 3 — Re-run headline analyses with corrected STAI'); print('='*72)
    print('\n--- A. AMI_Social → log(ω) univariate (sanity check) ---')
    sub = df[['log_omega_z','AMI_Social_z']].dropna()
    s = fit_t('log_omega_z ~ AMI_Social_z', sub)
    r = s.loc['AMI_Social_z']
    print(f'  β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  '
          f'{"★" if surv(r) else ""}  (should match prior: +0.122 ★)')
    results.append({'analysis':'A', 'term':'AMI_Social', 'outcome':'log_omega',
                    'mean':float(r['mean']), 'hdi_lo':float(r['hdi_2.5%']),
                    'hdi_hi':float(r['hdi_97.5%']), 'survives':bool(surv(r))})

    # --- Kitchen sink with corrected STAI but original AMI subscales
    print('\n--- B. Kitchen-sink ω with corrected STAI + original AMI subscales + κ control ---')
    all_scales_z = ['AMI_Social_z', 'AMI_Behavioural_z', 'AMI_Emotional_z',
                    'DASS21_Anxiety_z','DASS21_Stress_z','DASS21_Depression_z',
                    'OASIS_Total_z','STICSA_Total_z','STAI_Trait_corrected_z','MFIS_Total_z','PHQ9_Total_z']
    sub = df[['log_omega_z','log_kappa_z'] + all_scales_z].dropna()
    formula = 'log_omega_z ~ ' + ' + '.join(all_scales_z) + ' + log_kappa_z'
    s = fit_t(formula, sub)
    print(f'  Kitchen-sink ω model (Student-t, N={len(sub)}):')
    for term in all_scales_z + ['log_kappa_z']:
        r = s.loc[term]
        flag = '★' if surv(r) else ' '
        print(f'  {term:<32} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
        results.append({'analysis':'B', 'term':term, 'outcome':'log_omega',
                        'mean':float(r['mean']), 'hdi_lo':float(r['hdi_2.5%']),
                        'hdi_hi':float(r['hdi_97.5%']), 'survives':bool(surv(r))})

    # --- κ kitchen sink with corrected scales — verify the STAI → κ finding
    print('\n--- C. κ kitchen-sink with corrected STAI_Trait ---')
    formula = 'log_kappa_z ~ ' + ' + '.join(all_scales_z) + ' + log_omega_z'
    s = fit_t(formula, sub)
    print(f'  Kitchen-sink κ model (Student-t, N={len(sub)}):')
    for term in all_scales_z + ['log_omega_z']:
        r = s.loc[term]
        flag = '★' if surv(r) else ' '
        print(f'  {term:<32} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
        results.append({'analysis':'C', 'term':term, 'outcome':'log_kappa',
                        'mean':float(r['mean']), 'hdi_lo':float(r['hdi_2.5%']),
                        'hdi_hi':float(r['hdi_97.5%']), 'survives':bool(surv(r))})

    # Save
    out = REPO/'results/stats/affect_analysis/headline_corrected_results.csv'
    pd.DataFrame(results).to_csv(out, index=False)
    print(f'\nSaved: {out}')

    print('\n' + '='*72); print('SUMMARY — surviving effects from corrected analyses'); print('='*72)
    for r in results:
        if r['survives']:
            print(f"  [{r['analysis']}] {r['term']} → {r['outcome']}: "
                  f"β={r['mean']:+.3f}  HDI [{r['hdi_lo']:+.3f}, {r['hdi_hi']:+.3f}]")


if __name__ == '__main__':
    main()
