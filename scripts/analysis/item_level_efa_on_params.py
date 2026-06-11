"""
Item-level EFA on transdiagnostic psychiatric items, then test factor scores
against model parameters (omega, kappa).

Phase A — Data prep:
  - Load item-level wide CSVs from both samples, pool
  - Drop PHQ9 (user instruction)
  - Join with (omega, kappa)
  - Audit STAI reverse-coding (mean inter-item correlation should be positive
    and comparable to other questionnaires; if not, reverse-keying is needed)
  - Item quality screen (drop near-zero variance, high missing)

Phase B — Factor structure:
  - Horn's parallel analysis to pick n_factors
  - EFA with varimax and oblimin rotation at chosen n
  - Save loadings + factor scores

Phase C — Relate factors to (omega, kappa):
  - Joint Student-t Bayesian: log_omega_z ~ each factor_z (univariate)
  - Multivariate: log_omega_z ~ all factors + log_kappa_z (control)
  - log_kappa_z ~ all factors + log_omega_z (control)
  - Compare to baseline AMI_Social finding

Outputs:
  results/stats/affect_analysis/item_efa_loadings.csv
  results/stats/affect_analysis/item_efa_subject_scores.csv
  results/stats/affect_analysis/item_efa_param_regressions.csv
  results/stats/affect_analysis/item_efa_parallel_analysis.csv
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
from sklearn.decomposition import FactorAnalysis, PCA
from scipy.stats import bartlett, chi2
warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO); sys.path.insert(0, str(REPO/'notebooks'/'analysis'))
from load_data import load_both

ITEM_PATHS = {
    'exploratory':  REPO/'data/exploratory_350/processed/stage4_mental_health_20260403_133425/mental_health_items_wide.csv',
    'confirmatory': REPO/'data/confirmatory_350/processed/stage4_mental_health_20260403_142413/mental_health_items_wide.csv',
}
QUESTIONNAIRES = ['STAI', 'DASS21', 'MFIS', 'STICSA', 'AMI', 'OASIS']  # PHQ9 excluded per user
BKW = dict(draws=1000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)


# ============================================================================
# Phase A
# ============================================================================

def load_items_and_params():
    """Load pooled item data + model parameters. Return joined per-subject DF."""
    # Items: pool both samples
    rows = []
    for sample, p in ITEM_PATHS.items():
        df = pd.read_csv(p)
        df['sample'] = sample
        rows.append(df)
    items = pd.concat(rows, ignore_index=True)

    # Identifier column: 'participantID' in items file, 'subj' in master
    # Need a subject id that matches both files. Check map file.
    for sample, _ in ITEM_PATHS.items():
        # subject_mapping connects participantID -> subj
        map_path = REPO / f'data/{sample}_350/processed/stage5_filtered_data_20260403_{"133425" if sample=="exploratory" else "142413"}/subject_mapping.csv'
        if not map_path.exists():
            # try alternative
            sd = REPO/f'data/{sample}_350/processed'
            cand = list(sd.glob('stage5_filtered_data_*/subject_mapping.csv'))
            if cand: map_path = cand[0]
        if map_path.exists():
            mp = pd.read_csv(map_path)
            mask = items['sample'] == sample
            items.loc[mask, 'subj'] = items.loc[mask, 'participantID'].map(
                dict(zip(mp['participantID'], mp['subj'])))

    # Drop PHQ9 columns
    phq9_cols = [c for c in items.columns if c.startswith('PHQ9_')]
    items = items.drop(columns=phq9_cols)
    print(f'  Dropped {len(phq9_cols)} PHQ9 columns')

    # Subjects with valid model fits
    exp, conf = load_both()
    pm_rows = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa'])
        m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        pm_rows.append(m[['subj','sample','omega','kappa','log_omega','log_kappa',
                          'AMI_Social']])
    params = pd.concat(pm_rows, ignore_index=True)

    # Join items + params on (subj, sample)
    df = items.merge(params, on=['subj','sample'], how='inner')
    print(f'  Joined items + params: N={len(df)} subjects '
          f'(exp={(df["sample"]=="exploratory").sum()}, '
          f'conf={(df["sample"]=="confirmatory").sum()})')
    return df


def stai_reverse_audit(df):
    """Compute mean inter-item correlation within each questionnaire.
    If STAI is much lower than others, reverse-keying wasn't applied."""
    print('\n--- STAI reverse-coding audit ---')
    print(f'  {"questionnaire":<14} {"n_items":>8} {"mean inter-item r":>20}  {"% neg":>7}')
    for q in QUESTIONNAIRES:
        qcols = [c for c in df.columns if c.startswith(q+'_item_')]
        if not qcols: continue
        sub = df[qcols].dropna()
        cm = sub.corr().values
        offdiag = cm[np.triu_indices_from(cm, k=1)]
        offdiag = offdiag[~np.isnan(offdiag)]
        neg_pct = (offdiag < 0).mean() * 100
        flag = ' ⚠️ likely needs reverse-keying' if (q == 'STAI' and offdiag.mean() < 0.15) else ''
        print(f'  {q:<14} {len(qcols):>8} {offdiag.mean():>20.3f}  {neg_pct:>6.1f}%{flag}')

    # Detect reverse-keyed items using PC1 sign (correct approach):
    # PC1 captures the dominant covariance pattern (the construct itself).
    # Items with NEGATIVE loading on PC1 are oriented opposite to the construct.
    for q in QUESTIONNAIRES:
        qcols = [c for c in df.columns if c.startswith(q+'_item_')]
        if len(qcols) < 4: continue
        sub = df[qcols].dropna()
        cm_pre = sub.corr().values
        neg_pre_pct = (cm_pre[np.triu_indices_from(cm_pre, k=1)] < 0).mean() * 100
        if neg_pre_pct < 5:
            continue  # nothing to fix
        # Standardize and PCA
        X = (sub.values - sub.values.mean(0)) / sub.values.std(0, ddof=1)
        pc1 = PCA(n_components=1).fit(X)
        loadings = pc1.components_[0]
        neg_idx = np.where(loadings < 0)[0]
        if len(neg_idx) == 0:
            continue
        max_val = sub.values.max()
        rev_cols = [qcols[i] for i in neg_idx]
        print(f'\n  {q} PC1-sign reverse-coding: flipping {len(neg_idx)}/{len(qcols)} items')
        print(f'    Items reversed: {[int(c.split("_")[-1]) for c in rev_cols]}')
        df[rev_cols] = max_val - df[rev_cols]
        # Recompute
        sub_post = df[qcols].dropna()
        cm_post = sub_post.corr().values
        offdiag = cm_post[np.triu_indices_from(cm_post, k=1)]
        offdiag = offdiag[~np.isnan(offdiag)]
        print(f'    Pre-reverse:  mean r={cm_pre[np.triu_indices_from(cm_pre,k=1)].mean():+.3f}, {neg_pre_pct:.1f}% neg')
        print(f'    Post-reverse: mean r={offdiag.mean():+.3f}, {(offdiag<0).mean()*100:.1f}% neg')

    return df


def item_quality_screen(df, item_cols, max_missing_pct=10, min_variance=0.05):
    """Drop items with too much missing or near-zero variance."""
    dropped = []
    keep = []
    for c in item_cols:
        x = df[c]
        miss = x.isna().mean() * 100
        var = x.var()
        if miss > max_missing_pct:
            dropped.append((c, f'missing={miss:.1f}%'))
        elif var < min_variance:
            dropped.append((c, f'var={var:.3f}'))
        else:
            keep.append(c)
    print(f'\n  Item quality screen: kept {len(keep)}/{len(item_cols)} items')
    if dropped:
        print(f'    Dropped: {[d[0] for d in dropped[:5]]}{"..." if len(dropped)>5 else ""}')
    return keep


# ============================================================================
# Phase B
# ============================================================================

def parallel_analysis(X, n_iter=300, percentile=95, seed=42):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    Xc = (X - X.mean(0)) / X.std(0, ddof=1)
    actual = np.linalg.eigvalsh(np.corrcoef(Xc.T))[::-1]
    rand_eigs = np.zeros((n_iter, p))
    for i in range(n_iter):
        R = rng.standard_normal((n, p))
        rand_eigs[i] = np.linalg.eigvalsh(np.corrcoef(R.T))[::-1]
    rand_p = np.percentile(rand_eigs, percentile, axis=0)
    return actual, rand_p


def run_efa(X_df, n_factors, rotation='varimax'):
    """Returns (loadings_df, scores_array, FA object).
    Uses sklearn FactorAnalysis (varimax only; oblimin not supported here)."""
    if rotation == 'oblimin':
        rotation = 'varimax'  # fallback — sklearn lacks oblimin
    fa = FactorAnalysis(n_components=n_factors, rotation=rotation, random_state=42)
    scores = fa.fit_transform(X_df.values)
    loadings = pd.DataFrame(fa.components_.T, index=X_df.columns,
                            columns=[f'F{i+1}' for i in range(n_factors)])
    return loadings, scores, fa


def label_factors(loadings, top_k=8):
    """Print top-loading items per factor for naming."""
    print(f'\n  Top-{top_k} items per factor (|loading| > 0.3):')
    for f in loadings.columns:
        s = loadings[f].sort_values(key=abs, ascending=False)
        s = s[s.abs() > 0.3].head(top_k)
        if len(s) == 0:
            print(f'  {f}: (no items |loading| > 0.3)')
            continue
        print(f'  {f}:')
        for item, ld in s.items():
            print(f'    {ld:+.2f}  {item}')


# ============================================================================
# Phase C
# ============================================================================

def fit_t(formula, data):
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={"log_likelihood":False})
    return az.summary(fit, hdi_prob=0.95)


def survives(r):
    return (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)


def test_factors_vs_params(df, factor_cols):
    """Univariate and multivariate Student-t regressions of (log_omega, log_kappa) on factors."""
    # Within-sample z-score everything
    for c in factor_cols + ['log_omega', 'log_kappa', 'AMI_Social']:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            if x.std() > 0:
                df.loc[mask, f'{c}_z'] = (x - x.mean()) / x.std()

    rows = []

    # Univariate
    print('\n--- UNIVARIATE: log_omega_z ~ factor_z ---')
    print(f'  {"factor":<8} {"β(log_ω)":>10} {"HDI":>22}  {"":<3} {"β(log_κ)":>10} {"HDI":>22}')
    for f in factor_cols:
        sub = df[[f'log_omega_z', f'log_kappa_z', f'{f}_z']].dropna()
        # log_omega
        s = fit_t(f'log_omega_z ~ {f}_z', sub)
        ro = s.loc[f'{f}_z']
        # log_kappa
        s = fit_t(f'log_kappa_z ~ {f}_z', sub)
        rk = s.loc[f'{f}_z']
        flag_o = '★' if survives(ro) else ' '
        flag_k = '★' if survives(rk) else ' '
        print(f'  {f:<8} {ro["mean"]:+10.3f} [{ro["hdi_2.5%"]:+.3f},{ro["hdi_97.5%"]:+.3f}] {flag_o}  '
              f'{rk["mean"]:+10.3f} [{rk["hdi_2.5%"]:+.3f},{rk["hdi_97.5%"]:+.3f}] {flag_k}')
        rows.append({'test':'univariate', 'factor':f, 'outcome':'log_omega', 'N':len(sub),
                     'mean':float(ro['mean']), 'sd':float(ro['sd']),
                     'hdi_lo':float(ro['hdi_2.5%']), 'hdi_hi':float(ro['hdi_97.5%']),
                     'survives':bool(survives(ro))})
        rows.append({'test':'univariate', 'factor':f, 'outcome':'log_kappa', 'N':len(sub),
                     'mean':float(rk['mean']), 'sd':float(rk['sd']),
                     'hdi_lo':float(rk['hdi_2.5%']), 'hdi_hi':float(rk['hdi_97.5%']),
                     'survives':bool(survives(rk))})

    # Multivariate
    print('\n--- MULTIVARIATE: log_omega_z ~ all factors + log_kappa_z (control) ---')
    sub = df[['log_omega_z','log_kappa_z'] + [f'{f}_z' for f in factor_cols]].dropna()
    pred_str = ' + '.join([f'{f}_z' for f in factor_cols])
    s = fit_t(f'log_omega_z ~ {pred_str} + log_kappa_z', sub)
    for f in factor_cols:
        r = s.loc[f'{f}_z']
        flag = '★' if survives(r) else ' '
        print(f'  {f:<8} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
        rows.append({'test':'multivariate_omega', 'factor':f, 'outcome':'log_omega',
                     'N':len(sub), 'mean':float(r['mean']), 'sd':float(r['sd']),
                     'hdi_lo':float(r['hdi_2.5%']), 'hdi_hi':float(r['hdi_97.5%']),
                     'survives':bool(survives(r))})

    print('\n--- MULTIVARIATE: log_kappa_z ~ all factors + log_omega_z (control) ---')
    s = fit_t(f'log_kappa_z ~ {pred_str} + log_omega_z', sub)
    for f in factor_cols:
        r = s.loc[f'{f}_z']
        flag = '★' if survives(r) else ' '
        print(f'  {f:<8} β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
        rows.append({'test':'multivariate_kappa', 'factor':f, 'outcome':'log_kappa',
                     'N':len(sub), 'mean':float(r['mean']), 'sd':float(r['sd']),
                     'hdi_lo':float(r['hdi_2.5%']), 'hdi_hi':float(r['hdi_97.5%']),
                     'survives':bool(survives(r))})

    # Baseline comparison: AMI_Social
    print('\n--- BASELINE: log_omega_z ~ AMI_Social_z ---')
    sub = df[['log_omega_z','AMI_Social_z']].dropna()
    s = fit_t('log_omega_z ~ AMI_Social_z', sub)
    r = s.loc['AMI_Social_z']
    flag = '★' if survives(r) else ' '
    print(f'  AMI_Social   β={r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
    rows.append({'test':'baseline', 'factor':'AMI_Social', 'outcome':'log_omega',
                 'N':len(sub), 'mean':float(r['mean']), 'sd':float(r['sd']),
                 'hdi_lo':float(r['hdi_2.5%']), 'hdi_hi':float(r['hdi_97.5%']),
                 'survives':bool(survives(r))})
    return rows


# ============================================================================
# Main
# ============================================================================

def main():
    print('='*72); print('PHASE A — Data prep'); print('='*72)
    df = load_items_and_params()
    df = stai_reverse_audit(df)
    item_cols = [c for c in df.columns if any(c.startswith(q+'_item_') for q in QUESTIONNAIRES)]
    item_cols = item_quality_screen(df, item_cols)

    # Within-sample standardize each item (otherwise sample mean drift contaminates FA)
    print(f'\n  Within-sample z-scoring {len(item_cols)} items...')
    for c in item_cols:
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            if x.std() > 0:
                df.loc[mask, c] = (x - x.mean()) / x.std()

    # Build clean item matrix for EFA (drop subjects with any missing item)
    X_df = df[['subj','sample'] + item_cols].dropna()
    print(f'  Final item matrix for EFA: {X_df.shape[0]} subjects × {len(item_cols)} items')

    print('\n' + '='*72); print('PHASE B — Factor structure'); print('='*72)

    # Skip KMO/Bartlett (factor_analyzer incompatible with current sklearn)

    # Parallel analysis
    print('\n  Running Horn parallel analysis (300 iters)...')
    actual, rand_p95 = parallel_analysis(X_df[item_cols].values, n_iter=300)
    n_keep = int(np.sum(actual > rand_p95))
    print(f'  Recommended n_factors: {n_keep}')
    print(f'  First 12 eigenvalues:  actual vs random 95%:')
    for i in range(min(12, len(actual))):
        keep = '✓' if actual[i] > rand_p95[i] else ''
        print(f'    F{i+1:<3}  actual={actual[i]:6.2f}  random95={rand_p95[i]:5.2f}  {keep}')

    pa_df = pd.DataFrame({
        'factor': range(1, len(actual)+1),
        'actual_eigval': actual,
        'random_95pct': rand_p95,
        'keep': actual > rand_p95,
    })
    pa_df.to_csv(REPO/'results/stats/affect_analysis/item_efa_parallel_analysis.csv', index=False)

    # Try a reasonable range — but cap for sanity
    n_to_fit = min(n_keep, 10) if n_keep > 0 else 4
    if n_keep > 10:
        print(f'  ⚠ Parallel analysis recommends {n_keep} but capping at 10 for interpretability')

    # Fit varimax (sklearn FactorAnalysis doesn't support oblimin)
    all_loadings = {}
    all_scores = {}
    for rot in ['varimax']:
        print(f'\n  ----- {n_to_fit}-factor solution, rotation={rot} -----')
        loadings, scores, fa = run_efa(X_df[item_cols], n_to_fit, rotation=rot)
        label_factors(loadings, top_k=8)
        all_loadings[rot] = loadings
        all_scores[rot] = scores

    # Save loadings (use varimax as primary)
    loadings_long = []
    for rot, ld in all_loadings.items():
        ld2 = ld.reset_index().rename(columns={'index':'item'})
        ld2['rotation'] = rot
        loadings_long.append(ld2)
    pd.concat(loadings_long).to_csv(
        REPO/'results/stats/affect_analysis/item_efa_loadings.csv', index=False)

    # Use varimax scores (only rotation available without factor_analyzer)
    scores = all_scores['varimax']
    factor_cols = [f'F{i+1}' for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, columns=factor_cols, index=X_df.index)
    df_with_scores = df.loc[X_df.index].copy()
    for c in factor_cols:
        df_with_scores[c] = scores_df[c]

    # Save subject scores
    df_with_scores[['subj','sample'] + factor_cols].to_csv(
        REPO/'results/stats/affect_analysis/item_efa_subject_scores.csv', index=False)

    print('\n' + '='*72); print('PHASE C — Factors vs (ω, κ)'); print('='*72)
    rows = test_factors_vs_params(df_with_scores, factor_cols)
    pd.DataFrame(rows).to_csv(
        REPO/'results/stats/affect_analysis/item_efa_param_regressions.csv', index=False)

    print('\n' + '='*72); print('SUMMARY — surviving effects'); print('='*72)
    surv = [r for r in rows if r['survives']]
    if not surv:
        print('  None.')
    for r in surv:
        print(f"  [{r['test']:<18}] {r['factor']:<14} → {r['outcome']}: "
              f"β={r['mean']:+.3f}  HDI [{r['hdi_lo']:+.3f}, {r['hdi_hi']:+.3f}]")


if __name__ == '__main__':
    main()
