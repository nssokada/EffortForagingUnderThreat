"""
Re-run item-level EFA with FEWER factors (3, 4, 5) to find the most
interpretable solution that still recovers the ω signal.

Builds on item_level_efa_on_params.py setup but with reduced n_factors.
Same preprocessing (STAI reverse-coding, within-sample z, etc.)
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
from sklearn.decomposition import FactorAnalysis, PCA
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO); sys.path.insert(0, str(REPO/'notebooks'/'analysis'))
from load_data import load_both

ITEM_PATHS = {
    'exploratory':  REPO/'data/exploratory_350/processed/stage4_mental_health_20260403_133425/mental_health_items_wide.csv',
    'confirmatory': REPO/'data/confirmatory_350/processed/stage4_mental_health_20260403_142413/mental_health_items_wide.csv',
}
QUESTIONNAIRES = ['STAI', 'DASS21', 'MFIS', 'STICSA', 'AMI', 'OASIS']
BKW = dict(draws=1000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)

# ---- Reuse the data-prep logic ----

def load_and_prep():
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

    items = items.drop(columns=[c for c in items.columns if c.startswith('PHQ9_')])

    exp, conf = load_both()
    pm_rows = []
    for sn, d in [('exploratory', exp), ('confirmatory', conf)]:
        m = d['master'].reset_index().rename(columns={'index':'subj'}).copy()
        m['sample'] = sn
        m = m.dropna(subset=['omega','kappa']); m = m[(m['omega']>0)&(m['kappa']>0)]
        m['log_omega'] = np.log(m['omega']); m['log_kappa'] = np.log(m['kappa'])
        pm_rows.append(m[['subj','sample','omega','kappa','log_omega','log_kappa','AMI_Social']])
    params = pd.concat(pm_rows, ignore_index=True)
    df = items.merge(params, on=['subj','sample'], how='inner')

    # PC1-sign reverse-code (carry over from item_level_efa_on_params.py)
    for q in QUESTIONNAIRES:
        qcols = [c for c in df.columns if c.startswith(q+'_item_')]
        if len(qcols) < 4: continue
        sub = df[qcols].dropna()
        cm_pre = sub.corr().values
        neg_pre = (cm_pre[np.triu_indices_from(cm_pre, k=1)] < 0).mean()*100
        if neg_pre < 5: continue
        X = (sub.values - sub.values.mean(0)) / sub.values.std(0, ddof=1)
        loadings = PCA(n_components=1).fit(X).components_[0]
        neg_idx = np.where(loadings < 0)[0]
        if len(neg_idx) == 0: continue
        max_val = sub.values.max()
        rev_cols = [qcols[i] for i in neg_idx]
        df[rev_cols] = max_val - df[rev_cols]

    item_cols = [c for c in df.columns if any(c.startswith(q+'_item_') for q in QUESTIONNAIRES)]

    # Within-sample standardize each item
    for c in item_cols:
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            if x.std() > 0:
                df.loc[mask, c] = (x - x.mean()) / x.std()

    X_df = df[['subj','sample','log_omega','log_kappa','AMI_Social'] + item_cols].dropna()
    return X_df, item_cols


# ---- EFA ----

def run_efa(X_df, n, item_cols):
    fa = FactorAnalysis(n_components=n, rotation='varimax', random_state=42)
    scores = fa.fit_transform(X_df[item_cols].values)
    loadings = pd.DataFrame(fa.components_.T, index=item_cols,
                            columns=[f'F{i+1}' for i in range(n)])
    return loadings, scores


def label_loadings(loadings, top_k=8, threshold=0.30):
    """Print top items per factor."""
    for f in loadings.columns:
        s = loadings[f].sort_values(key=abs, ascending=False)
        s = s[s.abs() > threshold].head(top_k)
        print(f'  {f}:')
        # also show what questionnaire dominates
        prefixes = [item.split('_item_')[0] for item in s.index]
        counts = pd.Series(prefixes).value_counts()
        dom_q = counts.idxmax() if len(counts) > 0 else '?'
        print(f'    (dominated by {dom_q}, {counts.get(dom_q,0)}/{len(s)} top items)')
        for item, ld in s.items():
            print(f'    {ld:+.2f}  {item}')


def fit_t(formula, data):
    m = bmb.Model(formula, data=data, family='t')
    fit = m.fit(**BKW, idata_kwargs={'log_likelihood': False})
    return az.summary(fit, hdi_prob=0.95)


def test_factors(df, factor_cols):
    # Within-sample z
    for c in factor_cols + ['log_omega', 'log_kappa', 'AMI_Social']:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            if x.std() > 0:
                df.loc[mask, c] = (x - x.mean())/x.std()
                df.loc[mask, f'{c}_z'] = (x - x.mean())/x.std()
    results = []
    print(f'\n  {"factor":<8} {"β(log_ω)":>14} {"95% HDI":>22}  {"β(log_κ)":>14} {"95% HDI":>22}')
    for f in factor_cols:
        sub = df[[f'log_omega_z', f'log_kappa_z', f'{f}_z']].dropna()
        s_o = fit_t(f'log_omega_z ~ {f}_z', sub)
        s_k = fit_t(f'log_kappa_z ~ {f}_z', sub)
        ro = s_o.loc[f'{f}_z']; rk = s_k.loc[f'{f}_z']
        fo = '★' if (ro['hdi_2.5%']>0 or ro['hdi_97.5%']<0) else ' '
        fk = '★' if (rk['hdi_2.5%']>0 or rk['hdi_97.5%']<0) else ' '
        print(f'  {f:<8} {ro["mean"]:+14.3f} [{ro["hdi_2.5%"]:+.3f},{ro["hdi_97.5%"]:+.3f}] {fo}  '
              f'{rk["mean"]:+14.3f} [{rk["hdi_2.5%"]:+.3f},{rk["hdi_97.5%"]:+.3f}] {fk}')
        results.append({'factor':f, 'beta_omega':ro['mean'], 'hdi_o_lo':ro['hdi_2.5%'],
                        'hdi_o_hi':ro['hdi_97.5%'], 'surv_omega':bool(fo=='★'),
                        'beta_kappa':rk['mean'], 'hdi_k_lo':rk['hdi_2.5%'],
                        'hdi_k_hi':rk['hdi_97.5%'], 'surv_kappa':bool(fk=='★')})
    return results


def main():
    print('Loading and preprocessing item data...')
    X_df, item_cols = load_and_prep()
    print(f'N = {len(X_df)}, items = {len(item_cols)}')

    # Scree plot
    Xc = (X_df[item_cols].values - X_df[item_cols].values.mean(0)) / X_df[item_cols].values.std(0, ddof=1)
    eigvals = np.linalg.eigvalsh(np.corrcoef(Xc.T))[::-1]
    print('\nScree plot (top 15 eigenvalues):')
    for i, e in enumerate(eigvals[:15]):
        bar = '█' * int(e*2)
        print(f'  F{i+1:>3}  {e:6.2f}  {bar}')

    # Save scree plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, 21), eigvals[:20], 'o-', color='steelblue', markersize=6)
    ax.axhline(1, color='gray', lw=0.5, ls='--', label='Kaiser threshold')
    for n in [3, 4, 5, 8]:
        ax.axvline(n+0.5, color='red', lw=0.8, alpha=0.4)
        ax.annotate(f'{n}', xy=(n+0.5, ax.get_ylim()[1]*0.9), fontsize=9, color='red')
    ax.set_xlabel('Factor'); ax.set_ylabel('Eigenvalue')
    ax.set_title('Scree plot — top 20 eigenvalues')
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPO/'results/figs/affect_analysis/item_efa_scree.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  Saved scree plot: results/figs/affect_analysis/item_efa_scree.png')

    summary_rows = []
    for n in [3, 4, 5]:
        print('\n' + '='*72)
        print(f'  {n}-FACTOR SOLUTION')
        print('='*72)
        loadings, scores = run_efa(X_df, n, item_cols)
        label_loadings(loadings, top_k=8)

        # Add scores to df and test
        df_test = X_df.copy()
        for i, fname in enumerate([f'F{i+1}' for i in range(n)]):
            df_test[fname] = scores[:, i]
        print(f'\n  Testing {n} factors against (ω, κ):')
        results = test_factors(df_test, [f'F{i+1}' for i in range(n)])
        for r in results:
            r['n_factors'] = n
            summary_rows.append(r)

        # Save loadings for this solution
        loadings.reset_index().rename(columns={'index':'item'}).assign(n_factors=n).to_csv(
            REPO/f'results/stats/affect_analysis/item_efa_{n}factor_loadings.csv', index=False)

    pd.DataFrame(summary_rows).to_csv(
        REPO/'results/stats/affect_analysis/item_efa_reduced_summary.csv', index=False)

    print('\n' + '='*72); print('OVERALL — surviving ω effects across solutions'); print('='*72)
    print(f'\n  {"n":>3} {"factor":<8} {"β(log_ω)":>10} {"95% HDI":>22}')
    for r in summary_rows:
        if r['surv_omega']:
            print(f"  {r['n_factors']:>3} {r['factor']:<8} {r['beta_omega']:+10.3f} "
                  f"[{r['hdi_o_lo']:+.3f}, {r['hdi_o_hi']:+.3f}] ★")

    # Baseline reference
    print('\n  Baseline AMI_Social → log(ω): β=+0.124 (from prior analysis)')


if __name__ == '__main__':
    main()
