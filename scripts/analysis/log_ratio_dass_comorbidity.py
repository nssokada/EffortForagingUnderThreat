"""
Three-phase deep-dive on log(omega/kappa) clinical signal:

  Phase 1: Diagnose whether Student-t (nu=3) unfairly killed DASS scales.
           Spearman + Huber (RLM) + 5%-trimmed Normal Bayesian + Student-t (nu free).
           Scatter plot of DASS_Stress vs log_ratio for visual leverage check.

  Phase 2: Better factor analysis (parallel analysis to pick n_factors)
           + theory-grouped composite scores (ANX, DEP, APATHY, STRESS).
           Test composites univariately on log_ratio.

  Phase 3: Anxiety x depression comorbidity tests on composites:
           A. Polar decomposition (severity + discordance + discordance^2)
           B. 2x2 quadrant ANOVA (median-split ANX x DEP)
           C. Joint univariate vs additive composite model

Outputs:
  results/stats/affect_analysis/log_ratio_dass_diagnostic.csv
  results/stats/affect_analysis/log_ratio_composites.csv
  results/stats/affect_analysis/log_ratio_comorbidity.csv
  results/stats/affect_analysis/factor_analysis_parallel.csv
  results/figs/affect_analysis/dass_stress_vs_log_ratio.png
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import bambi as bmb
import arviz as az
import statsmodels.api as sm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
REPO = Path('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
os.chdir(REPO)
sys.path.insert(0, str(REPO / 'notebooks' / 'analysis'))
from load_data import load_both

SAMPLES = {
    "exploratory":  "data/exploratory_350/processed/stage5_filtered_data_20260403_133425",
    "confirmatory": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413",
}

BKW = dict(draws=1000, tune=1000, chains=4, target_accept=0.95,
           random_seed=42, progressbar=False)

ANX_SCALES = ['DASS21_Anxiety', 'STAI_Trait', 'OASIS_Total', 'STICSA_Total']
DEP_SCALES = ['DASS21_Depression', 'PHQ9_Total']
APATHY_SCALES = ['AMI_Behavioural', 'AMI_Social', 'AMI_Emotional', 'MFIS_Total']
ALL_SCALES = ANX_SCALES + DEP_SCALES + APATHY_SCALES + ['DASS21_Stress']


# ============================================================================
# Data prep
# ============================================================================

def build_pooled_master():
    exp, conf = load_both()
    rows = []
    for sample_name, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d['master'].reset_index().rename(columns={'index': 'subj'}).copy()
        m['sample'] = sample_name
        m = m.dropna(subset=['omega', 'kappa'])
        m = m[(m['omega'] > 0) & (m['kappa'] > 0)]
        m['log_ratio'] = np.log(m['omega']) - np.log(m['kappa'])
        rows.append(m)
    master = pd.concat(rows, ignore_index=True)

    factors = pd.read_csv(REPO / 'results/stats/clinical/factor_scores.csv')[
        ['subj', 'sample', 'F1', 'F2']]
    df = master.merge(factors, on=['subj', 'sample'], how='left')

    # Quality filter
    keep = pd.Series(True, index=df.index)
    for s in df['sample'].unique():
        ss = df[df['sample'] == s]
        z = (ss['log_ratio'] - ss['log_ratio'].mean()) / ss['log_ratio'].std()
        keep.loc[ss.index[z.abs() > 3]] = False
    n_drop = (~keep).sum()
    df = df[keep].copy()
    print(f'Quality filter: dropped {n_drop} (|log_ratio_z|>3)')

    # Within-sample z for log_ratio and all clinical scales
    cols_to_z = ['log_ratio'] + [c for c in ALL_SCALES if c in df.columns]
    for c in cols_to_z:
        df[f'{c}_z'] = np.nan
        for s in df['sample'].unique():
            mask = df['sample'] == s
            x = df.loc[mask, c]
            df.loc[mask, f'{c}_z'] = (x - x.mean()) / x.std()

    return df


# ============================================================================
# Phase 1: DASS diagnostic
# ============================================================================

def phase1_dass_diagnostic(df):
    print('\n' + '='*72)
    print('PHASE 1: DASS diagnostic — was Student-t too aggressive?')
    print('='*72)

    rows = []
    dass = ['DASS21_Anxiety', 'DASS21_Depression', 'DASS21_Stress']

    for pred in dass:
        z = f'{pred}_z'
        sub = df[['log_ratio_z', z, 'log_ratio', pred]].dropna()
        y = sub['log_ratio_z'].values
        x = sub[z].values
        N = len(sub)

        # (a) Spearman
        rho, p_sp = spearmanr(sub[pred], sub['log_ratio'])

        # (b) Huber RLM
        X = sm.add_constant(x)
        rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        b_h = rlm.params[1]; se_h = rlm.bse[1]; p_h = rlm.pvalues[1]

        # (c) Trimmed Normal Bayesian (5% on predictor)
        lo, hi = np.percentile(sub[z], [5, 95])
        tr = sub[(sub[z] >= lo) & (sub[z] <= hi)]
        m_tr = bmb.Model(f'log_ratio_z ~ {z}', data=tr, family='gaussian')
        fit_tr = m_tr.fit(**BKW, idata_kwargs={"log_likelihood": False})
        s_tr = az.summary(fit_tr, hdi_prob=0.95).loc[z]

        # (d) Student-t with nu estimated (less aggressive than nu=3)
        m_t = bmb.Model(f'log_ratio_z ~ {z}', data=sub, family='t')
        fit_t = m_t.fit(**BKW, idata_kwargs={"log_likelihood": False})
        s_t = az.summary(fit_t, hdi_prob=0.95).loc[z]
        nu_post = az.summary(fit_t, hdi_prob=0.95)
        nu_mean = float(nu_post.loc['nu', 'mean']) if 'nu' in nu_post.index else np.nan

        # (e) Normal Bayesian (reference: original signal)
        m_n = bmb.Model(f'log_ratio_z ~ {z}', data=sub, family='gaussian')
        fit_n = m_n.fit(**BKW, idata_kwargs={"log_likelihood": False})
        s_n = az.summary(fit_n, hdi_prob=0.95).loc[z]

        print(f'\n--- {pred} (N={N}) ---')
        print(f'  Spearman rho   = {rho:+.3f}  p = {p_sp:.4f}')
        print(f'  Huber RLM      β = {b_h:+.3f}  SE={se_h:.3f}  p = {p_h:.4f}')
        print(f'  Trimmed Normal β = {s_tr["mean"]:+.3f}  HDI [{s_tr["hdi_2.5%"]:+.3f}, {s_tr["hdi_97.5%"]:+.3f}]  (N_trim={len(tr)})')
        print(f'  Student-t (nu) β = {s_t["mean"]:+.3f}  HDI [{s_t["hdi_2.5%"]:+.3f}, {s_t["hdi_97.5%"]:+.3f}]  (nu={nu_mean:.1f})')
        print(f'  Normal         β = {s_n["mean"]:+.3f}  HDI [{s_n["hdi_2.5%"]:+.3f}, {s_n["hdi_97.5%"]:+.3f}]')

        rows.append({
            'predictor': pred, 'N': N,
            'spearman_rho': rho, 'spearman_p': p_sp,
            'huber_beta': b_h, 'huber_se': se_h, 'huber_p': p_h,
            'trimmed_normal_beta': float(s_tr['mean']),
            'trimmed_normal_hdi_lo': float(s_tr['hdi_2.5%']),
            'trimmed_normal_hdi_hi': float(s_tr['hdi_97.5%']),
            'studentt_free_beta': float(s_t['mean']),
            'studentt_free_hdi_lo': float(s_t['hdi_2.5%']),
            'studentt_free_hdi_hi': float(s_t['hdi_97.5%']),
            'studentt_nu_mean': nu_mean,
            'normal_beta': float(s_n['mean']),
            'normal_hdi_lo': float(s_n['hdi_2.5%']),
            'normal_hdi_hi': float(s_n['hdi_97.5%']),
        })

    out = REPO / 'results/stats/affect_analysis/log_ratio_dass_diagnostic.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nSaved: {out}')

    # Visualize DASS_Stress vs log_ratio
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, pred in zip(axes, ['DASS21_Anxiety', 'DASS21_Depression', 'DASS21_Stress']):
        sub = df[[pred, 'log_ratio', 'sample']].dropna()
        for samp, c in [('exploratory', 'C0'), ('confirmatory', 'C1')]:
            ss = sub[sub['sample'] == samp]
            ax.scatter(ss[pred], ss['log_ratio'], s=10, alpha=0.45, c=c, label=samp)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel(pred); ax.set_ylabel('log(ω/κ)')
        ax.set_title(pred); ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = REPO / 'results/figs/affect_analysis/dass_vs_log_ratio.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'Saved figure: {fig_path}')


# ============================================================================
# Phase 2: Better factor analysis + theory composites
# ============================================================================

def parallel_analysis(X, n_iter=500, percentile=95, seed=42):
    """Horn's parallel analysis. Returns (actual_eigvals, random_eigvals_percentile)."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    Xc = X - X.mean(0)
    Xc = Xc / Xc.std(0, ddof=1)
    actual = np.linalg.eigvalsh(np.corrcoef(Xc.T))[::-1]
    rand_eigs = np.zeros((n_iter, p))
    for i in range(n_iter):
        R = rng.standard_normal((n, p))
        rand_eigs[i] = np.linalg.eigvalsh(np.corrcoef(R.T))[::-1]
    rand_p = np.percentile(rand_eigs, percentile, axis=0)
    return actual, rand_p


def fit_efa(X, n_factors, scale_names):
    """Simple EFA via varimax rotation. Returns DataFrame of loadings + factor scores."""
    from sklearn.decomposition import FactorAnalysis
    fa = FactorAnalysis(n_components=n_factors, rotation='varimax', random_state=42)
    scores = fa.fit_transform(X)
    loadings = pd.DataFrame(fa.components_.T,
                            index=scale_names,
                            columns=[f'F{i+1}' for i in range(n_factors)])
    return loadings, scores


def phase2_factors_composites(df):
    print('\n' + '='*72)
    print('PHASE 2: Better factor analysis + theory composites')
    print('='*72)

    # Need all scales non-missing for FA
    fa_cols = [c for c in ALL_SCALES if c in df.columns]
    sub = df[['subj', 'sample'] + fa_cols].dropna()
    print(f'\nFactor analysis sample: N={len(sub)} (subjects with all {len(fa_cols)} scales)')

    # Within-sample z-score the scales for FA (else sample-mean drift contaminates loadings)
    X_df = sub.copy()
    for c in fa_cols:
        for s in X_df['sample'].unique():
            mask = X_df['sample'] == s
            x = X_df.loc[mask, c]
            X_df.loc[mask, c] = (x - x.mean()) / x.std()
    X = X_df[fa_cols].values

    # Parallel analysis
    actual, random_p95 = parallel_analysis(X)
    n_keep = int(np.sum(actual > random_p95))
    print(f'\nParallel analysis (Horn):')
    print(f'  {"Factor":<8} {"Actual":>10} {"Random 95%":>12}  Keep?')
    for i, (a, r) in enumerate(zip(actual, random_p95)):
        keep = '✓' if a > r else ''
        print(f'  F{i+1:<7} {a:>10.3f} {r:>12.3f}  {keep}')
    print(f'  → Parallel analysis recommends {n_keep} factors')

    pa_rows = [{'factor': i+1, 'actual_eigval': a, 'random_95pct': r, 'keep': bool(a > r)}
               for i, (a, r) in enumerate(zip(actual, random_p95))]
    pd.DataFrame(pa_rows).to_csv(
        REPO / 'results/stats/affect_analysis/factor_analysis_parallel.csv', index=False)

    # Fit EFA at the recommended n + 2 and 3 for comparison
    print('\n--- EFA loadings (varimax rotation) ---')
    for nf in sorted({2, 3, n_keep}):
        if nf < 1: continue
        loadings, scores = fit_efa(X, nf, fa_cols)
        print(f'\n{nf}-factor solution:')
        with pd.option_context('display.float_format', '{:+.2f}'.format):
            print(loadings.to_string())

    # Theory-grouped composites (z-mean within already-z-scored scales)
    print('\n--- Theory-grouped composites ---')
    comp_df = df.copy()
    comp_df['ANX_comp'] = comp_df[[f'{c}_z' for c in ANX_SCALES if f'{c}_z' in comp_df.columns]].mean(1)
    comp_df['DEP_comp'] = comp_df[[f'{c}_z' for c in DEP_SCALES if f'{c}_z' in comp_df.columns]].mean(1)
    comp_df['APATHY_comp'] = comp_df[[f'{c}_z' for c in APATHY_SCALES if f'{c}_z' in comp_df.columns]].mean(1)
    comp_df['STRESS_comp'] = comp_df['DASS21_Stress_z'] if 'DASS21_Stress_z' in comp_df.columns else np.nan

    # Re-z within-sample (composites have different SDs)
    for c in ['ANX_comp', 'DEP_comp', 'APATHY_comp', 'STRESS_comp']:
        for s in comp_df['sample'].unique():
            mask = comp_df['sample'] == s
            x = comp_df.loc[mask, c]
            if x.notna().sum() > 5:
                comp_df.loc[mask, c] = (x - x.mean()) / x.std()

    # Composite reliability and inter-correlations
    print('\nComposite correlations:')
    cor = comp_df[['ANX_comp', 'DEP_comp', 'APATHY_comp', 'STRESS_comp']].corr()
    with pd.option_context('display.float_format', '{:+.2f}'.format):
        print(cor.to_string())

    # Univariate Normal + Student-t tests of composites on log_ratio
    print('\n--- Composite → log_ratio (univariate, both Normal and Student-t) ---')
    print(f'{"composite":<14} {"family":<12} {"N":>5} {"β":>8} {"95% HDI":>22}  flag')
    rows = []
    for comp in ['ANX_comp', 'DEP_comp', 'APATHY_comp', 'STRESS_comp']:
        ss = comp_df[['log_ratio_z', comp]].dropna()
        if len(ss) < 50: continue
        for family in ['gaussian', 't']:
            mdl = bmb.Model(f'log_ratio_z ~ {comp}', data=ss, family=family)
            fit = mdl.fit(**BKW, idata_kwargs={"log_likelihood": False})
            r = az.summary(fit, hdi_prob=0.95).loc[comp]
            sv = (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)
            flag = '★' if sv else ' '
            print(f'{comp:<14} {family:<12} {len(ss):>5} {r["mean"]:+.3f}  [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
            rows.append({
                'composite': comp, 'family': family, 'N': len(ss),
                'mean': float(r['mean']), 'sd': float(r['sd']),
                'hdi_lo': float(r['hdi_2.5%']), 'hdi_hi': float(r['hdi_97.5%']),
                'survives': bool(sv),
            })

    out = REPO / 'results/stats/affect_analysis/log_ratio_composites.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nSaved: {out}')

    return comp_df


# ============================================================================
# Phase 3: Comorbidity tests (ANX × DEP)
# ============================================================================

def phase3_comorbidity(comp_df):
    print('\n' + '='*72)
    print('PHASE 3: Anxiety × Depression comorbidity on log(ω/κ)')
    print('='*72)

    sub = comp_df[['log_ratio_z', 'ANX_comp', 'DEP_comp', 'sample']].dropna().copy()
    sub['severity']    = sub['ANX_comp'] + sub['DEP_comp']
    sub['discordance'] = sub['ANX_comp'] - sub['DEP_comp']
    # Within-sample z the derived vars too
    for c in ['severity', 'discordance']:
        for s in sub['sample'].unique():
            mask = sub['sample'] == s
            x = sub.loc[mask, c]
            sub.loc[mask, c] = (x - x.mean()) / x.std()
    sub['discordance_sq'] = sub['discordance'] ** 2
    print(f'\nN = {len(sub)}')

    rows = []

    # ===== A. Polar decomposition (Normal + Student-t) =====
    print('\n--- A. Polar decomposition: severity + discordance + discordance² ---')
    for family in ['gaussian', 't']:
        mdl = bmb.Model('log_ratio_z ~ severity + discordance + discordance_sq',
                        data=sub, family=family)
        fit = mdl.fit(**BKW, idata_kwargs={"log_likelihood": False})
        s_summary = az.summary(fit, hdi_prob=0.95)
        print(f'\n  family={family}:')
        for term in ['severity', 'discordance', 'discordance_sq']:
            r = s_summary.loc[term]
            sv = (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)
            flag = '★' if sv else ' '
            print(f'    {term:<18} β = {r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
            rows.append({
                'test': 'A_polar', 'family': family, 'term': term, 'N': len(sub),
                'mean': float(r['mean']), 'sd': float(r['sd']),
                'hdi_lo': float(r['hdi_2.5%']), 'hdi_hi': float(r['hdi_97.5%']),
                'survives': bool(sv),
            })

    # ===== B. 2x2 quadrant ANOVA =====
    print('\n--- B. 2×2 quadrant (median-split ANX × DEP) ---')
    sub['anx_hi'] = (sub['ANX_comp'] > sub['ANX_comp'].median()).astype(int)
    sub['dep_hi'] = (sub['DEP_comp'] > sub['DEP_comp'].median()).astype(int)
    sub['quadrant'] = sub.apply(
        lambda r: 'healthy' if (r['anx_hi']==0 and r['dep_hi']==0)
        else ('pure_anx' if (r['anx_hi']==1 and r['dep_hi']==0)
        else ('pure_dep' if (r['anx_hi']==0 and r['dep_hi']==1)
        else 'comorbid')), axis=1)
    print(f'\n  Quadrant cell sizes:')
    print(sub['quadrant'].value_counts())
    print('\n  log_ratio_z by quadrant (mean ± SE):')
    g = sub.groupby('quadrant')['log_ratio_z'].agg(['mean', 'std', 'count'])
    g['se'] = g['std'] / np.sqrt(g['count'])
    print(g[['mean', 'se', 'count']].round(3).to_string())

    # Bayesian: log_ratio ~ quadrant
    mdl = bmb.Model('log_ratio_z ~ quadrant', data=sub, family='gaussian')
    fit = mdl.fit(**BKW, idata_kwargs={"log_likelihood": False})
    print('\n  Bayesian contrasts (vs comorbid baseline if first alpha):')
    s_summary = az.summary(fit, hdi_prob=0.95)
    for idx in s_summary.index:
        if idx.startswith('quadrant'):
            r = s_summary.loc[idx]
            sv = (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)
            flag = '★' if sv else ' '
            print(f'    {idx:<28} β = {r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
            rows.append({
                'test': 'B_quadrant', 'family': 'gaussian', 'term': idx, 'N': len(sub),
                'mean': float(r['mean']), 'sd': float(r['sd']),
                'hdi_lo': float(r['hdi_2.5%']), 'hdi_hi': float(r['hdi_97.5%']),
                'survives': bool(sv),
            })

    # ===== C. Univariate + joint composite =====
    print('\n--- C. Univariate ANX and DEP separately, then joint ---')
    for spec in ['log_ratio_z ~ ANX_comp',
                 'log_ratio_z ~ DEP_comp',
                 'log_ratio_z ~ ANX_comp + DEP_comp']:
        for family in ['gaussian', 't']:
            mdl = bmb.Model(spec, data=sub, family=family)
            fit = mdl.fit(**BKW, idata_kwargs={"log_likelihood": False})
            s_summary = az.summary(fit, hdi_prob=0.95)
            label = spec.replace('log_ratio_z ~ ', '')
            print(f'\n  {label}  ({family}):')
            for term in ['ANX_comp', 'DEP_comp']:
                if term in s_summary.index:
                    r = s_summary.loc[term]
                    sv = (r['hdi_2.5%'] > 0) or (r['hdi_97.5%'] < 0)
                    flag = '★' if sv else ' '
                    print(f'    {term:<14} β = {r["mean"]:+.3f}  HDI [{r["hdi_2.5%"]:+.3f}, {r["hdi_97.5%"]:+.3f}]  {flag}')
                    rows.append({
                        'test': f'C_{label}', 'family': family, 'term': term, 'N': len(sub),
                        'mean': float(r['mean']), 'sd': float(r['sd']),
                        'hdi_lo': float(r['hdi_2.5%']), 'hdi_hi': float(r['hdi_97.5%']),
                        'survives': bool(sv),
                    })

    out = REPO / 'results/stats/affect_analysis/log_ratio_comorbidity.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nSaved: {out}')


# ============================================================================
# Main
# ============================================================================

def main():
    df = build_pooled_master()
    print(f'\nN pooled (after quality filter): {len(df)}')
    print(f'  exp={(df["sample"]=="exploratory").sum()}  conf={(df["sample"]=="confirmatory").sum()}')
    phase1_dass_diagnostic(df)
    comp_df = phase2_factors_composites(df)
    phase3_comorbidity(comp_df)


if __name__ == '__main__':
    main()
