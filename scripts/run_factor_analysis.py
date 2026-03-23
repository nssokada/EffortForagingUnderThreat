"""
NB06-psych equivalent: Factor Analysis of Psychiatric Battery
EFA on psychiatric subscales, then test 5 params → factor scores.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path('/workspace')
DATA   = BASE / 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
STATS  = BASE / 'results/stats'
STATS.mkdir(parents=True, exist_ok=True)

# ── 1. Load psychiatric battery ────────────────────────────────────────────────
print("Loading data...")
psych = pd.read_csv(DATA / 'psych.csv')
print(f"Psych: {len(psych)} subjects, cols: {psych.columns.tolist()}")

# ── 2. Select subscales for EFA ────────────────────────────────────────────────
# Use subscales (not total scores which are sums of included subscales)
efa_cols = [
    'DASS21_Stress',
    'DASS21_Anxiety',
    'DASS21_Depression',
    'AMI_Behavioural',
    'AMI_Social',
    'AMI_Emotional',
    'MFIS_Physical',
    'MFIS_Cognitive',
    'MFIS_Psychosocial',
    'OASIS_Total',
    'PHQ9_Total',
    'STICSA_Total',
    'STAI_Trait',
]

# Verify columns exist
missing = [c for c in efa_cols if c not in psych.columns]
if missing:
    print(f"WARNING: missing columns: {missing}")
    efa_cols = [c for c in efa_cols if c in psych.columns]

print(f"\nEFA columns ({len(efa_cols)}): {efa_cols}")

# Drop subjects with any missing subscale
psych_clean = psych[['subj'] + efa_cols].dropna()
print(f"N after dropping NA: {len(psych_clean)}")

# ── 3. Z-score all subscales ───────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_raw  = psych_clean[efa_cols].values
X      = scaler.fit_transform(X_raw)
X_df   = pd.DataFrame(X, columns=efa_cols, index=psych_clean.index)

print("\nCorrelation matrix (key pairs):")
corr = X_df.corr()
# Show a few key correlations
for pair in [('DASS21_Anxiety', 'DASS21_Depression'),
             ('DASS21_Stress', 'OASIS_Total'),
             ('AMI_Behavioural', 'MFIS_Physical')]:
    if pair[0] in corr.columns and pair[1] in corr.columns:
        print(f"  {pair[0]} ~ {pair[1]}: r={corr.loc[pair[0], pair[1]]:.3f}")

# ── 4. EFA with sklearn FactorAnalysis ────────────────────────────────────────
print("\n" + "="*60)
print("EFA: 3 factors, varimax rotation (sklearn FactorAnalysis)")
print("="*60)

# Try factor_analyzer first
try:
    from factor_analyzer import FactorAnalyzer
    fa = FactorAnalyzer(n_factors=3, rotation='varimax', method='ml')
    fa.fit(X)
    loadings   = fa.loadings_
    factor_scores = fa.transform(X)
    ev, _ = fa.get_eigenvalues()
    communalities = fa.get_communalities()
    print("Using factor_analyzer (ML, varimax)")
    using_fa_lib = True
except ImportError:
    from sklearn.decomposition import FactorAnalysis
    # sklearn FactorAnalysis doesn't support varimax natively; apply manual varimax
    fa = FactorAnalysis(n_components=3, random_state=42)
    fa.fit(X)
    loadings_raw = fa.components_.T  # shape (n_features, n_factors)

    # Apply varimax rotation
    def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
        p, k = Phi.shape
        R = np.eye(k)
        d = 0
        for _ in range(q):
            d_old = d
            Lambda = Phi @ R
            u, s, vh = np.linalg.svd(
                Phi.T @ (Lambda**3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))
            )
            R = u @ vh
            d = np.sum(s)
            if d_old != 0 and d / d_old < 1 + tol:
                break
        return Phi @ R

    loadings = varimax(loadings_raw)

    # Factor scores via regression method
    factor_scores = X @ np.linalg.pinv(loadings.T)

    ev = np.var(loadings, axis=0) * len(efa_cols)
    communalities = np.sum(loadings**2, axis=1)
    using_fa_lib = False
    print("Using sklearn FactorAnalysis + manual varimax rotation")

# ── 5. Report factor loadings ─────────────────────────────────────────────────
factor_names = ['F1', 'F2', 'F3']
loadings_df = pd.DataFrame(loadings, index=efa_cols, columns=factor_names)

print("\nFactor loadings (varimax-rotated):")
print(loadings_df.round(3).to_string())

print("\nCommunalities:")
for i, col in enumerate(efa_cols):
    print(f"  {col}: {communalities[i]:.3f}")

print(f"\nEigenvalues: {ev[:5].round(3) if len(ev) >= 5 else ev.round(3)}")
print(f"Variance explained: {(ev[:3] / len(efa_cols) * 100).round(1)}%")

# Label factors by dominant loadings
print("\nFactor interpretation (loadings > |0.4|):")
for j, fname in enumerate(factor_names):
    high = [(efa_cols[i], loadings[i, j]) for i in range(len(efa_cols)) if abs(loadings[i, j]) > 0.4]
    high_sorted = sorted(high, key=lambda x: -abs(x[1]))
    print(f"  {fname}: {high_sorted}")

# ── 6. Build factor scores dataframe ──────────────────────────────────────────
scores_df = pd.DataFrame(factor_scores, columns=factor_names)
scores_df['subj'] = psych_clean['subj'].values
# Reorder
scores_df = scores_df[['subj'] + factor_names]

print(f"\nFactor scores: {scores_df.shape}")
print(scores_df[factor_names].describe().round(3))

# ── 7. Load behavioral parameters ─────────────────────────────────────────────
# k, β from L3_add (unified_3param_clean.csv)
choice_params = pd.read_csv(STATS / 'unified_3param_clean.csv')
print(f"\nChoice params: {len(choice_params)} subjects, cols: {choice_params.columns.tolist()}")

# α from vigor HBM
vigor_params = pd.read_csv(STATS / 'vigor_hbm_posteriors.csv')
print(f"Vigor params: {len(vigor_params)} subjects, cols: {vigor_params.columns.tolist()}")

# Merge all params
all_params = choice_params[['subj', 'k', 'beta']].merge(
    vigor_params[['subj', 'alpha_bayes', 'rho_bayes']].rename(
        columns={'alpha_bayes': 'alpha', 'rho_bayes': 'rho'}
    ),
    on='subj', how='inner'
)
print(f"Merged params: {len(all_params)} subjects")

# Merge with factor scores
analysis_df = scores_df.merge(all_params, on='subj', how='inner')
print(f"Analysis dataset: {len(analysis_df)} subjects")

# Z-score params for OLS
for col in ['k', 'beta', 'alpha', 'rho']:
    analysis_df[f'{col}_z'] = (analysis_df[col] - analysis_df[col].mean()) / analysis_df[col].std()

# ── 8. OLS: each factor ~ k + β + α ──────────────────────────────────────────
print("\n" + "="*60)
print("OLS: Factor score ~ k_z + beta_z + alpha_z")
print("="*60)

ols_results = []

for factor in factor_names:
    formula = f"{factor} ~ k_z + beta_z + alpha_z"
    ols = smf.ols(formula, data=analysis_df).fit()

    r2    = ols.rsquared
    r2adj = ols.rsquared_adj
    fstat = ols.fvalue
    fpval = ols.f_pvalue

    print(f"\n{factor} ~ k + β + α  (N={len(analysis_df)}):")
    print(f"  R²={r2:.4f}, R²adj={r2adj:.4f}, F={fstat:.3f}, p={fpval:.4f}")

    for pred in ['k_z', 'beta_z', 'alpha_z']:
        coef = ols.params[pred]
        se   = ols.bse[pred]
        tval = ols.tvalues[pred]
        pval = ols.pvalues[pred]
        print(f"  {pred}: β={coef:.4f}, SE={se:.4f}, t={tval:.3f}, p={pval:.4f}")
        ols_results.append({
            'factor':    factor,
            'predictor': pred,
            'beta':      coef,
            'se':        se,
            't':         tval,
            'p':         pval,
            'r2_model':  r2,
            'r2adj':     r2adj,
            'f_model':   fstat,
            'f_p':       fpval,
            'n':         len(analysis_df),
        })

ols_df = pd.DataFrame(ols_results)

# FDR correction
from statsmodels.stats.multitest import multipletests
reject, pvals_adj, _, _ = multipletests(ols_df['p'].values, method='fdr_bh')
ols_df['p_fdr'] = pvals_adj
ols_df['sig_fdr'] = reject

print("\n" + "="*60)
print("All OLS results with FDR correction:")
print(ols_df[['factor','predictor','beta','se','t','p','p_fdr','sig_fdr']].to_string(index=False))

# ── 9. Also run with rho included ─────────────────────────────────────────────
print("\n" + "="*60)
print("OLS: Factor score ~ k_z + beta_z + alpha_z + rho_z (full 4-param)")
print("="*60)

ols_results_full = []
for factor in factor_names:
    formula = f"{factor} ~ k_z + beta_z + alpha_z + rho_z"
    ols = smf.ols(formula, data=analysis_df).fit()

    r2    = ols.rsquared
    r2adj = ols.rsquared_adj
    fstat = ols.fvalue
    fpval = ols.f_pvalue

    print(f"\n{factor} ~ k + β + α + ρ  (N={len(analysis_df)}):")
    print(f"  R²={r2:.4f}, R²adj={r2adj:.4f}, F={fstat:.3f}, p={fpval:.4f}")

    for pred in ['k_z', 'beta_z', 'alpha_z', 'rho_z']:
        coef = ols.params[pred]
        se   = ols.bse[pred]
        tval = ols.tvalues[pred]
        pval = ols.pvalues[pred]
        print(f"  {pred}: β={coef:.4f}, SE={se:.4f}, t={tval:.3f}, p={pval:.4f}")
        ols_results_full.append({
            'factor':    factor,
            'predictor': pred,
            'model':     '4param',
            'beta':      coef,
            'se':        se,
            't':         tval,
            'p':         pval,
            'r2_model':  r2,
            'r2adj':     r2adj,
            'f_model':   fstat,
            'f_p':       fpval,
            'n':         len(analysis_df),
        })

ols_full_df = pd.DataFrame(ols_results_full)
reject_f, pvals_adj_f, _, _ = multipletests(ols_full_df['p'].values, method='fdr_bh')
ols_full_df['p_fdr'] = pvals_adj_f
ols_full_df['sig_fdr'] = reject_f

# Combine 3-param and 4-param results
ols_df['model'] = '3param'
ols_combined = pd.concat([ols_df, ols_full_df], ignore_index=True)

# ── 10. Save results ──────────────────────────────────────────────────────────
# Factor loadings
out_loadings = STATS / 'psych_factor_loadings.csv'
loadings_df.reset_index().rename(columns={'index': 'subscale'}).to_csv(out_loadings, index=False)
print(f"\nSaved factor loadings → {out_loadings}")

# Factor scores
out_scores = STATS / 'psych_factor_scores.csv'
scores_df.to_csv(out_scores, index=False)
print(f"Saved factor scores → {out_scores}")

# Params→factors OLS results
out_params_factors = STATS / 'psych_params_to_factors.csv'
ols_combined.to_csv(out_params_factors, index=False)
print(f"Saved params→factors results → {out_params_factors}")

print("\nDone.")
