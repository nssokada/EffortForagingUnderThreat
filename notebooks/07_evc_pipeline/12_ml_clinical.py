#!/usr/bin/env python3
"""
ML-based clinical prediction using EVC parameters + metacognitive measures.

Analyses:
  1. Elastic Net predicting each clinical subscale (z-scored)
  2. Elastic Net predicting factor scores (F1=distress, F2=fatigue, F3=amotivation)
  3. Ridge regression comparison (no sparsity)
  4. Feature importance (permutation) for best-predicted outcome
  5. Cross-validated R² comparison: params-only vs params+metacognition
  6. Summary figure (3 panels)

Output:
  results/stats/evc_ml_clinical.csv
  results/figs/paper/fig_ml_clinical.png
"""

import sys
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'axes.spines.right': False,
    'axes.spines.top': False,
})

# ── Load data ──────────────────────────────────────────────────────────────
params = pd.read_csv('/workspace/results/stats/oc_evc_final_params.csv')
psych = pd.read_csv('/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950/psych.csv')
factors = pd.read_csv('/workspace/results/stats/psych_factor_scores.csv')
feelings = pd.read_csv('/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950/feelings.csv')

print(f"Params N = {len(params)}, Psych N = {len(psych)}, Factors N = {len(factors)}")
print(f"Feelings rows = {len(feelings)}, unique subj = {feelings['subj'].nunique()}")

# ── Compute calibration & discrepancy per subject ─────────────────────────
# Filter to anxiety probes only
anx = feelings[feelings['questionLabel'] == 'anxiety'].copy()
print(f"Anxiety probes: {len(anx)} rows, {anx['subj'].nunique()} subjects")

# S_probe = (1 - T^0.21) + 0.098 * T^0.21 * 0.6
anx['S_probe'] = (1 - anx['threat']**0.21) + 0.098 * anx['threat']**0.21 * 0.6

# Calibration = within-subject correlation r(anxiety, 1 - S)
# Discrepancy = mean residual of anxiety after regressing out S
meta_records = []
for subj, grp in anx.groupby('subj'):
    anxiety_vals = grp['response'].values.astype(float)
    s_vals = grp['S_probe'].values

    # Calibration: r(anxiety, 1 - S)
    if len(grp) >= 3 and np.std(anxiety_vals) > 0 and np.std(1 - s_vals) > 0:
        cal, _ = pearsonr(anxiety_vals, 1 - s_vals)
    else:
        cal = np.nan

    # Discrepancy: mean residual of anxiety regressed on S
    if len(grp) >= 3:
        # Simple OLS: anxiety ~ S
        X_reg = np.column_stack([np.ones(len(s_vals)), s_vals])
        try:
            beta = np.linalg.lstsq(X_reg, anxiety_vals, rcond=None)[0]
            resid = anxiety_vals - X_reg @ beta
            disc = np.mean(resid)
        except:
            disc = np.nan
    else:
        disc = np.nan

    meta_records.append({'subj': subj, 'calibration': cal, 'discrepancy': disc})

meta_df = pd.DataFrame(meta_records)
print(f"Metacognitive measures computed: {len(meta_df)} subjects, "
      f"calibration NaN = {meta_df['calibration'].isna().sum()}, "
      f"discrepancy NaN = {meta_df['discrepancy'].isna().sum()}")

# ── Merge everything ──────────────────────────────────────────────────────
df = params.merge(psych, on='subj').merge(factors, on='subj').merge(meta_df, on='subj')
df = df.dropna(subset=['calibration', 'discrepancy', 'c_effort', 'c_death'])
print(f"\nFinal merged N = {len(df)}")

# Log-transform parameters
df['log_ce'] = np.log(df['c_effort'])
df['log_cd'] = np.log(df['c_death'])

# Clinical subscales (exclude RT columns and totals that are sums of subscales)
clinical_cols = [c for c in psych.columns
                 if c not in ('participantID', 'subj')
                 and '_RT' not in c]
print(f"Clinical subscales ({len(clinical_cols)}): {clinical_cols}")

# ── Define feature sets ───────────────────────────────────────────────────
# Full feature set
feature_names_full = ['log_ce', 'log_cd', 'discrepancy', 'calibration',
                      'log_ce_x_log_cd', 'log_ce_x_disc', 'log_cd_x_disc']

# Params-only feature set
feature_names_params = ['log_ce', 'log_cd']

# Create interaction features
df['log_ce_x_log_cd'] = df['log_ce'] * df['log_cd']
df['log_ce_x_disc'] = df['log_ce'] * df['discrepancy']
df['log_cd_x_disc'] = df['log_cd'] * df['discrepancy']

# Z-score all features
scaler = StandardScaler()
X_full = scaler.fit_transform(df[feature_names_full].values)
X_params = StandardScaler().fit_transform(df[feature_names_params].values)

# CV strategy
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)

# ── Analysis 1: Elastic Net for each clinical subscale ────────────────────
print("\n" + "="*80)
print("ANALYSIS 1: Elastic Net → Clinical Subscales")
print("="*80)

enet_results = []
for col in clinical_cols:
    y = zscore(df[col].values)

    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        n_alphas=50, cv=10, random_state=42, max_iter=10000
    )
    enet.fit(X_full, y)

    # CV R² via RepeatedKFold
    scores = cross_val_score(enet, X_full, y, cv=cv, scoring='r2')

    # Non-zero coefficients
    nonzero = {feature_names_full[i]: round(enet.coef_[i], 4)
               for i in range(len(feature_names_full)) if abs(enet.coef_[i]) > 1e-6}

    result = {
        'outcome': col,
        'model': 'ElasticNet',
        'feature_set': 'full',
        'cv_r2_mean': scores.mean(),
        'cv_r2_std': scores.std(),
        'l1_ratio': enet.l1_ratio_,
        'alpha': round(enet.alpha_, 6),
        'n_nonzero': len(nonzero),
        'nonzero_features': str(nonzero),
    }
    enet_results.append(result)

    print(f"\n{col}:")
    print(f"  CV R² = {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  l1_ratio = {enet.l1_ratio_}, alpha = {enet.alpha_:.6f}")
    print(f"  Non-zero ({len(nonzero)}): {nonzero}")

enet_df = pd.DataFrame(enet_results)

# ── Analysis 2: Elastic Net → Factor Scores ──────────────────────────────
print("\n" + "="*80)
print("ANALYSIS 2: Elastic Net → Factor Scores")
print("="*80)

factor_results = []
for fcol in ['F1', 'F2', 'F3']:
    y = zscore(df[fcol].values)

    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        n_alphas=50, cv=10, random_state=42, max_iter=10000
    )
    enet.fit(X_full, y)

    scores = cross_val_score(enet, X_full, y, cv=cv, scoring='r2')

    nonzero = {feature_names_full[i]: round(enet.coef_[i], 4)
               for i in range(len(feature_names_full)) if abs(enet.coef_[i]) > 1e-6}

    label = {'F1': 'F1_distress', 'F2': 'F2_fatigue', 'F3': 'F3_amotivation'}[fcol]
    result = {
        'outcome': label,
        'model': 'ElasticNet',
        'feature_set': 'full',
        'cv_r2_mean': scores.mean(),
        'cv_r2_std': scores.std(),
        'l1_ratio': enet.l1_ratio_,
        'alpha': round(enet.alpha_, 6),
        'n_nonzero': len(nonzero),
        'nonzero_features': str(nonzero),
    }
    factor_results.append(result)

    print(f"\n{label}:")
    print(f"  CV R² = {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  l1_ratio = {enet.l1_ratio_}, alpha = {enet.alpha_:.6f}")
    print(f"  Non-zero ({len(nonzero)}): {nonzero}")

factor_df = pd.DataFrame(factor_results)

# ── Analysis 3: Ridge Regression (comparison) ────────────────────────────
print("\n" + "="*80)
print("ANALYSIS 3: Ridge Regression Comparison")
print("="*80)

ridge_results = []
all_outcomes = clinical_cols + ['F1', 'F2', 'F3']
outcome_labels = clinical_cols + ['F1_distress', 'F2_fatigue', 'F3_amotivation']

for col, label in zip(all_outcomes, outcome_labels):
    y = zscore(df[col].values)

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=10)
    ridge.fit(X_full, y)

    scores = cross_val_score(ridge, X_full, y, cv=cv, scoring='r2')

    result = {
        'outcome': label,
        'model': 'Ridge',
        'feature_set': 'full',
        'cv_r2_mean': scores.mean(),
        'cv_r2_std': scores.std(),
        'alpha': round(ridge.alpha_, 6),
    }
    ridge_results.append(result)

    print(f"  {label}: CV R² = {scores.mean():.4f} ± {scores.std():.4f} (alpha={ridge.alpha_:.4f})")

ridge_df = pd.DataFrame(ridge_results)

# ── Analysis 4: Feature Importance (best outcome) ────────────────────────
print("\n" + "="*80)
print("ANALYSIS 4: Permutation Feature Importance")
print("="*80)

# Find best-predicted outcome across all ElasticNet results
all_enet = pd.concat([enet_df, factor_df], ignore_index=True)
best_idx = int(all_enet['cv_r2_mean'].idxmax())
best_outcome = all_enet.loc[best_idx, 'outcome']
best_r2 = float(all_enet.loc[best_idx, 'cv_r2_mean'])
print(f"Best-predicted outcome: {best_outcome} (CV R² = {best_r2:.4f})")

# Map back to column name
if best_outcome in clinical_cols:
    best_col = best_outcome
else:
    best_col = {'F1_distress': 'F1', 'F2_fatigue': 'F2', 'F3_amotivation': 'F3'}.get(best_outcome, best_outcome)

y_best = zscore(df[best_col].values)

# Fit final model
enet_best = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
    n_alphas=50, cv=10, random_state=42, max_iter=10000
)
enet_best.fit(X_full, y_best)

# Permutation importance
perm_imp = permutation_importance(enet_best, X_full, y_best,
                                   n_repeats=30, random_state=42, scoring='r2')

print(f"\nPermutation importance for {best_outcome}:")
imp_order = np.argsort(perm_imp.importances_mean)[::-1]
for i in imp_order:
    print(f"  {feature_names_full[i]:20s}: {perm_imp.importances_mean[i]:.4f} ± {perm_imp.importances_std[i]:.4f}")

# ── Analysis 5: Params-only vs Params+Meta comparison ────────────────────
print("\n" + "="*80)
print("ANALYSIS 5: Params-only vs Params+Meta CV R² Comparison")
print("="*80)

comparison_results = []
for col, label in zip(all_outcomes, outcome_labels):
    y = zscore(df[col].values)

    # Params only
    enet_p = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                          n_alphas=50, cv=10, random_state=42, max_iter=10000)
    enet_p.fit(X_params, y)
    scores_p = cross_val_score(enet_p, X_params, y, cv=cv, scoring='r2')

    # Params + Meta (full)
    enet_f = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                          n_alphas=50, cv=10, random_state=42, max_iter=10000)
    enet_f.fit(X_full, y)
    scores_f = cross_val_score(enet_f, X_full, y, cv=cv, scoring='r2')

    improvement = scores_f.mean() - scores_p.mean()

    comparison_results.append({
        'outcome': label,
        'params_only_r2_mean': scores_p.mean(),
        'params_only_r2_std': scores_p.std(),
        'params_meta_r2_mean': scores_f.mean(),
        'params_meta_r2_std': scores_f.std(),
        'improvement': improvement,
    })

    print(f"  {label:25s}: params={scores_p.mean():.4f}±{scores_p.std():.4f}  "
          f"full={scores_f.mean():.4f}±{scores_f.std():.4f}  Δ={improvement:+.4f}")

comp_df = pd.DataFrame(comparison_results)

# ── Save results ──────────────────────────────────────────────────────────
# Combine all results into one CSV
all_results = pd.concat([
    enet_df.assign(analysis='enet_clinical'),
    factor_df.assign(analysis='enet_factor'),
    ridge_df.assign(analysis='ridge_all'),
], ignore_index=True)

all_results.to_csv('/workspace/results/stats/evc_ml_clinical.csv', index=False)
comp_df.to_csv('/workspace/results/stats/evc_ml_clinical_comparison.csv', index=False)
print(f"\nSaved results to results/stats/evc_ml_clinical.csv")
print(f"Saved comparison to results/stats/evc_ml_clinical_comparison.csv")

# ── Analysis 6: Summary Figure ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: CV R² bar chart for clinical measures (params only vs params+meta)
ax = axes[0]
clinical_comp = comp_df[~comp_df['outcome'].str.startswith('F')]
x_pos = np.arange(len(clinical_comp))
width = 0.35

# Short labels
short_labels = []
for name in clinical_comp['outcome']:
    short = name.replace('DASS21_', 'D-').replace('_Total', '').replace('_', ' ')
    short = short.replace('MFIS ', 'MF-').replace('AMI ', 'A-')
    short = short.replace('OASIS Total', 'OASIS').replace('PHQ9 Total', 'PHQ9')
    short = short.replace('STICSA Total', 'STICSA').replace('STAI State', 'STAI-S')
    short = short.replace('STAI Trait', 'STAI-T')
    short_labels.append(short)

bars1 = ax.bar(x_pos - width/2, clinical_comp['params_only_r2_mean'], width,
               yerr=clinical_comp['params_only_r2_std'], label='Params only',
               color='#4C72B0', alpha=0.8, capsize=2)
bars2 = ax.bar(x_pos + width/2, clinical_comp['params_meta_r2_mean'], width,
               yerr=clinical_comp['params_meta_r2_std'], label='Params + Meta',
               color='#DD8452', alpha=0.8, capsize=2)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xticks(x_pos)
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('CV R²')
ax.set_title('A. Clinical Subscale Prediction')
ax.legend(fontsize=8)

# Panel B: Feature importance for best outcome
ax = axes[1]
sorted_idx = np.argsort(perm_imp.importances_mean)
feature_short = [f.replace('log_ce_x_log_cd', 'ce×cd')
                  .replace('log_ce_x_disc', 'ce×disc')
                  .replace('log_cd_x_disc', 'cd×disc')
                  .replace('log_ce', 'log(cₑ)')
                  .replace('log_cd', 'log(c_d)')
                  .replace('discrepancy', 'discrepancy')
                  .replace('calibration', 'calibration')
                 for f in feature_names_full]

colors = ['#DD8452' if perm_imp.importances_mean[i] > 0 else '#4C72B0' for i in sorted_idx]
ax.barh(range(len(sorted_idx)), perm_imp.importances_mean[sorted_idx],
        xerr=perm_imp.importances_std[sorted_idx], color=[colors[j] for j in range(len(sorted_idx))],
        alpha=0.8, capsize=2)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([feature_short[i] for i in sorted_idx], fontsize=9)
ax.set_xlabel('Permutation Importance (ΔR²)')
ax.set_title(f'B. Feature Importance ({best_outcome})')
ax.axvline(0, color='k', lw=0.5, ls='--')

# Panel C: CV R² for factor scores
ax = axes[2]
factor_comp = comp_df[comp_df['outcome'].str.startswith('F')]
x_pos_f = np.arange(len(factor_comp))
factor_labels = ['F1\nDistress', 'F2\nFatigue', 'F3\nAmotivation']

bars1 = ax.bar(x_pos_f - width/2, factor_comp['params_only_r2_mean'], width,
               yerr=factor_comp['params_only_r2_std'], label='Params only',
               color='#4C72B0', alpha=0.8, capsize=3)
bars2 = ax.bar(x_pos_f + width/2, factor_comp['params_meta_r2_mean'], width,
               yerr=factor_comp['params_meta_r2_std'], label='Params + Meta',
               color='#DD8452', alpha=0.8, capsize=3)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xticks(x_pos_f)
ax.set_xticklabels(factor_labels, fontsize=10)
ax.set_ylabel('CV R²')
ax.set_title('C. Factor Score Prediction')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/workspace/results/figs/paper/fig_ml_clinical.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: results/figs/paper/fig_ml_clinical.png")

# ── Final summary table ──────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY: Cross-validated R² Comparison Table")
print("="*80)
print(f"{'Outcome':25s} {'Params R²':>15s} {'Params+Meta R²':>18s} {'Δ':>10s}")
print("-"*70)
for _, row in comp_df.iterrows():
    print(f"{row['outcome']:25s} "
          f"{row['params_only_r2_mean']:>7.4f}±{row['params_only_r2_std']:.4f} "
          f"{row['params_meta_r2_mean']:>7.4f}±{row['params_meta_r2_std']:.4f} "
          f"{row['improvement']:>+8.4f}")

print("\n" + "="*80)
print("ElasticNet vs Ridge comparison (full features)")
print("="*80)
for _, erow in all_enet.iterrows():
    rrow = ridge_df[ridge_df['outcome'] == erow['outcome']]
    if len(rrow):
        print(f"  {erow['outcome']:25s}: ENet={erow['cv_r2_mean']:.4f}  Ridge={rrow.iloc[0]['cv_r2_mean']:.4f}")

print("\nDone.")
