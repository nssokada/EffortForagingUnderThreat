"""
04_coherent_shift.py — Joint strategy shift (H4)
=================================================
Tests whether subjects show a coherent joint strategy shift under threat:
both choosing lower-effort options AND increasing vigor simultaneously.

Key constructs:
  - choice_shift:  per-subject slope of P(high-effort choice) ~ danger (model-free)
  - excess_shift:  per-subject δ (excess effort ~ danger, from 03_vigor_excess_effort.py)
  - coherent_shift: product of choice_shift and excess_shift (both negative × positive
                    = coherent: avoid high effort in choice BUT put more vigor in)
  - k × δ:         effort discounting × vigor sensitivity (H4a: dissociation)
  - β × δ:         threat bias × vigor sensitivity (H4b: alignment)

Analyses:
  1. Compute per-subject choice_shift (P(high) ~ danger)
  2. Merge with excess_shift (δ) from vigor pipeline
  3. Test coupling: r(choice_shift, excess_shift)
  4. Cross-parameter correlations: k×δ, β×δ
  5. Define coherent_shift → test against zero
  6. Multiple regression: reward ~ α_v + δ + k + β
  7. Strategy profile table (quadrants: high/low choice shift × high/low vigor shift)
  8. Save strategy_results.csv

Outputs:
    results/stats/paper/strategy_results.csv

Usage:
    export PATH="$HOME/.local/bin:$PATH"
    python3 notebooks/06_paper_pipeline/04_coherent_shift.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path('/workspace')
DATA_DIR = ROOT / 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
STAT_DIR = ROOT / 'results/stats'
OUT_DIR  = ROOT / 'results/stats/paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAM = 2.0  # λ from L3_add

# ── Load data ──────────────────────────────────────────────────────────────────
print('=' * 70)
print('STEP 1: Loading data')
print('=' * 70)

behavior = pd.read_csv(DATA_DIR / 'behavior.csv')
print(f'behavior.csv: {len(behavior)} trials, {behavior["subj"].nunique()} subjects')

# Load choice parameters (paper pipeline first, then fallback)
params_path = OUT_DIR / 'choice_params.csv'
if not params_path.exists():
    params_path = STAT_DIR / 'unified_3param_clean.csv'
    print(f'Using fallback choice params: {params_path}')
choice_params = pd.read_csv(params_path)
print(f'Choice params: {len(choice_params)} subjects, columns={choice_params.columns.tolist()}')

# Load vigor parameters (from 03_vigor_excess_effort.py)
vigor_path = OUT_DIR / 'vigor_params.csv'
if not vigor_path.exists():
    raise FileNotFoundError(
        f'vigor_params.csv not found at {vigor_path}\n'
        'Run 03_vigor_excess_effort.py first.'
    )
vigor_params = pd.read_csv(vigor_path)
print(f'Vigor params: {len(vigor_params)} subjects, columns={vigor_params.columns.tolist()}')

# ── Compute per-subject choice_shift (P(high) ~ danger) ───────────────────────
print('\n' + '=' * 70)
print('STEP 2: Compute per-subject choice_shift')
print('=' * 70)

# Compute danger per trial
behavior['S_trial'] = (1 - behavior['threat']) + behavior['threat'] / (1 + LAM * behavior['distance_H'])
behavior['danger']  = 1 - behavior['S_trial']

# Per-subject: slope of choice ~ danger
choice_shift_rows = []
for s in behavior['subj'].unique():
    sub = behavior[behavior['subj'] == s].dropna(subset=['choice', 'danger'])
    if len(sub) < 5 or sub['danger'].std() < 1e-6:
        continue
    slope, intercept, r_val, p_val, se = stats.linregress(sub['danger'].values, sub['choice'].values)
    choice_shift_rows.append({
        'subj':          s,
        'choice_shift':  slope,
        'choice_r':      r_val,
        'choice_p':      p_val,
    })

choice_shift_df = pd.DataFrame(choice_shift_rows)
print(f'choice_shift computed for {len(choice_shift_df)} subjects')
print(f'choice_shift: mean={choice_shift_df["choice_shift"].mean():.4f}, '
      f'SD={choice_shift_df["choice_shift"].std():.4f}')

# H test: choice_shift < 0 (danger → fewer high-effort choices)
t_cs, p_cs = stats.ttest_1samp(choice_shift_df['choice_shift'].values, 0)
print(f'choice_shift ≠ 0: t({len(choice_shift_df)-1})={t_cs:.3f}, p={p_cs:.4f}'
      + (' *' if p_cs < 0.05 else ' n.s.'))

# ── Merge all parameters ───────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 3: Merge choice and vigor parameters')
print('=' * 70)

# Rename vigor params if needed
if 'alpha_v' in vigor_params.columns:
    vigor_rename = {'alpha_v': 'alpha_v', 'delta': 'delta'}
else:
    vigor_rename = {}

merged = (
    choice_shift_df
    .merge(choice_params[['subj', 'k', 'beta']], on='subj', how='inner')
    .merge(vigor_params[['subj', 'alpha_v', 'delta']], on='subj', how='inner')
)
print(f'Merged N = {len(merged)} subjects')

# Log-transform k and beta (right-skewed)
merged['logk']   = np.log(merged['k'].clip(lower=1e-6))
merged['logbeta'] = np.log(merged['beta'].clip(lower=1e-6))

# ── Test coupling r(choice_shift, δ) ──────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 4: Choice–vigor coupling test')
print('=' * 70)

r_couple, p_couple = stats.pearsonr(merged['choice_shift'].values, merged['delta'].values)
print(f'r(choice_shift, δ): r={r_couple:+.4f}, p={p_couple:.4f}'
      + (' *' if p_couple < 0.05 else ' n.s.'))

# Also Spearman
rho_couple, p_rho = stats.spearmanr(merged['choice_shift'].values, merged['delta'].values)
print(f'ρ(choice_shift, δ): ρ={rho_couple:+.4f}, p={p_rho:.4f}'
      + (' *' if p_rho < 0.05 else ' n.s.'))

# ── Cross-parameter correlations ───────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 5: Cross-parameter correlations (k×δ, β×δ)')
print('=' * 70)

cross_pairs = [
    ('logk',    'delta',        'k × δ'),
    ('logbeta', 'delta',        'β × δ'),
    ('logk',    'choice_shift', 'k × choice_shift'),
    ('logbeta', 'choice_shift', 'β × choice_shift'),
    ('logk',    'logbeta',      'k × β'),
    ('alpha_v', 'delta',        'α_v × δ'),
]
corr_results = []
for col_a, col_b, label in cross_pairs:
    if col_a not in merged.columns or col_b not in merged.columns:
        continue
    x = merged[col_a].values
    y = merged[col_b].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    r_val, p_val = stats.pearsonr(x, y)
    sig = '*' if p_val < 0.05 else 'n.s.'
    print(f'  {label}: r={r_val:+.4f}, p={p_val:.4f} {sig}, n={mask.sum()}')
    corr_results.append({'pair': label, 'r': r_val, 'p': p_val, 'n': mask.sum()})

# ── Coherent shift ─────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 6: Coherent shift test')
print('=' * 70)

# Coherent shift: negative choice_shift (avoid risky) AND positive delta (more vigor)
# Operationalise as: coherent = 1 if choice_shift < 0 AND delta > 0
merged['coherent'] = ((merged['choice_shift'] < 0) & (merged['delta'] > 0)).astype(int)
n_coherent = merged['coherent'].sum()
n_total    = len(merged)
prop       = n_coherent / n_total
# Binomial test vs 0.25 (chance: any of 4 quadrants)
from scipy.stats import binomtest
btest = binomtest(n_coherent, n_total, p=0.25, alternative='greater')
print(f'Coherent (avoid+vigor) subjects: {n_coherent}/{n_total} ({prop*100:.1f}%)')
print(f'Binomial test vs 25% chance: p={btest.pvalue:.4f}'
      + (' *' if btest.pvalue < 0.05 else ' n.s.'))

# Coherent shift as z-score product (continuous)
merged['cs_z']    = stats.zscore(merged['choice_shift'])
merged['delta_z'] = stats.zscore(merged['delta'])
# Positive coherent_shift_z = opposite-signed product because choice_shift should be < 0
merged['coherent_shift_z'] = -merged['cs_z'] * merged['delta_z']
t_cz, p_cz = stats.ttest_1samp(merged['coherent_shift_z'].values, 0)
print(f'Coherent shift index > 0: t({n_total-1})={t_cz:.3f}, p={p_cz:.4f}'
      + (' *' if p_cz < 0.05 else ' n.s.'))

# ── Multiple regression: reward ~ α_v + δ + k + β ─────────────────────────────
print('\n' + '=' * 70)
print('STEP 7: Multiple regression — reward ~ α_v + δ + k + β')
print('=' * 70)

# Compute per-subject mean reward from behavior
subj_reward = behavior.groupby('subj')['outcome'].mean().reset_index()
subj_reward.columns = ['subj', 'mean_reward']
merged_r = merged.merge(subj_reward, on='subj', how='inner')

# OLS via numpy
from numpy.linalg import lstsq

dv     = merged_r['mean_reward'].values
preds  = {
    'alpha_v': merged_r['alpha_v'].values,
    'delta':   merged_r['delta'].values,
    'logk':    merged_r['logk'].values,
    'logbeta': merged_r['logbeta'].values,
}

# Z-score predictors
X = np.column_stack([stats.zscore(v) for v in preds.values()])
X = np.hstack([np.ones((len(X), 1)), X])   # add intercept

beta_hat, residuals, rank, sv = lstsq(X, dv, rcond=None)
y_hat = X @ beta_hat
ss_res = np.sum((dv - y_hat) ** 2)
ss_tot = np.sum((dv - dv.mean()) ** 2)
r_sq   = 1 - ss_res / ss_tot
n, p   = X.shape
sigma2 = ss_res / (n - p)
cov    = sigma2 * np.linalg.pinv(X.T @ X)
se_hat = np.sqrt(np.diag(cov))
t_hat  = beta_hat / se_hat
p_hat  = 2 * stats.t.sf(np.abs(t_hat), df=n - p)

print(f'R² = {r_sq:.4f}  (adjusted R² = {1 - (1-r_sq)*(n-1)/(n-p-1):.4f})')
print(f'{"Predictor":12s} {"β":>8s} {"SE":>8s} {"t":>8s} {"p":>8s}')
print('-' * 52)
terms = ['intercept'] + list(preds.keys())
for i, term in enumerate(terms):
    sig = ' *' if p_hat[i] < 0.05 else ''
    print(f'{term:12s} {beta_hat[i]:>8.4f} {se_hat[i]:>8.4f} '
          f'{t_hat[i]:>8.3f} {p_hat[i]:>8.4f}{sig}')

# ── Strategy profile table ─────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 8: Strategy profile table (quadrants)')
print('=' * 70)

merged['choice_dir'] = np.where(merged['choice_shift'] < 0, 'avoid', 'approach')
merged['vigor_dir']  = np.where(merged['delta'] > 0, 'high_vigor', 'low_vigor')

quadrants = merged.groupby(['choice_dir', 'vigor_dir']).size().reset_index(name='count')
quadrants['pct'] = (quadrants['count'] / len(merged) * 100).round(1)
print(quadrants.to_string(index=False))

# ── Save outputs ───────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 9: Saving outputs')
print('=' * 70)

# Main strategy table
strategy_cols = [
    'subj', 'choice_shift', 'choice_r', 'choice_p',
    'k', 'beta', 'alpha_v', 'delta',
    'coherent', 'coherent_shift_z', 'choice_dir', 'vigor_dir'
]
strategy_df = merged_r[strategy_cols + ['mean_reward']].copy()
strategy_df.to_csv(OUT_DIR / 'strategy_results.csv', index=False)
print(f'Saved: {OUT_DIR}/strategy_results.csv')

# Correlation table
corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(OUT_DIR / 'strategy_correlations.csv', index=False)
print(f'Saved: {OUT_DIR}/strategy_correlations.csv')

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'N subjects: {len(merged)}')
print(f'choice_shift: mean={choice_shift_df["choice_shift"].mean():.4f}, t={t_cs:.3f}, p={p_cs:.4f}')
print(f'r(choice_shift, δ): r={r_couple:+.4f}, p={p_couple:.4f}')
print(f'Coherent subjects: {n_coherent}/{n_total} ({prop*100:.1f}%), binom p={btest.pvalue:.4f}')
print(f'Reward regression R² = {r_sq:.4f}')
print('\nDone.')
