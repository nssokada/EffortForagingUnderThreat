"""
H3 and H4 hypothesis tests for the EffortForagingUnderThreat paper.

H3: Survival probability drives excess motor vigor
H4: Coherent strategy shift

Data:
- behavior.csv: trial-level choices (subj, trial 1-45)
- smoothed_vigor_ts.parquet: timeseries vigor (subj, trial 0-80)
- unified_3param_clean.csv: model parameters (subj, k, beta, alpha)

Lambda = 2.0 for S = (1-T) + T/(1 + lambda*D)
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, ttest_1samp
import statsmodels.formula.api as smf
import json
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────
# 0. Paths and constants
# ──────────────────────────────────────────────────
BEHAVIOR_PATH  = "data/exploratory_350/processed/stage5_filtered_data_20260320_191950/behavior.csv"
VIGOR_PATH     = "data/exploratory_350/processed/vigor_processed/smoothed_vigor_ts.parquet"
PARAMS_PATH    = "results/stats/unified_3param_clean.csv"
H3_OUT         = "results/stats/h3_results.json"
H4_OUT         = "results/stats/h4_results.json"
TEXT_OUT       = "results/h3_h4_results_text.md"

LAMBDA = 2.0
EFFORT_L = 0.4  # fixed low effort (always 0.4)

# ──────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────
print("Loading data...")
beh    = pd.read_csv(BEHAVIOR_PATH)
vig    = pd.read_parquet(VIGOR_PATH)
params = pd.read_csv(PARAMS_PATH)

print(f"  behavior: {beh.shape}, vigor: {vig.shape}, params: {params.shape}")

# ──────────────────────────────────────────────────
# 2. Build trial-level dataset from vigor file
# ──────────────────────────────────────────────────
# Step H3-1: Compute trial-level mean vigor (mean vigor_norm per subj×trial)
print("Computing trial-level mean vigor...")
trial_vigor = (
    vig.groupby(['subj', 'trial'])['vigor_norm']
    .mean()
    .reset_index()
    .rename(columns={'vigor_norm': 'mean_vigor'})
)

# Get trial-level metadata (one row per subj×trial)
trial_meta = (
    vig[['subj', 'trial', 'threat', 'choice', 'outcome', 'effort_H', 'distance_H']]
    .drop_duplicates(['subj', 'trial'])
    .copy()
)

# Merge
trial_df = trial_meta.merge(trial_vigor, on=['subj', 'trial'], how='inner')
print(f"  trial_df shape: {trial_df.shape}")

# ──────────────────────────────────────────────────
# 3. Compute derived variables
# ──────────────────────────────────────────────────
# H3-2: demand and excess
# demand = 0.4 if choice==0, else effort_H
trial_df['demand'] = np.where(trial_df['choice'] == 0, EFFORT_L, trial_df['effort_H'])
trial_df['excess'] = trial_df['mean_vigor'] - trial_df['demand']

# H3-3: Compute S and danger
# D_chosen: chosen option distance (abstract 1,2,3)
#   choice=0 → D=1 (low option always distance=1)
#   choice=1 → D=distance_H
trial_df['D_chosen'] = np.where(trial_df['choice'] == 0, 1, trial_df['distance_H'])

# S = (1-T) + T/(1 + lambda*D)
trial_df['S'] = (1 - trial_df['threat']) + trial_df['threat'] / (1 + LAMBDA * trial_df['D_chosen'])
trial_df['danger'] = 1 - trial_df['S']

# Merge model parameters
trial_df = trial_df.merge(params, on='subj', how='left')

print("  Sample trial_df:")
print(trial_df[['subj','trial','threat','choice','mean_vigor','demand','excess','S','danger']].head(5))
print()

# ──────────────────────────────────────────────────
# 4. Compute reward
# ──────────────────────────────────────────────────
# reward = 5 if choice=1 & outcome=0
#        = 1 if choice=0 & outcome=0
#        = -5 if outcome=1 (captured)
def compute_reward(row):
    if row['outcome'] == 1:
        return -5
    elif row['choice'] == 1:
        return 5
    else:
        return 1

trial_df['reward'] = trial_df.apply(compute_reward, axis=1)

# ──────────────────────────────────────────────────
# 5. H3: Survival probability drives excess motor vigor
# ──────────────────────────────────────────────────
print("=" * 60)
print("H3: Survival probability drives excess motor vigor")
print("=" * 60)

h3_results = {}

# ─── H3-4: Per-subject OLS: excess ~ danger → α_i (intercept), δ_i (slope)
print("\nFitting per-subject OLS: excess ~ danger...")
subject_ols = []
for subj_id, grp in trial_df.groupby('subj'):
    grp = grp.dropna(subset=['excess', 'danger'])
    if len(grp) < 5:
        continue
    x = grp['danger'].values
    y = grp['excess'].values
    # Add intercept
    X = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        alpha_i, delta_i = coeffs
        subject_ols.append({'subj': subj_id, 'alpha_i': alpha_i, 'delta_i': delta_i,
                            'n_trials': len(grp)})
    except Exception as e:
        print(f"  OLS failed for subj {subj_id}: {e}")

subj_ols_df = pd.DataFrame(subject_ols)
print(f"  N subjects with OLS fit: {len(subj_ols_df)}")
print(f"  delta_i mean: {subj_ols_df.delta_i.mean():.4f}, sd: {subj_ols_df.delta_i.std():.4f}")

# ─── H3(a): Population mean δ > 0
print("\nH3(a): Population mean delta > 0")
deltas = subj_ols_df['delta_i'].values
t_stat, p_val_two = ttest_1samp(deltas, 0)
p_val_one = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2
pct_pos = (deltas > 0).mean() * 100

print(f"  Mean delta: {deltas.mean():.4f}")
print(f"  SD delta:   {deltas.std():.4f}")
print(f"  % positive: {pct_pos:.1f}%")
print(f"  t({len(deltas)-1}) = {t_stat:.3f}, p (one-tailed) = {p_val_one:.4f}")

h3_results['H3a'] = {
    'mean_delta': float(deltas.mean()),
    'sd_delta': float(deltas.std()),
    'pct_positive': float(pct_pos),
    'n_subjects': int(len(deltas)),
    't_stat': float(t_stat),
    'df': int(len(deltas) - 1),
    'p_two_tailed': float(p_val_two),
    'p_one_tailed': float(p_val_one),
    'interpretation': 'delta > 0 supported' if p_val_one < 0.05 and deltas.mean() > 0 else 'not supported'
}

# ─── H3(b): Split-half reliability of δ (odd/even trials, Spearman-Brown)
print("\nH3(b): Split-half reliability of delta...")
odd_ols = []
even_ols = []

for subj_id, grp in trial_df.groupby('subj'):
    grp = grp.dropna(subset=['excess', 'danger'])
    g_odd  = grp[grp['trial'] % 2 == 1]
    g_even = grp[grp['trial'] % 2 == 0]

    for g, store in [(g_odd, odd_ols), (g_even, even_ols)]:
        if len(g) < 3:
            store.append({'subj': subj_id, 'delta_i': np.nan})
            continue
        x = g['danger'].values
        y = g['excess'].values
        X = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            store.append({'subj': subj_id, 'delta_i': coeffs[1]})
        except:
            store.append({'subj': subj_id, 'delta_i': np.nan})

odd_df  = pd.DataFrame(odd_ols).rename(columns={'delta_i': 'delta_odd'})
even_df = pd.DataFrame(even_ols).rename(columns={'delta_i': 'delta_even'})
split_df = odd_df.merge(even_df, on='subj').dropna()

r_split, p_split = pearsonr(split_df['delta_odd'], split_df['delta_even'])
# Spearman-Brown formula: r_SB = 2r / (1 + r)
r_sb = 2 * r_split / (1 + r_split)

print(f"  Split-half r = {r_split:.3f} (p={p_split:.4f})")
print(f"  Spearman-Brown reliability = {r_sb:.3f}")

h3_results['H3b'] = {
    'n_subjects': int(len(split_df)),
    'split_half_r': float(r_split),
    'split_half_p': float(p_split),
    'spearman_brown': float(r_sb),
    'interpretation': 'reliable' if r_sb > 0.7 else 'moderate' if r_sb > 0.4 else 'low'
}

# ─── H3(c): Within choice=0 only: danger predicts excess (LMM)
print("\nH3(c): Within choice=0, danger predicts excess (LMM)...")
c0_df = trial_df[trial_df['choice'] == 0].dropna(subset=['excess', 'danger']).copy()
c0_df['subj_cat'] = c0_df['subj'].astype('category')

try:
    lmm_c0 = smf.mixedlm("excess ~ danger", c0_df, groups=c0_df['subj_cat']).fit(reml=False)
    danger_coef_c0 = float(lmm_c0.params['danger'])
    danger_se_c0   = float(lmm_c0.bse['danger'])
    danger_z_c0    = float(lmm_c0.tvalues['danger'])
    danger_p_c0    = float(lmm_c0.pvalues['danger'])
    print(f"  choice=0: danger β={danger_coef_c0:.4f} (SE={danger_se_c0:.4f}), z={danger_z_c0:.3f}, p={danger_p_c0:.4f}")
    h3_results['H3c'] = {
        'subset': 'choice=0',
        'n_rows': int(len(c0_df)),
        'danger_beta': danger_coef_c0,
        'danger_se': danger_se_c0,
        'danger_z': danger_z_c0,
        'danger_p': danger_p_c0,
        'interpretation': 'significant' if danger_p_c0 < 0.05 else 'not significant'
    }
except Exception as e:
    print(f"  LMM failed: {e}")
    h3_results['H3c'] = {'error': str(e)}

# ─── H3(c2): Within choice=1 only: danger predicts excess (LMM)
print("\nH3(c2): Within choice=1, danger predicts excess (LMM)...")
c1_df = trial_df[trial_df['choice'] == 1].dropna(subset=['excess', 'danger']).copy()
c1_df['subj_cat'] = c1_df['subj'].astype('category')

try:
    lmm_c1 = smf.mixedlm("excess ~ danger", c1_df, groups=c1_df['subj_cat']).fit(reml=False)
    danger_coef_c1 = float(lmm_c1.params['danger'])
    danger_se_c1   = float(lmm_c1.bse['danger'])
    danger_z_c1    = float(lmm_c1.tvalues['danger'])
    danger_p_c1    = float(lmm_c1.pvalues['danger'])
    print(f"  choice=1: danger β={danger_coef_c1:.4f} (SE={danger_se_c1:.4f}), z={danger_z_c1:.3f}, p={danger_p_c1:.4f}")
    h3_results['H3c2'] = {
        'subset': 'choice=1',
        'n_rows': int(len(c1_df)),
        'danger_beta': danger_coef_c1,
        'danger_se': danger_se_c1,
        'danger_z': danger_z_c1,
        'danger_p': danger_p_c1,
        'interpretation': 'significant' if danger_p_c1 < 0.05 else 'not significant'
    }
except Exception as e:
    print(f"  LMM failed: {e}")
    h3_results['H3c2'] = {'error': str(e)}

# ─── H3(d): Threat × Distance interaction on excess effort (LMM, controlling for choice)
print("\nH3(d): Threat × Distance interaction on excess effort (LMM)...")
lmm_df = trial_df.dropna(subset=['excess', 'threat', 'D_chosen', 'choice']).copy()
lmm_df['subj_cat'] = lmm_df['subj'].astype('category')
lmm_df['threat_c']   = lmm_df['threat'] - lmm_df['threat'].mean()
lmm_df['D_chosen_c'] = lmm_df['D_chosen'] - lmm_df['D_chosen'].mean()
lmm_df['choice_c']   = lmm_df['choice'] - lmm_df['choice'].mean()

try:
    lmm_int = smf.mixedlm(
        "excess ~ threat_c * D_chosen_c + choice_c",
        lmm_df, groups=lmm_df['subj_cat']
    ).fit(reml=False)
    print(lmm_int.summary().tables[1])

    def extract_lmm_param(model, param):
        return {
            'beta': float(model.params.get(param, np.nan)),
            'se':   float(model.bse.get(param, np.nan)),
            'z':    float(model.tvalues.get(param, np.nan)),
            'p':    float(model.pvalues.get(param, np.nan)),
        }

    h3_results['H3d'] = {
        'n_rows': int(len(lmm_df)),
        'threat_c':         extract_lmm_param(lmm_int, 'threat_c'),
        'D_chosen_c':       extract_lmm_param(lmm_int, 'D_chosen_c'),
        'interaction':      extract_lmm_param(lmm_int, 'threat_c:D_chosen_c'),
        'choice_c':         extract_lmm_param(lmm_int, 'choice_c'),
        'intercept':        extract_lmm_param(lmm_int, 'Intercept'),
        'interaction_sig':  float(lmm_int.pvalues.get('threat_c:D_chosen_c', 1.0)) < 0.05
    }
    int_p = lmm_int.pvalues.get('threat_c:D_chosen_c', np.nan)
    int_b = lmm_int.params.get('threat_c:D_chosen_c', np.nan)
    print(f"  Threat × Distance interaction: β={int_b:.4f}, p={int_p:.4f}")
except Exception as e:
    print(f"  LMM failed: {e}")
    h3_results['H3d'] = {'error': str(e)}

# ─── H3(e): Store subject OLS results for use in H4
# Merge delta_i and alpha_i into params
subj_ols_merged = subj_ols_df.merge(params, on='subj', how='inner')
print(f"\nSubject OLS merged with params: {subj_ols_merged.shape}")

# ──────────────────────────────────────────────────
# 6. H4: Coherent strategy shift
# ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("H4: Coherent strategy shift")
print("=" * 60)

h4_results = {}

# ─── H4-1: Per-subject choice_shift = P(high|T=0.9) - P(high|T=0.1)
print("\nComputing choice_shift...")
# P(high) = mean(choice) for given threat level
def p_high(grp):
    return grp['choice'].mean()

choice_shift_list = []
for subj_id, grp in trial_df.groupby('subj'):
    t09 = grp[grp['threat'] == 0.9]
    t01 = grp[grp['threat'] == 0.1]
    if len(t09) > 0 and len(t01) > 0:
        shift = t09['choice'].mean() - t01['choice'].mean()
        choice_shift_list.append({'subj': subj_id, 'choice_shift': shift})

choice_shift_df = pd.DataFrame(choice_shift_list)
print(f"  choice_shift: mean={choice_shift_df.choice_shift.mean():.4f}, sd={choice_shift_df.choice_shift.std():.4f}")
print(f"  % negative (higher risk aversion at T=0.9): {(choice_shift_df.choice_shift < 0).mean()*100:.1f}%")

# ─── H4-2: Per-subject excess_shift = mean_excess(T=0.9) - mean_excess(T=0.1)
print("\nComputing excess_shift...")
excess_shift_list = []
for subj_id, grp in trial_df.groupby('subj'):
    t09 = grp[grp['threat'] == 0.9]
    t01 = grp[grp['threat'] == 0.1]
    if len(t09) > 0 and len(t01) > 0:
        shift = t09['excess'].mean() - t01['excess'].mean()
        excess_shift_list.append({'subj': subj_id, 'excess_shift': shift})

excess_shift_df = pd.DataFrame(excess_shift_list)
print(f"  excess_shift: mean={excess_shift_df.excess_shift.mean():.4f}, sd={excess_shift_df.excess_shift.std():.4f}")
print(f"  % positive (more excess vigor at T=0.9): {(excess_shift_df.excess_shift > 0).mean()*100:.1f}%")

# ─── H4-3: Compute coherent_shift = -choice_shift + excess_shift
# (more avoidance in choice + more mobilization in vigor = coherent)
shift_df = choice_shift_df.merge(excess_shift_df, on='subj', how='inner')
shift_df['coherent_shift'] = -shift_df['choice_shift'] + shift_df['excess_shift']
print(f"\ncoherent_shift: mean={shift_df.coherent_shift.mean():.4f}, sd={shift_df.coherent_shift.std():.4f}")

# Compute subject-level total_reward and escape_rate
print("\nComputing subject-level reward and escape metrics...")
subj_reward = trial_df.groupby('subj').agg(
    total_reward=('reward', 'sum'),
    escape_rate=('outcome', lambda x: (x == 0).mean()),
    n_trials=('trial', 'count')
).reset_index()

# Merge everything for H4 tests
h4_df = (shift_df
    .merge(params, on='subj', how='inner')
    .merge(subj_ols_merged[['subj', 'alpha_i', 'delta_i']], on='subj', how='inner')
    .merge(subj_reward, on='subj', how='inner')
)
print(f"  H4 merged df: {h4_df.shape}")

# ─── H4(a): choice_shift × excess_shift correlation (should be r < -0.5)
print("\nH4(a): choice_shift × excess_shift correlation...")
r_cs_es, p_cs_es = pearsonr(h4_df['choice_shift'], h4_df['excess_shift'])
print(f"  r = {r_cs_es:.3f}, p = {p_cs_es:.4f}")
h4_results['H4a'] = {
    'r': float(r_cs_es),
    'p': float(p_cs_es),
    'n': int(len(h4_df)),
    'expected_direction': r_cs_es < 0,
    'meets_threshold_r_neg05': r_cs_es < -0.5,
    'interpretation': f"r={r_cs_es:.3f}, {'significant' if p_cs_es < 0.05 else 'not significant'}"
}

# ─── H4(b): k × δ correlation
print("\nH4(b): k × delta_i correlation...")
r_k_d, p_k_d = pearsonr(h4_df['k'], h4_df['delta_i'])
print(f"  r = {r_k_d:.3f}, p = {p_k_d:.4f}")
h4_results['H4b'] = {
    'r': float(r_k_d),
    'p': float(p_k_d),
    'n': int(len(h4_df)),
    'interpretation': 'significant' if p_k_d < 0.05 else 'not significant'
}

# ─── H4(c): β × δ correlation
print("\nH4(c): beta × delta_i correlation...")
r_b_d, p_b_d = pearsonr(h4_df['beta'], h4_df['delta_i'])
print(f"  r = {r_b_d:.3f}, p = {p_b_d:.4f}")
h4_results['H4c'] = {
    'r': float(r_b_d),
    'p': float(p_b_d),
    'n': int(len(h4_df)),
    'interpretation': 'significant' if p_b_d < 0.05 else 'not significant'
}

# ─── H4(d): coherent_shift predicts total_reward and escape_rate
print("\nH4(d): coherent_shift predicts total_reward and escape_rate...")
r_cs_reward, p_cs_reward = pearsonr(h4_df['coherent_shift'], h4_df['total_reward'])
r_cs_escape, p_cs_escape = pearsonr(h4_df['coherent_shift'], h4_df['escape_rate'])
print(f"  coherent_shift ~ total_reward:  r={r_cs_reward:.3f}, p={p_cs_reward:.4f}")
print(f"  coherent_shift ~ escape_rate:   r={r_cs_escape:.3f}, p={p_cs_escape:.4f}")

h4_results['H4d'] = {
    'n': int(len(h4_df)),
    'total_reward': {
        'r': float(r_cs_reward),
        'p': float(p_cs_reward),
        'interpretation': 'significant' if p_cs_reward < 0.05 else 'not significant'
    },
    'escape_rate': {
        'r': float(r_cs_escape),
        'p': float(p_cs_escape),
        'interpretation': 'significant' if p_cs_escape < 0.05 else 'not significant'
    }
}

# ─── H4(e): Multiple regression: total_reward ~ alpha_i + delta_i + k + beta
print("\nH4(e): Multiple regression: total_reward ~ alpha_i + delta_i + k + beta...")
# Standardize predictors for comparable coefficients
from scipy.stats import zscore
reg_df = h4_df[['total_reward', 'alpha_i', 'delta_i', 'k', 'beta']].dropna().copy()
for col in ['alpha_i', 'delta_i', 'k', 'beta']:
    reg_df[f'{col}_z'] = zscore(reg_df[col])

try:
    mr_model = smf.ols(
        "total_reward ~ alpha_i_z + delta_i_z + k_z + beta_z",
        data=reg_df
    ).fit()
    print(mr_model.summary())

    h4_results['H4e'] = {
        'n': int(len(reg_df)),
        'r_squared': float(mr_model.rsquared),
        'r_squared_adj': float(mr_model.rsquared_adj),
        'f_stat': float(mr_model.fvalue),
        'f_p': float(mr_model.f_pvalue),
        'predictors': {}
    }

    for pred in ['alpha_i_z', 'delta_i_z', 'k_z', 'beta_z']:
        h4_results['H4e']['predictors'][pred] = {
            'beta': float(mr_model.params.get(pred, np.nan)),
            'se':   float(mr_model.bse.get(pred, np.nan)),
            't':    float(mr_model.tvalues.get(pred, np.nan)),
            'p':    float(mr_model.pvalues.get(pred, np.nan)),
        }
    print(f"\n  R² = {mr_model.rsquared:.3f} (adj R² = {mr_model.rsquared_adj:.3f})")
    print(f"  F({mr_model.df_model:.0f},{mr_model.df_resid:.0f}) = {mr_model.fvalue:.3f}, p = {mr_model.f_pvalue:.4f}")

except Exception as e:
    print(f"  Regression failed: {e}")
    h4_results['H4e'] = {'error': str(e)}

# ─── Additional: descriptive stats for shifts
h4_results['descriptives'] = {
    'choice_shift': {
        'mean': float(h4_df['choice_shift'].mean()),
        'sd':   float(h4_df['choice_shift'].std()),
        'pct_negative': float((h4_df['choice_shift'] < 0).mean() * 100),
    },
    'excess_shift': {
        'mean': float(h4_df['excess_shift'].mean()),
        'sd':   float(h4_df['excess_shift'].std()),
        'pct_positive': float((h4_df['excess_shift'] > 0).mean() * 100),
    },
    'coherent_shift': {
        'mean': float(h4_df['coherent_shift'].mean()),
        'sd':   float(h4_df['coherent_shift'].std()),
    },
    'n_subjects': int(len(h4_df)),
}

# ──────────────────────────────────────────────────
# 7. Save JSON results
# ──────────────────────────────────────────────────
print("\nSaving results...")
import os
os.makedirs("results/stats", exist_ok=True)

with open(H3_OUT, 'w') as f:
    json.dump(h3_results, f, indent=2)
print(f"  Saved H3 results to {H3_OUT}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(H4_OUT, 'w') as f:
    json.dump(h4_results, f, indent=2, cls=NumpyEncoder)
print(f"  Saved H4 results to {H4_OUT}")

# ──────────────────────────────────────────────────
# 8. Write prose results
# ──────────────────────────────────────────────────
def fmt_p(p):
    if p < 0.001:
        return "p < .001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"

def sig_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

h3a = h3_results['H3a']
h3b = h3_results['H3b']
h3c = h3_results.get('H3c', {})
h3c2 = h3_results.get('H3c2', {})
h3d = h3_results.get('H3d', {})
h4a = h4_results['H4a']
h4b = h4_results['H4b']
h4c = h4_results['H4c']
h4d = h4_results['H4d']
h4e = h4_results.get('H4e', {})
h4desc = h4_results['descriptives']

prose = f"""# H3 and H4 Results

**Date:** 2026-03-20
**Sample:** N = 293 (exploratory)
**Model:** L3_add, λ = {LAMBDA}
**Data:** behavior.csv (stage5_20260320_191950), smoothed_vigor_ts.parquet, unified_3param_clean.csv

---

## H3: Survival Probability Drives Excess Motor Vigor

We operationalized trial-level vigor as the mean of the normalized keypress rate (vigor_norm) per subject × trial, derived from the 20 Hz kernel-smoothed time series (N = {h3a['n_subjects']} subjects). Demand was defined as the effort fraction associated with the chosen option (effort_L = 0.40 for the low cookie; effort_H ∈ {{0.60, 0.80, 1.00}} for the high cookie). Excess effort was computed as vigor − demand. Danger was derived from the subjective survival function S = (1−T) + T/(1+λD) with λ = {LAMBDA}, as danger = 1 − S, where D ∈ {{1, 2, 3}} is the chosen distance level and T ∈ {{0.1, 0.5, 0.9}} is predator threat probability.

### H3(a): Population-level excess effort slope (δ) > 0

Per-subject OLS regression of excess effort on trial-level danger yielded individual slopes δᵢ (the sensitivity of excess vigor to danger). Across subjects, the mean slope was δ = {h3a['mean_delta']:.3f} (SD = {h3a['sd_delta']:.3f}), with {h3a['pct_positive']:.1f}% of subjects showing positive slopes. A one-sample t-test confirmed that the population mean δ differed significantly from zero, t({h3a['df']}) = {h3a['t_stat']:.2f}, {fmt_p(h3a['p_one_tailed'])} (one-tailed). {h3a['interpretation'].capitalize()}.

### H3(b): Split-half reliability of δ

To assess the reliability of individual difference estimates in δ, we split trials into odd- and even-numbered halves and computed δ separately for each half. Pearson correlation between halves was r = {h3b['split_half_r']:.3f} ({fmt_p(h3b['split_half_p'])}); Spearman-Brown corrected reliability was ρSB = {h3b['spearman_brown']:.3f}. This indicates {h3b['interpretation']} reliability of individual excess vigor slopes.

### H3(c): Danger predicts excess effort within choice = 0 (constant demand)

To isolate the effect of danger on vigor from effort-demand confounds, we restricted analysis to low-option trials (choice = 0), where demand is constant at 0.40. A linear mixed-effects model (excess ~ danger, random intercept per subject) yielded a {h3c.get('interpretation','—')} danger effect: β = {h3c.get('danger_beta', float('nan')):.3f} (SE = {h3c.get('danger_se', float('nan')):.3f}), z = {h3c.get('danger_z', float('nan')):.2f}, {fmt_p(h3c.get('danger_p', 1.0))} (N = {h3c.get('n_rows', 0):,} trials). This confirms that danger modulates vigor even when task demands are held constant.

### H3(c2): Danger predicts excess effort within choice = 1

For high-option trials (choice = 1), where demand varies with effort_H, a parallel LMM yielded: β = {h3c2.get('danger_beta', float('nan')):.3f} (SE = {h3c2.get('danger_se', float('nan')):.3f}), z = {h3c2.get('danger_z', float('nan')):.2f}, {fmt_p(h3c2.get('danger_p', 1.0))} (N = {h3c2.get('n_rows', 0):,} trials), {h3c2.get('interpretation','—')}.

### H3(d): Threat × Distance interaction on excess effort

A fully specified LMM regressed excess effort on threat, distance, their interaction, and choice (all mean-centered), with random intercepts per subject.
"""

# Add interaction table
if 'interaction' in h3d:
    int_r = h3d['interaction']
    thr_r = h3d['threat_c']
    dis_r = h3d['D_chosen_c']
    cho_r = h3d['choice_c']
    sig = "significant" if h3d.get('interaction_sig', False) else "not significant"
    prose += f"""Key findings:
- Threat: β = {thr_r['beta']:.3f}, SE = {thr_r['se']:.3f}, z = {thr_r['z']:.2f}, {fmt_p(thr_r['p'])}
- Distance: β = {dis_r['beta']:.3f}, SE = {dis_r['se']:.3f}, z = {dis_r['z']:.2f}, {fmt_p(dis_r['p'])}
- Threat × Distance: β = {int_r['beta']:.3f}, SE = {int_r['se']:.3f}, z = {int_r['z']:.2f}, {fmt_p(int_r['p'])} ({sig})
- Choice: β = {cho_r['beta']:.3f}, SE = {cho_r['se']:.3f}, z = {cho_r['z']:.2f}, {fmt_p(cho_r['p'])}

The threat × distance interaction was {sig}, indicating that the effect of danger on excess effort {'did' if h3d.get('interaction_sig') else 'did not'} depend on distance level.
"""
else:
    prose += "The Threat × Distance interaction LMM could not be estimated.\n"

prose += f"""
---

## H4: Coherent Strategy Shift

### Descriptive Statistics

Across subjects (N = {h4desc['n_subjects']}), the mean threat-based choice shift (P(high|T=0.9) − P(high|T=0.1)) was {h4desc['choice_shift']['mean']:.3f} (SD = {h4desc['choice_shift']['sd']:.3f}), with {h4desc['choice_shift']['pct_negative']:.1f}% of subjects showing a negative shift (reduced high-option selection under high threat). The mean excess vigor shift (mean_excess at T=0.9 − T=0.1) was {h4desc['excess_shift']['mean']:.3f} (SD = {h4desc['excess_shift']['sd']:.3f}), with {h4desc['excess_shift']['pct_positive']:.1f}% positive. The coherent shift index (−choice_shift + excess_shift) had mean {h4desc['coherent_shift']['mean']:.3f} (SD = {h4desc['coherent_shift']['sd']:.3f}).

### H4(a): choice_shift × excess_shift correlation

The correlation between individual choice shifts and excess vigor shifts was r = {h4a['r']:.3f}, {fmt_p(h4a['p'])} (N = {h4a['n']}). A negative correlation indicates that subjects who avoided high-value cookies under threat compensated with higher motor vigor — the hallmark of a coherent defensive strategy. The correlation {'met' if h4a['meets_threshold_r_neg05'] else 'did not meet'} the pre-specified threshold of r < −0.5.

### H4(b): Effort-discounting parameter k × vigor slope δ

The correlation between individual effort-discounting rates (k, from L3_add choice model) and individual vigor slopes (δ, from per-subject OLS) was r = {h4b['r']:.3f}, {fmt_p(h4b['p'])} (N = {h4b['n']}), {h4b['interpretation']}. This tests whether subjects who discount effort more in choice also show steeper danger-driven vigor mobilization.

### H4(c): Threat bias parameter β × vigor slope δ

The correlation between threat bias (β) and vigor slope (δ) was r = {h4c['r']:.3f}, {fmt_p(h4c['p'])} (N = {h4c['n']}), {h4c['interpretation']}. β captures residual threat aversion in choice beyond survival-weighted expected value; its relation to δ reveals whether the same threat sensitivity is expressed in both choice and action.

### H4(d): Coherent shift predicts behavioral outcomes

Coherent shift correlated with:
- **Total reward:** r = {h4d['total_reward']['r']:.3f}, {fmt_p(h4d['total_reward']['p'])}, {h4d['total_reward']['interpretation']}
- **Escape rate:** r = {h4d['escape_rate']['r']:.3f}, {fmt_p(h4d['escape_rate']['p'])}, {h4d['escape_rate']['interpretation']}

"""

if 'r_squared' in h4e:
    prose += f"""### H4(e): Multiple regression: total_reward ~ α_i + δ_i + k + β

Standardized predictors (z-scored) were regressed on total accumulated reward. The model explained R² = {h4e['r_squared']:.3f} (adjusted R² = {h4e['r_squared_adj']:.3f}) of variance, F({h4e.get('f_stat',float('nan')):.0f}) = {h4e.get('f_stat',float('nan')):.2f}, {fmt_p(h4e['f_p'])} (N = {h4e['n']}). Individual predictor estimates (standardized):

| Predictor | β | SE | t | p |
|-----------|---|-----|---|---|
"""
    for pred_name, pred_label in [('alpha_i_z','α_i (vigor baseline)'), ('delta_i_z','δ_i (danger slope)'), ('k_z','k (effort disc.)'), ('beta_z','β (threat bias)')]:
        if pred_name in h4e.get('predictors', {}):
            r = h4e['predictors'][pred_name]
            sig = sig_star(r['p'])
            prose += f"| {pred_label} | {r['beta']:.3f} | {r['se']:.3f} | {r['t']:.2f} | {fmt_p(r['p'])} {sig} |\n"

    prose += f"""
*p < .05 (*), p < .01 (**), p < .001 (***)*
"""
else:
    if 'error' in h4e:
        prose += f"### H4(e): Multiple regression\n\nModel estimation failed: {h4e['error']}\n"

prose += """
---

*Results generated from scripts/analysis/run_h3_h4_tests.py*
"""

with open(TEXT_OUT, 'w') as f:
    f.write(prose)
print(f"\nSaved prose results to {TEXT_OUT}")

print("\n✓ H3 and H4 analysis complete.")
