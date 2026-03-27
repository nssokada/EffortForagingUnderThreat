#!/usr/bin/env python3
"""
NB09 Final Stats — Model-free vigor analyses.
Runs all Step 1 LMM tests and saves results to results/stats/step1_modelfree_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

ROOT        = Path('/workspace')
VIGOR_PROC  = ROOT / 'data' / 'exploratory_350' / 'processed' / 'vigor_processed'
RESULTS_DIR = ROOT / 'results' / 'stats'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──
print('Loading phase_vigor_metrics.parquet ...')
pm = pd.read_parquet(VIGOR_PROC / 'phase_vigor_metrics.parquet')
df_trial = pm.copy()

# ── Column name harmonization ──
col_map = {
    'enc_spike_resid':    'encounter_spike_resid',
    'enc_spike_norm':     'encounter_spike_norm',
    'enc_pre_mean_resid': 'pre_encounter_vigor_resid',
    'enc_pre_mean_norm':  'pre_encounter_vigor_norm',
    'enc_post_mean_resid':'post_encounter_vigor_resid',
    'enc_post_mean_norm': 'post_encounter_vigor_norm',
    'term_mean_resid':    'terminal_mean_resid',
    'term_mean_norm':     'terminal_mean_norm',
    'term_slope_resid':   'terminal_auc_resid',
    'term_slope_norm':    'terminal_auc_norm',
}
df_trial = df_trial.rename(columns=col_map)
df_trial['onset_peak_resid'] = df_trial['onset_mean_resid']
df_trial['onset_peak_norm']  = df_trial['onset_mean_norm']

# ── Standardize column names ──
df_trial['attack'] = df_trial['isAttackTrial'].astype(int)

# ── Escaped ──
if 'escaped' not in df_trial.columns:
    if df_trial['outcome'].dtype == 'object':
        df_trial['escaped'] = (df_trial['outcome'] == 'escaped').astype(int)
    else:
        df_trial['escaped'] = (df_trial['outcome'] == 0).astype(int)

# ── Center predictors ──
df_trial['threat_c'] = df_trial['threat'] - 0.5
df_trial['attack_c'] = df_trial['attack'] - df_trial['attack'].mean()
df_trial['startDistance_z'] = (
    (df_trial['startDistance'] - df_trial['startDistance'].mean())
    / df_trial['startDistance'].std()
)

# ── Build df_attack ──
df_attack = df_trial[df_trial['attack'] == 1].copy()

# ── Confirm DVs exist ──
main_dvs = [
    'onset_slope_norm', 'onset_slope_resid',
    'onset_mean_resid', 'onset_peak_resid',
    'encounter_spike_norm', 'encounter_spike_resid',
    'pre_encounter_vigor_norm', 'pre_encounter_vigor_resid',
    'post_encounter_vigor_resid',
    'terminal_mean_norm', 'terminal_mean_resid',
    'terminal_auc_resid',
]
missing = [dv for dv in main_dvs if dv not in df_trial.columns]
assert len(missing) == 0, f'Missing DVs: {missing}'

print(f'df_trial: {len(df_trial):,} trials, {df_trial["subj"].nunique()} subjects')
print(f'df_attack: {len(df_attack):,} attack trials')
print(f'All {len(main_dvs)} main DVs confirmed present.')


# ── Helper: run LMM ──
def run_lmm(formula, data, group_col='subj'):
    """Fit a linear mixed model and return a tidy results dict for each predictor."""
    dv = formula.split('~')[0].strip()
    clean = data.dropna(subset=[dv]).reset_index(drop=True)
    model = smf.mixedlm(formula, data=clean, groups=clean[group_col])
    result = model.fit(reml=True, method='lbfgs')

    rows = []
    for pred in result.params.index:
        b  = result.params[pred]
        se = result.bse.get(pred, np.nan)
        p  = result.pvalues.get(pred, np.nan)
        rows.append({
            'predictor': pred,
            'beta': b, 'se': se, 'p': p,
            'ci_lo': b - 1.96 * se,
            'ci_hi': b + 1.96 * se,
        })

    coef_df = pd.DataFrame(rows)
    info = {
        'nobs':      int(result.nobs),
        'n_groups':  int(len(np.unique(result.model.groups))),
        'converged': result.converged,
        'aic':       result.aic,
    }
    return coef_df, info, result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Model-free tests
# ══════════════════════════════════════════════════════════════════════════════

summary_rows = []

# ─── 1A: Threat scales onset slope ───────────────────────────────────────────
print('\n=== STEP 1A: Threat scales onset slope ===')
for stream in ['norm', 'resid']:
    col = f'onset_slope_{stream}'
    desc = df_trial.groupby('threat')[col].agg(['mean', 'std', 'count'])
    print(f'\n  {col}:')
    for thr, row in desc.iterrows():
        print(f'    Threat {thr}: M = {row["mean"]:.4f}, SD = {row["std"]:.4f}, N = {int(row["count"])}')

    res, info, _ = run_lmm(f'{col} ~ threat_c + C(choice)', df_trial)
    r = res[res['predictor'] == 'threat_c'].iloc[0]
    tag = '*** ' if r['p'] < 0.001 else ('** ' if r['p'] < 0.01 else ('* ' if r['p'] < 0.05 else ''))
    print(f'  threat_c: β={r["beta"]:.4f}, SE={r["se"]:.4f}, p={r["p"]:.4f} {tag}')
    print(f'  N={info["nobs"]}, groups={info["n_groups"]}, converged={info["converged"]}')

    summary_rows.append({
        'Step': '1A',
        'Claim': 'Threat scales anticipatory vigor',
        'Model': f'onset_slope_{stream} ~ threat_c + C(choice) + (1|subj)',
        'Sample': 'All trials',
        'Key predictor': 'threat_c',
        'Stream': stream,
        'beta': r['beta'], 'SE': r['se'], 'p': r['p'],
        'CI_lo': r['ci_lo'], 'CI_hi': r['ci_hi'],
        'N_obs': int(info['nobs']), 'N_subj': int(info['n_groups']),
    })

# ─── 1B: Attack triggers encounter spike ─────────────────────────────────────
print('\n=== STEP 1B: Attack triggers encounter spike ===')
formula_1b = '{dv} ~ attack_c + threat_c + C(choice)'

for stream in ['norm', 'resid']:
    dv_spike = f'encounter_spike_{stream}'
    dv_pre   = f'pre_encounter_vigor_{stream}'

    print(f'\n  Stream: {stream}')

    # Encounter spike model
    res, info, _ = run_lmm(formula_1b.format(dv=dv_spike), df_trial)
    r = res[res['predictor'] == 'attack_c'].iloc[0]
    tag = '*** ' if r['p'] < 0.001 else ('** ' if r['p'] < 0.01 else ('* ' if r['p'] < 0.05 else ''))
    print(f'  {dv_spike}: attack_c β={r["beta"]:.4f}, p={r["p"]:.4f} {tag}, N={info["nobs"]}')
    summary_rows.append({
        'Step': '1B',
        'Claim': 'Attack triggers phasic spike',
        'Model': f'encounter_spike_{stream} ~ attack_c + threat_c + C(choice) + (1|subj)',
        'Sample': 'All trials',
        'Key predictor': 'attack_c',
        'Stream': stream,
        'beta': r['beta'], 'SE': r['se'], 'p': r['p'],
        'CI_lo': r['ci_lo'], 'CI_hi': r['ci_hi'],
        'N_obs': int(info['nobs']), 'N_subj': int(info['n_groups']),
    })

    # Manipulation check: pre_encounter_vigor ~ attack_c (should be n.s.)
    res_pre, info_pre, _ = run_lmm(formula_1b.format(dv=dv_pre), df_trial)
    r_pre = res_pre[res_pre['predictor'] == 'attack_c'].iloc[0]
    tag_pre = '*** ' if r_pre['p'] < 0.001 else ('** ' if r_pre['p'] < 0.01 else ('* ' if r_pre['p'] < 0.05 else 'n.s.'))
    print(f'  {dv_pre}: attack_c β={r_pre["beta"]:.4f}, p={r_pre["p"]:.4f} {tag_pre}')
    summary_rows.append({
        'Step': '1B-check',
        'Claim': 'Manipulation check (expect n.s.)',
        'Model': f'pre_encounter_vigor_{stream} ~ attack_c + threat_c + C(choice) + (1|subj)',
        'Sample': 'All trials',
        'Key predictor': 'attack_c',
        'Stream': stream,
        'beta': r_pre['beta'], 'SE': r_pre['se'], 'p': r_pre['p'],
        'CI_lo': r_pre['ci_lo'], 'CI_hi': r_pre['ci_hi'],
        'N_obs': int(info_pre['nobs']), 'N_subj': int(info_pre['n_groups']),
    })

# 1B: Tonic-phasic correlation
print('\n=== STEP 1B: Tonic-phasic correlation (attack trials) ===')
for stream in ['norm', 'resid']:
    pre_col   = f'pre_encounter_vigor_{stream}'
    spike_col = f'encounter_spike_{stream}'
    valid = df_attack.dropna(subset=[pre_col, spike_col])
    r_all, p_all = pearsonr(valid[pre_col], valid[spike_col])
    tag = '*** ' if p_all < 0.001 else ('** ' if p_all < 0.01 else ('* ' if p_all < 0.05 else 'n.s.'))
    print(f'  {stream}: r={r_all:.3f}, p={p_all:.2e} {tag}, N={len(valid)}')

    # By threat level
    for thr in sorted(valid['threat'].unique()):
        sub = valid[valid['threat'] == thr]
        r_t, p_t = pearsonr(sub[pre_col], sub[spike_col])
        print(f'    Threat {thr}: r={r_t:.3f}, p={p_t:.3f}, N={len(sub)}')

    summary_rows.append({
        'Step': '1B-corr',
        'Claim': 'Tonic-phasic tradeoff',
        'Model': f'Pearson r(pre_encounter_vigor_{stream}, encounter_spike_{stream})',
        'Sample': 'Attack trials',
        'Key predictor': 'r',
        'Stream': stream,
        'beta': r_all, 'SE': np.nan, 'p': p_all,
        'CI_lo': np.nan, 'CI_hi': np.nan,
        'N_obs': int(len(valid)), 'N_subj': int(valid['subj'].nunique()),
    })

# ─── 1C: Terminal vigor predicts escape ──────────────────────────────────────
print('\n=== STEP 1C: Terminal vigor predicts escape ===')
print(f'  Overall escape rate: {df_attack["escaped"].mean():.3f} '
      f'({df_attack["escaped"].sum()}/{len(df_attack)})')
for thr in sorted(df_attack['threat'].unique()):
    sub = df_attack[df_attack['threat'] == thr]
    print(f'  Threat {thr}: {sub["escaped"].mean():.3f} ({sub["escaped"].sum()}/{len(sub)})')

for stream in ['norm', 'resid']:
    dv       = f'terminal_mean_{stream}'
    dv_z     = f'{dv}_z'
    valid    = df_attack.dropna(subset=[dv, 'escaped', 'startDistance_z']).copy()
    valid[dv_z] = (valid[dv] - valid[dv].mean()) / valid[dv].std()
    res, info, _ = run_lmm(
        f'escaped ~ {dv_z} + threat_c + startDistance_z + C(choice)', valid)
    r = res[res['predictor'] == dv_z].iloc[0]
    tag = '*** ' if r['p'] < 0.001 else ('** ' if r['p'] < 0.01 else ('* ' if r['p'] < 0.05 else 'n.s.'))
    print(f'\n  {stream}: terminal_mean β={r["beta"]:.4f}, SE={r["se"]:.4f}, p={r["p"]:.4f} {tag}')
    print(f'  N={info["nobs"]}, groups={info["n_groups"]}')

    summary_rows.append({
        'Step': '1C',
        'Claim': 'Terminal vigor predicts escape',
        'Model': f'escaped ~ terminal_mean_{stream}_z + threat_c + startDistance_z + C(choice) + (1|subj)',
        'Sample': 'Attack trials',
        'Key predictor': f'terminal_mean_{stream}_z',
        'Stream': stream,
        'beta': r['beta'], 'SE': r['se'], 'p': r['p'],
        'CI_lo': r['ci_lo'], 'CI_hi': r['ci_hi'],
        'N_obs': int(info['nobs']), 'N_subj': int(info['n_groups']),
    })

# ── Save ──
summary_df = pd.DataFrame(summary_rows)
out_path = RESULTS_DIR / 'step1_modelfree_results.csv'
summary_df.to_csv(out_path, index=False)
print(f'\nRaw results saved → {out_path}')
print(f'{len(summary_df)} rows collected.')
print('\nSummary:')
print(summary_df[['Step', 'Stream', 'Key predictor', 'beta', 'SE', 'p']].to_string(index=False))
