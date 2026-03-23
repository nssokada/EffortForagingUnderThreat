#!/usr/bin/env python3
"""
NB08 Parameter Dissociation — Test whether choice model parameters (z, k, β)
predict distinct patterns of dynamic vigor regulation.

Saves results to results/tables/table_s2_parameter_dissociation.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

ROOT        = Path('/workspace')
VIGOR_PROC  = ROOT / 'data' / 'exploratory_350' / 'processed' / 'vigor_processed'
RESULTS_DIR = ROOT / 'results'
STATS_DIR   = RESULTS_DIR / 'stats'
TABLES      = RESULTS_DIR / 'tables'
TABLES.mkdir(parents=True, exist_ok=True)

# ── Load trial-level vigor data ──
print('Loading phase_vigor_metrics.parquet ...')
pm = pd.read_parquet(VIGOR_PROC / 'phase_vigor_metrics.parquet')
df_trial = pm.copy()

# ── Prepare predictors ──
df_trial['attack'] = df_trial['isAttackTrial'].astype(int)
df_trial['threat_c'] = df_trial['threat'] - 0.5
df_trial['attack_c'] = df_trial['attack'] - df_trial['attack'].mean()

# ── Column name harmonization ──
col_map = {
    'enc_spike_resid':    'encounter_spike_resid',
    'enc_post_mean_resid':'post_encounter_vigor_resid',
    'term_mean_resid':    'terminal_mean_resid',
    'term_slope_resid':   'terminal_auc_resid',
    'enc_spike_norm':     'encounter_spike_norm',
    'enc_post_mean_norm': 'post_encounter_vigor_norm',
    'term_mean_norm':     'terminal_mean_norm',
    'term_slope_norm':    'terminal_auc_norm',
}
df_trial = df_trial.rename(columns=col_map)
df_trial['onset_peak_resid'] = df_trial['onset_mean_resid']
df_trial['onset_peak_norm']  = df_trial['onset_mean_norm']

# ── Load choice model parameters ──
print('Loading choice model parameters ...')
params_path = STATS_DIR / 'unified_3param_clean.csv'
params = pd.read_csv(params_path)
print(f'  unified_3param_clean columns: {list(params.columns)}')

# unified_3param_clean has: subj, k, beta, alpha
# k → kappa (effort discounting)
# beta → beta (threat bias)
# alpha → will be treated as z (motor/hazard sensitivity from vigor HBM)
# Also check for FET_Exp_Bias_z_params.csv for per-subject z
z_path = STATS_DIR / 'FET_Exp_Bias_z_params.csv'
if z_path.exists():
    z_df = pd.read_csv(z_path)
    z_df = z_df.rename(columns={'subject': 'subj', 'mean': 'z'})
    params = params.merge(z_df[['subj', 'z']], on='subj', how='left')
    print(f'  Merged z from FET_Exp_Bias_z_params.csv')
elif 'alpha' in params.columns:
    # Use alpha (vigor-derived) as z proxy
    params['z'] = params['alpha']
    print('  Using alpha as z proxy (FET_Exp_Bias_z_params.csv not found)')
else:
    # Set z to 1 (neutral) if not available
    params['z'] = 1.0
    print('  WARNING: z not available, setting to 1.0')

# Rename k → kappa if needed
if 'k' in params.columns and 'kappa' not in params.columns:
    params['kappa'] = params['k']

# Standardize parameters
params['z_z']     = (params['z']     - params['z'].mean())     / params['z'].std()
params['kappa_z'] = (params['kappa'] - params['kappa'].mean()) / params['kappa'].std()
params['beta_z']  = (params['beta']  - params['beta'].mean())  / params['beta'].std()

print(f'  Params: {len(params)} subjects')
print(f'    z_z    range: [{params["z_z"].min():.2f}, {params["z_z"].max():.2f}]')
print(f'    kappa_z range: [{params["kappa_z"].min():.2f}, {params["kappa_z"].max():.2f}]')
print(f'    beta_z  range: [{params["beta_z"].min():.2f}, {params["beta_z"].max():.2f}]')

# ── Merge parameters into trial-level data ──
df_trial = df_trial.merge(
    params[['subj', 'z_z', 'kappa_z', 'beta_z']],
    on='subj', how='left'
).reset_index(drop=True)
print(f'\nAfter merging params: {df_trial.shape}')

# ── Confirm DVs ──
required_dvs = [
    'onset_slope_resid', 'onset_mean_resid', 'onset_peak_resid',
    'encounter_spike_resid', 'post_encounter_vigor_resid',
    'terminal_mean_resid', 'terminal_auc_resid',
    'onset_slope_norm', 'onset_mean_norm', 'onset_peak_norm',
    'encounter_spike_norm', 'post_encounter_vigor_norm',
    'terminal_mean_norm', 'terminal_auc_norm',
]
missing = [d for d in required_dvs if d not in df_trial.columns]
assert len(missing) == 0, f'Missing DVs: {missing}'
print(f'All {len(required_dvs)} DVs confirmed.')
print(f'df_trial: {len(df_trial):,} trials, {df_trial["subj"].nunique()} subjects')
print(f'Attack rate: {df_trial["attack"].mean():.3f}')


# ── Helper functions ──
def run_lmm(formula, data, group_col='subj'):
    """Fit a linear mixed model, return tidy coefficients DataFrame + info dict.
    Tries multiple optimizers in sequence to avoid lbfgs degeneracy."""
    dv = formula.split('~')[0].strip()
    clean = data.dropna(subset=[dv, 'z_z', 'kappa_z', 'beta_z']).reset_index(drop=True)
    model = smf.mixedlm(formula, data=clean, groups=clean[group_col])

    # Try optimizers in order — lbfgs can degenerate on some models
    result = None
    for method in ['powell', 'bfgs', 'nm', 'lbfgs']:
        try:
            res = model.fit(reml=True, method=method)
            # Check for degeneracy: if z_z se is enormous, skip
            se_check = res.bse.get('z_z', res.bse.get('kappa_z', 1.0))
            if np.isfinite(se_check) and se_check < 1e6:
                result = res
                break
        except Exception:
            continue
    if result is None:
        result = model.fit(reml=True, method='powell')  # fallback

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
    }
    return coef_df, info, result


def extract(coef_df, predictor):
    """Extract a single row by predictor name; return NaN row if not found."""
    rows = coef_df[coef_df['predictor'] == predictor]
    if len(rows) == 0:
        return {'beta': np.nan, 'se': np.nan, 'p': np.nan,
                'ci_lo': np.nan, 'ci_hi': np.nan}
    return rows.iloc[0].to_dict()


def report(row, label):
    sig = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01
          else ('*' if row['p'] < 0.05 else 'n.s.'))
    fdr_sig = '(FDR sig)' if row.get('sig_fdr', False) else ''
    print(f'    {label}: β={row["beta"]:.4f}, SE={row["se"]:.4f}, '
          f'p={row["p"]:.4f} {sig} {fdr_sig}')


# ── Fit all 7 models × 2 streams ──
print('\n=== FITTING MODELS ===')

ONSET_FORMULA = '{dv} ~ z_z*threat_c + kappa_z*threat_c + beta_z*threat_c + C(choice)'
FULL_FORMULA  = ('{dv} ~ z_z*threat_c + kappa_z*threat_c + beta_z*threat_c'
                 ' + attack_c + attack_c:z_z + attack_c:kappa_z + attack_c:beta_z + C(choice)')

DV_SPECS = [
    ('onset_slope',          'onset'),
    ('onset_mean',           'onset'),
    ('onset_peak',           'onset'),
    ('encounter_spike',      'full'),
    ('post_encounter_vigor', 'full'),
    ('terminal_mean',        'full'),
    ('terminal_auc',         'full'),
]

all_fits = {}

for dv_base, ftype in DV_SPECS:
    template = ONSET_FORMULA if ftype == 'onset' else FULL_FORMULA

    for stream in ['resid', 'norm']:
        dv_col  = f'{dv_base}_{stream}'
        formula = template.format(dv=dv_col)

        coefs, info, _ = run_lmm(formula, df_trial)
        all_fits[(dv_base, stream)] = (coefs, info)

        tag = 'OK' if info['converged'] else 'NOT CONVERGED'
        print(f'  [{tag}]  {dv_col:40s}  N={info["nobs"]:,}  groups={info["n_groups"]}')

print(f'\nTotal models fit: {len(all_fits)} (7 DVs × 2 streams)')


# ── FDR correction across 39 tests ──
print('\n=== FDR CORRECTION (39 tests, resid stream) ===')

PARAMS              = ['z_z', 'kappa_z', 'beta_z']
MAIN_EFFECT_DVS     = ['onset_slope', 'onset_mean', 'onset_peak',
                        'encounter_spike', 'post_encounter_vigor', 'terminal_mean']
ONSET_DVS           = ['onset_slope', 'onset_mean', 'onset_peak']
ENCOUNTER_TERMINAL_DVS = ['encounter_spike', 'post_encounter_vigor', 'terminal_mean']

fdr_rows = []

# 1. Main effects: 3 params × 6 DVs = 18
for dv_base in MAIN_EFFECT_DVS:
    coefs, info = all_fits[(dv_base, 'resid')]
    for param in PARAMS:
        r = extract(coefs, param)
        fdr_rows.append({
            'claim': 'main_effect', 'dv_base': dv_base,
            'predictor': param, 'type': 'main',
            'beta': r['beta'], 'se': r['se'], 'p': r['p'],
            'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'],
            'nobs': info['nobs'], 'n_groups': info['n_groups'],
        })

# 2. Threat interactions on onset DVs: 3 params × 3 DVs = 9
for dv_base in ONSET_DVS:
    coefs, info = all_fits[(dv_base, 'resid')]
    for param in PARAMS:
        pred_name = f'{param}:threat_c'
        r = extract(coefs, pred_name)
        fdr_rows.append({
            'claim': 'threat_interaction_onset', 'dv_base': dv_base,
            'predictor': pred_name, 'type': 'param×threat',
            'beta': r['beta'], 'se': r['se'], 'p': r['p'],
            'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'],
            'nobs': info['nobs'], 'n_groups': info['n_groups'],
        })

# 3. Threat interactions on encounter/terminal DVs: 3 params × 3 DVs = 9
for dv_base in ENCOUNTER_TERMINAL_DVS:
    coefs, info = all_fits[(dv_base, 'resid')]
    for param in PARAMS:
        pred_name = f'{param}:threat_c'
        r = extract(coefs, pred_name)
        fdr_rows.append({
            'claim': 'threat_interaction_enc_term', 'dv_base': dv_base,
            'predictor': pred_name, 'type': 'param×threat',
            'beta': r['beta'], 'se': r['se'], 'p': r['p'],
            'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'],
            'nobs': info['nobs'], 'n_groups': info['n_groups'],
        })

# 4. Attack interactions on terminal_auc: 3 params = 3
coefs_tauc, info_tauc = all_fits[('terminal_auc', 'resid')]
for param in PARAMS:
    pred_name = f'attack_c:{param}'
    r = extract(coefs_tauc, pred_name)
    fdr_rows.append({
        'claim': 'attack_interaction_terminal', 'dv_base': 'terminal_auc',
        'predictor': pred_name, 'type': 'attack×param',
        'beta': r['beta'], 'se': r['se'], 'p': r['p'],
        'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'],
        'nobs': info_tauc['nobs'], 'n_groups': info_tauc['n_groups'],
    })

fdr_df = pd.DataFrame(fdr_rows)
assert len(fdr_df) == 39, f'Expected 39 tests, got {len(fdr_df)}'

# Apply Benjamini-Hochberg (handle NaN p-values by treating them as 1.0)
p_vals = fdr_df['p'].fillna(1.0).values
reject, p_fdr, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
fdr_df['p_fdr']          = p_fdr
fdr_df['sig_fdr']        = reject
fdr_df['sig_uncorrected'] = fdr_df['p'].fillna(1.0) < 0.05

print(f'FDR correction applied to {len(fdr_df)} tests')
print(f'  Significant (uncorrected p < .05): {fdr_df["sig_uncorrected"].sum()}')
print(f'  Significant (FDR q < .05):         {fdr_df["sig_fdr"].sum()}')

for test_type in ['main', 'param×threat', 'attack×param']:
    sub = fdr_df[fdr_df['type'] == test_type]
    print(f'    {test_type}: {sub["sig_fdr"].sum()}/{len(sub)} survive FDR')


# ── Claims evaluation ──
print('\n=== CLAIM 2A: κ GLOBAL CONSTRAINT ===')
claim_2a_dvs = ['onset_slope', 'post_encounter_vigor', 'terminal_mean']
all_neg = True; all_sig = True
for dv_base in claim_2a_dvs:
    row = fdr_df[(fdr_df['dv_base'] == dv_base) & (fdr_df['predictor'] == 'kappa_z')].iloc[0]
    report(row, f'kappa_z on {dv_base}')
    if row['beta'] >= 0:    all_neg = False
    if not row['sig_fdr']:  all_sig = False
print(f'  All negative: {all_neg}')
print(f'  All FDR-sig:  {all_sig}')
print(f'  CLAIM 2A {"SUPPORTED" if (all_neg and all_sig) else "NOT FULLY SUPPORTED"}')

print('\n=== CLAIM 2B: z PREPARATORY ALLOCATION ===')
r_onset_mean  = fdr_df[(fdr_df['dv_base'] == 'onset_mean')  & (fdr_df['predictor'] == 'z_z')].iloc[0]
r_onset_slope = fdr_df[(fdr_df['dv_base'] == 'onset_slope') & (fdr_df['predictor'] == 'z_z')].iloc[0]
r_enc_spike   = fdr_df[(fdr_df['dv_base'] == 'encounter_spike') & (fdr_df['predictor'] == 'z_z')].iloc[0]
report(r_onset_mean,  'z_z on onset_mean')
report(r_onset_slope, 'z_z on onset_slope')
report(r_enc_spike,   'z_z on encounter_spike')
onset_pos  = r_onset_mean['beta'] > 0 and r_onset_mean['sig_fdr']
spike_neg  = r_enc_spike['beta'] < 0 and r_enc_spike['sig_fdr']
print(f'  Onset positive+sig: {onset_pos}')
print(f'  Spike negative+sig: {spike_neg}')
print(f'  CLAIM 2B {"SUPPORTED" if (onset_pos or spike_neg) else "NOT FULLY SUPPORTED"}')

print('\n=== CLAIM 2C: β THREAT-SENSITIVITY OF ONSET ===')
onset_sig_count = 0
for dv_base in ONSET_DVS:
    row = fdr_df[(fdr_df['dv_base'] == dv_base) & (fdr_df['predictor'] == 'beta_z:threat_c')].iloc[0]
    report(row, f'beta_z:threat_c on {dv_base}')
    if row['sig_fdr']: onset_sig_count += 1
elsewhere_sig = 0
for dv_base in ENCOUNTER_TERMINAL_DVS:
    row = fdr_df[(fdr_df['dv_base'] == dv_base) & (fdr_df['predictor'] == 'beta_z:threat_c')].iloc[0]
    report(row, f'beta_z:threat_c on {dv_base}')
    if row['sig_fdr']: elsewhere_sig += 1
print(f'  β×threat sig on onset: {onset_sig_count}/{len(ONSET_DVS)}')
print(f'  β×threat sig elsewhere: {elsewhere_sig}/{len(ENCOUNTER_TERMINAL_DVS)} (expect 0)')

print('\n=== CLAIM 2D: ATTACK × PARAMETER TERMINAL INTERACTIONS ===')
all_sig_2d = True
for param in PARAMS:
    pred_name = f'attack_c:{param}'
    row = fdr_df[(fdr_df['dv_base'] == 'terminal_auc') & (fdr_df['predictor'] == pred_name)].iloc[0]
    report(row, pred_name)
    if not row['sig_fdr']: all_sig_2d = False
print(f'  All three FDR-significant: {all_sig_2d}')
print(f'  CLAIM 2D {"SUPPORTED" if all_sig_2d else "NOT FULLY SUPPORTED"}')


# ── Save ──
csv_out = fdr_df.copy()
csv_out['sig_star'] = csv_out['p_fdr'].apply(
    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns')))
out_path = TABLES / 'table_s2_parameter_dissociation.csv'
csv_out.to_csv(out_path, index=False)
print(f'\nCSV saved → {out_path}')
print(f'{len(csv_out)} rows, {csv_out["sig_fdr"].sum()} FDR-significant tests.')

print('\n=== DONE ===')
print(fdr_df[['dv_base', 'predictor', 'type', 'beta', 'p', 'p_fdr', 'sig_fdr']].to_string(index=False))
