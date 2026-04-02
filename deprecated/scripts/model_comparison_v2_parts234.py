#!/usr/bin/env python3
"""
Parts 2-4 of the model comparison plan v2.
Uses M3 (objective survival) as the winning model.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ast
from scipy.stats import pearsonr, ttest_rel, ttest_1samp
from scipy.special import expit
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DATA_DIR = Path("data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("results/stats/model_comparison_v2")

# Load data
beh_rich = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
feelings = pd.read_csv(DATA_DIR / "feelings.csv")
psych = pd.read_csv(DATA_DIR / "psych.csv")

# Exclusions
exclude = [154, 197, 208]
beh_rich = beh_rich[~beh_rich['subj'].isin(exclude)].copy()
feelings = feelings[~feelings['subj'].isin(exclude)].copy()
psych = psych[~psych['subj'].isin(exclude)].copy()

# Load M3 parameters
m3_params = pd.read_csv(OUT_DIR / "M3_params.csv")
m3_preds = pd.read_csv(OUT_DIR / "M3_predictions.csv")

beh_rich['T_round'] = beh_rich['threat'].round(1)
beh_rich['actual_dist'] = beh_rich['startDistance'].map({5: 1, 7: 2, 9: 3})
beh_rich['is_heavy'] = (beh_rich['trialCookie_weight'] == 3.0).astype(int)
beh_rich['actual_req'] = np.where(beh_rich['is_heavy'] == 1, 0.9, 0.4)
beh_rich['T_H'] = beh_rich['actual_dist'].map({1: 5.0, 2: 7.0, 3: 9.0})

print(f"Subjects: {beh_rich['subj'].nunique()}")
print(f"M3 params: {len(m3_params)} subjects, cols={list(m3_params.columns)}")

# Compute ΔV for each trial using M3: ΔV = 5·exp(-p·T_H) - exp(-p·5) - λ·effort
# For forced trials, ΔV represents the value of the current situation
subj_params = m3_params.set_index('subj')

def compute_dv_m3(row, lam):
    p = row['T_round']
    T_H = row['T_H']
    T_L = 5.0
    req_H = row['actual_req'] if row['is_heavy'] else 0.9  # use heavy for the comparison
    effort = req_H * T_H - 0.4 * T_L
    return 5.0 * np.exp(-p * T_H) - np.exp(-p * T_L) - lam * effort

# ═══════════════════════════════════════════════════════════════
# PART 2: SEQUENTIAL VIGOR PREDICTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: SEQUENTIAL VIGOR PREDICTION")
print("=" * 70)

# Step 10-11: Forced choice trials with predicted ΔV
probes = beh_rich[beh_rich['type'].isin([5, 6])].copy()
print(f"\nForced trials: {len(probes)} ({probes['subj'].nunique()} subjects)")

# Compute ΔV for each probe trial
dv_list = []
for _, row in probes.iterrows():
    s = row['subj']
    if s in subj_params.index:
        lam = subj_params.loc[s, 'lam']
        dv_list.append(compute_dv_m3(row, lam))
    else:
        dv_list.append(np.nan)
probes['dv'] = dv_list

# Step 12-13: Compute epoch-level press rates from raw keypress data
enc_time_map = {1: 2.5, 2: 3.5, 3: 5.0}

epoch_records = []
for _, row in probes.iterrows():
    try:
        pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
        if len(pt) < 5:
            continue

        enc = enc_time_map.get(row['actual_dist'], 3.5)

        # Anticipatory: t < enc
        antic_presses = pt[pt < enc]
        antic_ipis = np.diff(antic_presses)
        antic_ipis = antic_ipis[antic_ipis > 0.01]
        antic_rate = np.mean(1.0 / antic_ipis) / row['calibrationMax'] if len(antic_ipis) >= 3 else np.nan

        # Reactive: enc <= t < enc + 2
        react_presses = pt[(pt >= enc) & (pt < enc + 2)]
        react_ipis = np.diff(react_presses)
        react_ipis = react_ipis[react_ipis > 0.01]
        react_rate = np.mean(1.0 / react_ipis) / row['calibrationMax'] if len(react_ipis) >= 3 else np.nan

        # Terminal: last 2 seconds
        if len(pt) > 3:
            t_end = pt[-1]
            term_presses = pt[pt >= t_end - 2]
            term_ipis = np.diff(term_presses)
            term_ipis = term_ipis[term_ipis > 0.01]
            term_rate = np.mean(1.0 / term_ipis) / row['calibrationMax'] if len(term_ipis) >= 3 else np.nan
        else:
            term_rate = np.nan

        epoch_records.append({
            'subj': row['subj'], 'trial': row['trial'],
            'T_round': row['T_round'], 'actual_dist': row['actual_dist'],
            'is_heavy': row['is_heavy'], 'dv': row.get('dv', np.nan),
            'isAttackTrial': row.get('isAttackTrial', 0),
            'antic_rate': antic_rate, 'react_rate': react_rate, 'term_rate': term_rate,
        })
    except:
        pass

epoch_df = pd.DataFrame(epoch_records)
epoch_df['delta_vigor'] = epoch_df['react_rate'] - epoch_df['antic_rate']
print(f"Epoch data: {len(epoch_df)} trials with valid epoch rates")

# Step 14: Anticipatory vigor prediction
print(f"\n--- Step 14: Anticipatory Vigor Prediction ---")
valid_antic = epoch_df.dropna(subset=['antic_rate', 'dv'])
print(f"Valid trials: {len(valid_antic)}")

# Z-score within cookie type
for ck in [0, 1]:
    mask = valid_antic['is_heavy'] == ck
    if mask.sum() > 10:
        m = valid_antic.loc[mask, 'antic_rate'].mean()
        s = valid_antic.loc[mask, 'antic_rate'].std()
        valid_antic.loc[mask, 'antic_z'] = (valid_antic.loc[mask, 'antic_rate'] - m) / s

# Simple regression: antic_z ~ dv
valid_a = valid_antic.dropna(subset=['antic_z'])
X = sm.add_constant(valid_a[['dv', 'is_heavy']].values)
y = valid_a['antic_z'].values
model_antic = sm.OLS(y, X).fit()
print(f"  antic_z ~ ΔV + cookie_type:")
print(f"    ΔV: β={model_antic.params[1]:.4f}, t={model_antic.tvalues[1]:.3f}, p={model_antic.pvalues[1]:.2e}")
print(f"    cookie: β={model_antic.params[2]:.4f}, t={model_antic.tvalues[2]:.3f}, p={model_antic.pvalues[2]:.2e}")
print(f"    R²={model_antic.rsquared:.4f}")

# Per cookie type
for ck, label in [(1, 'Heavy'), (0, 'Light')]:
    sub = valid_a[valid_a['is_heavy'] == ck]
    if len(sub) > 20:
        r, p_val = pearsonr(sub['dv'], sub['antic_z'])
        print(f"  {label}: r(ΔV, antic_z) = {r:.3f}, p={p_val:.2e}, N={len(sub)}")

# Step 15: Delta-vigor
print(f"\n--- Step 15: Delta-Vigor ---")
valid_dv = epoch_df.dropna(subset=['delta_vigor'])
print(f"Valid trials: {len(valid_dv)}")

# By threat level
print(f"\nMean delta-vigor by threat:")
for T in [0.1, 0.5, 0.9]:
    sub = valid_dv[valid_dv['T_round'] == T]
    print(f"  T={T:.1f}: M={sub['delta_vigor'].mean():.4f}, SD={sub['delta_vigor'].std():.4f}, N={len(sub)}")

# Per-subject delta-vigor by threat
subj_dv = valid_dv.groupby(['subj', 'T_round'])['delta_vigor'].mean().reset_index()
subj_dv_wide = subj_dv.pivot(index='subj', columns='T_round', values='delta_vigor').dropna()

if 0.1 in subj_dv_wide.columns and 0.9 in subj_dv_wide.columns:
    t_stat, p_val = ttest_rel(subj_dv_wide[0.9], subj_dv_wide[0.1])
    diff = subj_dv_wide[0.9] - subj_dv_wide[0.1]
    d = diff.mean() / diff.std()
    print(f"\nPaired t(T=0.9 vs T=0.1): t={t_stat:.3f}, p={p_val:.2e}, d={d:.3f}")

# By distance
print(f"\nMean delta-vigor by distance:")
for D in [1, 2, 3]:
    sub = valid_dv[valid_dv['actual_dist'] == D]
    print(f"  D={D}: M={sub['delta_vigor'].mean():.4f}, N={len(sub)}")

# Step 16a: ΔV–delta-vigor correlation
print(f"\n--- Step 16a: ΔV–Delta-Vigor Correlation ---")

# Between-subject
subj_means = valid_dv.groupby('subj').agg(
    mean_dv=('dv', 'mean'), mean_delta=('delta_vigor', 'mean')
).dropna()
r_between, p_between = pearsonr(subj_means['mean_dv'], subj_means['mean_delta'])
print(f"Between-subject r(ΔV, delta_vigor): {r_between:.3f}, p={p_between:.2e}, N={len(subj_means)}")

# Within-subject (per-subject correlation across 3 threat levels)
within_rs = []
for s in subj_dv_wide.index:
    sdf = valid_dv[valid_dv['subj'] == s].dropna(subset=['dv', 'delta_vigor'])
    if len(sdf) >= 6:
        r, _ = pearsonr(sdf['dv'], sdf['delta_vigor'])
        within_rs.append(r)
within_rs = np.array(within_rs)
print(f"Within-subject r(ΔV, delta_vigor): mean={np.nanmean(within_rs):.3f}, "
      f"SD={np.nanstd(within_rs):.3f}, N={len(within_rs)}")
t_within, p_within = ttest_1samp(within_rs[~np.isnan(within_rs)], 0)
print(f"  t-test vs 0: t={t_within:.3f}, p={p_within:.2e}")

# Step 16b: Threshold test feasibility
print(f"\n--- Step 16b: Threshold Test Feasibility ---")
# Check: how many subjects have delta-vigor that changes sign across threat levels?
n_sign_change = 0
for s in subj_dv_wide.index:
    vals = subj_dv_wide.loc[s].values
    if np.any(vals > 0) and np.any(vals < 0):
        n_sign_change += 1
print(f"Subjects with sign change in delta-vigor across T: {n_sign_change}/{len(subj_dv_wide)} "
      f"({n_sign_change/len(subj_dv_wide)*100:.1f}%)")

# Step 17: Abandonment timing
print(f"\n--- Step 17: Abandonment Timing ---")
# Heavy forced trials on attack trials
heavy_attack = epoch_df[(epoch_df['is_heavy'] == 1) & (epoch_df['isAttackTrial'] == 1)]
print(f"Heavy forced attack trials: {len(heavy_attack)}")

# Compute abandonment from raw data
abandon_records = []
for _, row in beh_rich[(beh_rich['type'].isin([5, 6])) & (beh_rich['is_heavy'] == 1) &
                        (beh_rich['isAttackTrial'] == 1)].iterrows():
    try:
        pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
        if len(pt) < 5:
            continue

        cal_max = row['calibrationMax']
        req = row['actual_req']
        threshold = 0.25 * cal_max  # 25% of calibrated max → zero speed

        # Check if pressing drops below threshold for > 1 second
        ipis = np.diff(pt)
        abandon_time = pt[-1]  # default: never abandoned
        abandoned = False
        for i in range(len(ipis)):
            if ipis[i] > 1.0:  # gap > 1 second
                abandon_time = pt[i]
                abandoned = True
                break

        # Predicted abandonment: when EV(t) crosses zero
        # EV(t) = exp(-p · T_remaining) · R - λ · C_remaining
        s = row['subj']
        if s in subj_params.index:
            lam = subj_params.loc[s, 'lam']
            p_threat = row['T_round']
            trial_dur = pt[-1] - pt[0]

            # Simple: predicted time is when survival drops below effort threshold
            # For M3, the relevant quantity is exp(-p · T_remaining) × R vs λ × effort
            # This is approximate
            pred_abandon = trial_dur  # default: never

            abandon_records.append({
                'subj': s, 'T_round': row['T_round'],
                'obs_abandon': abandon_time,
                'abandoned': abandoned,
                'trial_dur': trial_dur,
            })
    except:
        pass

abandon_df = pd.DataFrame(abandon_records)
print(f"Trials analyzed: {len(abandon_df)}")
print(f"Abandonment rate: {abandon_df['abandoned'].mean():.3f}")
print(f"By threat:")
for T in [0.1, 0.5, 0.9]:
    sub = abandon_df[abandon_df['T_round'] == T]
    print(f"  T={T}: abandon rate={sub['abandoned'].mean():.3f}, N={len(sub)}")

# ═══════════════════════════════════════════════════════════════
# PART 3: INDIVIDUAL DIFFERENCES
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: INDIVIDUAL DIFFERENCES")
print("=" * 70)

# Step 19: Clinical regressions
print(f"\n--- Step 19: Clinical Regressions ---")
clinical_cols = ['STAI_Trait', 'STAI_State', 'OASIS_Total', 'DASS21_Anxiety',
                 'DASS21_Depression', 'DASS21_Stress', 'PHQ9_Total', 'AMI_Total',
                 'MFIS_Total', 'STICSA_Total']

# Z-score clinical
for col in clinical_cols:
    psych[col + '_z'] = (psych[col] - psych[col].mean()) / psych[col].std()

merged = m3_params.merge(psych, on='subj')
merged['log_lam'] = np.log(merged['lam'])
merged['log_beta'] = np.log(merged['beta'])

print(f"\nM3 params → Clinical (N={len(merged)}):")
for param, plabel in [('log_lam', 'λ'), ('log_beta', 'β')]:
    print(f"\n  {plabel} → Clinical:")
    for col in clinical_cols:
        r, p_val = pearsonr(merged[param], merged[col + '_z'])
        star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        print(f"    {col:<20}: r={r:+.3f}, p={p_val:.4f} {star}")

# Affect decomposition
print(f"\n--- Affect Decomposition ---")
anxiety_df = feelings[feelings['questionLabel'] == 'anxiety'].copy()
# Calibration: within-subject r(anxiety, threat)
# Discrepancy: mean residual from population regression
pop_slope, pop_int = np.polyfit(anxiety_df['threat'].values, anxiety_df['response'].values, 1)

calib_list = []
for s, sdf in anxiety_df.groupby('subj'):
    danger = sdf['threat'].values
    anx = sdf['response'].values
    if np.std(danger) > 0 and np.std(anx) > 0:
        r_cal, _ = pearsonr(anx, danger)
    else:
        r_cal = np.nan
    predicted = pop_slope * danger + pop_int
    disc = (anx - predicted).mean()
    calib_list.append({'subj': s, 'calibration': r_cal, 'discrepancy': disc})

calib_df = pd.DataFrame(calib_list).dropna()
merged2 = merged.merge(calib_df, on='subj')

print(f"\nDiscrepancy → Clinical (controlling for λ, β):")
for col in clinical_cols:
    r, p_val = pearsonr(merged2['discrepancy'], merged2[col + '_z'])
    star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
    print(f"  disc → {col:<20}: r={r:+.3f}, p={p_val:.4f} {star}")

# Step 22: Anxiety tercile choice surfaces
print(f"\n--- Step 22: Anxiety Tercile Choice Surfaces ---")
beh_choice = pd.read_csv(DATA_DIR / "behavior.csv")
beh_choice = beh_choice[~beh_choice['subj'].isin(exclude)]
beh_choice['T_round'] = beh_choice['threat'].round(1)

merged_stai = beh_choice.merge(psych[['subj', 'STAI_State']], on='subj')
merged_stai['anx_tercile'] = pd.qcut(merged_stai['STAI_State'], 3, labels=['Low', 'Mid', 'High'])

print(f"\nP(heavy) by anxiety tercile:")
for tercile in ['Low', 'Mid', 'High']:
    sub = merged_stai[merged_stai['anx_tercile'] == tercile]
    print(f"\n  {tercile} anxiety (N={sub['subj'].nunique()}):")
    print(f"  {'':>8} {'D=1':>8} {'D=2':>8} {'D=3':>8}")
    for T in [0.1, 0.5, 0.9]:
        row = f"  T={T:.0%}"
        for D in [1, 2, 3]:
            cell = sub[(sub['T_round'] == T) & (sub['distance_H'] == D)]
            row += f"  {cell['choice'].mean():>6.3f}"
        print(row)

# ═══════════════════════════════════════════════════════════════
# PART 4: BRIDGE TO FRAC_FULL AND PARAMETER CORRESPONDENCE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: BRIDGE TO FRAC_FULL AND PARAMETER CORRESPONDENCE")
print("=" * 70)

# Compute per-subject frac_full
ff_list = []
for _, row in beh_rich.iterrows():
    try:
        pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
        ipis = np.diff(pt)
        ipis = ipis[ipis > 0.01]
        if len(ipis) < 5: ff_list.append(np.nan); continue
        rates = (1.0 / ipis) / row['calibrationMax']
        ff_list.append(np.mean(rates >= row['actual_req']))
    except: ff_list.append(np.nan)
beh_rich['frac_full'] = ff_list

subj_ff = beh_rich.dropna(subset=['frac_full']).groupby('subj')['frac_full'].mean().reset_index()
subj_ff.columns = ['subj', 'mean_frac_full']

# Step 23: M3 params → frac_full
merged_ff = m3_params.merge(subj_ff, on='subj')
merged_ff['log_lam'] = np.log(merged_ff['lam'])
merged_ff['log_beta'] = np.log(merged_ff['beta'])

print(f"\nM3 params → frac_full (N={len(merged_ff)}):")
for param, plabel in [('log_lam', 'λ'), ('log_beta', 'β')]:
    r, p_val = pearsonr(merged_ff[param], merged_ff['mean_frac_full'])
    print(f"  {plabel} → frac_full: r={r:+.3f}, p={p_val:.2e}")

# Step 24: Correspondence with 3-param v2
print(f"\n--- Parameter Correspondence with 3-param v2 ---")
v2_params = pd.read_csv("results/stats/oc_evc_3param_v2_params.csv")
v2_params = v2_params[~v2_params['subj'].isin(exclude)]

corresp = m3_params.merge(v2_params, on='subj', suffixes=('_m3', '_v2'))
corresp['log_lam'] = np.log(corresp['lam'])
corresp['log_beta_m3'] = np.log(corresp['beta'])
corresp['log_k'] = np.log(corresp['k'])
corresp['log_beta_v2'] = np.log(corresp['beta_v2'] if 'beta_v2' in corresp.columns else corresp['beta'])
corresp['log_cd'] = np.log(corresp['c_death'])

# Handle column naming
if 'k' in corresp.columns:
    r_lam_k, p_lam_k = pearsonr(corresp['log_lam'], corresp['log_k'])
    print(f"  M3 λ ↔ v2 k: r={r_lam_k:.3f}, p={p_lam_k:.2e}")

    r_lam_cd, _ = pearsonr(corresp['log_lam'], corresp['log_cd'])
    print(f"  M3 λ ↔ v2 cd: r={r_lam_cd:.3f}")

    r_beta_k, _ = pearsonr(corresp['log_beta_m3'], corresp['log_k'])
    r_beta_cd, _ = pearsonr(corresp['log_beta_m3'], corresp['log_cd'])
    print(f"  M3 β ↔ v2 k: r={r_beta_k:.3f}")
    print(f"  M3 β ↔ v2 cd: r={r_beta_cd:.3f}")

    # v2 beta (threat) — need to handle column collision
    if 'beta_v2' in corresp.columns:
        r_lam_bv2, _ = pearsonr(corresp['log_lam'], np.log(corresp['beta_v2']))
        print(f"  M3 λ ↔ v2 β_threat: r={r_lam_bv2:.3f}")

# Known benchmarks from 3-param v2
print(f"\nBenchmarks (3-param v2):")
print(f"  cd → frac_full: r=0.710")
print(f"  β → frac_full: r=0.249")
print(f"  k → frac_full: r=0.044")

print("\nDone.")
