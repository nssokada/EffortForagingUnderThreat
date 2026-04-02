"""
fig_imminence_timecourse.py — Core figure for the imminence paper.

4 panels:
A. Onset excess effort by threat (trial start aligned, shows pre-encounter ramp)
B. Attack effect encounter-aligned (attack - no attack difference)  
C. β tertiles: excess effort by threat level in anticipatory epoch
D. cd tertiles: excess effort encounter-aligned (post-encounter divergence)
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("/workspace/data/exploratory_350/processed/vigor_processed")
PARAMS_FILE = Path("/workspace/results/stats/full_analysis/part1_params_full.csv")
OUT = Path("/workspace/results/figs/paper/fig_imminence_timecourse.png")
EXCLUDE = [154, 197, 208]

# Colors
C_LOW = '#2196F3'    # blue
C_MID = '#9E9E9E'    # gray
C_HIGH = '#F44336'   # red
C_T01 = '#2196F3'
C_T05 = '#9E9E9E'
C_T09 = '#F44336'

print("Loading data...")
ts = pd.read_parquet(DATA_DIR / "smoothed_vigor_ts.parquet")
ts = ts[~ts['subj'].isin(EXCLUDE)]
ts['T_round'] = ts['threat'].round(1)

# Compute excess effort = vigor_norm - effort_chosen
ts['effort_chosen'] = np.where(ts['choice'] == 1, ts['effort_H'], 0.4)
ts['excess'] = ts['vigor_norm'] - ts['effort_chosen']

# Encounter-aligned time
ts['t_enc'] = ts['t'] - ts['encounterTime']

# Load params for tertile splits
params = pd.read_csv(PARAMS_FILE)
beta_z = params.set_index('subj')['log_beta_z']
cd_z = params.set_index('subj')['log_cd_z']

# Map tertiles using vectorized approach (fast)
beta_q33, beta_q67 = beta_z.quantile(0.33), beta_z.quantile(0.67)
cd_q33, cd_q67 = cd_z.quantile(0.33), cd_z.quantile(0.67)

subj_beta_tert = beta_z.map(lambda v: 'Low' if v <= beta_q33 else ('High' if v >= beta_q67 else 'Mid'))
subj_cd_tert = cd_z.map(lambda v: 'Low' if v <= cd_q33 else ('High' if v >= cd_q67 else 'Mid'))

ts['beta_tert'] = ts['subj'].map(subj_beta_tert)
ts['cd_tert'] = ts['subj'].map(subj_cd_tert)

print(f"Loaded {len(ts):,} rows, {ts['subj'].nunique()} subjects")

# ═══════════════════════════════════════════════════════════════
# PANEL DATA
# ═══════════════════════════════════════════════════════════════

BIN = 0.1  # 100ms bins

# Panel A: Onset-aligned excess effort by threat
print("Panel A: Onset-aligned by threat...")
ts['t_bin_onset'] = (ts['t'] / BIN).round() * BIN
onset_window = ts[(ts['t'] >= 0) & (ts['t'] <= 3.0)]
panel_a = onset_window.groupby(['T_round', 't_bin_onset'])['excess'].agg(['mean', 'sem', 'count']).reset_index()
panel_a = panel_a[panel_a['count'] > 500]

# Panel B: Attack effect (attack - no attack), encounter-aligned
print("Panel B: Attack effect...")
ts['t_bin_enc'] = (ts['t_enc'] / BIN).round() * BIN
enc_window = ts[(ts['t_enc'] >= -2) & (ts['t_enc'] <= 4)]

# Compute per bin: attack mean - no attack mean
atk_means = enc_window[enc_window['isAttackTrial'] == 1].groupby('t_bin_enc')['excess'].agg(['mean', 'count']).reset_index()
noatk_means = enc_window[enc_window['isAttackTrial'] == 0].groupby('t_bin_enc')['excess'].agg(['mean', 'count']).reset_index()
panel_b = atk_means.merge(noatk_means, on='t_bin_enc', suffixes=('_atk', '_noatk'))
panel_b['diff'] = panel_b['mean_atk'] - panel_b['mean_noatk']
# Bootstrap SE for the difference
panel_b['se_diff'] = np.sqrt(
    (enc_window[enc_window['isAttackTrial']==1].groupby('t_bin_enc')['excess'].sem().reindex(panel_b['t_bin_enc']).values)**2 +
    (enc_window[enc_window['isAttackTrial']==0].groupby('t_bin_enc')['excess'].sem().reindex(panel_b['t_bin_enc']).values)**2
)
panel_b = panel_b[(panel_b['count_atk'] > 300) & (panel_b['count_noatk'] > 300)]

# Panel C: β tertiles, onset-aligned excess at each threat level
print("Panel C: β tertiles onset-aligned...")
panel_c = onset_window.groupby(['beta_tert', 'T_round', 't_bin_onset'])['excess'].agg(['mean', 'sem', 'count']).reset_index()
panel_c = panel_c[panel_c['count'] > 100]

# Panel D: cd tertiles, encounter-aligned
print("Panel D: cd tertiles encounter-aligned...")
atk_only = enc_window[enc_window['isAttackTrial'] == 1]
panel_d = atk_only.groupby(['cd_tert', 't_bin_enc'])['excess'].agg(['mean', 'sem', 'count']).reset_index()
panel_d = panel_d[panel_d['count'] > 100]

# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════
print("Drawing figure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── Panel A: Onset excess effort by threat ──
ax = axes[0, 0]
for T, color, label in [(0.1, C_T01, 'T = 0.1'), (0.5, C_T05, 'T = 0.5'), (0.9, C_T09, 'T = 0.9')]:
    d = panel_a[panel_a['T_round'] == T].sort_values('t_bin_onset')
    ax.plot(d['t_bin_onset'], d['mean'], color=color, lw=2, label=label)
    ax.fill_between(d['t_bin_onset'], d['mean'] - d['sem'], d['mean'] + d['sem'],
                     color=color, alpha=0.15)
ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.4)
ax.legend(fontsize=9, frameon=False, loc='upper left')
ax.set_xlabel('Time from trial start (s)')
ax.set_ylabel('Excess effort')
ax.set_title('A. Excess effort by threat', fontweight='bold', fontsize=12, loc='left')

# ── Panel B: Attack effect ──
ax = axes[0, 1]
ax.fill_between([0, 2], -0.02, 0.16, color='#FFEBEE', alpha=0.5, zorder=0)
d = panel_b.sort_values('t_bin_enc')
ax.plot(d['t_bin_enc'], d['diff'], color=C_T09, lw=2)
ax.fill_between(d['t_bin_enc'], d['diff'] - d['se_diff'], d['diff'] + d['se_diff'],
                 color=C_T09, alpha=0.15)
ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.4)
ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('Time from encounter (s)')
ax.set_ylabel('Δ Excess effort (attack − no attack)')
ax.set_title('B. Encounter effect', fontweight='bold', fontsize=12, loc='left')
ax.annotate('Predator appears', xy=(0, d['diff'].max()*0.95), fontsize=8, color='gray',
            ha='left', va='top')

# ── Panel C: β tertiles by threat at onset ──
ax = axes[1, 0]
# Show high T only (T=0.9) split by β tertile
for tert, color, label in [('Low', C_LOW, 'Low β'), ('Mid', C_MID, 'Mid β'), ('High', C_HIGH, 'High β')]:
    d = panel_c[(panel_c['beta_tert'] == tert) & (panel_c['T_round'] == 0.9)].sort_values('t_bin_onset')
    ax.plot(d['t_bin_onset'], d['mean'], color=color, lw=2, label=label)
    ax.fill_between(d['t_bin_onset'], d['mean'] - d['sem'], d['mean'] + d['sem'],
                     color=color, alpha=0.12)
# Also show T=0.1 for high β as dashed
d_low_threat = panel_c[(panel_c['beta_tert'] == 'High') & (panel_c['T_round'] == 0.1)].sort_values('t_bin_onset')
ax.plot(d_low_threat['t_bin_onset'], d_low_threat['mean'], color=C_HIGH, lw=1.5, ls='--', label='High β, T=0.1')
ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.4)
ax.legend(fontsize=8, frameon=False, loc='upper left')
ax.set_xlabel('Time from trial start (s)')
ax.set_ylabel('Excess effort')
ax.set_title('C. β modulates pre-encounter vigor (T=0.9)', fontweight='bold', fontsize=12, loc='left')

# ── Panel D: cd tertiles encounter-aligned ──
ax = axes[1, 1]
ax.fill_between([0, 2], -0.5, 0.3, color='#FFEBEE', alpha=0.3, zorder=0)
for tert, color, label in [('Low', C_LOW, 'Low cd'), ('Mid', C_MID, 'Mid cd'), ('High', C_HIGH, 'High cd')]:
    d = panel_d[panel_d['cd_tert'] == tert].sort_values('t_bin_enc')
    ax.plot(d['t_bin_enc'], d['mean'], color=color, lw=2, label=label)
    ax.fill_between(d['t_bin_enc'], d['mean'] - d['sem'], d['mean'] + d['sem'],
                     color=color, alpha=0.12)
ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.4)
ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
ax.legend(fontsize=9, frameon=False, loc='lower right')
ax.set_xlabel('Time from encounter (s)')
ax.set_ylabel('Excess effort')
ax.set_title('D. cd diverges post-encounter', fontweight='bold', fontsize=12, loc='left')

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f"Saved: {OUT}")

# Quick numerics
print("\n--- Panel A: threat separation at t=2.5s ---")
for T in [0.1, 0.5, 0.9]:
    d = panel_a[(panel_a['T_round']==T) & (panel_a['t_bin_onset'].between(2.3, 2.7))]
    print(f"  T={T}: excess = {d['mean'].mean():.4f}")

print("\n--- Panel B: peak encounter effect ---")
peak = panel_b.loc[panel_b['diff'].idxmax()]
print(f"  Peak at t={peak['t_bin_enc']:.1f}s: Δ = {peak['diff']:.4f}")

print("\n--- Panel D: cd divergence at t_enc=2s ---")
for tert in ['Low', 'Mid', 'High']:
    d = panel_d[(panel_d['cd_tert']==tert) & (panel_d['t_bin_enc'].between(1.5, 2.5))]
    print(f"  {tert} cd: excess = {d['mean'].mean():.4f}")
