#!/usr/bin/env python3
"""
Part 2: What happens at encounter?

Analysis 2.1: Encounter-aligned timecourse (5Hz, 600ms smooth)
Analysis 2.2: Per-subject encounter spike
Analysis 2.3: Metric comparison (which shows cleanest encounter effect?)
Analysis 2.4: Onset timing of encounter effect
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, ast
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_rel, pearsonr
from pathlib import Path

DATA_DIR = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("/workspace/results/stats/vigor_analysis")
EXCLUDE = [154, 197, 208]


def run_part2(metrics_df=None):
    print(f"\n{'='*70}")
    print("PART 2: ENCOUNTER DYNAMICS")
    print(f"{'='*70}")

    # ═══════════════════════════════════════════════════════════════
    # 2.2: Per-subject encounter spike for each metric
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- 2.2: Per-subject encounter spike (attack - non-attack, reactive epoch) ---")

    if metrics_df is None:
        metrics_df = pd.read_csv(OUT_DIR / "vigor_metrics.csv")

    reactive = metrics_df[metrics_df['epoch'] == 'reactive'].copy()

    METRICS = ['median_ipi', 'norm_rate', 'relative_vigor', 'frac_full',
               'press_sd', 'pause_freq', 'n_presses']

    print(f"\n  {'Metric':<20} {'Spike':>8} {'t':>8} {'p':>12} {'d':>8} {'%pos':>6} {'cd_r':>8} {'cd_p':>10}")
    print(f"  {'-'*85}")

    params = pd.read_csv("results/stats/full_analysis/part1_params_full.csv")
    cd_map = params.set_index('subj')['log_cd_z'] if 'log_cd_z' in params.columns else None

    spike_results = {}
    for metric in METRICS:
        sub_atk = reactive[reactive['is_attack'] == 1].groupby('subj')[metric].mean()
        sub_noatk = reactive[reactive['is_attack'] == 0].groupby('subj')[metric].mean()
        shared = sorted(set(sub_atk.index) & set(sub_noatk.index))
        if len(shared) < 50:
            continue

        spike = sub_atk.loc[shared] - sub_noatk.loc[shared]
        t, p = ttest_1samp(spike, 0)
        d = spike.mean() / spike.std()
        pct_pos = (spike > 0).mean() * 100

        # cd correlation
        if cd_map is not None:
            shared_cd = sorted(set(shared) & set(cd_map.index))
            r_cd, p_cd = pearsonr(spike.loc[shared_cd], cd_map.loc[shared_cd])
        else:
            r_cd, p_cd = np.nan, np.nan

        sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))
        print(f"  {metric:<20} {spike.mean():>+8.4f} {t:>8.2f} {p:>12.2e} {d:>+8.3f} {pct_pos:>5.1f}% {r_cd:>+8.3f} {p_cd:>10.4f} {sig}")

        spike_results[metric] = {'spike': spike, 'd': abs(d), 'r_cd': r_cd}

    # ═══════════════════════════════════════════════════════════════
    # 2.3: Which metric shows cleanest encounter effect?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- 2.3: Metric ranking by encounter effect size ---")
    ranked = sorted(spike_results.items(), key=lambda x: x[1]['d'], reverse=True)
    print(f"\n  {'Rank':>4} {'Metric':<20} {'|d|':>8} {'cd_r':>8}")
    for i, (metric, res) in enumerate(ranked):
        print(f"  {i+1:>4} {metric:<20} {res['d']:>8.3f} {res['r_cd']:>+8.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 2.2b: Threat modulation of spike
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- 2.2b: Is the encounter spike threat-modulated? ---")
    best_metric = ranked[0][0]
    print(f"  Using best metric: {best_metric}")

    for T in [0.1, 0.5, 0.9]:
        sub_atk = reactive[(reactive['is_attack'] == 1) & (reactive['T_round'] == T)].groupby('subj')[best_metric].mean()
        sub_noatk = reactive[(reactive['is_attack'] == 0) & (reactive['T_round'] == T)].groupby('subj')[best_metric].mean()
        shared = sorted(set(sub_atk.index) & set(sub_noatk.index))
        if len(shared) > 20:
            spike = sub_atk.loc[shared] - sub_noatk.loc[shared]
            print(f"  T={T}: spike mean={spike.mean():+.4f}, SD={spike.std():.4f}")

    # Test threat modulation
    spikes_by_T = {}
    for T in [0.1, 0.5, 0.9]:
        sub_atk = reactive[(reactive['is_attack'] == 1) & (reactive['T_round'] == T)].groupby('subj')[best_metric].mean()
        sub_noatk = reactive[(reactive['is_attack'] == 0) & (reactive['T_round'] == T)].groupby('subj')[best_metric].mean()
        shared = sorted(set(sub_atk.index) & set(sub_noatk.index))
        spikes_by_T[T] = sub_atk.loc[shared] - sub_noatk.loc[shared]

    shared_all = sorted(set(spikes_by_T[0.1].index) & set(spikes_by_T[0.9].index))
    if len(shared_all) > 50:
        t_mod, p_mod = ttest_rel(spikes_by_T[0.9].loc[shared_all], spikes_by_T[0.1].loc[shared_all])
        print(f"\n  T=0.9 vs T=0.1 spike: t={t_mod:.3f}, p={p_mod:.4f}")
        print(f"  {'Threat-INDEPENDENT' if p_mod > 0.05 else 'Threat-MODULATED'}")

    # ═══════════════════════════════════════════════════════════════
    # 2.1: Encounter-aligned timecourse (5Hz from raw keypresses)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- 2.1: Encounter-aligned timecourse ---")
    print("  Computing from raw keypresses at 5Hz...")

    beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)]
    beh['T_round'] = beh['threat'].round(1)
    beh['is_heavy'] = (beh['trialCookie_weight'] == 3.0).astype(int)
    beh['enc_t'] = pd.to_numeric(beh['encounterTime'], errors='coerce')
    beh['is_attack'] = beh['isAttackTrial'].astype(int)

    BIN = 0.2
    WIN = np.arange(-2.0, 5.2, BIN)

    enc_records = []
    for _, row in beh.iterrows():
        try:
            pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
            if len(pt) < 5: continue
            cal = row['calibrationMax']; enc = row['enc_t']
            req = 0.9 if row['is_heavy'] else 0.4
            if cal <= 0 or pd.isna(enc): continue

            ipis = np.diff(pt)
            midpoints = (pt[:-1] + pt[1:]) / 2
            rates = np.where(ipis > 0.01, (1.0 / ipis) / cal, np.nan)
            enc_mid = midpoints - enc

            for t_start in WIN:
                mask = (enc_mid >= t_start) & (enc_mid < t_start + BIN)
                valid = rates[mask]; valid = valid[~np.isnan(valid)]
                if len(valid) >= 1:
                    enc_records.append({
                        'subj': row['subj'], 'T_round': row['T_round'],
                        'cookie': row['is_heavy'], 'is_attack': row['is_attack'],
                        't_bin': round(t_start + BIN / 2, 2),
                        'rate': np.median(valid),
                        'rel_vigor': np.median(valid) / req,
                    })
        except: pass

    enc_df = pd.DataFrame(enc_records)
    print(f"  Records: {len(enc_df):,}")

    # Per-subject attack-noattack difference for relative vigor
    subj_ts = enc_df.groupby(['subj', 'is_attack', 't_bin'])['rel_vigor'].mean().reset_index()
    att = subj_ts[subj_ts['is_attack'] == 1].rename(columns={'rel_vigor': 'att'}).drop(columns='is_attack')
    non = subj_ts[subj_ts['is_attack'] == 0].rename(columns={'rel_vigor': 'non'}).drop(columns='is_attack')
    diff = att.merge(non, on=['subj', 't_bin'])
    diff['effect'] = diff['att'] - diff['non']
    panel_b = diff.groupby('t_bin')['effect'].agg(['mean', 'sem']).reset_index()

    # Smooth with 3-point moving average
    panel_b['mean_smooth'] = panel_b['mean'].rolling(3, center=True, min_periods=1).mean()
    panel_b['sem_smooth'] = panel_b['sem'].rolling(3, center=True, min_periods=1).mean()

    # ── 2.4: Onset timing ──
    print(f"\n--- 2.4: Encounter effect onset timing ---")
    post = panel_b[panel_b['t_bin'] > 0].sort_values('t_bin')
    pre_mean = panel_b[panel_b['t_bin'] < 0]['mean'].mean()
    pre_sd = panel_b[panel_b['t_bin'] < 0]['mean'].std()

    first_sig = None
    for _, r in post.iterrows():
        # Test each bin vs pre-encounter distribution
        if r['mean'] > pre_mean + 1.96 * r['sem']:
            first_sig = r['t_bin']
            break

    peak = panel_b.loc[panel_b['mean'].idxmax()]
    print(f"  Pre-encounter baseline: {pre_mean:.4f}")
    print(f"  First significant bin: {first_sig}s" if first_sig else "  No significant bin found")
    print(f"  Peak effect: t={peak['t_bin']:.1f}s, Δ={peak['mean']:.4f}")

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: by threat
    subj_threat = enc_df.groupby(['subj', 'T_round', 't_bin'])['rel_vigor'].mean().reset_index()
    threat_ts = subj_threat.groupby(['T_round', 't_bin'])['rel_vigor'].agg(['mean', 'sem']).reset_index()

    ax = axes[0]
    colors = {0.1: '#2196F3', 0.5: '#9E9E9E', 0.9: '#F44336'}
    for T in [0.1, 0.5, 0.9]:
        d = threat_ts[threat_ts['T_round'] == T].sort_values('t_bin')
        m = d['mean'].rolling(3, center=True, min_periods=1).mean()
        s = d['sem'].rolling(3, center=True, min_periods=1).mean()
        ax.plot(d['t_bin'], m, color=colors[T], lw=2, label=f'T={T}')
        ax.fill_between(d['t_bin'], m - s, m + s, color=colors[T], alpha=0.15)
    ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Time from encounter (s)')
    ax.set_ylabel('Relative vigor (rate / required)')
    ax.set_title('A. Relative vigor by threat', fontweight='bold', loc='left')
    ax.legend(fontsize=9, frameon=False)

    # Panel B: attack effect
    ax = axes[1]
    ax.fill_between([0, 5], -0.05, panel_b['mean_smooth'].max() * 1.3, color='#FFEBEE', alpha=0.4, zorder=0)
    ax.plot(panel_b['t_bin'], panel_b['mean_smooth'], color='#F44336', lw=2)
    ax.fill_between(panel_b['t_bin'],
                     panel_b['mean_smooth'] - 1.96 * panel_b['sem_smooth'],
                     panel_b['mean_smooth'] + 1.96 * panel_b['sem_smooth'],
                     color='#F44336', alpha=0.2)
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.5)
    ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
    if first_sig:
        ax.axvline(first_sig, color='green', ls=':', lw=1.5, alpha=0.7, label=f'First sig: {first_sig}s')
        ax.legend(fontsize=9, frameon=False)
    ax.set_xlabel('Time from encounter (s)')
    ax.set_ylabel('Δ Relative vigor (attack - no attack)')
    ax.set_title('B. Encounter effect', fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig_encounter_timecourse.png', dpi=200, bbox_inches='tight')
    print(f"\n  Saved: {OUT_DIR / 'fig_encounter_timecourse.png'}")

    print(f"\n  Part 2 complete.")


if __name__ == '__main__':
    run_part2()
