#!/usr/bin/env python3
"""
Functional Data Analysis of encounter-aligned vigor timecourse.

1. Extract per-subject encounter-aligned pressing timecourse (5Hz, cookie-controlled)
2. Functional mixed model: rate(t) ~ threat × subject effects
3. Cluster permutation test for temporal significance
4. Functional regression: do ce, cd predict the SHAPE of the encounter response?
5. Attack vs non-attack functional contrast with pointwise confidence

Output: per-subject functional features + publication figure
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_1samp, pearsonr, ttest_ind
from pathlib import Path

DATA_DIR = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("/workspace/results/stats/vigor_analysis")
FIG_DIR = Path("/workspace/results/figs/paper")
EXCLUDE = [154, 197, 208]

BIN = 0.2  # 200ms
SMOOTH = 3
T_WIN = np.arange(-2.0, 5.0, BIN)  # encounter-aligned window
T_CENTERS = np.round(T_WIN + BIN/2, 2)


def smooth(arr, w=SMOOTH):
    s = pd.Series(arr)
    return s.rolling(w, center=True, min_periods=1).mean().values


def extract_timecourses():
    """Extract per-subject, per-trial encounter-aligned pressing timecourse."""
    print("Loading data...")
    beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)]
    beh['T_round'] = beh['threat'].round(1)
    beh['is_heavy'] = (beh['trialCookie_weight'] == 3.0).astype(int)
    beh['enc_t'] = pd.to_numeric(beh['encounterTime'], errors='coerce')
    beh['is_attack'] = beh['isAttackTrial'].astype(int)

    print("Extracting encounter-aligned timecourses from raw keypresses...")
    records = []
    for _, row in beh.iterrows():
        try:
            pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
            if len(pt) < 5: continue
            cal = row['calibrationMax']; enc = row['enc_t']
            if cal <= 0 or pd.isna(enc): continue

            ipis = np.diff(pt)
            midpoints = (pt[:-1] + pt[1:]) / 2
            rates = np.where(ipis > 0.01, (1.0 / ipis) / cal, np.nan)

            enc_mid = midpoints - enc
            trial_rates = np.full(len(T_CENTERS), np.nan)

            for i, t0 in enumerate(T_WIN):
                mask = (enc_mid >= t0) & (enc_mid < t0 + BIN)
                v = rates[mask]; v = v[~np.isnan(v)]
                if len(v) >= 1:
                    trial_rates[i] = np.median(v)

            records.append({
                'subj': row['subj'], 'trial': row['trial'],
                'T_round': row['T_round'], 'cookie': row['is_heavy'],
                'is_attack': row['is_attack'],
                'rates': trial_rates,
            })
        except: pass

    print(f"  {len(records)} trial timecourses extracted")
    return records


def build_subject_functions(records):
    """Build per-subject mean functions, cookie-controlled."""
    # Cookie-center: at each time bin, subtract cookie-type mean, add grand mean
    all_rates = np.array([r['rates'] for r in records])
    cookies = np.array([r['cookie'] for r in records])
    attacks = np.array([r['is_attack'] for r in records])
    threats = np.array([r['T_round'] for r in records])
    subjs = np.array([r['subj'] for r in records])

    n_bins = len(T_CENTERS)

    # Cookie-center per bin
    rates_cc = all_rates.copy()
    for i in range(n_bins):
        col = all_rates[:, i]
        valid = ~np.isnan(col)
        if valid.sum() < 10: continue
        grand_mean = np.nanmean(col)
        for ck in [0, 1]:
            mask = valid & (cookies == ck)
            if mask.sum() > 0:
                cookie_mean = np.nanmean(col[mask])
                rates_cc[mask, i] = col[mask] - cookie_mean + grand_mean

    # Per-subject mean functions by condition
    unique_subj = sorted(set(subjs))
    subj_functions = {}

    for s in unique_subj:
        s_mask = subjs == s
        subj_functions[s] = {}

        # Overall mean
        s_rates = rates_cc[s_mask]
        subj_functions[s]['overall'] = np.nanmean(s_rates, axis=0)

        # By threat
        for T in [0.1, 0.5, 0.9]:
            t_mask = s_mask & (threats == T)
            if t_mask.sum() > 0:
                subj_functions[s][f'T{T}'] = np.nanmean(rates_cc[t_mask], axis=0)

        # Attack vs non-attack
        atk_mask = s_mask & (attacks == 1)
        noatk_mask = s_mask & (attacks == 0)
        if atk_mask.sum() > 0 and noatk_mask.sum() > 0:
            subj_functions[s]['attack'] = np.nanmean(rates_cc[atk_mask], axis=0)
            subj_functions[s]['no_attack'] = np.nanmean(rates_cc[noatk_mask], axis=0)
            subj_functions[s]['encounter_effect'] = (
                subj_functions[s]['attack'] - subj_functions[s]['no_attack']
            )

    return subj_functions, rates_cc, subjs, threats, attacks, cookies


def cluster_permutation_test(data_matrix, n_perms=1000, alpha=0.05):
    """
    Cluster permutation test for a matrix of subject × time.
    Tests whether the mean at each timepoint differs from zero.
    Returns significant clusters.
    """
    n_subj, n_time = data_matrix.shape

    # Observed t-stats
    obs_t = np.zeros(n_time)
    for i in range(n_time):
        col = data_matrix[:, i]
        valid = ~np.isnan(col)
        if valid.sum() > 10:
            t, _ = ttest_1samp(col[valid], 0)
            obs_t[i] = t
        else:
            obs_t[i] = 0

    # Threshold
    t_thresh = 2.0  # approximately p < 0.05 two-tailed

    # Find observed clusters
    def find_clusters(t_vals, thresh):
        clusters = []
        in_cluster = False
        for i in range(len(t_vals)):
            if abs(t_vals[i]) > thresh:
                if not in_cluster:
                    cluster_start = i
                    in_cluster = True
            else:
                if in_cluster:
                    cluster_mass = np.sum(t_vals[cluster_start:i])
                    clusters.append((cluster_start, i, abs(cluster_mass)))
                    in_cluster = False
        if in_cluster:
            cluster_mass = np.sum(t_vals[cluster_start:len(t_vals)])
            clusters.append((cluster_start, len(t_vals), abs(cluster_mass)))
        return clusters

    obs_clusters = find_clusters(obs_t, t_thresh)

    if len(obs_clusters) == 0:
        return obs_t, [], []

    # Permutation distribution of max cluster mass
    max_cluster_masses = []
    for perm in range(n_perms):
        # Randomly flip signs
        signs = np.random.choice([-1, 1], size=n_subj)
        perm_data = data_matrix * signs[:, None]
        perm_t = np.zeros(n_time)
        for i in range(n_time):
            col = perm_data[:, i]
            valid = ~np.isnan(col)
            if valid.sum() > 10:
                t, _ = ttest_1samp(col[valid], 0)
                perm_t[i] = t
        perm_clusters = find_clusters(perm_t, t_thresh)
        if len(perm_clusters) > 0:
            max_cluster_masses.append(max(c[2] for c in perm_clusters))
        else:
            max_cluster_masses.append(0)

    max_cluster_masses = np.array(max_cluster_masses)

    # Test each observed cluster
    sig_clusters = []
    for start, end, mass in obs_clusters:
        p = (max_cluster_masses >= mass).mean()
        sig_clusters.append((start, end, mass, p))

    return obs_t, obs_clusters, sig_clusters


def extract_functional_features(subj_functions, unique_subj):
    """Extract per-subject scalar features from the functional data."""
    features = []

    for s in unique_subj:
        sf = subj_functions.get(s)
        if sf is None: continue

        overall = sf.get('overall')
        if overall is None: continue

        # Feature 1: Baseline (mean rate in pre-encounter window, t < -1s)
        pre_mask = T_CENTERS < -1.0
        baseline = np.nanmean(overall[pre_mask]) if pre_mask.sum() > 0 else np.nan

        # Feature 2: Cruising rate (mean rate in steady-state, t = -1 to 0)
        cruise_mask = (T_CENTERS >= -1.0) & (T_CENTERS < 0)
        cruising = np.nanmean(overall[cruise_mask]) if cruise_mask.sum() > 0 else np.nan

        # Feature 3: Post-encounter rate (mean rate t = 0 to 2)
        post_mask = (T_CENTERS >= 0) & (T_CENTERS < 2)
        post_rate = np.nanmean(overall[post_mask]) if post_mask.sum() > 0 else np.nan

        # Feature 4: Encounter spike magnitude
        enc_eff = sf.get('encounter_effect')
        if enc_eff is not None:
            spike_mag = np.nanmean(enc_eff[(T_CENTERS >= 0) & (T_CENTERS < 2)])
            spike_peak = np.nanmax(enc_eff[(T_CENTERS >= 0) & (T_CENTERS < 3)])
            spike_peak_time = T_CENTERS[(T_CENTERS >= 0) & (T_CENTERS < 3)][
                np.nanargmax(enc_eff[(T_CENTERS >= 0) & (T_CENTERS < 3)])]
        else:
            spike_mag = spike_peak = spike_peak_time = np.nan

        # Feature 5: Threat gain (slope across T levels at cruising)
        t01 = sf.get('T0.1'); t09 = sf.get('T0.9')
        if t01 is not None and t09 is not None:
            threat_gain_pre = (np.nanmean(t09[cruise_mask]) - np.nanmean(t01[cruise_mask]))
            threat_gain_post = (np.nanmean(t09[post_mask]) - np.nanmean(t01[post_mask]))
        else:
            threat_gain_pre = threat_gain_post = np.nan

        # Feature 6: Ramp-up speed (how quickly rate increases from onset)
        onset_mask = T_CENTERS < 0
        onset_rates = overall[onset_mask]
        valid_onset = ~np.isnan(onset_rates)
        if valid_onset.sum() > 3:
            from scipy.stats import linregress
            slope, _, _, _, _ = linregress(T_CENTERS[onset_mask][valid_onset], onset_rates[valid_onset])
            ramp_speed = slope
        else:
            ramp_speed = np.nan

        features.append({
            'subj': s,
            'baseline': baseline,
            'cruising_rate': cruising,
            'post_encounter_rate': post_rate,
            'encounter_spike': spike_mag,
            'spike_peak': spike_peak,
            'spike_peak_time': spike_peak_time,
            'threat_gain_pre': threat_gain_pre,
            'threat_gain_post': threat_gain_post,
            'ramp_speed': ramp_speed,
        })

    return pd.DataFrame(features)


def run():
    # Extract timecourses
    records = extract_timecourses()
    subj_functions, rates_cc, subjs, threats, attacks, cookies = build_subject_functions(records)
    unique_subj = sorted(subj_functions.keys())
    n_subj = len(unique_subj)

    print(f"\n{n_subj} subjects with functional data")

    # ═══════════════════════════════════════════════════════════
    # 1. POPULATION ENCOUNTER EFFECT WITH CLUSTER PERMUTATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("1. ENCOUNTER EFFECT — CLUSTER PERMUTATION TEST")
    print(f"{'='*60}")

    # Build matrix: subject × time for encounter effect
    enc_matrix = np.full((n_subj, len(T_CENTERS)), np.nan)
    for i, s in enumerate(unique_subj):
        eff = subj_functions[s].get('encounter_effect')
        if eff is not None:
            enc_matrix[i] = eff

    obs_t, obs_clusters, sig_clusters = cluster_permutation_test(enc_matrix, n_perms=1000)

    print(f"\n  Observed clusters:")
    for start, end, mass, p in sig_clusters:
        t_start = T_CENTERS[start]; t_end = T_CENTERS[min(end, len(T_CENTERS)-1)]
        print(f"    t=[{t_start:.1f}, {t_end:.1f}]s, mass={mass:.1f}, p={p:.3f} {'***' if p<.001 else ('**' if p<.01 else ('*' if p<.05 else ''))}")

    # ═══════════════════════════════════════════════════════════
    # 2. THREAT MODULATION — POINTWISE T-TESTS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("2. THREAT MODULATION — POINTWISE")
    print(f"{'='*60}")

    # Build per-subject T=0.9 minus T=0.1 difference function
    threat_diff_matrix = np.full((n_subj, len(T_CENTERS)), np.nan)
    for i, s in enumerate(unique_subj):
        t09 = subj_functions[s].get('T0.9')
        t01 = subj_functions[s].get('T0.1')
        if t09 is not None and t01 is not None:
            threat_diff_matrix[i] = t09 - t01

    obs_t_threat, _, sig_clusters_threat = cluster_permutation_test(threat_diff_matrix, n_perms=1000)

    print(f"\n  Threat effect clusters (T=0.9 - T=0.1):")
    for start, end, mass, p in sig_clusters_threat:
        t_start = T_CENTERS[start]; t_end = T_CENTERS[min(end, len(T_CENTERS)-1)]
        print(f"    t=[{t_start:.1f}, {t_end:.1f}]s, mass={mass:.1f}, p={p:.3f} {'***' if p<.001 else ('**' if p<.01 else ('*' if p<.05 else ''))}")

    # ═══════════════════════════════════════════════════════════
    # 3. FUNCTIONAL REGRESSION: DO ce, cd PREDICT THE SHAPE?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("3. FUNCTIONAL REGRESSION — ce, cd → TIMECOURSE SHAPE")
    print(f"{'='*60}")

    params = pd.read_csv("results/stats/oc_evc_final_params.csv")
    params = params[~params['subj'].isin(EXCLUDE)]
    params['log_ce'] = np.log(params['c_effort'])
    params['log_cd'] = np.log(params['c_death'])

    ce_map = params.set_index('subj')['log_ce']
    cd_map = params.set_index('subj')['log_cd']

    # At each time bin, correlate ce/cd with the encounter effect
    r_cd_ts = np.full(len(T_CENTERS), np.nan)
    r_ce_ts = np.full(len(T_CENTERS), np.nan)

    for i in range(len(T_CENTERS)):
        vals = []
        cd_vals = []
        ce_vals = []
        for j, s in enumerate(unique_subj):
            eff = enc_matrix[j, i]
            if np.isnan(eff): continue
            if s not in cd_map.index: continue
            vals.append(eff)
            cd_vals.append(cd_map[s])
            ce_vals.append(ce_map[s])

        if len(vals) > 30:
            r_cd_ts[i], _ = pearsonr(vals, cd_vals)
            r_ce_ts[i], _ = pearsonr(vals, ce_vals)

    print(f"\n  Peak cd correlation with encounter effect:")
    valid_cd = ~np.isnan(r_cd_ts)
    if valid_cd.any():
        peak_idx = np.nanargmax(np.abs(r_cd_ts))
        print(f"    t={T_CENTERS[peak_idx]:.1f}s, r={r_cd_ts[peak_idx]:+.3f}")

    print(f"  Peak ce correlation with encounter effect:")
    valid_ce = ~np.isnan(r_ce_ts)
    if valid_ce.any():
        peak_idx = np.nanargmax(np.abs(r_ce_ts))
        print(f"    t={T_CENTERS[peak_idx]:.1f}s, r={r_ce_ts[peak_idx]:+.3f}")

    # ═══════════════════════════════════════════════════════════
    # 4. EXTRACT FUNCTIONAL FEATURES
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("4. FUNCTIONAL FEATURES")
    print(f"{'='*60}")

    features = extract_functional_features(subj_functions, unique_subj)
    features = features.merge(params[['subj', 'c_effort', 'c_death', 'log_ce', 'log_cd']], on='subj')

    print(f"\n  Features extracted for {len(features)} subjects")
    print(f"\n  {'Feature':<25} {'Mean':>8} {'SD':>8} {'r(cd)':>8} {'r(ce)':>8}")
    print(f"  {'-'*60}")
    for feat in ['baseline', 'cruising_rate', 'post_encounter_rate', 'encounter_spike',
                  'threat_gain_pre', 'threat_gain_post', 'ramp_speed']:
        valid = features.dropna(subset=[feat, 'log_cd', 'log_ce'])
        r_cd, _ = pearsonr(valid['log_cd'], valid[feat])
        r_ce, _ = pearsonr(valid['log_ce'], valid[feat])
        print(f"  {feat:<25} {valid[feat].mean():>8.4f} {valid[feat].std():>8.4f} "
              f"{r_cd:>+8.3f} {r_ce:>+8.3f}")

    features.to_csv(OUT_DIR / 'functional_features.csv', index=False)
    print(f"\n  Saved: {OUT_DIR / 'functional_features.csv'}")

    # ═══════════════════════════════════════════════════════════
    # 5. FIGURE
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("5. FIGURE")
    print(f"{'='*60}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Population encounter effect with significance shading
    ax = axes[0, 0]
    mean_enc = np.nanmean(enc_matrix, axis=0)
    sem_enc = np.nanstd(enc_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(enc_matrix), axis=0))
    mean_sm = smooth(mean_enc)
    sem_sm = smooth(sem_enc)

    ax.plot(T_CENTERS, mean_sm, color='#F44336', lw=2.5)
    ax.fill_between(T_CENTERS, mean_sm - 1.96*sem_sm, mean_sm + 1.96*sem_sm,
                     color='#F44336', alpha=0.2)
    # Shade significant clusters
    for start, end, mass, p in sig_clusters:
        if p < 0.05:
            ax.axvspan(T_CENTERS[start], T_CENTERS[min(end-1, len(T_CENTERS)-1)],
                       color='#F44336', alpha=0.1)
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.5)
    ax.axvline(0, color='gray', ls='--', lw=1.5, alpha=0.5)
    ax.set_xlabel('Time from encounter (s)')
    ax.set_ylabel('Δ Press rate (attack − no attack)')
    ax.set_title('A. Encounter effect (cluster-corrected)', fontweight='bold')

    # Panel B: Threat modulation with significance
    ax = axes[0, 1]
    mean_threat = np.nanmean(threat_diff_matrix, axis=0)
    sem_threat = np.nanstd(threat_diff_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(threat_diff_matrix), axis=0))
    mean_t_sm = smooth(mean_threat)
    sem_t_sm = smooth(sem_threat)

    ax.plot(T_CENTERS, mean_t_sm, color='#9C27B0', lw=2.5)
    ax.fill_between(T_CENTERS, mean_t_sm - 1.96*sem_t_sm, mean_t_sm + 1.96*sem_t_sm,
                     color='#9C27B0', alpha=0.2)
    for start, end, mass, p in sig_clusters_threat:
        if p < 0.05:
            ax.axvspan(T_CENTERS[start], T_CENTERS[min(end-1, len(T_CENTERS)-1)],
                       color='#9C27B0', alpha=0.1)
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.5)
    ax.axvline(0, color='gray', ls='--', lw=1.5, alpha=0.5)
    ax.set_xlabel('Time from encounter (s)')
    ax.set_ylabel('Δ Press rate (T=0.9 − T=0.1)')
    ax.set_title('B. Threat effect (cluster-corrected)', fontweight='bold')

    # Panel C: cd correlation across time
    ax = axes[1, 0]
    ax.plot(T_CENTERS, smooth(r_cd_ts), color='#F44336', lw=2.5, label='cd')
    ax.plot(T_CENTERS, smooth(r_ce_ts), color='#2196F3', lw=2.5, label='ce')
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.5)
    ax.axvline(0, color='gray', ls='--', lw=1.5, alpha=0.5)
    # Significance threshold lines
    r_crit = 2 / np.sqrt(n_subj)  # approximate r for p < 0.05
    ax.axhline(r_crit, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(-r_crit, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Time from encounter (s)')
    ax.set_ylabel('r with encounter effect')
    ax.set_title('C. Parameter-timecourse correlation', fontweight='bold')
    ax.legend(fontsize=10, frameon=False)

    # Panel D: Individual encounter functions (top/bottom cd quartiles)
    ax = axes[1, 1]
    cd_vals_for_split = np.array([cd_map.get(s, np.nan) for s in unique_subj])
    q25, q75 = np.nanpercentile(cd_vals_for_split, [25, 75])

    hi_cd = [i for i, s in enumerate(unique_subj) if cd_map.get(s, np.nan) >= q75]
    lo_cd = [i for i, s in enumerate(unique_subj) if cd_map.get(s, np.nan) <= q25]

    hi_mean = smooth(np.nanmean(enc_matrix[hi_cd], axis=0))
    lo_mean = smooth(np.nanmean(enc_matrix[lo_cd], axis=0))
    hi_sem = smooth(np.nanstd(enc_matrix[hi_cd], axis=0) / np.sqrt(len(hi_cd)))
    lo_sem = smooth(np.nanstd(enc_matrix[lo_cd], axis=0) / np.sqrt(len(lo_cd)))

    ax.plot(T_CENTERS, hi_mean, color='#F44336', lw=2.5, label=f'High cd (N={len(hi_cd)})')
    ax.fill_between(T_CENTERS, hi_mean - hi_sem, hi_mean + hi_sem, color='#F44336', alpha=0.15)
    ax.plot(T_CENTERS, lo_mean, color='#2196F3', lw=2.5, label=f'Low cd (N={len(lo_cd)})')
    ax.fill_between(T_CENTERS, lo_mean - lo_sem, lo_mean + lo_sem, color='#2196F3', alpha=0.15)
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.5)
    ax.axvline(0, color='gray', ls='--', lw=1.5, alpha=0.5)
    ax.set_xlabel('Time from encounter (s)')
    ax.set_ylabel('Δ Press rate (attack − no attack)')
    ax.set_title('D. Encounter effect by cd quartile', fontweight='bold')
    ax.legend(fontsize=10, frameon=False)

    plt.tight_layout()
    fig_path = FIG_DIR / 'fig_fda_encounter.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"\n  Saved: {fig_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    run()
