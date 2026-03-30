#!/usr/bin/env python3
"""Part 3: The Threat Imminence Gradient (Steps 3.1–3.6)"""

import sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *


def run_part3(p1=None, p2=None):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PART 3: THE THREAT IMMINENCE GRADIENT")
    print("=" * 70)

    df = p2['df'] if p2 else pd.read_csv(OUT_DIR / 'part2_epoch_data.csv')
    param_df = p1['param_df'] if p1 else pd.read_csv(OUT_DIR / 'part1_params_full.csv')

    # Ensure param columns present
    if 'log_k_z' not in df.columns:
        df = df.merge(param_df[['subj', 'log_k_z', 'log_beta_z', 'log_cd_z']], on='subj', how='left')

    # Rename for formula
    df['beta_z'] = df['log_beta_z']
    df['k_z'] = df['log_k_z']
    df['cd_z'] = df['log_cd_z']

    formula = ("epoch_resid ~ predator_probability + distance "
               "+ beta_z + k_z + cd_z "
               "+ predator_probability:beta_z + predator_probability:k_z + predator_probability:cd_z "
               "+ distance:beta_z + distance:k_z + distance:cd_z")

    # ── Step 3.1: Core test ──
    print("\n--- Step 3.1: Core epoch-by-epoch interaction test ---")
    stop_threat_beta_antic = True
    stop_dist_cd_react = True

    results_3_1 = {}
    for epoch, col in [('ANTICIPATORY', 'antic_vigor_resid'), ('REACTIVE', 'react_vigor_resid'),
                        ('TERMINAL', 'term_vigor_resid')]:
        edf = df.dropna(subset=[col]).copy()
        edf['epoch_resid'] = edf[col]
        print(f"\n  --- {epoch} (N={len(edf)}) ---")
        try:
            m = smf.mixedlm(formula, edf, groups=edf["subj"]).fit(reml=False)
            print_table(f"{epoch} fixed effects", m)

            # Check stopping criteria
            if epoch == 'ANTICIPATORY':
                p_tb = m.pvalues.get('predator_probability:beta_z', 1.0)
                if p_tb < 0.05:
                    stop_threat_beta_antic = False
            if epoch == 'REACTIVE':
                p_dcd = m.pvalues.get('distance:cd_z', 1.0)
                if p_dcd < 0.05:
                    stop_dist_cd_react = False

            results_3_1[epoch] = {t: {'coef': m.fe_params[t], 'z': m.tvalues.get(t, np.nan),
                                       'p': m.pvalues.get(t, np.nan)}
                                   for t in m.fe_params.index}
        except Exception as e:
            print(f"    FAILED: {e}")
            results_3_1[epoch] = {}

    if stop_threat_beta_antic and stop_dist_cd_react:
        return {'STOP': 'threat×β null in antic AND dist×cd null in reactive'}

    # ── Step 3.2: Threat independence ──
    print("\n--- Step 3.2: Threat independence test ---")
    rdf = df.dropna(subset=['react_vigor_resid']).copy()
    rdf['epoch_resid'] = rdf['react_vigor_resid']
    rdf['attack_trial'] = rdf['is_attack'].astype(float)
    m32 = smf.mixedlm("epoch_resid ~ attack_trial + predator_probability + attack_trial:predator_probability",
                       rdf, groups=rdf["subj"]).fit(reml=False)
    print_table("Reactive: attack_trial + threat + interaction", m32)

    # ── Step 3.3: Sliding window ──
    print("\n--- Step 3.3: Sliding window analysis ---")
    # Reload raw data for within-trial timecourse
    beh_rich = p1.get('beh_rich') if p1 else load_all_trials()

    windows = np.arange(0, 12, 0.5)  # 0 to 12 sec in 500ms steps
    r_threat_ts = []
    r_cd_ts = []

    for w_start in windows:
        w_end = w_start + 0.5
        # For each trial, compute vigor in this window
        rates_in_window = []
        for _, row in beh_rich.iterrows():
            try:
                pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
                if len(pt) < 5:
                    rates_in_window.append(np.nan)
                    continue
                cal = row['calibrationMax']
                if cal <= 0:
                    rates_in_window.append(np.nan)
                    continue
                w_pts = pt[(pt >= w_start) & (pt < w_end)]
                if len(w_pts) < 3:
                    rates_in_window.append(np.nan)
                    continue
                ipis = np.diff(w_pts)
                ipis = ipis[ipis > 0.01]
                if len(ipis) < 2:
                    rates_in_window.append(np.nan)
                    continue
                rates_in_window.append(np.median((1.0 / ipis) / cal))
            except:
                rates_in_window.append(np.nan)

        beh_rich['w_vigor'] = rates_in_window
        valid_w = beh_rich.dropna(subset=['w_vigor']).copy()

        if len(valid_w) < 1000:
            r_threat_ts.append(np.nan)
            r_cd_ts.append(np.nan)
            continue

        # Partial correlation: threat → vigor | cookie_type, distance
        try:
            X_controls = sm.add_constant(valid_w[['is_heavy', 'actual_dist']].values.astype(float))
            resid_threat = sm.OLS(valid_w['T_round'].values, X_controls).fit().resid
            resid_vigor = sm.OLS(valid_w['w_vigor'].values, X_controls).fit().resid
            rt, _ = pearsonr(resid_threat, resid_vigor)
        except:
            rt = np.nan

        # Partial correlation: cd → vigor | cookie_type, distance, threat
        valid_w2 = valid_w.merge(param_df[['subj', 'log_cd_z']], on='subj', how='left').dropna(subset=['log_cd_z'])
        try:
            X_c2 = sm.add_constant(valid_w2[['is_heavy', 'actual_dist', 'T_round']].values.astype(float))
            resid_cd = sm.OLS(valid_w2['log_cd_z'].values, X_c2).fit().resid
            resid_v2 = sm.OLS(valid_w2['w_vigor'].values, X_c2).fit().resid
            rc, _ = pearsonr(resid_cd, resid_v2)
        except:
            rc = np.nan

        r_threat_ts.append(rt)
        r_cd_ts.append(rc)

    ts_df = pd.DataFrame({'time': windows + 0.25, 'r_threat': r_threat_ts, 'r_cd': r_cd_ts})
    save_df(ts_df, 'part3_sliding_window')

    # Find crossing point
    valid_ts = ts_df.dropna()
    crossing = None
    for i in range(1, len(valid_ts)):
        if (valid_ts.iloc[i-1]['r_threat'] > valid_ts.iloc[i-1]['r_cd'] and
            valid_ts.iloc[i]['r_threat'] <= valid_ts.iloc[i]['r_cd']):
            crossing = valid_ts.iloc[i]['time']
            break
    mean_enc = np.mean(list(ENC_TIMES.values()))
    print(f"  Crossing point: {crossing if crossing else 'not found'}s (mean enc={mean_enc:.1f}s)")
    print(f"  Time series saved to part3_sliding_window.csv")

    # ── Step 3.4: Pre-encounter figure ──
    print("\n--- Step 3.4: Pre-encounter figure (β tertiles × threat) ---")
    bv = param_df.set_index('subj')['log_beta_z']
    df['bt'] = df['subj'].map(lambda s: 'Low' if bv.get(s, 0) <= bv.quantile(.33)
                               else ('High' if bv.get(s, 0) >= bv.quantile(.67) else 'Mid'))
    for ep, col, lab in [('Antic', 'antic_vigor_resid', 'ANTICIPATORY'), ('React', 'react_vigor_resid', 'REACTIVE')]:
        edf = df.dropna(subset=[col])
        print(f"\n  {lab}:")
        print(f"  {'Tert':<6} {'T=0.1':>12} {'T=0.5':>12} {'T=0.9':>12}")
        for tert in ['Low', 'Mid', 'High']:
            row = f"  {tert:<6}"
            for T in [0.1, 0.5, 0.9]:
                sub = edf[(edf['bt'] == tert) & (edf['T_round'] == T)]
                m_ = sub[col].mean()
                se = sub[col].std() / np.sqrt(len(sub)) if len(sub) > 0 else 0
                row += f"  {m_:>6.4f}({se:.4f})"
            print(row)

    # ── Step 3.5: Post-encounter figure ──
    print("\n--- Step 3.5: Post-encounter figure (cd tertiles × distance) ---")
    cv = param_df.set_index('subj')['log_cd_z']
    df['ct'] = df['subj'].map(lambda s: 'Low' if cv.get(s, 0) <= cv.quantile(.33)
                               else ('High' if cv.get(s, 0) >= cv.quantile(.67) else 'Mid'))
    for ep, col, lab in [('Antic', 'antic_vigor_resid', 'ANTICIPATORY'), ('React', 'react_vigor_resid', 'REACTIVE')]:
        edf = df.dropna(subset=[col])
        print(f"\n  {lab}:")
        print(f"  {'Tert':<6} {'D=1':>12} {'D=2':>12} {'D=3':>12}")
        for tert in ['Low', 'Mid', 'High']:
            row = f"  {tert:<6}"
            for D in [1, 2, 3]:
                sub = edf[(edf['ct'] == tert) & (edf['distance'] == D)]
                m_ = sub[col].mean()
                se = sub[col].std() / np.sqrt(len(sub)) if len(sub) > 0 else 0
                row += f"  {m_:>6.4f}({se:.4f})"
            print(row)

    # ── Step 3.6: Handoff summary table ──
    print("\n--- Step 3.6: Parameter handoff summary ---")
    key_interactions = [
        ('threat × β', 'predator_probability:beta_z'),
        ('threat × k', 'predator_probability:k_z'),
        ('threat × cd', 'predator_probability:cd_z'),
        ('dist × β', 'distance:beta_z'),
        ('dist × k', 'distance:k_z'),
        ('dist × cd', 'distance:cd_z'),
        ('cd main', 'cd_z'),
    ]
    print(f"\n  {'Interaction':<15} {'Antic':>20} {'React':>20} {'Term':>20}")
    print(f"  {'-' * 77}")
    for label, term in key_interactions:
        row = f"  {label:<15}"
        for epoch in ['ANTICIPATORY', 'REACTIVE', 'TERMINAL']:
            r = results_3_1.get(epoch, {}).get(term, {})
            z = r.get('z', np.nan)
            p = r.get('p', np.nan)
            sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))
            row += f"  z={z:>5.2f} p={p:.3f}{sig:>3}"
        print(row)

    elapsed = (time.time() - t0) / 60
    print(f"\n  Part 3 complete ({elapsed:.1f} min)")

    return {'df': df, 'results_3_1': results_3_1, 'ts_df': ts_df, 'crossing': crossing}


if __name__ == '__main__':
    p1 = {'param_df': pd.read_csv(OUT_DIR / 'part1_params_full.csv'),
           'beh_rich': load_all_trials()}
    p2 = run_part2(p1) if 'run_part2' in dir() else None
    if p2 is None:
        from part2_vigor import run_part2
        p2 = run_part2(p1)
    p3 = run_part3(p1, p2)
