#!/usr/bin/env python3
"""Part 2: Vigor Decomposition (Steps 2.1–2.5)"""

import sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *


def run_part2(p1=None):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PART 2: VIGOR DECOMPOSITION")
    print("=" * 70)

    # Load and compute epoch vigor
    beh_rich = p1['beh_rich'] if p1 and 'beh_rich' in p1 else load_all_trials()
    param_df = p1['param_df'] if p1 else pd.read_csv(OUT_DIR / 'part1_params_full.csv')

    print("\n--- Step 2.1: Computing epoch-level vigor ---")
    df = compute_epoch_vigor(beh_rich)
    df = df.merge(param_df[['subj', 'log_k_z', 'log_beta_z', 'log_cd_z']], on='subj', how='left')
    print(f"  Trials: {len(df)}, Subjects: {df['subj'].nunique()}")
    print(f"  Valid: antic={df['antic_vigor'].notna().sum()}, "
          f"react={df['react_vigor'].notna().sum()}, term={df['term_vigor'].notna().sum()}")

    # Step 2.2: Residualize
    print("\n--- Step 2.2: Residualizing epochs ---")
    for epoch, col in [('Antic', 'antic_vigor'), ('React', 'react_vigor'), ('Term', 'term_vigor')]:
        resid_df = residualize_epoch(df, col)
        df = df.merge(resid_df, on=['subj', 'trial'], how='left')

    # Step 2.3: Verify
    print("\n--- Step 2.3: Verification ---")
    stop = True
    for epoch, col in [('Antic', 'antic_vigor_resid'), ('React', 'react_vigor_resid'), ('Term', 'term_vigor_resid')]:
        edf = df.dropna(subset=[col])
        subj_m = edf.groupby('subj')[col].mean()
        rt, pt = pearsonr(edf['predator_probability'], edf[col])
        rd, pd_ = pearsonr(edf['distance'], edf[col])
        print(f"\n  {epoch} (N={len(edf)}):")
        print(f"    r(threat) = {rt:.4f}, p = {pt:.2e}")
        print(f"    r(distance) = {rd:.4f}, p = {pd_:.2e}")
        print(f"    Between-subj SD = {subj_m.std():.5f}")
        if epoch in ['Antic', 'React'] and pt < 0.05:
            stop = False

    if stop:
        return {'STOP': 'Threat does not predict vigor_resid in anticipatory AND reactive'}

    # Step 2.4: Variance
    print("\n--- Step 2.4: Between-subject variance ---")
    sa = df.groupby('subj')['antic_vigor_resid'].mean().dropna()
    sr = df.groupby('subj')['react_vigor_resid'].mean().dropna()
    st_ = df.groupby('subj')['term_vigor_resid'].mean().dropna()
    print(f"  Antic SD:  {sa.std():.5f}")
    print(f"  React SD:  {sr.std():.5f}")
    print(f"  Term SD:   {st_.std():.5f}")
    print(f"  Antic/React ratio: {sa.std()/sr.std():.1f}×")

    shared = sorted(set(sa.index) & set(sr.index) & set(st_.index))
    s1, p1_ = levene(sa.loc[shared], sr.loc[shared], st_.loc[shared])
    print(f"  Levene's (3 epochs): F={s1:.3f}, p={p1_:.4f}")

    # Step 2.5: Cross-epoch
    print("\n--- Step 2.5: Cross-epoch correlations ---")
    mat = pd.DataFrame({'Antic': sa.loc[shared], 'React': sr.loc[shared], 'Term': st_.loc[shared]})
    print(f"  N={len(shared)}")
    for c1 in ['Antic', 'React', 'Term']:
        vals = []
        for c2 in ['Antic', 'React', 'Term']:
            if c1 == c2:
                vals.append("1.000 ")
            else:
                r, p = pearsonr(mat[c1], mat[c2])
                vals.append(f"{r:.3f}{'*' if p < .05 else ' '}")
        print(f"  {c1:<6} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

    save_df(df[['subj', 'trial', 'T_round', 'distance', 'cookie_type', 'is_attack',
                'predator_probability', 'antic_vigor_resid', 'react_vigor_resid', 'term_vigor_resid',
                'log_k_z', 'log_beta_z', 'log_cd_z']].dropna(subset=['antic_vigor_resid']),
            'part2_epoch_data')

    elapsed = (time.time() - t0) / 60
    print(f"\n  Part 2 complete ({elapsed:.1f} min)")

    return {'df': df, 'shared_subj': shared, 'sa': sa, 'sr': sr, 'st': st_}


if __name__ == '__main__':
    p1 = {'param_df': pd.read_csv(OUT_DIR / 'part1_params_full.csv'),
           'beh_rich': load_all_trials()}
    p2 = run_part2(p1)
    if p2.get('STOP'):
        print(f"\n*** STOPPED: {p2['STOP']} ***")
