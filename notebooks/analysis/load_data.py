"""
Shared data loading for analysis notebooks.
Loads behavioral, vigor, affect, and model data for one or both samples.

Usage:
    from load_data import load_sample_data, load_both
    exp_data = load_sample_data(EXP)
    exp_data, conf_data = load_both()
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore, pearsonr, linregress
from pathlib import Path


def load_sample_data(sample_config):
    """Load all data for a single sample. Returns a dict of DataFrames + indices."""
    s = sample_config
    data = {'config': s}

    # Behavioral
    beh = pd.read_csv(s.data_dir / 'behavior_rich.csv', low_memory=False)
    beh = beh[~beh['subj'].isin(s.exclude)]
    beh['T_round'] = beh['threat'].round(1)
    data['beh'] = beh

    # Choice trials
    cdf = beh[beh['type'] == 1].copy()
    cdf['T_round'] = cdf['threat'].round(1)
    data['choice'] = cdf

    # Behavior (slim)
    beh_slim_path = s.data_dir / 'behavior.csv'
    if beh_slim_path.exists():
        beh_slim = pd.read_csv(beh_slim_path)
        beh_slim = beh_slim[~beh_slim['subj'].isin(s.exclude)]
        beh_slim['threat_z'] = zscore(beh_slim['threat'])
        beh_slim['dist_z'] = zscore(beh_slim['distance_H'])
        beh_slim['T_round'] = beh_slim['threat'].round(1)
        data['beh_slim'] = beh_slim

    # Trial vigor
    tv_path = s.data_dir / 'trial_vigor.csv'
    if tv_path.exists():
        tv = pd.read_csv(tv_path)
        tv = tv[~tv['subj'].isin(s.exclude)]
        tv['T_round'] = tv['T_round'].round(1)
        tv['is_heavy'] = tv['cookie']
        tv['threat_z'] = zscore(tv['T_round'])
        data['vigor'] = tv
        data['vigor_valid'] = tv[tv['norm_rate'].notna()].copy()

    # Vigor metrics (epoch-level)
    vm_path = s.vigor_dir / 'vigor_metrics.csv'
    if vm_path.exists():
        vm = pd.read_csv(vm_path)
        vm = vm[~vm['subj'].isin(s.exclude)]
        data['vigor_metrics'] = vm

    # Cell means
    cm_path = s.vigor_dir / f'{s.name}/cell_means.csv'
    if not cm_path.exists():
        cm_path = s.vigor_dir / 'cell_means.csv'
    if cm_path.exists():
        cm = pd.read_csv(cm_path)
        cm = cm[~cm['subj'].isin(s.exclude)]
        data['cell_means'] = cm

    # Feelings
    feel_path = s.data_dir / 'feelings.csv'
    if feel_path.exists():
        feel = pd.read_csv(feel_path)
        feel = feel[~feel['subj'].isin(s.exclude)]
        feel['threat_z'] = zscore(feel['threat'].astype(float))
        feel['dist_z'] = zscore(feel['distance'].astype(float))
        data['feelings'] = feel

    # Psych
    psych_path = s.data_dir / 'psych.csv'
    if psych_path.exists():
        data['psych'] = pd.read_csv(psych_path)

    # Model params
    if s.params_path.exists():
        mp = pd.read_csv(s.params_path).set_index('subj')
        mp['log_omega'] = np.log(mp['omega'])
        mp['log_kappa'] = np.log(mp['kappa'])
        mp['omega_z'] = zscore(mp['log_omega'].values)
        mp['kappa_z'] = zscore(mp['log_kappa'].values)
        data['params'] = mp

    # Model comparison
    mc_path = s.model_dir / 'mcmc_model_comparison.csv'
    if mc_path.exists():
        data['model_comparison'] = pd.read_csv(mc_path)

    # Build master subject dataframe
    data['master'] = build_master(data)

    return data


def build_master(data):
    """Build per-subject master dataframe with params + behavioral indices + affect."""
    s = data['config']

    if 'params' not in data:
        return pd.DataFrame()

    master = data['params'].copy()
    beh = data['beh']
    cdf = data['choice']

    # Escape rate
    att = beh[beh['isAttackTrial'] == 1]
    master['escape_rate'] = att.groupby('subj').apply(
        lambda x: (x['trialEndState'] == 'escaped').mean())

    # Earnings
    master['earnings'] = beh.groupby('subj')['trialReward'].sum()

    # P(heavy)
    master['p_heavy'] = cdf.groupby('subj')['choice'].mean()

    # Choice shift
    lo = cdf[cdf['T_round'] == 0.1].groupby('subj')['choice'].mean()
    hi = cdf[cdf['T_round'] == 0.9].groupby('subj')['choice'].mean()
    master['choice_shift'] = lo - hi

    # Mean vigor
    if 'cell_means' in data:
        master['mean_vigor'] = data['cell_means'].groupby('subj')['mean_rate'].mean()

    # Vigor slope
    if 'vigor_valid' in data:
        v = data['vigor_valid']
        vslope = {}
        for subj in v['subj'].unique():
            sv = v[v['subj'] == subj].groupby('T_round')['norm_rate'].mean()
            if len(sv) >= 3:
                vslope[subj] = linregress(sv.index, sv.values).slope
        master['vigor_slope'] = pd.Series(vslope)

    # Affect indices
    if 'feelings' in data:
        feel = data['feelings']
        anx = feel[feel['questionLabel'] == 'anxiety']
        conf = feel[feel['questionLabel'] == 'confidence']
        for subj in master.index:
            sa = anx[anx['subj'] == subj]
            sc = conf[conf['subj'] == subj]
            if len(sa) >= 5:
                master.loc[subj, 'anx_calibration'] = pearsonr(sa['threat'], sa['response'])[0]
                master.loc[subj, 'anx_slope'] = linregress(
                    sa['threat'].values, sa['response'].values).slope
                master.loc[subj, 'mean_anxiety'] = sa['response'].mean()
            if len(sc) >= 5:
                master.loc[subj, 'mean_confidence'] = sc['response'].mean()

    # Optimality
    opt_map = {}
    for (T, D), grp in cdf.groupby(['T_round', 'distance_H']):
        er_h = grp[grp['choice'] == 1]['trialReward'].mean() if (grp['choice'] == 1).sum() > 0 else -99
        er_l = grp[grp['choice'] == 0]['trialReward'].mean() if (grp['choice'] == 0).sum() > 0 else -99
        opt_map[(T, D)] = 1 if er_h > er_l else 0
    cdf = cdf.copy()
    cdf['optimal'] = cdf.apply(lambda r: opt_map.get((r['T_round'], r['distance_H']), np.nan), axis=1)
    cdf['is_opt'] = (cdf['choice'] == cdf['optimal']).astype(int)
    cdf['err_type'] = np.where(cdf['is_opt'] == 1, 'correct',
                               np.where(cdf['choice'] == 0, 'overcautious', 'reckless'))
    so = cdf.groupby('subj').agg(
        pct_opt=('is_opt', 'mean'),
        n_oc=('err_type', lambda x: (x == 'overcautious').sum()),
        n_rk=('err_type', lambda x: (x == 'reckless').sum()),
        n_err=('is_opt', lambda x: (x == 0).sum()),
    )
    so['oc_ratio'] = so['n_oc'] / so['n_err'].clip(1)
    for c in so.columns:
        if c in master.columns:
            master = master.drop(columns=[c])
    master = master.join(so, how='left')

    # Angle
    master['angle'] = np.arctan2(master['kappa_z'], master['omega_z'])
    master['angle_z'] = zscore(master['angle'].values, nan_policy='omit')

    # Merge psych if available
    if 'psych' in data:
        psych = data['psych'].set_index('subj')
        shared_cols = [c for c in psych.columns if c not in master.columns and c != 'participantID']
        if shared_cols:
            master = master.join(psych[shared_cols], how='left')

    return master


def load_both():
    """Load both samples. Returns (exp_data, conf_data) dicts."""
    from config import EXP, CONF
    exp_data = load_sample_data(EXP) if EXP else None
    conf_data = load_sample_data(CONF) if CONF else None
    return exp_data, conf_data
