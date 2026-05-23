"""
Posterior Predictive Checks for the Joint Model (M4).

PPC 1: Group-level choice — observed vs predicted P(heavy) by threat × distance
PPC 2: Subject-level — choice + vigor predictions across trials for selected subjects

Usage:
  python scripts/plotting/plot_ppc.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import expit
from pathlib import Path

from plotter import Colors, set_plot_style, style_axis, plot_parameter_half_violin_glow

set_plot_style()
plt.rcParams.update({'savefig.dpi': 300, 'figure.dpi': 300})

OUT_DIR = Path('results/figs/h3')
OUT_DIR.mkdir(parents=True, exist_ok=True)

C = 5.0  # capture penalty


# ============================================================
# Reconstruct model predictions from posterior means
# ============================================================

def _load_pop_params(sample_name):
    """Extract M4 population parameter means from convergence diagnostics."""
    diag = pd.read_csv(f'results/stats/joint_optimal/{sample_name}/mcmc_convergence_diagnostics.csv')
    m4 = diag[diag['model'] == 'M4'].set_index('parameter')

    return {
        'gamma': float(np.clip(np.exp(m4.loc['gr', 'mean']), 0.1, 3.0)),
        'hazard': float(np.exp(m4.loc['hr', 'mean'])),
        'tau': float(np.clip(np.exp(m4.loc['tr', 'mean']), 0.01, 50.)),
        'sp': float(np.clip(np.exp(m4.loc['spr', 'mean']), 0.01, 1.0)),
        'bc': float(m4.loc['bc', 'mean']),
        'sv': float(m4.loc['sv', 'mean']),
    }


def _eu_sat_numpy(om, kap, T, D, R, req, gamma, hazard, sp, ug):
    """Numpy version of eu_sat: returns (u_star, V_star) for each trial."""
    u = ug[None, :]  # (1, U)
    speed = expit((u - 0.25 * req[:, None]) / sp)
    S = np.exp(-hazard * np.power(T[:, None], gamma) * D[:, None]
               / np.clip(speed, 0.01, None))
    W = (S * R[:, None]
         - (1.0 - S) * om[:, None] * (R[:, None] + C)
         - kap[:, None] * (u - req[:, None]) ** 2 * D[:, None])
    # Soft argmax (matching model)
    w = np.exp(W * 20.0)
    w = w / w.sum(axis=1, keepdims=True)
    u_star = (w * u).sum(axis=1)
    V_star = (w * W).sum(axis=1)
    return u_star, V_star


def predict_choice(choice_df, params_df, pop, subj_col='subj_idx'):
    """Predict P(heavy) for each trial using posterior mean parameters."""
    ct = choice_df.copy()
    # Map subject params
    pm = params_df.set_index('subj_idx') if 'subj_idx' in params_df.columns else params_df
    om = pm.loc[ct[subj_col].values, 'omega'].values
    kap = pm.loc[ct[subj_col].values, 'kappa'].values

    ug = np.linspace(0.1, 1.5, 40)
    N = len(ct)
    T = ct['threat'].values.astype(float)
    DH = ct['distance_H'].values.astype(float)

    # Heavy cookie
    _, VH_base = _eu_sat_numpy(om, kap, T, DH,
                                np.full(N, 5.0), np.full(N, 0.9),
                                pop['gamma'], pop['hazard'], pop['sp'], ug)
    VH = VH_base - kap * 0.9 * DH

    # Light cookie (distance always 1)
    _, VL_base = _eu_sat_numpy(om, kap, T, np.ones(N),
                                np.full(N, 1.0), np.full(N, 0.4),
                                pop['gamma'], pop['hazard'], pop['sp'], ug)
    VL = VL_base - kap * 0.4 * 1.0

    pH = expit(np.clip((VH - VL) / pop['tau'], -20, 20))
    ct['p_heavy'] = pH
    return ct


def predict_vigor(vigor_df, params_df, pop, subj_col='subj_idx'):
    """Predict vigor (u*) for each cell using posterior mean parameters."""
    vd = vigor_df.copy()
    pm = params_df.set_index('subj_idx') if 'subj_idx' in params_df.columns else params_df
    om = pm.loc[vd[subj_col].values, 'omega'].values
    kap = pm.loc[vd[subj_col].values, 'kappa'].values

    ug = np.linspace(0.1, 1.5, 40)
    T = vd['T_round'].values.astype(float)
    D = vd['actual_dist'].values.astype(float)
    R = vd['actual_R'].values.astype(float)
    req = vd['actual_req'].values.astype(float)
    cookie = vd['is_heavy'].values.astype(float)

    u_star, _ = _eu_sat_numpy(om, kap, T, D, R, req,
                               pop['gamma'], pop['hazard'], pop['sp'], ug)
    vd['pred_vigor'] = u_star + pop['bc'] * cookie
    return vd


def predict_trial_vigor(trial_vigor_df, params_df, pop, subj_map):
    """Predict vigor for individual trials from trial_vigor.csv."""
    tv = trial_vigor_df.copy()
    # Map subj → subj_idx
    tv = tv.merge(subj_map[['subj', 'subj_idx']], on='subj', how='left')
    tv = tv.dropna(subset=['subj_idx'])
    tv['subj_idx'] = tv['subj_idx'].astype(int)

    pm = params_df.set_index('subj_idx')
    om = pm.loc[tv['subj_idx'].values, 'omega'].values
    kap = pm.loc[tv['subj_idx'].values, 'kappa'].values

    ug = np.linspace(0.1, 1.5, 40)
    T = tv['T_round'].values.astype(float)
    D = (tv['distance'].values + 1).astype(float)  # 0,1,2 → 1,2,3
    is_heavy = tv['cookie'].values.astype(float)
    R = np.where(is_heavy, 5.0, 1.0)
    req = np.where(is_heavy, 0.9, 0.4)

    u_star, _ = _eu_sat_numpy(om, kap, T, D, R, req,
                               pop['gamma'], pop['hazard'], pop['sp'], ug)
    tv['pred_vigor'] = u_star + pop['bc'] * is_heavy
    return tv


# ============================================================
# Load data for both samples
# ============================================================

def load_sample_data(sample_name):
    """Load model input + fitted params for a sample."""
    input_dir = Path(f'data/model_input_{sample_name}')
    choice = pd.read_csv(input_dir / 'choice_trials.csv')
    vigor = pd.read_csv(input_dir / 'vigor_cell_means.csv')
    subj_map = pd.read_csv(input_dir / 'subject_mapping.csv')

    # Trial-level vigor
    base = Path(f'data/{sample_name}_350/processed')
    s5 = sorted(base.glob('stage5_*'))[-1]
    trial_vigor = pd.read_csv(s5 / 'trial_vigor.csv')

    params = pd.read_csv(f'results/stats/joint_optimal/{sample_name}/mcmc_m4_params.csv')
    params = params.merge(subj_map, on='subj')

    pop = _load_pop_params(sample_name)

    return {
        'choice': choice, 'vigor': vigor, 'trial_vigor': trial_vigor,
        'params': params, 'subj_map': subj_map, 'pop': pop,
        'label': 'Exploratory' if sample_name == 'exploratory' else 'Confirmatory',
    }


samples = {}
for name in ['exploratory', 'confirmatory']:
    try:
        samples[name] = load_sample_data(name)
    except Exception as e:
        print(f'Skipping {name}: {e}')

print(f'Loaded: {", ".join(s["label"] for s in samples.values())}')


# ============================================================
# PPC 1: Group-level choice by T × D (like ppc_1.png)
# ============================================================

def _draw_group_choice_panel(ax, ct, label, show_legend=False):
    """Draw a single group-level PPC panel."""
    threats = [0.1, 0.5, 0.9]
    distances = [1, 2, 3]
    threat_colors = {0.1: Colors.CERULEAN2, 0.5: Colors.SLATE, 0.9: Colors.RUBY1}

    ct = ct.copy()
    ct['T_round'] = ct['threat'].round(1)
    bar_width = 0.12
    x_pos = np.arange(len(distances))

    for t_idx, T in enumerate(threats):
        obs_means, obs_sems = [], []
        pred_means, pred_sems = [], []
        for D in distances:
            sub = ct[(ct['T_round'] == T) & (ct['distance_H'] == D)]
            sg = sub.groupby('subj_idx').agg(
                obs=('choice', 'mean'), pred=('p_heavy', 'mean'))
            obs_means.append(sg['obs'].mean())
            obs_sems.append(sg['obs'].sem())
            pred_means.append(sg['pred'].mean())
            pred_sems.append(sg['pred'].sem())

        offset = (t_idx - 1) * (bar_width * 2.3)
        color = threat_colors[T]

        # Observed: solid fill
        ax.bar(x_pos + offset - bar_width * 0.55, obs_means, bar_width,
               yerr=[se * 1.96 for se in obs_sems], capsize=2,
               color=color, alpha=0.85, edgecolor='white', linewidth=0.5,
               zorder=3)
        # Predicted: hatched outline
        ax.bar(x_pos + offset + bar_width * 0.55, pred_means, bar_width,
               yerr=[se * 1.96 for se in pred_sems], capsize=2,
               facecolor='none', edgecolor=color, linewidth=1.3,
               hatch='///', zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'D = {d}' for d in distances], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(label, fontsize=11, color=Colors.DARK_GREY, pad=10)
    style_axis(ax, xlabel='Distance', ylabel='P(choose heavy)')
    ax.set_facecolor('#FCFCFD')

    if show_legend:
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=Colors.INK, alpha=0.85, label='Observed'),
            Patch(facecolor='none', edgecolor=Colors.INK, linewidth=1.2,
                  hatch='///', label='Predicted'),
        ]
        for T, c in threat_colors.items():
            handles.append(Patch(facecolor=c, alpha=0.7, label=f'T = {T}'))
        leg = ax.legend(handles=handles, fontsize=7, frameon=True,
                        labelcolor=Colors.INK, loc='upper right')
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('#E5E7EB')
        leg.get_frame().set_linewidth(0.8)


def plot_ppc_group_choice():
    """Combined: both samples side by side."""
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(5.0 * n_samples, 3.5), sharey=True)
    if n_samples == 1:
        axes = [axes]

    for i, (sname, s) in enumerate(samples.items()):
        ct = predict_choice(s['choice'], s['params'], s['pop'])
        ax = axes[i]
        _draw_group_choice_panel(ax, ct, s['label'], show_legend=(i == 0))
        if i > 0:
            ax.set_ylabel(None)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'ppc_group_choice.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'ppc_group_choice.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved ppc_group_choice')


def plot_ppc_group_choice_individual():
    """Individual plots: one per sample."""
    for sname, s in samples.items():
        ct = predict_choice(s['choice'], s['params'], s['pop'])
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        _draw_group_choice_panel(ax, ct, s['label'], show_legend=True)
        plt.tight_layout()
        fig.savefig(OUT_DIR / f'ppc_group_choice_{sname}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUT_DIR / f'ppc_group_choice_{sname}.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Saved ppc_group_choice_{sname}')


# ============================================================
# PPC 2: Subject-level choice + vigor across trials
# ============================================================

def plot_ppc_subject(seed=42):
    """One figure per subject: choice and vigor stacked vertically.
    Generates a 3×3 grid of subjects across (omega, kappa) param space."""
    from plotter import _gradient_fill_under_curve
    from matplotlib.lines import Line2D

    sname = 'confirmatory' if 'confirmatory' in samples else list(samples.keys())[0]
    s = samples[sname]

    ct = predict_choice(s['choice'], s['params'], s['pop'])
    tv = predict_trial_vigor(s['trial_vigor'], s['params'], s['pop'], s['subj_map'])

    # Pick subjects spread across (omega, kappa) param space
    params = s['params'].copy()
    params['omega_rank'] = params['omega'].rank(pct=True)
    params['kappa_rank'] = params['kappa'].rank(pct=True)

    selected = []
    for q_o in [0.1, 0.5, 0.9]:
        for q_k in [0.1, 0.5, 0.9]:
            dist = (params['omega_rank'] - q_o)**2 + (params['kappa_rank'] - q_k)**2
            selected.append(params.loc[[dist.idxmin()]])
    selected = pd.concat(selected).drop_duplicates(subset=['subj'])

    for fig_i, (_, subj_row) in enumerate(selected.iterrows()):
        sidx = subj_row['subj_idx']
        omega = subj_row['omega']
        kappa = subj_row['kappa']
        subj_id = int(subj_row['subj'])

        fig, axes = plt.subplots(2, 1, figsize=(10.2, 4.5),
                                  gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.4})

        # ── Row 0: Choice ──
        ax_c = axes[0]
        sc = ct[ct['subj_idx'] == sidx].copy().reset_index(drop=True)
        trials = np.arange(len(sc))

        # Raw choice dots (0/1)
        heavy = sc['choice'] == 1
        ax_c.scatter(trials[heavy], sc.loc[heavy, 'choice'], s=18, alpha=0.5,
                     color=Colors.CERULEAN2, edgecolors='none', zorder=2, marker='o')
        ax_c.scatter(trials[~heavy], sc.loc[~heavy, 'choice'], s=18, alpha=0.5,
                     color=Colors.SLATE, edgecolors='none', zorder=2, marker='o')

        # Observed: smoothed (3-trial centered rolling mean)
        obs_smooth = sc['choice'].rolling(3, center=True, min_periods=1).mean().values
        ax_c.plot(trials, obs_smooth, color=Colors.SLATE, lw=2.2, zorder=3,
                  label='Observed (smoothed)')
        _gradient_fill_under_curve(ax_c, trials, obs_smooth, baseline=0.0,
                                    color=Colors.SLATE, alpha_top=0.10)

        # Predicted P(heavy)
        p = sc['p_heavy'].values
        ax_c.plot(trials, p, color=Colors.CERULEAN2, lw=2.2, zorder=4,
                  label='Model P(heavy)')
        _gradient_fill_under_curve(ax_c, trials, p, baseline=0.0,
                                    color=Colors.CERULEAN2, alpha_top=0.15)

        ax_c.set_ylim(-0.08, 1.08)
        ax_c.set_title(f'Subject {subj_id}  —  $\\omega$ = {omega:.2f},  $\\kappa$ = {kappa:.2f}',
                        fontsize=11, color=Colors.DARK_GREY, pad=8)
        style_axis(ax_c, ylabel='P(choose heavy)', xlabel='Trial')
        ax_c.set_facecolor('#FCFCFD')

        leg = ax_c.legend(fontsize=8, frameon=True, labelcolor=Colors.INK,
                          loc='lower left')
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('#E5E7EB')
        leg.get_frame().set_linewidth(0.8)

        # ── Row 1: Vigor — all trials ──
        ax_v = axes[1]
        subj_real = selected[selected['subj_idx'] == sidx]['subj'].values[0]
        sv = tv[(tv['subj'] == subj_real) & (tv['type'] == 1)].copy()
        sv = sv.sort_values('trial').reset_index(drop=True)

        if len(sv) == 0 or sv['norm_rate'].isna().all():
            ax_v.text(0.5, 0.5, 'No vigor data', transform=ax_v.transAxes,
                      ha='center', fontsize=10, color=Colors.INK)
            style_axis(ax_v)
            ax_v.set_facecolor('#FCFCFD')
        else:
            trials_v = np.arange(len(sv))
            obs = sv['norm_rate'].values
            pred = sv['pred_vigor'].values

            # Observed: grey line with gradient fill
            ax_v.plot(trials_v, obs, color=Colors.SLATE, lw=2.0, zorder=2,
                      label='Observed')
            _gradient_fill_under_curve(ax_v, trials_v, obs,
                                        baseline=0,
                                        color=Colors.SLATE, alpha_top=0.10)

            # Predicted: red line with gradient fill
            ax_v.plot(trials_v, pred, color=Colors.RUBY1, lw=2.0, zorder=3,
                      label='Predicted')
            _gradient_fill_under_curve(ax_v, trials_v, pred,
                                        baseline=0,
                                        color=Colors.RUBY1, alpha_top=0.12)

            # Per-subject auto-range with padding, snapped to nearest 0.1
            all_vals = np.concatenate([obs[np.isfinite(obs)], pred[np.isfinite(pred)]])
            if len(all_vals) > 0:
                lo = np.floor((all_vals.min() - 0.05) * 10) / 10
                hi = np.ceil((all_vals.max() + 0.05) * 10) / 10
                ax_v.set_ylim(max(0, lo), hi)

            style_axis(ax_v, ylabel='Normalized press rate', xlabel='Trial')
            ax_v.set_facecolor('#FCFCFD')

            leg = ax_v.legend(fontsize=8, frameon=True, labelcolor=Colors.INK,
                              loc='upper right')
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#E5E7EB')
            leg.get_frame().set_linewidth(0.8)

        plt.tight_layout()
        fig.savefig(OUT_DIR / f'ppc_subject_{fig_i+1}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUT_DIR / f'ppc_subject_{fig_i+1}.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Saved ppc_subject_{fig_i+1} (subj {subj_id}, ω={omega:.2f})')


# ============================================================
# PPC 3: Clean 3-panel paper figure (subjects spanning omega)
# ============================================================

def plot_ppc_subject_paper(subjects=(75, 35)):
    """One figure per subject — choice (top) and vigor (bottom) by trial.
    Legends sit outside the axes to the right. Saved as
    ppc_subject_paper_{subj_id}.png/pdf for each subject.

    Default subjects contrast cautiousness: 75 has high ω (≈4.85, mostly
    chooses light), 35 has low ω (≈0.82, mostly chooses heavy). Both have
    good model fit and clean vigor data.
    """
    from plotter import _gradient_fill_under_curve

    sname = 'confirmatory' if 'confirmatory' in samples else list(samples.keys())[0]
    s = samples[sname]

    ct = predict_choice(s['choice'], s['params'], s['pop'])
    tv = predict_trial_vigor(s['trial_vigor'], s['params'], s['pop'], s['subj_map'])
    sm = s['subj_map']
    params = s['params']

    for subj_id in subjects:
        sidx = sm[sm['subj'] == subj_id]['subj_idx'].values[0]
        p = params[params['subj'] == subj_id].iloc[0]
        omega, kappa = p['omega'], p['kappa']

        fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.0),
                                  gridspec_kw={'height_ratios': [1, 1],
                                               'hspace': 0.42})

        # ── Row 0: Choice ──
        ax_c = axes[0]
        sc = ct[ct['subj_idx'] == sidx].copy().reset_index(drop=True)
        trials = np.arange(len(sc))

        heavy = sc['choice'] == 1
        ax_c.scatter(trials[heavy], sc.loc[heavy, 'choice'], s=14, alpha=0.5,
                     color=Colors.CERULEAN2, edgecolors='none', zorder=2, marker='o')
        ax_c.scatter(trials[~heavy], sc.loc[~heavy, 'choice'], s=14, alpha=0.5,
                     color=Colors.SLATE, edgecolors='none', zorder=2, marker='o')

        obs_smooth = sc['choice'].rolling(3, center=True, min_periods=1).mean().values
        ax_c.plot(trials, obs_smooth, color=Colors.SLATE, lw=2.0, zorder=3,
                  label='Observed')
        _gradient_fill_under_curve(ax_c, trials, obs_smooth, baseline=0.0,
                                    color=Colors.SLATE, alpha_top=0.10)

        pH = sc['p_heavy'].values
        ax_c.plot(trials, pH, color=Colors.CERULEAN2, lw=2.0, zorder=4,
                  label='Predicted')
        _gradient_fill_under_curve(ax_c, trials, pH, baseline=0.0,
                                    color=Colors.CERULEAN2, alpha_top=0.15)

        ax_c.set_ylim(-0.08, 1.08)
        ax_c.set_title(f'Subject {subj_id}  —  $\\omega$ = {omega:.2f},  $\\kappa$ = {kappa:.2f}',
                        fontsize=11, color=Colors.DARK_GREY, pad=8)
        style_axis(ax_c, ylabel='P(choose heavy)', xlabel=None)
        ax_c.set_facecolor('#FCFCFD')
        leg_c = ax_c.legend(fontsize=8, frameon=True, labelcolor=Colors.INK,
                             loc='center left', bbox_to_anchor=(1.02, 0.5))
        leg_c.get_frame().set_facecolor('white')
        leg_c.get_frame().set_edgecolor('#E5E7EB')
        leg_c.get_frame().set_linewidth(0.8)

        # ── Row 1: Vigor ──
        ax_v = axes[1]
        sv = tv[(tv['subj'] == subj_id) & (tv['type'] == 1)].copy()
        sv = sv.sort_values('trial').reset_index(drop=True)
        trials_v = np.arange(len(sv))
        obs = sv['norm_rate'].values
        pred = sv['pred_vigor'].values

        ax_v.plot(trials_v, obs, color=Colors.SLATE, lw=1.8, zorder=2,
                  label='Observed')
        _gradient_fill_under_curve(ax_v, trials_v, obs, baseline=0,
                                    color=Colors.SLATE, alpha_top=0.10)

        ax_v.plot(trials_v, pred, color=Colors.RUBY1, lw=1.8, zorder=3,
                  label='Predicted')
        _gradient_fill_under_curve(ax_v, trials_v, pred, baseline=0,
                                    color=Colors.RUBY1, alpha_top=0.12)

        all_vals = np.concatenate([obs[np.isfinite(obs)], pred[np.isfinite(pred)]])
        if len(all_vals) > 0:
            lo = np.floor((all_vals.min() - 0.05) * 10) / 10
            hi = np.ceil((all_vals.max() + 0.05) * 10) / 10
            ax_v.set_ylim(max(0, lo), hi)

        style_axis(ax_v, ylabel='Normalized press rate', xlabel='Trial')
        ax_v.set_facecolor('#FCFCFD')
        leg_v = ax_v.legend(fontsize=8, frameon=True, labelcolor=Colors.INK,
                             loc='center left', bbox_to_anchor=(1.02, 0.5))
        leg_v.get_frame().set_facecolor('white')
        leg_v.get_frame().set_edgecolor('#E5E7EB')
        leg_v.get_frame().set_linewidth(0.8)

        plt.tight_layout()
        fig.savefig(OUT_DIR / f'ppc_subject_paper_{subj_id}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUT_DIR / f'ppc_subject_paper_{subj_id}.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Saved ppc_subject_paper_{subj_id}')


# ============================================================
# PPC: Group-level vigor — observed vs predicted scatter
# ============================================================

def _draw_group_vigor_panel(ax, vd, label, show_legend=False):
    """Scatter of group-mean observed vs predicted vigor per (T × D × cookie) cell."""
    threat_colors = {0.1: Colors.CERULEAN2, 0.5: Colors.SLATE, 0.9: Colors.RUBY1}

    # Group mean ± SEM across subjects
    g = vd.groupby(['T_round', 'actual_dist', 'is_heavy']).agg(
        obs_mean=('mean_rate', 'mean'),
        obs_sem=('mean_rate', 'sem'),
        pred_mean=('pred_vigor', 'mean'),
        pred_sem=('pred_vigor', 'sem'),
    ).reset_index()

    # Identity line
    lo = min(g['obs_mean'].min(), g['pred_mean'].min()) - 0.05
    hi = max(g['obs_mean'].max(), g['pred_mean'].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=Colors.INK, lw=0.8, ls='--',
            alpha=0.4, zorder=1)

    # One point per cell, colored by threat, marker by cookie type
    for _, row in g.iterrows():
        c = threat_colors[row['T_round']]
        marker = 'o' if row['is_heavy'] == 1 else 's'
        face = c if row['is_heavy'] == 1 else 'white'
        ax.errorbar(row['obs_mean'], row['pred_mean'],
                    xerr=row['obs_sem'] * 1.96, yerr=row['pred_sem'] * 1.96,
                    fmt='none', ecolor=c, elinewidth=1.0, alpha=0.4,
                    capsize=2, zorder=2)
        ax.scatter(row['obs_mean'], row['pred_mean'],
                   s=80, c=face, edgecolors=c, linewidths=1.5,
                   marker=marker, zorder=3, alpha=0.9)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(label, fontsize=11, color=Colors.DARK_GREY, pad=10)
    style_axis(ax, xlabel='Observed mean vigor', ylabel='Predicted mean vigor')
    ax.set_facecolor('#FCFCFD')

    if show_legend:
        from matplotlib.lines import Line2D
        handles = []
        for T, c in threat_colors.items():
            handles.append(Line2D([0], [0], marker='o', color=c, lw=0,
                                   markersize=7, label=f'T = {T}'))
        handles.append(Line2D([0], [0], marker='o', color=Colors.INK, lw=0,
                               markersize=7, label='Heavy'))
        handles.append(Line2D([0], [0], marker='s', color=Colors.INK, lw=0,
                               markersize=7, markerfacecolor='white',
                               markeredgewidth=1.2, label='Light'))
        leg = ax.legend(handles=handles, fontsize=7, frameon=True,
                        labelcolor=Colors.INK, loc='lower right')
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('#E5E7EB')
        leg.get_frame().set_linewidth(0.8)


def plot_ppc_group_vigor():
    """Combined: both samples side by side."""
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.2 * n_samples, 4.0))
    if n_samples == 1:
        axes = [axes]

    for i, (sname, s) in enumerate(samples.items()):
        vd = predict_vigor(s['vigor'], s['params'], s['pop'])
        _draw_group_vigor_panel(axes[i], vd, s['label'],
                                 show_legend=(i == 0))

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'ppc_group_vigor.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'ppc_group_vigor.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved ppc_group_vigor')


def _draw_subject_vigor_panel(ax, vd, label, obs_cap=1.5):
    """Scatter of per-subject mean vigor: observed vs predicted.

    Subjects with observed mean vigor above ``obs_cap`` are dropped from the
    panel because the model's vigor grid (u ∈ [0.1, 1.5]) cannot represent
    them — they sit far off the identity line and distort the scatter.
    """
    # Per-subject weighted mean (weighted by n_trials)
    g = vd.groupby('subj_idx').apply(
        lambda x: pd.Series({
            'obs': (x['mean_rate'] * x['n_trials']).sum() / x['n_trials'].sum(),
            'pred': (x['pred_vigor'] * x['n_trials']).sum() / x['n_trials'].sum(),
        })
    ).reset_index()

    g_in = g[g['obs'] <= obs_cap]

    # Identity line
    lo = min(g_in['obs'].min(), g_in['pred'].min()) - 0.05
    hi = max(g_in['obs'].max(), g_in['pred'].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=Colors.INK, lw=0.8, ls='--',
            alpha=0.4, zorder=1)

    ax.scatter(g_in['obs'], g_in['pred'], s=22, c=Colors.CERULEAN2,
               edgecolors='white', linewidths=0.5, alpha=0.7, zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(label, fontsize=11, color=Colors.DARK_GREY, pad=10)
    style_axis(ax, xlabel='Observed mean vigor', ylabel='Predicted mean vigor')
    ax.set_facecolor('#FCFCFD')


def plot_ppc_subject_vigor():
    """Subject-mean version (one point per subject)."""
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.2 * n_samples, 4.0))
    if n_samples == 1:
        axes = [axes]

    for i, (sname, s) in enumerate(samples.items()):
        vd = predict_vigor(s['vigor'], s['params'], s['pop'])
        _draw_subject_vigor_panel(axes[i], vd, s['label'])

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'ppc_subject_vigor.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'ppc_subject_vigor.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved ppc_subject_vigor')


def plot_ppc_group_vigor_individual():
    """Individual plots: one per sample."""
    for sname, s in samples.items():
        vd = predict_vigor(s['vigor'], s['params'], s['pop'])
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        _draw_group_vigor_panel(ax, vd, s['label'], show_legend=True)
        plt.tight_layout()
        fig.savefig(OUT_DIR / f'ppc_group_vigor_{sname}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUT_DIR / f'ppc_group_vigor_{sname}.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Saved ppc_group_vigor_{sname}')


def plot_ppc_subject_vigor_individual():
    """Individual subject-mean vigor scatter: one per sample."""
    for sname, s in samples.items():
        vd = predict_vigor(s['vigor'], s['params'], s['pop'])
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        _draw_subject_vigor_panel(ax, vd, s['label'])
        plt.tight_layout()
        fig.savefig(OUT_DIR / f'ppc_subject_vigor_{sname}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUT_DIR / f'ppc_subject_vigor_{sname}.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Saved ppc_subject_vigor_{sname}')


# ============================================================
# PPC: Parameter estimate raincloud plots (per sample)
# ============================================================

def _draw_raincloud(ax, vals, color, *, ridge_height=0.6,
                     point_alpha=0.30, point_size=14, jitter_sd=0.05,
                     gridsize=600, density_cut=0.01, gradient_alpha_top=0.30):
    """Single-row raincloud: KDE on top, box+points below at y=0."""
    from matplotlib.patches import Rectangle
    from plotter import (_kde_gaussian_1d, _gradient_fill_under_curve_oriented)

    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    vmin, vmax = float(vals.min()), float(vals.max())
    span = vmax - vmin + 1e-9
    pad = 0.04 * span
    grid = np.linspace(vmin - pad, vmax + pad, gridsize)

    dens = _kde_gaussian_1d(vals, grid, bw=None)
    if dens.max() > 0:
        dens = dens / dens.max()
    mask = dens >= density_cut
    if mask.any():
        grid_eff, dens_eff = grid[mask], dens[mask]
    else:
        grid_eff, dens_eff = grid, dens

    # KDE ridge above the baseline (y=0)
    edge = ridge_height * dens_eff
    _gradient_fill_under_curve_oriented(
        ax, grid_eff, edge,
        baseline=0.0, color=color, alpha_top=gradient_alpha_top,
    )
    ax.plot(grid_eff, edge, color=color, alpha=0.85, lw=1.6, zorder=2)

    # Jittered points just below the baseline
    rng = np.random.default_rng(42)
    yj = -0.20 + rng.normal(0, jitter_sd, size=vals.size)
    ax.scatter(vals, yj, s=point_size, color=color, alpha=point_alpha,
               edgecolors='none', zorder=2)

    # Box & whiskers at y=-0.20
    q5, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
    y0 = -0.20
    box_h = 0.18

    ax.hlines(y0, q5, q95, color=Colors.INK, lw=1.3, zorder=3)
    ax.vlines([q5, q95], y0 - 0.10, y0 + 0.10, color=Colors.INK, lw=1.1, zorder=3)
    rect = Rectangle((q25, y0 - box_h / 2), q75 - q25, box_h,
                     facecolor='none', edgecolor=Colors.INK, lw=1.3, zorder=3.2)
    ax.add_patch(rect)
    ax.vlines(q50, y0 - box_h / 2, y0 + box_h / 2,
              color=Colors.INK, lw=1.8, zorder=3.3)


def _draw_vertical_box_jitter(ax, vals, color, *, x_pos=0,
                               jitter_sd=0.07, point_alpha=0.30,
                               point_size=22, box_w=0.32):
    """Vertical box (q25-q75, median line) + whiskers (q5-q95) + jittered points."""
    from matplotlib.patches import Rectangle

    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    # Jittered scatter
    rng = np.random.default_rng(42)
    x_jit = x_pos + rng.normal(0, jitter_sd, size=vals.size)
    ax.scatter(x_jit, vals, s=point_size, color=color, alpha=point_alpha,
               edgecolors='none', zorder=2)

    q5, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])

    # Whiskers
    ax.vlines(x_pos, q5, q95, color=Colors.INK, lw=1.3, zorder=3)
    ax.hlines([q5, q95], x_pos - 0.10, x_pos + 0.10,
              color=Colors.INK, lw=1.1, zorder=3)

    # Box
    rect = Rectangle((x_pos - box_w / 2, q25), box_w, q75 - q25,
                     facecolor='white', edgecolor=Colors.INK,
                     lw=1.3, zorder=3.2, alpha=0.85)
    ax.add_patch(rect)

    # Median
    ax.hlines(q50, x_pos - box_w / 2, x_pos + box_w / 2,
              color=Colors.INK, lw=1.8, zorder=3.3)


def _gradient_fill_curve_horizontal(ax, x_curve, y_grid, *, baseline,
                                     color, alpha_top=0.30):
    """Horizontal-fade gradient between a curve (x as function of y) and a
    vertical baseline. Vertical analogue of plotter._gradient_fill_under_curve_oriented.
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    import matplotlib.colors as mcolors

    x_curve = np.asarray(x_curve)
    y_grid = np.asarray(y_grid)

    verts = np.vstack([
        np.column_stack([x_curve, y_grid]),
        np.column_stack([np.full_like(y_grid, baseline)[::-1], y_grid[::-1]]),
    ])
    clip_path = Path(verts)
    clip_patch = PathPatch(clip_path, facecolor='none', edgecolor='none',
                            transform=ax.transData)
    ax.add_patch(clip_patch)

    x0 = min(baseline, np.nanmin(x_curve))
    x1 = max(baseline, np.nanmax(x_curve))

    n = 256
    fade = np.linspace(0.0, 1.0, n).reshape(1, n)  # alpha varies with x
    # If curve is to the left of baseline, alpha should be max at the curve.
    if baseline > np.nanmedian(x_curve):
        fade = fade[:, ::-1]

    r, g, b = mcolors.to_rgb(color)
    rgba = np.dstack([np.full((1, n), r),
                      np.full((1, n), g),
                      np.full((1, n), b),
                      fade * alpha_top])

    ax.imshow(
        rgba,
        extent=[x0, x1, y_grid.min(), y_grid.max()],
        origin='lower',
        aspect='auto',
        interpolation='bicubic',
        zorder=0,
        clip_path=clip_patch,
        clip_on=True,
    )


def _draw_vertical_raincloud(ax, vals, color, *, x_pos=0,
                              ridge_width=0.35, point_offset=0.18,
                              jitter_sd=0.045, point_alpha=0.30,
                              point_size=22, box_w=0.14):
    """Vertical raincloud column at x_pos: KDE rises to the left, box and
    jittered points sit to the right of the baseline.
    """
    from matplotlib.patches import Rectangle
    from plotter import _kde_gaussian_1d

    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    vmin, vmax = float(vals.min()), float(vals.max())
    span = vmax - vmin + 1e-9
    pad = 0.04 * span
    grid = np.linspace(vmin - pad, vmax + pad, 600)

    dens = _kde_gaussian_1d(vals, grid, bw=None)
    if dens.max() > 0:
        dens = dens / dens.max()
    mask = dens >= 0.01
    if mask.any():
        grid_eff, dens_eff = grid[mask], dens[mask]
    else:
        grid_eff, dens_eff = grid, dens

    # KDE rises to the LEFT of x_pos
    thickness = ridge_width * dens_eff
    x_curve = x_pos - thickness

    _gradient_fill_curve_horizontal(
        ax, x_curve, grid_eff, baseline=x_pos, color=color, alpha_top=0.30,
    )
    ax.plot(x_curve, grid_eff, color=color, alpha=0.85, lw=1.6, zorder=1.1)

    # Jittered points to the right of the baseline
    rng = np.random.default_rng(42)
    x_jit = x_pos + point_offset + rng.normal(0, jitter_sd, size=vals.size)
    ax.scatter(x_jit, vals, s=point_size, color=color, alpha=point_alpha,
               edgecolors='none', zorder=2)

    # Box + whiskers slightly right of the baseline
    box_x = x_pos + 0.06
    q5, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])

    ax.vlines(box_x, q5, q95, color=Colors.INK, lw=1.3, zorder=3)
    ax.hlines([q5, q95], box_x - 0.04, box_x + 0.04,
              color=Colors.INK, lw=1.1, zorder=3)

    rect = Rectangle((box_x - box_w / 2, q25), box_w, q75 - q25,
                     facecolor='white', edgecolor=Colors.INK,
                     lw=1.3, zorder=3.2, alpha=0.85)
    ax.add_patch(rect)

    ax.hlines(q50, box_x - box_w / 2, box_x + box_w / 2,
              color=Colors.INK, lw=1.8, zorder=3.3)


def plot_param_estimates_vertical():
    """Vertical raincloud (KDE on the left, box + jittered points on the
    right) for log(omega) and log(kappa) per sample. Vertical analogue of
    plot_param_estimates.
    """
    plot_order = ('omega', 'kappa')
    param_labels = {
        'omega': r'$\log\,\omega$',
        'kappa': r'$\log\,\kappa$',
    }
    param_colors = {
        'omega': Colors.RUBY1,
        'kappa': Colors.CERULEAN2,
    }

    for sname, s in samples.items():
        params = s['params'].copy()
        n_subj = params['subj'].nunique()

        fig, ax = plt.subplots(figsize=(4.6, 5.2))

        all_log = []
        for i, p in enumerate(plot_order):
            v = params[p].dropna().values
            log_v = np.log(v[v > 0])
            _draw_vertical_raincloud(ax, log_v, param_colors[p], x_pos=i)
            all_log.append(log_v)

        flat = np.concatenate(all_log)
        span = flat.max() - flat.min()
        ax.set_ylim(flat.min() - 0.05 * span, flat.max() + 0.10 * span)
        ax.set_xlim(-0.6, len(plot_order) - 0.4)

        ax.set_xticks(range(len(plot_order)))
        ax.set_xticklabels([param_labels[p] for p in plot_order],
                           fontsize=14, color=Colors.INK)

        ax.axhline(0, color=Colors.INK, lw=0.7, ls=':', alpha=0.4, zorder=1)

        ax.set_ylabel('log estimate', fontsize=11, color=Colors.INK)
        ax.set_facecolor('#FCFCFD')
        ax.grid(True, axis='y', color='#E5E7EB', linewidth=0.8, alpha=0.45)
        ax.grid(False, axis='x')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#D1D5DB')
        ax.spines['bottom'].set_color('#E5E7EB')
        ax.tick_params(axis='y', colors=Colors.INK, labelsize=10)
        ax.tick_params(axis='x', colors=Colors.INK, labelsize=12, length=0)

        label = 'Exploratory' if sname == 'exploratory' else 'Confirmatory'
        ax.set_title(f'{label} (N = {n_subj})', fontsize=12,
                     color=Colors.DARK_GREY, pad=10)

        plt.tight_layout()
        fig.savefig(OUT_DIR / f'param_estimates_vertical_{sname}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUT_DIR / f'param_estimates_vertical_{sname}.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Saved param_estimates_vertical_{sname}')


def plot_param_estimates():
    """Wide horizontal raincloud (KDE + box + points) for log(omega) and
    log(kappa) — matches example_parameter estimate.png style. Log space
    puts both params on a shared scale and makes the densities symmetric.
    """
    # Bottom-to-top order: function indexes y=0 at bottom row.
    plot_order = ('kappa', 'omega')
    param_labels = {
        'omega': r'$\log\,\omega$',
        'kappa': r'$\log\,\kappa$',
    }
    param_colors = {
        'omega': Colors.RUBY1,
        'kappa': Colors.CERULEAN2,
    }

    for sname, s in samples.items():
        params = s['params'].copy()
        n_subj = params['subj'].nunique()

        # Log-transform (drop any non-positive just in case)
        log_df = params.copy()
        for p in plot_order:
            v = log_df[p].values
            log_df[p] = np.where(v > 0, np.log(v), np.nan)

        fig, ax = plot_parameter_half_violin_glow(
            log_df,
            params=plot_order,
            mean_suffix='',
            labels=param_labels,
            colors=param_colors,
            xlabel='log estimate',
            figsize=(4.4, 3.6),
            side='upper',
            point_alpha=0.30,
        )

        # Reference line at log(1) = 0
        ax.axvline(0, color=Colors.INK, lw=0.7, ls=':', alpha=0.4, zorder=1)

        label = 'Exploratory' if sname == 'exploratory' else 'Confirmatory'
        fig.suptitle(f'{label} (N = {n_subj})', fontsize=12,
                     color=Colors.DARK_GREY, y=1.00)

        plt.tight_layout()
        fig.savefig(OUT_DIR / f'param_estimates_{sname}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUT_DIR / f'param_estimates_{sname}.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Saved param_estimates_{sname}')


# ============================================================
# Delta WAIC: model comparison bars
# ============================================================

SHOW_MODELS_DWAIC = ['M1', 'M2', 'M3b', 'M4']
MODEL_LABELS_SHORT = {
    'M1': 'Effort-only', 'M2': 'Threat-only',
    'M3b': 'Single-param', 'M4': 'Joint',
}


def _draw_dwaic_panel(ax, df, *, show_title=False, label=''):
    """Vertical ΔWAIC forest on an existing ax."""
    df = df.set_index('Model') if 'Model' in df.columns else df
    models = [m for m in SHOW_MODELS_DWAIC if m in df.index]
    positions = np.arange(len(models))

    for i, m in enumerate(models):
        is_winner = (m == 'M4')
        c = Colors.RUBY1 if is_winner else Colors.INK
        dw = df.loc[m, 'dWAIC']
        se = df.loc[m, 'SE_WAIC']

        ax.scatter(i, dw, s=80, c=c, edgecolors='white',
                   linewidths=1.2, zorder=3, alpha=0.95)
        ax.errorbar(i, dw, yerr=se, fmt='none', ecolor=c,
                    elinewidth=1.5, capsize=4, capthick=1.5,
                    alpha=0.5, zorder=2)

    ax.axhline(0, color=Colors.RUBY1, lw=1.5, ls='--', alpha=0.7, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABELS_SHORT.get(m, m) for m in models],
                       fontsize=8, rotation=30, ha='right')
    if show_title and label:
        ax.set_title(label, fontsize=10, color=Colors.DARK_GREY, pad=6)
    ax.set_ylabel('\u0394WAIC (vs Joint)', fontsize=9, color=Colors.INK)
    ax.set_facecolor('#FCFCFD')
    ax.grid(True, which='major', axis='y', color='#E5E7EB',
            linewidth=0.8, alpha=0.25, zorder=0)
    ax.grid(False, axis='x')
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.tick_params(colors=Colors.INK, labelsize=8)


def plot_dwaic():
    """Separate vertical ΔWAIC forest plots for exploratory and confirmatory."""
    for sname in samples.keys():
        df = pd.read_csv(f'results/stats/joint_optimal/{sname}/mcmc_model_comparison.csv')

        fig, ax = plt.subplots(figsize=(3.4, 4.0))
        label = 'Exploratory' if sname == 'exploratory' else 'Confirmatory'
        n_subj = samples[sname]['params']['subj'].nunique()
        _draw_dwaic_panel(ax, df, show_title=True,
                          label=f'{label} (N = {n_subj})')

        plt.tight_layout()
        fig.savefig(OUT_DIR / f'dwaic_{sname}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUT_DIR / f'dwaic_{sname}.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Saved dwaic_{sname}')


# ============================================================
# Combined H3 row figure: choice PPC | vigor PPC | ΔWAIC | params
# ============================================================

def plot_h3_row():
    """Single-row 4-panel figure per sample, matching example_layout.png:
    (1) group choice PPC, (2) group vigor PPC, (3) ΔWAIC, (4) param raincloud.

    All four panels are forced to the same display-box aspect via
    set_box_aspect, so the equal-aspect vigor scatter doesn't get squished.
    """
    plot_order = ('omega', 'kappa')
    param_labels = {'omega': r'$\log\,\omega$', 'kappa': r'$\log\,\kappa$'}
    param_colors = {'omega': Colors.RUBY1, 'kappa': Colors.CERULEAN2}

    PANEL_BOX_ASPECT = 0.95  # height/width ratio for every panel

    for sname, s in samples.items():
        n_subj = s['params']['subj'].nunique()
        label = 'Exploratory' if sname == 'exploratory' else 'Confirmatory'

        fig = plt.figure(figsize=(15.0, 3.8))
        gs = GridSpec(
            1, 4, figure=fig,
            width_ratios=[1.0, 1.0, 1.0, 1.0],
            wspace=0.45,
        )

        # 1) Group choice PPC
        ax1 = fig.add_subplot(gs[0, 0])
        ct = predict_choice(s['choice'], s['params'], s['pop'])
        _draw_group_choice_panel(ax1, ct, f'{label} (N = {n_subj})',
                                  show_legend=True)

        # 2) Group vigor PPC
        ax2 = fig.add_subplot(gs[0, 1])
        vd = predict_vigor(s['vigor'], s['params'], s['pop'])
        _draw_group_vigor_panel(ax2, vd, '', show_legend=True)
        # Drop the equal-aspect constraint — set_box_aspect handles it.
        ax2.set_aspect('auto')

        # 3) ΔWAIC
        ax3 = fig.add_subplot(gs[0, 2])
        dw_df = pd.read_csv(
            f'results/stats/joint_optimal/{sname}/mcmc_model_comparison.csv')
        _draw_dwaic_panel(ax3, dw_df)

        # 4) Parameter raincloud (log space)
        ax4 = fig.add_subplot(gs[0, 3])
        log_df = s['params'].copy()
        for p in plot_order:
            v = log_df[p].values
            log_df[p] = np.where(v > 0, np.log(v), np.nan)
        plot_parameter_half_violin_glow(
            log_df,
            params=('kappa', 'omega'),  # bottom-to-top → omega on top
            mean_suffix='',
            labels=param_labels,
            colors=param_colors,
            xlabel='log estimate',
            side='upper',
            point_alpha=0.30,
            ax=ax4,
        )
        ax4.axvline(0, color=Colors.INK, lw=0.7, ls=':', alpha=0.4, zorder=1)

        # Force every panel to the same display-box aspect.
        for a in (ax1, ax2, ax3, ax4):
            a.set_box_aspect(PANEL_BOX_ASPECT)

        fig.savefig(OUT_DIR / f'h3_row_{sname}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUT_DIR / f'h3_row_{sname}.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Saved h3_row_{sname}')


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    print('Generating PPC figures...')
    plot_ppc_group_choice()
    plot_ppc_group_choice_individual()
    plot_ppc_group_vigor()
    plot_ppc_group_vigor_individual()
    plot_ppc_subject_vigor()
    plot_ppc_subject_vigor_individual()
    plot_ppc_subject()
    plot_ppc_subject_paper()
    plot_param_estimates()
    plot_param_estimates_vertical()
    plot_dwaic()
    plot_h3_row()
    print(f'Done. Saved to {OUT_DIR}/')
