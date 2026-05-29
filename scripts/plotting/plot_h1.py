"""
H1 Results Figures — Publication quality using plotter.py style.

Generates:
  h1a_choice_surface.png    — P(heavy) by threat × distance (heatmap, both samples)
  h1b_affect.png            — Anxiety and confidence by threat (both samples)
  h1c_vigor_by_threat.png   — Normalized press rate, grouped by cookie × threat (both samples)

Usage:
  python scripts/plotting/plot_h1.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'analysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import zscore, ttest_rel
from pathlib import Path

from plotter import Colors, set_plot_style, style_axis

# Apply global style
set_plot_style()
plt.rcParams.update({'savefig.dpi': 300, 'figure.dpi': 300})

OUT_DIR = Path('results/figs/h1')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load data
# ============================================================

def load_sample(name):
    """Load a sample's data for H1 figures."""
    base = Path(f'data/{name}_350/processed')
    s5 = sorted(base.glob('stage5_*'))[-1]

    beh = pd.read_csv(s5 / 'behavior_rich.csv', low_memory=False)
    feel = pd.read_csv(s5 / 'feelings.csv')
    tv_path = s5 / 'trial_vigor.csv'
    tv = pd.read_csv(tv_path) if tv_path.exists() else None

    # Exclusions
    exclude = [154, 197, 208] if name == 'exploratory' else []
    beh = beh[~beh['subj'].isin(exclude)]
    feel = feel[~feel['subj'].isin(exclude)]
    if tv is not None:
        tv = tv[~tv['subj'].isin(exclude)]

    return {
        'beh': beh, 'feelings': feel, 'vigor': tv,
        'label': f'{"Exploratory" if name == "exploratory" else "Confirmatory"} (N={beh["subj"].nunique()})',
        'name': name,
    }

samples = {}
for name in ['exploratory', 'confirmatory']:
    try:
        samples[name] = load_sample(name)
    except Exception as e:
        print(f'Skipping {name}: {e}')

labels = [s["label"] for s in samples.values()]
print(f'Loaded: {", ".join(labels)}')


# ============================================================
# H1a: Choice surface — P(heavy) by threat × distance
# ============================================================

def plot_h1a():
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.0 * n_samples, 3.5), sharey=True)
    if n_samples == 1:
        axes = [axes]

    threat_colors = {0.1: Colors.CERULEAN2, 0.5: Colors.SLATE, 0.9: Colors.RUBY1}
    threat_labels = {0.1: 'T = 0.1', 0.5: 'T = 0.5', 0.9: 'T = 0.9'}
    bar_width = 0.25
    distances = [1, 2, 3]
    x_pos = np.arange(len(distances))

    for ax, (name, s) in zip(axes, samples.items()):
        beh = s['beh']
        cdf = beh[beh['type'] == 1].copy()
        cdf['T_round'] = cdf['threat'].round(1)

        for t_idx, T in enumerate([0.1, 0.5, 0.9]):
            means = []
            sems = []
            for D in distances:
                sub = cdf[(cdf['T_round'] == T) & (cdf['distance_H'] == D)]
                subj_means = sub.groupby('subj')['choice'].mean()
                means.append(subj_means.mean())
                sems.append(subj_means.sem())

            offset = (t_idx - 1) * bar_width
            ax.bar(x_pos + offset, means, bar_width * 0.88,
                   yerr=[se * 1.96 for se in sems], capsize=3,
                   color=threat_colors[T], edgecolor='white', linewidth=0.5,
                   alpha=0.85, label=threat_labels[T], zorder=3)

        style_axis(ax, xlabel='Distance',
                   ylabel='P(choose heavy)' if ax == axes[0] else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['D = 1', 'D = 2', 'D = 3'], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(s['label'], fontsize=11, color=Colors.DARK_GREY, pad=10)

        if ax == axes[0]:
            leg = ax.legend(fontsize=8, frameon=True, labelcolor=Colors.INK)
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#E5E7EB')
            leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h1a_choice_surface.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h1a_choice_surface.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h1a_choice_surface')


# ============================================================
# H1b: Affect by threat — anxiety up, confidence down
# ============================================================

def plot_h1b():
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.0 * n_samples, 3.5), sharey=True)
    if n_samples == 1:
        axes = [axes]

    affect_styles = {
        'anxiety':    {'color': Colors.RUBY1,     'marker': 'o'},
        'confidence': {'color': Colors.CERULEAN2, 'marker': 's'},
    }
    dist_styles = {
        1: {'ls': ':',  'lw': 2.0, 'offset': -0.03},
        2: {'ls': '--', 'lw': 2.0, 'offset':  0.0},
        3: {'ls': '-',  'lw': 2.0, 'offset': +0.03},
    }
    threats = [0.1, 0.5, 0.9]

    for ax, (name, s) in zip(axes, samples.items()):
        feel = s['feelings'].copy()
        feel['D'] = feel['distance'] + 1

        for q_type, asty in affect_styles.items():
            sub = feel[feel['questionLabel'] == q_type]
            for D, dsty in dist_styles.items():
                sub_d = sub[sub['D'] == D]
                means, sems = [], []
                for T in threats:
                    subj_means = sub_d[sub_d['threat'].round(1) == T].groupby('subj')['response'].mean()
                    means.append(subj_means.mean())
                    sems.append(subj_means.sem())

                x = [t + dsty['offset'] for t in threats]
                ax.errorbar(x, means, yerr=[se * 1.96 for se in sems],
                           color=asty['color'], marker=asty['marker'], ms=6,
                           lw=dsty['lw'], ls=dsty['ls'],
                           capsize=3, zorder=3)

        style_axis(ax, xlabel='Threat probability',
                  ylabel='Rating (1-10)' if ax == axes[0] else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_xticks(threats)
        ax.set_title(s['label'], fontsize=11, color=Colors.DARK_GREY, pad=10)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=Colors.RUBY1, marker='o', ms=6, lw=2, label='Anxiety'),
        Line2D([0], [0], color=Colors.CERULEAN2, marker='s', ms=6, lw=2, label='Confidence'),
        Line2D([0], [0], color=Colors.INK, lw=1.5, ls=':',  label='D = 1'),
        Line2D([0], [0], color=Colors.INK, lw=1.5, ls='--', label='D = 2'),
        Line2D([0], [0], color=Colors.INK, lw=1.5, ls='-',  label='D = 3'),
    ]
    leg = fig.legend(handles=handles, fontsize=8, frameon=True, labelcolor=Colors.INK,
                    loc='center right', bbox_to_anchor=(1.12, 0.5))
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#E5E7EB')
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h1b_affect.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h1b_affect.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h1b_affect')


# ============================================================
# H1c: Vigor by threat, within cookie type (grouped bars)
# ============================================================

def _sig_stars(p):
    """Convert a p-value to APA-style significance stars."""
    if p < 1e-3:  return '***'
    if p < 1e-2:  return '**'
    if p < 5e-2:  return '*'
    return 'n.s.'


def plot_h1c():
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.5 * n_samples, 3.8), sharey=True)
    if n_samples == 1:
        axes = [axes]

    threat_colors = {0.1: Colors.CERULEAN2, 0.5: Colors.SLATE, 0.9: Colors.RUBY1}
    threat_labels = {0.1: 'T = 0.1', 0.5: 'T = 0.5', 0.9: 'T = 0.9'}
    threat_levels = [0.1, 0.5, 0.9]
    cookies = [(1, 'Heavy', 0.9), (0, 'Light', 0.4)]
    bar_width = 0.25
    x_pos = np.arange(len(cookies))

    for ax, (name, s) in zip(axes, samples.items()):
        tv = s['vigor']
        if tv is None:
            ax.text(0.5, 0.5, 'No vigor data', transform=ax.transAxes, ha='center')
            continue

        # Restrict to type==1 (free-choice trials). The prereg's H1c specifies
        # "within each chosen effort level" — "chosen" implies type==1 (where the
        # subject freely selected). Probe trials (type 5, 6) are forced and not
        # part of the prereg-named test. The within-subject selection-into-heavy
        # at high threat is part of the question the prereg asks, not an artifact.
        tv_valid = tv[(tv['type'] == 1) & tv['norm_rate'].notna()].copy()
        tv_valid['T_round'] = tv_valid['T_round'].round(1)

        # Within (subject × cookie) demeaning: removes baseline differences
        # between subjects AND between cookies, isolating the within-subject
        # threat effect within each cookie type.
        sc_mean = tv_valid.groupby(['subj', 'cookie'])['norm_rate'].mean().rename('sc_mean')
        tv_valid = tv_valid.merge(sc_mean, on=['subj', 'cookie'])
        tv_valid['vigor_delta'] = tv_valid['norm_rate'] - tv_valid['sc_mean']

        # Track per-cookie bar tops so significance brackets clear the highest bar.
        # tops_by_cookie[cookie_x_index] = max(mean + 1.96*sem) across the three threats
        tops_by_cookie = {ci: -np.inf for ci in range(len(cookies))}

        for t_idx, T in enumerate(threat_levels):
            means, sems = [], []
            for ci, (cookie_val, _, _) in enumerate(cookies):
                sub = tv_valid[(tv_valid['T_round'] == T) & (tv_valid['cookie'] == cookie_val)]
                subj_means = sub.groupby('subj')['vigor_delta'].mean()
                m = subj_means.mean()
                se = subj_means.sem()
                means.append(m); sems.append(se)
                tops_by_cookie[ci] = max(tops_by_cookie[ci], m + 1.96 * se)

            offset = (t_idx - 1) * bar_width
            ax.bar(x_pos + offset, means, bar_width * 0.88,
                   yerr=[se * 1.96 for se in sems], capsize=3,
                   color=threat_colors[T], edgecolor='white', linewidth=0.5,
                   alpha=0.85, label=threat_labels[T], zorder=3)

        ax.axhline(0, color=Colors.INK, lw=0.8, ls='--', alpha=0.4, zorder=1)
        style_axis(ax, xlabel='Cookie type',
                  ylabel='Within-subject Δ press rate' if ax == axes[0] else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{lbl}\n(req = {req})' for _, lbl, req in cookies], fontsize=9)
        ax.set_title(s['label'], fontsize=11, color=Colors.DARK_GREY, pad=10)

        # ── Significance brackets: paired-t per threat-pair, within cookie ──
        # Three brackets per cookie:
        #   short, low:  T=0.1 vs T=0.5   (adjacent, EXPLORATORY — Bonferroni×2 within cookie)
        #   short, low:  T=0.5 vs T=0.9   (adjacent, EXPLORATORY — Bonferroni×2 within cookie)
        #   long, high:  T=0.1 vs T=0.9   (PREREGISTERED, uncorrected — prereg line 221)
        bracket_color = Colors.INK
        # Common bracket baseline across both cookies in this panel for visual alignment.
        baseline = max(tops_by_cookie.values())
        y_span = baseline - min(0.0, *tops_by_cookie.values())
        pad = 0.08 * max(y_span, 0.02)         # gap unit above the tallest bar
        tick = 0.30 * pad                       # downward tick at bracket ends

        def _paired_p(sub_df, t_lo, t_hi):
            ps = (sub_df[sub_df['T_round'].isin([t_lo, t_hi])]
                  .groupby(['subj', 'T_round'])['norm_rate'].mean()
                  .unstack('T_round')
                  .dropna(subset=[t_lo, t_hi]))
            _, p = ttest_rel(ps[t_hi], ps[t_lo])
            return p

        def _draw_bracket(x_lo, x_hi, y_top, stars, fontsize, weight):
            ax.plot([x_lo, x_lo, x_hi, x_hi],
                    [y_top - tick, y_top, y_top, y_top - tick],
                    color=bracket_color, lw=0.9, zorder=5)
            ax.text((x_lo + x_hi) / 2, y_top + 0.2 * pad, stars,
                    ha='center', va='bottom', fontsize=fontsize,
                    color=bracket_color, zorder=5, fontweight=weight)

        for ci, (cookie_val, _, _) in enumerate(cookies):
            sub_c = tv_valid[tv_valid['cookie'] == cookie_val]

            # x-positions of T=0.1 / T=0.5 / T=0.9 bar centers within this cookie
            x_01 = x_pos[ci] - bar_width
            x_05 = x_pos[ci]
            x_09 = x_pos[ci] + bar_width

            # ── Adjacent comparisons (EXPLORATORY) — Bonferroni × 2 within cookie ──
            p_01_05 = _paired_p(sub_c, 0.1, 0.5) * 2  # Bonferroni for 2 adjacent tests
            p_05_09 = _paired_p(sub_c, 0.5, 0.9) * 2
            y_adj = baseline + pad
            _draw_bracket(x_01, x_05, y_adj, _sig_stars(p_01_05), fontsize=9, weight='normal')
            _draw_bracket(x_05, x_09, y_adj, _sig_stars(p_05_09), fontsize=9, weight='normal')

            # ── Preregistered comparison (UNCORRECTED): T=0.1 vs T=0.9 ──
            p_01_09 = _paired_p(sub_c, 0.1, 0.9)
            y_ext = baseline + 3.0 * pad   # high enough to clear adjacent stars
            _draw_bracket(x_01, x_09, y_ext, _sig_stars(p_01_09), fontsize=11, weight='bold')

        # Headroom for the highest bracket + stars
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, max(ymax, baseline + 4.2 * pad))

        if ax == axes[0]:
            leg = ax.legend(fontsize=8, frameon=True, labelcolor=Colors.INK,
                            loc='lower left')
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#E5E7EB')
            leg.get_frame().set_linewidth(0.8)

    # Caption note: bracket convention + correction policy
    fig.text(0.5, -0.04,
             'Top bracket (bold): preregistered paired t-test, T=0.9 vs T=0.1, within cookie (uncorrected).  '
             'Lower brackets: adjacent threat-pair paired t-tests (exploratory; Bonferroni ×2 within cookie).\n'
             '* p<.05    ** p<.01    *** p<.001',
             ha='center', va='top', fontsize=8, color=Colors.DARK_GREY)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h1c_vigor_by_threat.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h1c_vigor_by_threat.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h1c_vigor_by_threat')


# ============================================================
# Combined: H1 summary panel (3 rows)
# ============================================================

def plot_h1_combined():
    """Single figure with all H1 results, both samples side by side."""
    n_samples = len(samples)
    fig = plt.figure(figsize=(4.5 * n_samples, 10.5))
    gs = GridSpec(3, n_samples, hspace=0.4, wspace=0.35)

    threat_colors = {0.1: Colors.CERULEAN2, 0.5: Colors.PERSIMMON3, 0.9: Colors.RUBY1}
    bar_width = 0.25

    for col, (name, s) in enumerate(samples.items()):
        beh = s['beh']
        feel = s['feelings']
        tv = s['vigor']
        label = s['label']

        # ── Row 1: Choice bar plot ──
        ax = fig.add_subplot(gs[0, col])
        cdf = beh[beh['type'] == 1].copy()
        cdf['T_round'] = cdf['threat'].round(1)
        distances = [1, 2, 3]
        x_pos = np.arange(len(distances))

        for t_idx, T in enumerate([0.1, 0.5, 0.9]):
            means, sems = [], []
            for D in distances:
                sub = cdf[(cdf['T_round'] == T) & (cdf['distance_H'] == D)]
                sm = sub.groupby('subj')['choice'].mean()
                means.append(sm.mean()); sems.append(sm.sem())
            offset = (t_idx - 1) * bar_width
            ax.bar(x_pos + offset, means, bar_width * 0.88,
                   yerr=[se * 1.96 for se in sems], capsize=2,
                   color=threat_colors[T], edgecolor='white', linewidth=0.5,
                   alpha=0.85, label=f'T={T}', zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(['D=1', 'D=2', 'D=3'], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=11, color=Colors.DARK_GREY, pad=8)
        style_axis(ax, xlabel='Distance',
                   ylabel='P(choose heavy)' if col == 0 else None)
        ax.set_facecolor('#FCFCFD')
        if col == 0:
            leg = ax.legend(fontsize=7, frameon=True, labelcolor=Colors.INK)
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#E5E7EB')
            leg.get_frame().set_linewidth(0.8)

        # ── Row 2: Affect ──
        ax = fig.add_subplot(gs[1, col])
        for q_type, marker, offset in [('anxiety', 'o', -0.03), ('confidence', 's', +0.03)]:
            sub = feel[feel['questionLabel'] == q_type].copy()
            sub['T_round'] = sub['threat'].round(1)
            means = sub.groupby('T_round')['response'].agg(['mean', 'sem']).reset_index()
            color = Colors.RUBY1 if q_type == 'anxiety' else Colors.CERULEAN2
            ax.errorbar(means['T_round'] + offset, means['mean'],
                       yerr=means['sem'] * 1.96, color=color, marker=marker, ms=6,
                       lw=2.0, capsize=3, label=q_type.capitalize(), zorder=3)

        style_axis(ax, xlabel='Threat probability',
                  ylabel='Rating (1-10)' if col == 0 else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_xticks([0.1, 0.5, 0.9])
        leg = ax.legend(fontsize=8, frameon=True, labelcolor=Colors.INK)
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('#E5E7EB')
        leg.get_frame().set_linewidth(0.8)

        # ── Row 3: Vigor (grouped by cookie type, within-subject Δ) ──
        ax = fig.add_subplot(gs[2, col])
        if tv is not None:
            tv_valid = tv[(tv['type'] == 1) & tv['norm_rate'].notna()].copy()
            tv_valid['T_round'] = tv_valid['T_round'].round(1)

            sc_mean = tv_valid.groupby(['subj', 'cookie'])['norm_rate'].mean().rename('sc_mean')
            tv_valid = tv_valid.merge(sc_mean, on=['subj', 'cookie'])
            tv_valid['vigor_delta'] = tv_valid['norm_rate'] - tv_valid['sc_mean']

            cookies_v = [(1, 'Heavy', 0.9), (0, 'Light', 0.4)]
            xv = np.arange(len(cookies_v))
            for t_idx, T in enumerate([0.1, 0.5, 0.9]):
                means, sems = [], []
                for cookie_val, _, _ in cookies_v:
                    sub = tv_valid[(tv_valid['T_round'] == T) & (tv_valid['cookie'] == cookie_val)]
                    sm = sub.groupby('subj')['vigor_delta'].mean()
                    means.append(sm.mean()); sems.append(sm.sem())
                offset = (t_idx - 1) * bar_width
                ax.bar(xv + offset, means, bar_width * 0.88,
                       yerr=[se * 1.96 for se in sems], capsize=2,
                       color=threat_colors[T], edgecolor='white', linewidth=0.5,
                       alpha=0.85, zorder=3)
            ax.axhline(0, color=Colors.INK, lw=0.8, ls='--', alpha=0.4, zorder=1)
            ax.set_xticks(xv)
            ax.set_xticklabels([f'{lbl}\n(req={req})' for _, lbl, req in cookies_v], fontsize=9)

        style_axis(ax, xlabel='Cookie type',
                  ylabel='Within-subject Δ press rate' if col == 0 else None)
        ax.set_facecolor('#FCFCFD')

    plt.savefig(OUT_DIR / 'h1_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_DIR / 'h1_combined.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h1_combined')


# ============================================================
# Run all
# ============================================================

if __name__ == '__main__':
    print('Generating H1 figures...')
    plot_h1a()
    plot_h1b()
    plot_h1c()
    plot_h1_combined()
    print(f'Done. Saved to {OUT_DIR}/')
