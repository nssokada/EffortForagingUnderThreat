"""
H3 Results Figures — Model Comparison (Joint Model).

Forest-plot style: ΔWAIC only (M4 as reference at 0).
Models shown: M1 (effort-only), M2 (threat-only), M3b (single + scaling), M4 (joint).
M3 (unscaled single-param) dropped — M3b is the fairer test.

Generates:
  h3_forest.png      — ΔWAIC forest plot (both samples side by side)
  h3_fit_quality.png  — Choice accuracy vs Vigor r² (both samples)
  h3_combined.png     — Combined panel for the paper

Usage:
  python scripts/plotting/plot_h3.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from plotter import Colors, set_plot_style, style_axis

# Apply global style
set_plot_style()
plt.rcParams.update({'savefig.dpi': 300, 'figure.dpi': 300})

OUT_DIR = Path('results/figs/h3')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Config
# ============================================================

MODEL_COMP_DIR = Path('results/stats/joint_optimal')

# Models to show (drop M3, keep M3b as the fair single-param test)
SHOW_MODELS = ['M4', 'M3b', 'M2', 'M1']

MODEL_LABELS = {
    'M1':  'M1: Effort-only ($\\kappa$)',
    'M2':  'M2: Threat-only ($\\omega$)',
    'M3b': 'M3: Single-param ($\\theta$)',
    'M4':  'M4: Joint ($\\omega$ + $\\kappa$)',
}

MODEL_LABELS_SHORT = {
    'M1': 'Effort-only', 'M2': 'Threat-only',
    'M3b': 'Single-param', 'M4': 'Joint',
}

# ============================================================
# Load data
# ============================================================

def load_model_comparison():
    samples = {}
    for name in ['exploratory', 'confirmatory']:
        csv_path = MODEL_COMP_DIR / name / 'mcmc_model_comparison.csv'
        try:
            df = pd.read_csv(csv_path)
            samples[name] = {
                'df': df.set_index('Model'),
                'label': 'Exploratory' if name == 'exploratory' else 'Confirmatory',
            }
        except Exception as e:
            print(f'Skipping {name}: {e}')
    return samples


samples = load_model_comparison()
print(f'Loaded: {", ".join(s["label"] for s in samples.values())}')


# ============================================================
# H3 Forest plot — ΔWAIC only, both samples
# ============================================================

def _draw_delta_forest(ax, df, label):
    """Draw a single ΔWAIC forest panel."""
    models = [m for m in SHOW_MODELS if m in df.index]
    n = len(models)
    positions = np.arange(n)

    for i, m in enumerate(models):
        is_winner = (m == 'M4')
        c = Colors.RUBY1 if is_winner else Colors.INK
        dw = df.loc[m, 'dWAIC']
        se = df.loc[m, 'SE_WAIC']

        ax.scatter(dw, i, s=100, c=c, edgecolors='white',
                   linewidths=1.5, zorder=3, alpha=0.95)
        ax.errorbar(dw, i, xerr=se, fmt='none', ecolor=c,
                    elinewidth=1.8, capsize=5, capthick=1.8,
                    alpha=0.5, zorder=2)

        # Value label
        offset = max(50, se * 1.2)
        ax.text(dw + offset, i, f'{dw:,.0f}',
                va='center', ha='left', fontsize=8.5,
                color=c, fontweight='medium')

    # Reference line at 0
    ax.axvline(0, color=Colors.RUBY1, lw=2, ls='--', alpha=0.7, zorder=1)

    ax.set_yticks(positions)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=9.5)

    max_dw = max(df.loc[m, 'dWAIC'] for m in models)
    ax.set_xlim(-max_dw * 0.05, max_dw * 1.25)

    ax.set_xlabel('\u0394WAIC (vs M4)', fontsize=10, color=Colors.INK)
    ax.set_title(label, fontsize=11, color=Colors.DARK_GREY, pad=10)

    ax.set_facecolor('#FCFCFD')
    ax.grid(True, which='major', axis='x', color='#E5E7EB',
            linewidth=0.8, alpha=0.25, zorder=0)
    ax.grid(False, axis='y')
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.tick_params(colors=Colors.INK, labelsize=9)


def plot_forest():
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(5.5 * n_samples, 3.2),
                              sharey=True)
    if n_samples == 1:
        axes = [axes]

    for ax, (name, s) in zip(axes, samples.items()):
        _draw_delta_forest(ax, s['df'], s['label'])
        # Only show y-labels on leftmost panel
        if ax != axes[0]:
            ax.set_yticklabels([])

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h3_forest.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h3_forest.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h3_forest')


# ============================================================
# H3 Fit quality — choice accuracy vs vigor r²
# ============================================================

def plot_fit_quality():
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.5 * n_samples, 3.8), sharey=True)
    if n_samples == 1:
        axes = [axes]

    bar_width = 0.35
    x_pos = np.arange(len(SHOW_MODELS))

    for ax, (name, s) in zip(axes, samples.items()):
        df = s['df']

        choice_acc = [df.loc[m, 'choice_acc'] for m in SHOW_MODELS]
        vigor_r2 = [df.loc[m, 'vigor_r2'] for m in SHOW_MODELS]

        ax.bar(x_pos - bar_width / 2, choice_acc, bar_width * 0.88,
               color=Colors.CERULEAN2, edgecolor='white', linewidth=0.5,
               alpha=0.85, label='Choice acc.', zorder=3)
        ax.bar(x_pos + bar_width / 2, vigor_r2, bar_width * 0.88,
               color=Colors.RUBY1, edgecolor='white', linewidth=0.5,
               alpha=0.85, label='Vigor r\u00B2', zorder=3)

        # Highlight M4
        m4_idx = SHOW_MODELS.index('M4')
        ax.axvspan(m4_idx - 0.45, m4_idx + 0.45,
                   color=Colors.CERULEAN2, alpha=0.06, zorder=0)

        for i, (ca, vr) in enumerate(zip(choice_acc, vigor_r2)):
            ax.text(i - bar_width / 2, ca + 0.015, f'{ca:.2f}',
                    ha='center', va='bottom', fontsize=7, color=Colors.INK)
            ax.text(i + bar_width / 2, vr + 0.015, f'{vr:.2f}',
                    ha='center', va='bottom', fontsize=7, color=Colors.INK)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([MODEL_LABELS_SHORT.get(m, m) for m in SHOW_MODELS],
                           fontsize=9)
        ax.set_ylim(0, 1.0)

        style_axis(ax, xlabel='Model',
                   ylabel='Fit metric' if ax == axes[0] else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_title(s['label'], fontsize=11, color=Colors.DARK_GREY, pad=10)

        if ax == axes[0]:
            leg = ax.legend(fontsize=8, frameon=True, labelcolor=Colors.INK,
                           loc='upper right')
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#E5E7EB')
            leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h3_fit_quality.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h3_fit_quality.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h3_fit_quality')


# ============================================================
# H3 Combined: forest + fit quality
# ============================================================

def plot_h3_combined():
    n_samples = len(samples)
    fig = plt.figure(figsize=(5.5 * n_samples, 7.5))
    gs = GridSpec(2, n_samples, height_ratios=[1, 1],
                  hspace=0.45, wspace=0.45)

    # ── Row 0: ΔWAIC forest ──
    for i, (name, s) in enumerate(samples.items()):
        ax = fig.add_subplot(gs[0, i])
        _draw_delta_forest(ax, s['df'], s['label'])
        if i > 0:
            ax.set_yticklabels([])

    # ── Row 1: Fit quality ──
    bar_width = 0.35
    x_pos = np.arange(len(SHOW_MODELS))

    for i, (name, s) in enumerate(samples.items()):
        ax = fig.add_subplot(gs[1, i])
        df = s['df']

        choice_acc = [df.loc[m, 'choice_acc'] for m in SHOW_MODELS]
        vigor_r2 = [df.loc[m, 'vigor_r2'] for m in SHOW_MODELS]

        ax.bar(x_pos - bar_width / 2, choice_acc, bar_width * 0.88,
               color=Colors.CERULEAN2, edgecolor='white', linewidth=0.5,
               alpha=0.85, label='Choice acc.', zorder=3)
        ax.bar(x_pos + bar_width / 2, vigor_r2, bar_width * 0.88,
               color=Colors.RUBY1, edgecolor='white', linewidth=0.5,
               alpha=0.85, label='Vigor r\u00B2', zorder=3)

        m4_idx = SHOW_MODELS.index('M4')
        ax.axvspan(m4_idx - 0.45, m4_idx + 0.45,
                   color=Colors.CERULEAN2, alpha=0.06, zorder=0)

        for j, (ca, vr) in enumerate(zip(choice_acc, vigor_r2)):
            ax.text(j - bar_width / 2, ca + 0.015, f'{ca:.2f}',
                    ha='center', va='bottom', fontsize=7, color=Colors.INK)
            ax.text(j + bar_width / 2, vr + 0.015, f'{vr:.2f}',
                    ha='center', va='bottom', fontsize=7, color=Colors.INK)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([MODEL_LABELS_SHORT.get(m, m) for m in SHOW_MODELS],
                           fontsize=9)
        ax.set_ylim(0, 1.0)
        style_axis(ax, xlabel='Model',
                   ylabel='Fit metric' if i == 0 else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_title(s['label'], fontsize=11, color=Colors.DARK_GREY, pad=8)

        if i == 0:
            leg = ax.legend(fontsize=8, frameon=True, labelcolor=Colors.INK,
                           loc='upper right')
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#E5E7EB')
            leg.get_frame().set_linewidth(0.8)

    plt.savefig(OUT_DIR / 'h3_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_DIR / 'h3_combined.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h3_combined')


# ============================================================
# Run all
# ============================================================

# ============================================================
# Tight ΔWAIC panels — horizontal and vertical versions
# ============================================================

def plot_forest_horizontal_tight():
    """Horizontal forest: models on y-axis, ΔWAIC on x-axis. Both samples stacked."""
    n_samples = len(samples)
    models = [m for m in SHOW_MODELS]
    n = len(models)

    fig, axes = plt.subplots(n_samples, 1, figsize=(4.5, 1.6 * n_samples),
                              sharex=True)
    if n_samples == 1:
        axes = [axes]

    for ax, (name, s) in zip(axes, samples.items()):
        df = s['df']
        positions = np.arange(n)

        for i, m in enumerate(models):
            is_winner = (m == 'M4')
            c = Colors.RUBY1 if is_winner else Colors.INK
            dw = df.loc[m, 'dWAIC']
            se = df.loc[m, 'SE_WAIC']

            ax.scatter(dw, i, s=80, c=c, edgecolors='white',
                       linewidths=1.2, zorder=3, alpha=0.95)
            ax.errorbar(dw, i, xerr=se, fmt='none', ecolor=c,
                        elinewidth=1.5, capsize=4, capthick=1.5,
                        alpha=0.5, zorder=2)

        ax.axvline(0, color=Colors.RUBY1, lw=1.5, ls='--', alpha=0.7, zorder=1)

        ax.set_yticks(positions)
        ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=8.5)

        max_dw = max(df.loc[m, 'dWAIC'] for m in models)
        ax.set_xlim(-max_dw * 0.04, max_dw * 1.08)

        ax.set_title(s['label'], fontsize=10, color=Colors.DARK_GREY, pad=6)
        ax.set_facecolor('#FCFCFD')
        ax.grid(True, which='major', axis='x', color='#E5E7EB',
                linewidth=0.8, alpha=0.25, zorder=0)
        ax.grid(False, axis='y')
        ax.spines['left'].set_color('#D1D5DB')
        ax.spines['bottom'].set_color('#E5E7EB')
        ax.tick_params(colors=Colors.INK, labelsize=8)

    axes[-1].set_xlabel('\u0394WAIC (vs Joint)', fontsize=9, color=Colors.INK)

    plt.tight_layout(h_pad=1.0)
    fig.savefig(OUT_DIR / 'h3_dwaic_horizontal.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h3_dwaic_horizontal.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h3_dwaic_horizontal')


def plot_forest_vertical_tight():
    """Vertical forest: models on x-axis, ΔWAIC on y-axis. Both samples side by side."""
    n_samples = len(samples)
    models = [m for m in SHOW_MODELS]
    n = len(models)

    fig, axes = plt.subplots(1, n_samples, figsize=(3.0 * n_samples, 4.0),
                              sharey=True)
    if n_samples == 1:
        axes = [axes]

    for ax, (name, s) in zip(axes, samples.items()):
        df = s['df']
        positions = np.arange(n)

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

        ax.set_title(s['label'], fontsize=10, color=Colors.DARK_GREY, pad=6)
        ax.set_facecolor('#FCFCFD')
        ax.grid(True, which='major', axis='y', color='#E5E7EB',
                linewidth=0.8, alpha=0.25, zorder=0)
        ax.grid(False, axis='x')
        ax.spines['left'].set_color('#D1D5DB')
        ax.spines['bottom'].set_color('#E5E7EB')
        ax.tick_params(colors=Colors.INK, labelsize=8)

    axes[0].set_ylabel('\u0394WAIC (vs Joint)', fontsize=9, color=Colors.INK)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h3_dwaic_vertical.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h3_dwaic_vertical.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h3_dwaic_vertical')


# ============================================================
# Run all
# ============================================================

if __name__ == '__main__':
    print('Generating H3 figures...')
    plot_forest()
    plot_fit_quality()
    plot_h3_combined()
    plot_forest_horizontal_tight()
    plot_forest_vertical_tight()
    print(f'Done. Saved to {OUT_DIR}/')
