"""Render Figure 5: H5 metacognitive monitoring (4 panels + forest)."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/notebooks/analysis')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, zscore
from load_data import load_both

exp, conf = load_both()
samples = {'Exploratory': exp['master'], 'Confirmatory': conf['master']}
COLORS = {'Exploratory': '#1f1f1f', 'Confirmatory': '#7a7a7a'}

# Z-score affect indices within sample
for name, m in samples.items():
    for col, zcol in [('mean_confidence', 'confidence_z'),
                      ('mean_anxiety', 'anxiety_z'),
                      ('anx_calibration', 'calibration_z'),
                      ('anx_slope', 'anx_slope_z')]:
        v = m[col]
        m[zcol] = (v - v.mean()) / v.std()

fig = plt.figure(figsize=(12, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.40,
                      left=0.07, right=0.97, top=0.93, bottom=0.08)

# ---------- Panel a: H5a ΔELPD bars ----------
# Recompute simply via OLS partial R² as a stand-in summary; or use stored values.
# Use stored numbers from §2.5: dELPD = 4.8/3.5/3.1 (ostensibly pooled).
# For panel: report calibration partial standardized β on each outcome by sample.
ax = fig.add_subplot(gs[0, 0])
outcomes = [('pct_opt', 'Optimality'), ('escape_rate', 'Escape'), ('earnings', 'Earnings')]
xpos = np.arange(len(outcomes))
width = 0.36
for i, (sample, m) in enumerate(samples.items()):
    bs, los, his = [], [], []
    for col, _ in outcomes:
        df = m[['omega_z', 'kappa_z', 'calibration_z', col]].dropna()
        # standardize outcome
        y = (df[col] - df[col].mean()) / df[col].std()
        X = df[['omega_z', 'kappa_z', 'calibration_z']].values
        X = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = (resid ** 2).sum() / (len(y) - X.shape[1])
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))[3]
        b = beta[3]
        bs.append(b); los.append(b - 1.96 * se); his.append(b + 1.96 * se)
    bs = np.array(bs); los = np.array(los); his = np.array(his)
    err = np.vstack([bs - los, his - bs])
    ax.bar(xpos + (i - 0.5) * width, bs, width, yerr=err, capsize=3,
           color=COLORS[sample], label=sample, alpha=0.85, edgecolor='none')
ax.axhline(0, color='k', lw=0.5)
ax.set_xticks(xpos)
ax.set_xticklabels([o[1] for o in outcomes], fontsize=9)
ax.set_ylabel('Anxiety calibration β\n(partialling ω, κ)', fontsize=10)
ax.set_title('a  H5a: calibration → outcomes\nbeyond ω + κ', fontsize=10, loc='left')
ax.legend(fontsize=8, frameon=False, loc='upper right')

# ---------- Panel b: H5b anxiety reactivity → choice shift ----------
ax = fig.add_subplot(gs[0, 1])
for sample, m in samples.items():
    df = m[['anx_slope_z', 'choice_shift']].dropna()
    ax.scatter(df['anx_slope_z'], df['choice_shift'], s=14, alpha=0.5,
               color=COLORS[sample], label=sample, edgecolors='none')
    # OLS line
    sl, ic, r, p, _ = linregress(df['anx_slope_z'], df['choice_shift'])
    xs = np.linspace(df['anx_slope_z'].min(), df['anx_slope_z'].max(), 50)
    ax.plot(xs, ic + sl * xs, color=COLORS[sample], lw=1.6)
ax.set_xlabel('Anxiety reactivity slope (z)', fontsize=10)
ax.set_ylabel('Choice shift across threat', fontsize=10)
ax.set_title('b  H5b: reactivity → choice shift', fontsize=10, loc='left')
ax.legend(fontsize=8, frameon=False, loc='upper left')

# ---------- Panel c: H5c ω → confidence vs ω → anxiety forest ----------
ax = fig.add_subplot(gs[0, 2])
rows = []
for sample, m in samples.items():
    for ycol, ylab in [('mean_confidence', 'ω → confidence'),
                       ('mean_anxiety', 'ω → anxiety')]:
        df = m[['omega_z', ycol]].dropna()
        y = (df[ycol] - df[ycol].mean()) / df[ycol].std()
        x = df['omega_z'].values
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = (resid ** 2).sum() / (len(y) - 2)
        se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
        rows.append((sample, ylab, beta[1], beta[1] - 1.96 * se, beta[1] + 1.96 * se))
labels = ['ω → confidence', 'ω → anxiety']
ypos = np.arange(len(labels))
for i, sample in enumerate(samples):
    bs = [r[2] for r in rows if r[0] == sample]
    los = [r[3] for r in rows if r[0] == sample]
    his = [r[4] for r in rows if r[0] == sample]
    offset = (i - 0.5) * 0.25
    ax.errorbar(bs, ypos + offset, xerr=[np.array(bs) - np.array(los), np.array(his) - np.array(bs)],
                fmt='o', color=COLORS[sample], label=sample, capsize=3, ms=6)
# ROPE shading
ax.axvspan(-0.10, 0.10, color='gold', alpha=0.20, label='ROPE [−.10, +.10]')
ax.axvline(0, color='k', lw=0.5)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Standardized β', fontsize=10)
ax.set_title('c  H5c: confidence ≠ anxiety', fontsize=10, loc='left')
ax.set_ylim(-0.6, 1.6)
ax.legend(fontsize=7, frameon=False, loc='lower right')

# ---------- Panel d: H5d confidence → error type ----------
ax = fig.add_subplot(gs[1, 0])
xpos = np.arange(2)
width = 0.36
errlabels = ['Overcautious', 'Reckless']
for i, (sample, m) in enumerate(samples.items()):
    bs, los, his = [], [], []
    for col in ['n_oc', 'n_rk']:
        df = m[['confidence_z', col]].dropna()
        y = df[col].values.astype(float)
        x = df['confidence_z'].values
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = (resid ** 2).sum() / (len(y) - 2)
        se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
        bs.append(beta[1]); los.append(beta[1] - 1.96 * se); his.append(beta[1] + 1.96 * se)
    bs = np.array(bs); los = np.array(los); his = np.array(his)
    err = np.vstack([bs - los, his - bs])
    ax.bar(xpos + (i - 0.5) * width, bs, width, yerr=err, capsize=3,
           color=COLORS[sample], label=sample, alpha=0.85, edgecolor='none')
ax.axhline(0, color='k', lw=0.5)
ax.set_xticks(xpos)
ax.set_xticklabels(errlabels, fontsize=10)
ax.set_ylabel('Error count β / +1 SD confidence', fontsize=10)
ax.set_title('d  H5d: confidence shifts\nerror type', fontsize=10, loc='left')
ax.legend(fontsize=8, frameon=False)

# ---------- Panel e: forest plot of all H5 effects ----------
ax = fig.add_subplot(gs[1, 1:])

# Build effects table
effects = []  # (label, sample, beta, lo, hi)
def add_std(label, sample, df, xcol, ycol):
    d = df[[xcol, ycol]].dropna()
    y = (d[ycol] - d[ycol].mean()) / d[ycol].std()
    x = (d[xcol] - d[xcol].mean()) / d[xcol].std()
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = (resid ** 2).sum() / (len(y) - 2)
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
    effects.append((label, sample, beta[1], beta[1] - 1.96 * se, beta[1] + 1.96 * se))

for sample, m in samples.items():
    add_std('H5a: calib → optimality', sample, m, 'calibration_z', 'pct_opt')
    add_std('H5a: calib → escape',     sample, m, 'calibration_z', 'escape_rate')
    add_std('H5a: calib → earnings',   sample, m, 'calibration_z', 'earnings')
    add_std('H5b: reactivity → shift', sample, m, 'anx_slope_z', 'choice_shift')
    add_std('H5c: ω → confidence',     sample, m, 'omega_z', 'mean_confidence')
    add_std('H5c: ω → anxiety (null)', sample, m, 'omega_z', 'mean_anxiety')
    add_std('H5d: conf → overcautious',sample, m, 'confidence_z', 'n_oc')
    add_std('H5d: conf → reckless',    sample, m, 'confidence_z', 'n_rk')

labels = list(dict.fromkeys([e[0] for e in effects]))
ypos = np.arange(len(labels))[::-1]
for i, sample in enumerate(samples):
    rows_s = [e for e in effects if e[1] == sample]
    bs = np.array([r[2] for r in rows_s])
    los = np.array([r[3] for r in rows_s])
    his = np.array([r[4] for r in rows_s])
    offset = (i - 0.5) * 0.30
    ax.errorbar(bs, ypos + offset, xerr=[bs - los, his - bs],
                fmt='o', color=COLORS[sample], label=sample, capsize=2.5, ms=5)
ax.axvline(0, color='k', lw=0.5)
ax.axvspan(-0.10, 0.10, color='gold', alpha=0.15)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Standardized β (95% CI)', fontsize=10)
ax.set_title('e  All H5 effects, both samples', fontsize=10, loc='left')
ax.legend(fontsize=8, frameon=False, loc='lower right')

import os
os.makedirs('/workspace/data/figures', exist_ok=True)
os.makedirs('/workspace/results/figs/affect_analysis', exist_ok=True)
out1 = '/workspace/data/figures/fig5_metacognition.png'
out2 = '/workspace/results/figs/affect_analysis/fig5_metacognition.png'
fig.savefig(out1, dpi=200, bbox_inches='tight')
fig.savefig(out2, dpi=200, bbox_inches='tight')
print('Saved:', out1)
print('Saved:', out2)
