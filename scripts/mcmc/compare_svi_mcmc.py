"""
Post-hoc comparison of SVI and MCMC parameter estimates.

Creates a 2-panel figure (log(ce) and log(cd)) comparing SVI vs MCMC
posterior means with identity lines.

Usage:
    python scripts/mcmc/compare_svi_mcmc.py \
        --svi results/stats/oc_evc_final_params.csv \
        --mcmc results/stats/mcmc/oc_evc_mcmc_params.csv \
        --out results/figs/paper/fig_s_mcmc_validation.png
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='Compare SVI and MCMC parameter estimates')
    parser.add_argument('--svi', type=str,
                        default='results/stats/oc_evc_final_params.csv',
                        help='Path to SVI parameter CSV')
    parser.add_argument('--mcmc', type=str,
                        default='results/stats/mcmc/oc_evc_mcmc_params.csv',
                        help='Path to MCMC parameter CSV')
    parser.add_argument('--out', type=str,
                        default='results/figs/paper/fig_s_mcmc_validation.png',
                        help='Output figure path')
    args = parser.parse_args()

    # Load data
    svi_df = pd.read_csv(args.svi)
    mcmc_df = pd.read_csv(args.mcmc)

    # Merge on subject
    merged = mcmc_df.merge(svi_df, on='subj', suffixes=('_mcmc', '_svi'))
    print(f"Matched {len(merged)} subjects")

    # Log-transform
    log_ce_svi = np.log(merged['c_effort_svi'].values)
    log_ce_mcmc = np.log(merged['c_effort_mcmc'].values)
    log_cd_svi = np.log(merged['c_death_svi'].values)
    log_cd_mcmc = np.log(merged['c_death_mcmc'].values)

    r_ce, p_ce = pearsonr(log_ce_svi, log_ce_mcmc)
    r_cd, p_cd = pearsonr(log_cd_svi, log_cd_mcmc)

    # Mean absolute difference
    mad_ce = np.mean(np.abs(log_ce_mcmc - log_ce_svi))
    mad_cd = np.mean(np.abs(log_cd_mcmc - log_cd_svi))

    print(f"\nlog(ce): r = {r_ce:.4f} (p = {p_ce:.2e}), MAD = {mad_ce:.4f}")
    print(f"log(cd): r = {r_cd:.4f} (p = {p_cd:.2e}), MAD = {mad_cd:.4f}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, log_svi, log_mcmc, label, r_val, p_val in [
        (axes[0], log_ce_svi, log_ce_mcmc,
         r'$\log(c_e)$', r_ce, p_ce),
        (axes[1], log_cd_svi, log_cd_mcmc,
         r'$\log(c_d)$', r_cd, p_cd),
    ]:
        ax.scatter(log_svi, log_mcmc, alpha=0.4, s=15, c='#2166ac',
                   edgecolors='none')

        # Identity line
        lo = min(log_svi.min(), log_mcmc.min())
        hi = max(log_svi.max(), log_mcmc.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', alpha=0.5, linewidth=1)

        ax.set_xlabel(f'SVI {label}', fontsize=12)
        ax.set_ylabel(f'MCMC {label}', fontsize=12)
        ax.set_title(f'{label}: r = {r_val:.3f}', fontsize=13)

        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=10)

    fig.suptitle('SVI vs MCMC Parameter Recovery', fontsize=14, y=1.02)
    fig.tight_layout()

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {args.out}")

    # Print summary for paper
    print(f"\nPaper-ready:")
    print(f'  "SVI parameter estimates were validated against full MCMC '
          f'(per-subject correlation: log(ce) r = {r_ce:.3f}, '
          f'log(cd) r = {r_cd:.3f})."')


if __name__ == '__main__':
    main()
