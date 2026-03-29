"""
Quick MCMC validation of the EVC 2+2 model.

Same model as run_mcmc.py but with reduced sampling for testing:
    2 chains, 200 warmup + 200 samples

Use this to verify the script works before committing to the full run.

Usage:
    python scripts/mcmc/run_mcmc_quick.py --data_dir data/exploratory_350/processed/stage5_filtered_data_20260320_191950
"""

import sys
import os

# Import everything from the full run script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_mcmc import (
    prepare_data, make_model, fit_mcmc, compute_diagnostics,
    compare_with_svi, write_summary
)

import argparse
import time
import jax
import numpy as np
import numpyro

jax.config.update('jax_enable_x64', True)


def main():
    parser = argparse.ArgumentParser(
        description='Quick MCMC test of EVC 2+2 model '
                    '(2 chains, 200+200 samples)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed data directory '
                             '(containing behavior_rich.csv)')
    parser.add_argument('--svi_params', type=str,
                        default='results/stats/oc_evc_final_params.csv',
                        help='Path to SVI parameter CSV for comparison')
    parser.add_argument('--out_dir', type=str,
                        default='results/stats/mcmc',
                        help='Output directory for MCMC results')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    NUM_CHAINS = 2
    NUM_WARMUP = 200
    NUM_SAMPLES = 200

    t0_total = time.time()

    # Try to set up parallel chains
    try:
        numpyro.set_host_device_count(NUM_CHAINS)
    except RuntimeError:
        print(f"WARNING: Could not set host device count to {NUM_CHAINS}. "
              f"Chains will run sequentially.")

    # Report device
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"\n*** QUICK TEST MODE: {NUM_CHAINS} chains x "
          f"{NUM_WARMUP} warmup + {NUM_SAMPLES} samples ***\n")

    # Load data
    print("Loading data...")
    behavior_path = os.path.join(args.data_dir, 'behavior_rich.csv')
    psych_path = os.path.join(args.data_dir, 'psych.csv')
    if not os.path.exists(psych_path):
        psych_path = None

    data = prepare_data(behavior_path, psych_path=psych_path)
    print(f"N_subjects={data['N_S']}")
    print(f"N_choice={data['N_choice']}, N_vigor={data['N_vigor']}")

    # Fit MCMC (quick settings)
    mcmc, elapsed = fit_mcmc(
        data,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        target_accept_prob=0.85,
        max_tree_depth=10,
        seed=args.seed,
    )

    # Diagnostics
    result = compute_diagnostics(mcmc, data, num_chains=NUM_CHAINS)

    # Divergences
    extra = mcmc.get_extra_fields()
    n_divergent = 0
    if 'diverging' in extra:
        n_divergent = int(np.sum(np.array(extra['diverging'])))
    print(f"\nDivergent transitions: {n_divergent}")

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)

    param_path = os.path.join(args.out_dir, 'oc_evc_mcmc_params.csv')
    result['param_df'].to_csv(param_path, index=False)
    print(f"\nSaved: {param_path}")

    pop_path = os.path.join(args.out_dir, 'oc_evc_mcmc_population.csv')
    result['pop_df'].to_csv(pop_path, index=False)
    print(f"Saved: {pop_path}")

    diag_path = os.path.join(args.out_dir, 'oc_evc_mcmc_diagnostics.csv')
    result['diag_df'].to_csv(diag_path, index=False)
    print(f"Saved: {diag_path}")

    # Compare with SVI
    comparison = None
    if os.path.exists(args.svi_params):
        comparison = compare_with_svi(result, args.svi_params)
    else:
        print(f"\nWARNING: SVI params not found at {args.svi_params}, "
              f"skipping comparison")

    # Write text summary
    summary_path = os.path.join(args.out_dir, 'oc_evc_mcmc_summary.txt')
    write_summary(summary_path, result, comparison, elapsed,
                  NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS, n_divergent)

    total_elapsed = time.time() - t0_total
    print(f"\nTotal time: {total_elapsed / 60:.1f} min")
    print(f"\n*** Quick test complete. If convergence looks reasonable, "
          f"run the full version with run_mcmc.py ***")


if __name__ == '__main__':
    main()
