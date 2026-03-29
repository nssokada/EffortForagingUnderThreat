# MCMC Validation for EVC 2+2 Model

## Requirements
- Python 3.11+
- JAX with GPU support (jax[cuda12] or jax[cuda11])
- NumPyro
- pandas, numpy, scipy

## Setup on GPU machine
pip install jax[cuda12] numpyro pandas scipy

## Quick test (5-10 min on GPU, ~20 min on CPU)
python scripts/mcmc/run_mcmc_quick.py --data_dir data/exploratory_350/processed/stage5_filtered_data_20260320_191950

## Full run (30-60 min on GPU)
python scripts/mcmc/run_mcmc.py --data_dir data/exploratory_350/processed/stage5_filtered_data_20260320_191950

## Post-hoc comparison figure (after MCMC completes)
python scripts/mcmc/compare_svi_mcmc.py

## Expected output
- R-hat < 1.05 for all population parameters
- R-hat < 1.10 for per-subject parameters (with 1000 samples)
- SVI-MCMC correlation: r > 0.99 for both log(ce) and log(cd)
- Zero or near-zero divergent transitions

## Output files
- `results/stats/mcmc/oc_evc_mcmc_params.csv` — per-subject ce, cd posterior means
- `results/stats/mcmc/oc_evc_mcmc_population.csv` — population param posteriors with HDIs
- `results/stats/mcmc/oc_evc_mcmc_diagnostics.csv` — R-hat, ESS per param
- `results/stats/mcmc/oc_evc_mcmc_summary.txt` — text summary of results
- `results/figs/paper/fig_s_mcmc_validation.png` — SVI vs MCMC comparison figure

## What to report in the paper
"SVI parameter estimates were validated against MCMC (NUTS, 4 chains x 1000 samples;
all population R-hat < 1.05; per-subject parameter correlation with SVI:
log(ce) r = X.XXX, log(cd) r = X.XXX; N divergent transitions = X)."
