# MCMC Model Comparison — Joint Optimal-Control Models (M1–M4)

Production MCMC inference for the joint fitness model. This is the current model
line (per-subject **ω** = capture cost and **κ** = effort cost). The older EVC
"2+2" model (`ce`/`cd`) and its scripts (`run_mcmc.py`, `run_mcmc_quick.py`,
`compare_svi_mcmc.py`) have been retired to `deprecated/scripts/`.

## Files

- `run_model_comparison_mcmc.py` — the runner (NUTS fit + WAIC/LOO + convergence).
- Model **definitions** live in
  [`scripts/modeling/joint_optimal/model_comparison_cm.py`](../modeling/joint_optimal/model_comparison_cm.py)
  (`make_m1`…`make_m4`, the shared `eu_sat` optimum, `pop_vigor_params`,
  and the `prepare_data` loader). The runner imports them directly.

## Requirements

- Python 3.11+ (conda env `effort_foraging_threat`)
- JAX with GPU support (`jax[cuda12]`) — falls back to 4 CPU chains if no GPU
- NumPyro, ArviZ, pandas, numpy, scipy

## Input

`prepare_data()` reads pre-built model-input CSVs from `data/model_input/`:
`choice_trials.csv`, `vigor_cell_means.csv` (cell means with `n_trials` weights),
`subject_mapping.csv`.

## Run

```bash
# Preregistered model comparison (default)
python scripts/mcmc/run_model_comparison_mcmc.py --models M1,M2,M3,M4

# Exploratory variants
python scripts/mcmc/run_model_comparison_mcmc.py --models M3b,M_sep
```

Inference (identical across models): **NUTS, 4 chains × 2000 warmup + 4000 samples,
target_accept = 0.95, max_tree_depth = 10**, with a short SVI/AutoNormal warm-start
for initialization. Chains run vectorized on one GPU when available
(`chain_method='vectorized'`); otherwise sequential on 4 CPU cores.
Run time ≈ 1 hour on an RTX-class GPU for M4 at N≈290.

## Models

| Model | Per-subject | Description |
|---|---|---|
| M1 | κ | Effort-only (no threat; intercept-only vigor) |
| M2 | ω | Threat-only (population κ) |
| M3 | θ (=ω=κ) | Single-parameter |
| **M4** | **ω + κ** | **Joint W(u) — both enter choice and vigor** (the paper's model) |
| M3b | θ, α·θ | Scaled single-parameter (exploratory) |
| M_sep | λ, ω | Separate choice/vigor equations (exploratory) |

## Outputs (`results/stats/joint_optimal/`)

- `mcmc_m4_params.csv` — per-subject ω, κ posterior means (the file downstream analyses read)
- `mcmc_model_comparison.csv` — WAIC (primary) + PSIS-LOO per model
- `mcmc_convergence_diagnostics.csv` — R-hat and ESS per parameter, per model

Per-sample committed copies live in `results/stats/joint_optimal/exploratory/` and
`…/confirmatory/` (the runner reads whichever sample's `data/model_input/` is in
place and writes to the top-level dir; copy outputs into the sample subdir).

## Expected / sanity checks

- Convergence: max R-hat < 1.01, high ESS (M4 on real exploratory data: max R-hat = 1.0016).
- Model comparison favors **M4** over M1/M2/M3 by WAIC (see results 202–204).

## Related

- **Parameter recovery:** `scripts/modeling/joint_optimal/param_recovery_m4_mcmc.py`
  (imports `make_m4`/`eu_sat` from `model_comparison_cm.py` so it can't drift from
  production) → write-up `result_205`.
- SVI fitting of the same models: `fit_model()` inside `model_comparison_cm.py`.
