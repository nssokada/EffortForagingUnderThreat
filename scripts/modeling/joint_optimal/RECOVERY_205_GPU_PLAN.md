# Result 205 — MCMC parameter-recovery run + writeup-revision plan

**Audience:** the Claude agent running on the GPU device.
**Goal:** run the MCMC-based parameter recovery for the M4 (V8c) joint optimal-control
model, decide whether the κ-recovery failure documented in `result_205` is real or an
SVI artifact, and revise the writeup accordingly.

Read `instructions/memory/MEMORY.md` and `write-up/results/result_205_parameter_recovery.md`
before starting.

---

## Background — why this run exists

`result_205` (current text) claims **κ is essentially unrecoverable at the empirical
population spread** (r_κ ≈ −0.07 in the "fitted distribution" scenario) while ω recovers
well (r_ω ≈ 0.89–0.99). That conclusion was produced with **SVI / AutoNormal** inference.

Two facts cast doubt on that being a genuine identifiability limit rather than an
inference artifact:

1. **SVI coverage is broken.** The SVI full run
   (`param_recovery_v8c_full.py`, σ_log(κ)=1.359) gives r_κ = −0.017 **with 80% CI
   coverage of 1.2%** and 17% coverage for ω. AutoNormal is known to produce
   over-confident, poorly-calibrated posteriors — exactly this signature.
2. **MCMC smoke test recovers κ.** A 20-subject NUTS smoke run on the **same** wide
   spread (σ_log(κ)=1.359) recovered **r_κ ≈ 0.86** with ~80–90% coverage.

So the empirical κ spread is actually **wide** (σ_log(κ)=1.359, not narrow), and MCMC
appears to recover κ where SVI does not. The full N=500 NUTS run settles it.

**Key inputs (already verified to exist):**
- `results/stats/joint_optimal/exploratory/mcmc_m4_params.csv` →
  μ_log(ω)=+0.078, σ_log(ω)=0.871; μ_log(κ)=−1.555, **σ_log(κ)=1.359**; r(logω,logκ)≈0.37
- `results/stats/joint_optimal/confirmatory/mcmc_m4_params.csv` (for the optional second run)

These are the "proper sigma" — the script loads them directly; do **not** hand-pick σ.

---

## Run steps (GPU)

1. **Confirm JAX sees the GPU** (otherwise it silently falls back to sequential CPU chains):
   ```bash
   python -c "import jax; print(jax.devices())"   # must show a cuda/gpu device
   ```

2. **GPU smoke test (~2 min)** — confirm the vectorized path engages. Look for
   `[chain_method = vectorized (GPU)]` in the log:
   ```bash
   python scripts/modeling/joint_optimal/param_recovery_v8c_mcmc.py \
       --n_subj 20 --num_warmup 200 --num_samples 200 --num_chains 4 \
       --out_prefix _smoke_gpu
   ```

3. **Full exploratory run** (single dataset, 500 subjects, proper σ — all defaults):
   ```bash
   python scripts/modeling/joint_optimal/param_recovery_v8c_mcmc.py \
       --n_subj 500 --num_warmup 2000 --num_samples 4000 \
       --num_chains 4 --target_accept 0.95 --seed 42 --sample exploratory \
       2>&1 | tee results/stats/joint_optimal/recovery_mcmc_run.log
   ```
   Use `nohup`/`tmux` if the session may drop. The SVI warm-start does 10k steps
   **before** NUTS begins — don't kill it thinking it's hung.

4. **(Optional) confirmatory run** with a distinct prefix so nothing is overwritten:
   ```bash
   python scripts/modeling/joint_optimal/param_recovery_v8c_mcmc.py \
       --sample confirmatory --seed 43 \
       --out_prefix param_recovery_v8c_mcmc_confirmatory
   ```

**Outputs** (prefix `param_recovery_v8c_mcmc`):
`*.csv` (per-subject), `*_summary.csv` (metrics), `*_population.csv` (pop params),
`*_diagnostics.csv` (R-hat/ESS), `*_samples.npz` (raw draws).

---

## What to check before trusting anything

1. **Convergence first.** In `*_summary.csv`, `convergence_passed` must be `True`
   (max R-hat < 1.01, min ESS > 400). If `WARN`, increase `--num_warmup` /
   `--num_samples` and rerun — do **not** interpret recovery from a non-converged fit.
2. **Headline metrics** in `*_summary.csv`:
   - `r_kappa`, `rho_kappa`, `cov80_kappa`, `cov95_kappa`  ← the decisive numbers
   - `r_omega`, `cov80_omega`, `cov95_omega`  ← should stay strong (~0.9, ~80%)
   - `cross_talk_om_to_kp`, `cross_talk_kp_to_om`  ← should be modest
3. **Population recovery** in `*_population.csv`: σ_log(κ) recovered ≈ 1.36, μ's and
   γ/hazard near their true values.

---

## Decision tree → how to revise `result_205`

Let r_κ and cov80_κ be the converged values from the exploratory `*_summary.csv`.

### Outcome A — MCMC recovers κ (expected): r_κ ≳ 0.6 AND cov80_κ ≈ 0.7–0.9
The earlier "κ unrecoverable" finding was an **SVI artifact**. Revise 205 substantially:
- **`status:`** `partial` → `supported`.
- **`title:`** change to reflect that **both ω and κ are well-recovered under the
  paper's actual MCMC inference**, and that the apparent κ failure was an inference-method
  artifact (SVI/AutoNormal mis-calibration), not an identifiability limit.
- **Overview / Result table:** replace the three SVI scenarios with the single MCMC run
  (report r_ω, r_κ, ρ, RMSE, coverage, cross-talk, population recovery). Add the SVI vs
  MCMC contrast (r_κ=−0.02 @ 1.2% coverage under SVI → r_κ=<value> @ <coverage> under MCMC,
  same σ=1.359) as the explanatory pivot.
- **Interpretation:** drop the "per-subject κ is unreliable" caveat. State that
  subject-level κ **is** interpretable. Note coverage is now calibrated (the SVI run's
  ~1–17% coverage was the tell-tale sign of over-confident variational posteriors).
- **Downstream:** relax the H4d / angle-test caveat in 205 **and** check
  `result_208` and the H4-family writeups for the same "interpret κ cautiously" hedge —
  flag those for update (don't silently edit them; list them for the user).
- **`notebooks:`/`scripts:`/`outputs:`** point to
  `scripts/modeling/joint_optimal/param_recovery_v8c_mcmc.py` and the new
  `param_recovery_v8c_mcmc*` outputs (the current `run_mcmc_pipeline.py` reference is stale).
- Add a recovery figure to `figures:` (currently `TODO`) if one is generated.

### Outcome B — κ still fails under MCMC: r_κ ≲ 0.3 OR cov80_κ poor
The limitation is **real**, not an SVI artifact — strengthen 205's existing claim:
- Keep `status: partial`. Keep the κ caveat, but now state it survives proper MCMC
  inference (so it's a genuine data/identifiability limit, not a method choice).
- Replace the SVI table with the MCMC numbers; note coverage is now well-calibrated for ω.
- Keep the cautious-κ guidance for `result_208` / H4 family.

### Outcome C — ambiguous / mixed (e.g. good r_κ but poor coverage, or vice-versa)
Report both metrics honestly; do **not** round up to "recovered." Keep `status: partial`,
describe the split (e.g. "rank order recovered but CIs mis-calibrated"), and ask the user
how they want to frame it before editing downstream files.

### In all outcomes
- Run the writeup through the `write-result` skill's fail-loud validation so the headline
  numbers in the prose match the CSVs (T3).
- If a confirmatory run was done, add it as a replication row / sample to the writeup.
- Update `instructions/memory/` (`joint_model_development.md`, `next_steps.md`) to record
  the MCMC recovery result and retire the "MCMC not run" item.

---

## Guardrails
- **Do not** hand-edit σ or hard-code recovery numbers into prose — pull every number from
  the generated CSVs.
- **Do not** edit `result_208` or other H4 writeups silently; surface them to the user.
- If convergence fails, fix convergence before interpreting — a non-converged fit can
  produce a misleadingly low *or* high r_κ.
- The SVI scripts (`param_recovery_v8c_full.py`) and the stale `run_mcmc_pipeline.py`
  reference in 205's frontmatter are superseded by `param_recovery_v8c_mcmc.py` for this result.
