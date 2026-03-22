---
name: Next Steps
description: Priority action items for Nature Comms submission — MCMC, confirmatory sample, remaining figures
type: project
---

# Next Steps (as of 2026-03-22)

## Priority 1: MCMC on GPU

**Script:** `scripts/run_mcmc_pipeline.py`
**Command:** `python3 scripts/run_mcmc_pipeline.py --platform gpu`

Three stages:
1. Choice NUTS → proper λ, k_i, β_i posteriors (~30-60 min GPU)
2. Vigor NUTS (λ fixed) → proper α_i, δ_i posteriors (~30-60 min GPU)
3. Joint NUTS (LKJ correlated) → proper ρ posteriors with Rhat/ESS (~2-4 hrs GPU)

**Why:** Proper posteriors, convergence diagnostics, no variational approximation. May resolve the λ sensitivity issue in the joint model. More credible for reviewers.

**Status:** 🔲 Script ready, awaiting GPU access.

---

## Priority 2: Remaining figures

- **Fig 1** (task design + 5-model comparison) — needs notebook
- **Fig 2** (vigor δ distribution + within-choice controls + affect) — needs notebook
- **PPCs** for choice model (predicted vs observed by condition) — supplementary
- **Parameter recovery** simulation — supplementary

Existing figures (done):
- ✅ Fig 3 — Coherent shift (`fig4_coherent_shift.pdf`)
- ✅ Fig 4 — Metacognitive bridge (`fig_s_metacognition.pdf`)
- ✅ Fig 5 — Joint model (`fig5_joint_model.pdf`)
- ✅ Fig S — Mental health (`fig_s_mental_health.pdf`)

---

## Priority 3: Confirmatory sample (N=350)

1. Preprocess through stages 1-5
2. Fit choice model (L3_add) → k, β, λ
3. Fit vigor HBM (λ fixed from choice) → α, δ
4. Test pre-registered H1-H6
5. Compute independent Bayesian correlations
6. Run joint model as robustness check

**Preregistration:** Updated at `drafts/preregistration.md` with H5 (joint model) and H6 (metacognition).

**Status:** 🔲 Data collected, not preprocessed.

---

## Priority 4: Final polish

- Finalize supplementary tables (model comparison, parameter estimates)
- Update all figure legends to match tightened 4-Results structure
- Add Supplementary Note 1 on λ sensitivity
- Code/data availability statements
- References (currently incomplete)
