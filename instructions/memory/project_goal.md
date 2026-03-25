---
name: Project Goal
description: Nature Comms submission — effort reallocation under threat, joint choice-vigor coupling via shared survival computation S
type: project
---

**Title:** Humans reallocate effort across decision and action when foraging under threat
**Authors:** Okada, Garg, Wise, Mobbs
**Target:** Nature Communications
**Status:** Draft complete, figures generated, preregistration updated. Awaiting MCMC runs + confirmatory sample.

## Core claim

Humans foraging under threat don't avoid effort — they **reallocate** it. A survival computation S governs both what people choose (shifting to safer targets) and how hard they press (deploying excess motor effort). This coupling is a structural population feature, not a post-hoc observation.

## Paper structure (4 Results sections)

1. **Choice model** — 5-model comparison. Winner: additive effort + hyperbolic survival (SV = R·S − k·E − β·(1−S)). Effort is a flat physical cost.
2. **Vigor** — Danger drives excess effort (δ = +0.21, 99% positive). S also predicts trial-level anxiety/confidence.
3. **Joint model** — Behavioral coupling (r = −0.78), independent Bayesian r(β,δ) = +0.55, joint LKJ model confirms structural coupling. Predicts foraging earnings (R² = 0.32).
4. **Metacognitive bridge** — δ predicts affect calibration to S. High-δ people are less anxious overall but more *accurately* anxious.

## What remains

- [x] ~~Run MCMC pipeline on GPU~~ — DONE. Choice, vigor, and joint MCMC outputs in `results/stats/mcmc_*.csv`
- [ ] Run confirmatory sample (N=350) through full pipeline
- [ ] Generate remaining figures (Fig 1: task + choice model, Fig 2: vigor)
- [ ] PPCs for choice and vigor models
- [ ] Parameter recovery analysis
- [ ] Finalize supplementary materials

**How to apply:** All work should bring the manuscript closer to submission. The story is set — execution remains.
