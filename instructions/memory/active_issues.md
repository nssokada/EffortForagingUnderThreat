---
name: Active Issues
description: Currently blocking or requiring attention for Nature Comms submission
type: issues
---

# Active Issues

## BLOCKING

*None currently blocking the draft. Confirmatory sample is the main remaining execution item.*

---

## MAJOR OPEN WORK

### Confirmatory sample (N=350) not started
- **Issue:** Raw data exists in `data/confirmatory_350/raw/` but not preprocessed.
- **Resolution:** Run full pipeline, then refit all models, test preregistered H1-H6.
- **Preregistration:** Updated at `drafts/preregistration.md` with H5 (joint model) and H6 (metacognition).

### Missing figures
- **Fig 1** (task design + choice model comparison) — not yet generated
- **Fig 2** (vigor + affect) — not yet generated
- **PPCs** for choice and vigor models — not yet computed
- **Parameter recovery** — not yet run on current model

---

## TECHNICAL DEBT

### Old model files may cause confusion
- `results/model_fits/exploratory/FET_Exp_Bias_fit.pkl` — superseded by L3_add
- `results/stats/joint_model_*.csv` — superseded by `joint_correlated_*.csv` and `independent_bayesian_*.csv`
- `results/stats/FET_Exp_Bias_*.csv` — superseded by `unified_3param_clean.csv`

### λ = 2.0 references may persist in old notebooks
- The original affect analyses used λ = 2.0 which was incorrect
- Current correct λ = 13.9 (±0.6) from L3_add SVI fit
- Affect results are robust to this change (S correlations r > 0.99 between λ values)
- Check any notebook that computes S_probe to ensure it uses λ = 13.9

---

## RESOLVED

- ~~Draft needs rewriting~~ → Complete rewrite done (2026-03-22), 4 Results sections
- ~~Joint model σ_δ collapsed~~ → Fixed with LKJ + AutoMultivariateNormal
- ~~β unidentified in joint model~~ → Fixed with option-specific S
- ~~λ discrepancy~~ → Resolved: both guides give λ ≈ 14, documented in Methods
- ~~Factor analysis blocked~~ → Complete (3 factors, α → apathy)
- ~~Bayesian MH analysis~~ → Complete with ROPE equivalence testing
