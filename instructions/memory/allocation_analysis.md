---
name: Allocation Analysis Results
description: Binary-E model breakthrough, β-δ coupling as threat-reactivity dimension, allocation angle vs magnitude, psychiatric correlates (δ→apathy), E-scaling test (failed)
type: project
---

# Allocation Analysis (2026-03-24)

## MAJOR MODEL BREAKTHROUGH: Binary Effort Specification

### The problem
With graded E ∈ {0.6, 0.8, 1.0} perfectly confounded with D ∈ {1, 2, 3}, k·E absorbed all distance effects, forcing λ=14 and making S flat on distance (corr(S,T)=-0.999).

### The fix
Binary effort: E = 1 (high option) or 0 (low option). Distance enters ONLY through S.

**SV = R·S − k·I(high) − β·(1−S), S = (1−T) + T/(1+λD)**

### Results
| Model | ELBO | λ | k | β |
|---|---|---|---|---|
| L3_add original (graded E) | -6275.4 | 13.76 | 6.12 | 54.90 |
| **L3_add binary E** | **-6187.5** | **0.80** | **2.73** | **10.65** |

**+88 ELBO improvement.** λ drops from 14 to 0.8. S now has real distance gradient:
- S(T=0.9, D=1) = 0.599, S(T=0.9, D=3) = 0.364 (gradient = 0.235 vs 0.040 before)
- Parameters on interpretable scales

### E-scaling also tested (failed)
E = effort_rate × distance made model significantly worse (ΔELBO = -293). The data prefer binary E.

### L4a_binary (α in S) also tested
Hurt the model (ΔELBO = -44 vs original). Don't use α in S with binary E.

## Vigor Model Update

### Effort-controlled vigor model
**excess = α + δ·(1−S) + γ·E_chosen**

Using λ=0.8 from binary-E choice model:
- γ = -0.535 (effort demand constraint — higher demand → less excess)
- μ_δ = 0.456, 98.6% positive
- ELBO improves by +1870 over uncontrolled model
- **CRITICAL**: Must use `smoothed_vigor_ts.parquet` data (20Hz timeseries averaged per trial), NOT `mean_trial_effort` from behavior_rich.csv. Current quick SVI tests used behavior_rich as proxy. Need to verify with actual parquet data — **requires pyarrow installation**.

### β-δ coupling with new model
- r(log(β), δ) = +0.326, p < 10⁻⁸ (cleaned of distance confounds; was +0.528 with old model)
- PC1 explains 66.3% of joint variance, equal loadings (0.707, 0.707)

## Allocation Angle vs Magnitude

**The key finding: direction matters, intensity doesn't.**

| Predictor | → Earnings | → Escape rate |
|---|---|---|
| Angle (direction) | **r = +0.513, p < 10⁻²¹** | **r = +0.537, p < 10⁻²³** |
| Magnitude (intensity) | r = -0.089, p = 0.13 | r = -0.091, p = 0.12 |

### Tercile breakdown
| Group | Earnings | Escape rate |
|---|---|---|
| β-heavy (choice-stuck) | **-61 pts** | **18.5%** |
| Balanced | +31 pts | 45.7% |
| δ-heavy (vigor-focused) | **+51 pts** | **51.1%** |

### Top vs bottom earners (quartiles)
Same magnitude (p=0.10), opposite angles (t=11.7, p≈0). Top earners: +67°, bottom: -82°.

### Quadrant analysis
88% "shift both" (only profitable strategy). Choice-only = catastrophic (2% escape, -142 pts).

### Continuous allocation model
- Vigor shift: β = 56.1 (dominant), choice shift: n.s., interaction: β = 14.2 (p<0.001)
- R² = 0.473

## Psychiatric Correlates

### All 4 params (k, β, α, δ) → psych factors (Bambi)
- F1 (distress): **null** for all params
- F2 (fatigue): **null** for all params
- **F3 (apathy): δ only, β = -0.319, HDI [-0.44, -0.20]**

k, β, α are psychiatrically silent. Only δ relates to mental health.

### All 4 params → individual subscales (94% HDI)
| Param | Outcome | β | HDI |
|---|---|---|---|
| **δ** | **AMI Behavioural** | **+0.237** | [+0.114, +0.360] |
| **δ** | **AMI Social** | **+0.231** | [+0.110, +0.355] |
| **δ** | **PHQ9 (depression)** | **+0.131** | [+0.009, +0.260] |

Borderline: OASIS × β (negative, 89% HDI only), DASS Depression × δ (89% only).

### Subscales → allocation angle (Bambi multivariate)
- **DASS21_Anxiety → angle: β = -0.424, HDI [-0.793, -0.043]** (anxious → choice-stuck)
- **AMI_Social → angle: β = +0.328, HDI [+0.070, +0.577]** (apathetic → vigor-focused)

### Interpretation: "Adaptive apathy"
High-δ individuals report more apathy and mild depression but mobilize powerfully under threat. They're selectively engaged, not globally impaired. The task reveals motor competence that self-report misses.

Anxiety predicts maladaptive allocation (choice avoidance without motor compensation).

## Figures Generated

All in `results/figs/paper/`:
- `fig_h2_coupling_scatter.pdf` — Δchoice vs Δvigor (cerulean, clean)
- `fig_allocation_scatter.pdf` — β-δ space colored by earnings
- `fig_allocation_surface.pdf` — predicted earnings contours + zero boundary
- `fig_allocation_punchline.pdf` — magnitude (null) vs angle (significant)
- `fig_excess_by_threat_choice_distance.pdf` — excess effort decomposed by choice type × distance × threat (capacity constraint story)

## Scripts Created/Modified

- `scripts/analysis/run_h1_lmm_tests.py` — H1a-c LMM tests (complete, tested)
- `scripts/analysis/run_h2_coupling_tests.py` — H2a-b coupling + split-half (complete, tested)
- `scripts/analysis/run_h3_optimality_tests.py` — H3a-b with empirical escape rates (complete, tested)
- `scripts/run_effort_scaled_comparison.py` — E-scaling comparison (graded vs binary)
- `scripts/plotting/plot_h2_coupling.py` — H2 scatter (multiple versions)
- `scripts/plotting/plot_h2_coupling_v5.py` — clean cerulean scatter
- `scripts/plotting/plot_allocation_space.py` — allocation dimension figures
- `scripts/analysis/generate_results_tables.py` — HTML results tables

## NEXT STEP (BLOCKING)

**Must verify the binary-E choice model and effort-controlled vigor model work with the actual vigor timeseries data from `smoothed_vigor_ts.parquet`.** Current results used `mean_trial_effort` from `behavior_rich.csv` as a proxy. Need pyarrow to read parquet files. The qualitative results should hold but exact numbers may differ.

Steps:
1. Install pyarrow (needs network or pre-built package)
2. Rerun effort-scaled comparison with parquet-derived vigor_norm
3. Rerun vigor HBM with proper data
4. Verify β-δ coupling, allocation angle results hold
5. Then run full MCMC pipeline with new model spec
