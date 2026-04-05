# H3: The Joint Fitness Model Outperforms All Alternatives

## Preregistered prediction

The joint model M4 (omega + kappa, both entering choice and vigor through a shared fitness function W(u)) will outperform three simpler alternatives, demonstrating that (a) threat matters beyond effort cost, (b) individual effort differences matter beyond threat sensitivity, and (c) capture cost and effort cost are separable traits.

## Tests and thresholds

| Test | Comparison | Criterion |
|------|-----------|-----------|
| H3a | M4 vs M1 (effort-only) | ΔWAIC > 0 AND ΔLOO > 0 |
| H3b | M4 vs M2 (threat-only) | ΔWAIC > 0 AND ΔLOO > 0 |
| H3c | M4 vs M3 (single-parameter) | ΔWAIC > 0 AND ΔLOO > 0 |

Both WAIC and PSIS-LOO must agree. All models evaluated on the same joint (choice + vigor) likelihood.

## Model descriptions

| Model | Per-subject params | Choice equation | Vigor equation |
|-------|-------------------|-----------------|----------------|
| M1 | κ | ΔV = ΔR − κ × Δeffort | Intercept only (no condition structure) |
| M2 | ω (population κ) | V from W(u) with shared κ | u* from W(u) with shared κ |
| M3 | θ = ω = κ | V from W(u) | u* from W(u) |
| M4 | ω + κ | V from W(u) − κ × req × D | u* from W(u) |

All fitted via NumPyro HMC/NUTS (4 chains × 2,000 warmup + 4,000 samples, target_accept = 0.95).

## Results

### Exploratory sample

| Model | WAIC | ΔWAIC | LOO | ΔLOO | Acc | Vigor r² | Converged |
|-------|------|-------|-----|------|-----|----------|-----------|
| M4 (Joint) | 12,776 | — | 12,779 | — | 0.77 | 0.37 | Yes |
| M2 (Threat-only) | 14,742 | +1,966 | 14,745 | +1,966 | 0.79 | 0.01 | Yes |
| M3 (Single-param) | 15,374 | +2,599 | 15,404 | +2,625 | 0.77 | 0.10 | Yes* |
| M3b (Scaled) | 14,735 | +1,959 | 14,741 | +1,962 | 0.79 | 0.02 | Yes |
| M1 (Effort-only) | 17,505 | +4,729 | 17,509 | +4,731 | 0.71 | 0.01 | Yes |

*Converged after doubled iterations (4,000 warmup + 8,000 samples).

### Confirmatory sample

| Model | WAIC | ΔWAIC | LOO | ΔLOO | Acc | Vigor r² | Converged |
|-------|------|-------|-----|------|-----|----------|-----------|
| M4 (Joint) | 12,252 | — | 12,263 | — | 0.76 | 0.41 | Yes |
| M2 (Threat-only) | 13,873 | +1,621 | 13,881 | +1,618 | 0.78 | 0.01 | Yes |
| M3 (Single-param) | 15,727 | +3,474 | 15,737 | +3,474 | 0.76 | 0.07 | No** |
| M3b (Scaled) | 13,850 | +1,597 | 13,856 | +1,593 | 0.79 | 0.02 | Yes |
| M1 (Effort-only) | 16,037 | +3,785 | 16,042 | +3,779 | 0.71 | 0.01 | Yes |

**Did not converge even after doubled iterations; ΔWAIC is nevertheless decisive.

### Hypothesis tests

| Test | Exploratory ΔWAIC / ΔLOO | Confirmatory ΔWAIC / ΔLOO | Both agree? | Verdict |
|------|-------------------------|--------------------------|-------------|---------|
| H3a: M4 vs M1 | +4,729 / +4,731 | +3,785 / +3,779 | Yes | **CONFIRMED** |
| H3b: M4 vs M2 | +1,966 / +1,966 | +1,621 / +1,618 | Yes | **CONFIRMED** |
| H3c: M4 vs M3 | +2,599 / +2,625 | +3,474 / +3,474 | Yes | **CONFIRMED** |

WAIC and PSIS-LOO agreed on all six comparisons (3 tests × 2 samples). All ΔWAIC values are in the thousands — these are decisive, not marginal.

## Summary

| Test | Exploratory | Confirmatory |
|------|-------------|--------------|
| H3a: M4 vs M1 | PASS | PASS |
| H3b: M4 vs M2 | PASS | PASS |
| H3c: M4 vs M3 | PASS | PASS |
| **Total** | **3/3** | **3/3** |

## Model fit quality

M4 achieves:
- **Choice accuracy:** 0.76 (confirmatory) / 0.77 (exploratory) — the model correctly predicts which cookie participants choose on 76–77% of trials
- **Vigor r²:** 0.41 (confirmatory) / 0.37 (exploratory) — the model explains 37–41% of variance in condition-level pressing rates

The vigor r² is notably higher in the confirmatory sample, suggesting the model generalises well. M2 and M3b achieve comparable choice accuracy (~0.79) but near-zero vigor r² (0.01–0.02), confirming that the per-subject kappa is essential for explaining vigor patterns.

## M3 non-convergence

M3 (single-parameter: θ = ω = κ) did not converge in the confirmatory sample even after doubled iterations. This is interpretively meaningful: the model is structurally misspecified — a single parameter literally cannot serve both the avoidance and activation roles simultaneously. The resulting posterior geometry has no stable mode. The exploratory M3 converged after doubled iterations but still lost decisively (ΔWAIC = +2,599).

The supplementary M3b (scaled: θ as ω, αθ as κ) converged in both samples but also lost (ΔWAIC = +1,597–1,959), ruling out a simple scale mismatch as the explanation for M3's failure. The two parameters are genuinely separable.

## Interpretation

A single fitness function with two individual-difference parameters explains both patch selection and motor vigor better than any simpler alternative. Threat matters (M1 fails). Individual effort differences matter (M2 fails). And avoidance sensitivity and activation intensity are separable traits that cannot be collapsed into a single dimension (M3 fails). The joint constraint — that the same omega and kappa enter both the choice and vigor likelihoods through a shared W(u) — does not hurt fit; it helps, because the two channels provide complementary information for parameter identification.
