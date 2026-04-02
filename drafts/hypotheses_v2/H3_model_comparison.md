# H3: A Joint Fitness Function Governs Both Patch Selection and Motor Vigor

## Overview

We test whether a single fitness function W(u) with two per-subject parameters — capture cost (ω) and effort cost (κ) — best explains both patch selection and motor vigor. We compare four models representing alternative theories of the effort-threat tradeoff.

### Theoretical grounding

1. **Optimal foraging under predation** (Bednekoff 2007; Brown 1999): W(u) = S(u)·V(u) jointly determines foraging effort and patch selection.
2. **Risk allocation** (Lima & Bednekoff 1999): The same threat evaluation governs both avoidance and mobilization.
3. **Motor vigor** (Shadmehr & Krakauer 2008; Yoon et al. 2018): Movement speed and harvest decisions derive from a single reward-rate optimization.

---

## The Fitness Function

W(u) = S(u)·R − (1 − S(u))·ω·(R + C) − κ·(u − req)²·D

- S(u, T, D) = exp(−h·T^γ·D / speed(u)), speed(u) = sigmoid((u − 0.25·req) / σ_sp)
- **Choice:** V_j = max_u W_j(u) − κ·req_j·D_j (total demand cost). P(heavy) = sigmoid((V_H − V_L) / τ)
- **Vigor:** Cell-mean rate ~ Normal(u*, σ_v/√n) where u* = argmax_u W(u) (quadratic cost only)

---

## The Models

### M1: Effort-only (κ)
People avoid effort. Threat is irrelevant. κ per-subject, no survival function, no vigor model.

### M2: Threat-only (ω)
Only survival matters. ω per-subject, population κ. No per-subject effort term in choice.

### M3: Single-parameter (θ = ω = κ)
One trait governs both channels. θ enters W as both capture and effort cost.

### M4: Joint W(u) (ω + κ)
Two separable costs in one fitness function. ω and κ both enter choice and vigor through W.

---

## Sub-hypotheses

**H3a.** M4 outperforms M1 (ΔWAIC > 0). Threat matters beyond effort.

**H3b.** M4 outperforms M2 (ΔWAIC > 0). Individual effort differences matter.

**H3c.** M4 outperforms M3 (ΔWAIC > 0). Capture cost and effort cost are separable.

All models fitted with identical MCMC inference (NumPyro NUTS, 4 chains × 2000 warmup + 4000 samples, target_accept=0.95). Primary criterion: WAIC. Robustness: PSIS-LOO. Hypothesis confirmed only if both agree.

---

## Vigor Data

Per-subject condition cell means (subject × threat × distance × cookie, ~5,200 cells, ~18 per subject). Weighted by √n in the likelihood. This denoises motor variability while preserving the condition structure the model predicts.

---

## Exploratory Benchmarks (Discovery Sample, N = 290)

| Metric | M4 (joint) |
|--------|-----------|
| Choice acc | 0.779 |
| Choice r² | 0.894 |
| Vigor r² (cell means) | 0.386 |
| ω recovery | r = 0.94 |
| κ recovery | r = 0.92 |
| ω↔κ | r = 0.27 |

---

## Confirmation Plan

| Test | Criterion | Threshold |
|------|-----------|-----------|
| H3a: M4 vs M1 | ΔWAIC + ΔLOO | Both > 0 |
| H3b: M4 vs M2 | ΔWAIC + ΔLOO | Both > 0 |
| H3c: M4 vs M3 | ΔWAIC + ΔLOO | Both > 0 |
