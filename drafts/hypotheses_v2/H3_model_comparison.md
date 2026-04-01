# H3: A Joint Fitness Function Governs Both Patch Selection and Motor Vigor

## Overview

This hypothesis tests whether human foraging under predation risk is best described by a single fitness function W(u) that jointly determines which patch to select and how intensely to forage. Individual differences in capture cost (ω) and effort cost (κ) enter the same objective function, producing both choice and vigor as outputs of one optimization.

We compare five models, each representing an alternative theoretical account of how organisms manage the effort-threat tradeoff.

### Theoretical grounding

The joint model draws on four literatures:

1. **Optimal foraging under predation** (Bednekoff 2007; Brown 1999): A single fitness function W(u) = S(u)·V(u) jointly determines foraging effort and patch selection. The organism optimizes one objective to produce both behaviors.

2. **Risk allocation** (Lima & Bednekoff 1999): Organisms allocate effort across danger levels based on survival probability. The same threat evaluation governs both strategic avoidance (which patches to visit) and tactical mobilization (how hard to work).

3. **Motor vigor and foraging** (Shadmehr & Krakauer 2008; Yoon et al. 2018): Movement vigor is chosen to maximize reward rate. Both harvest decisions and movement speed derive from a single normative principle.

4. **Predation risk decomposition** (Lima & Dill 1990): P(death) = 1 − exp(−α·d·T). Each component (encounter rate, lethality, exposure time) is partially under behavioral control through both patch selection and motor execution.

---

## The Fitness Function

W(u) = S(u)·R − (1 − S(u))·ω·(R + C) − κ·(u − req)²·D

where:
- S(u, T, D) = exp(−h·T^γ·D / speed(u)) — survival probability
- speed(u) = sigmoid((u − 0.25·req) / σ_sp) — movement speed, saturating above req
- ω_i = per-subject cost of capture
- κ_i = per-subject cost of effort
- R, req, D = reward, required pressing rate, distance for cookie j
- C = 5 (capture penalty), T = threat probability

**Choice:** Compare V_H = max_u W_H(u) vs V_L = max_u W_L(u). P(heavy) = sigmoid((V_H − V_L) / τ).
**Vigor:** u* = argmax_u W_chosen(u). Observed cell-mean rate ~ Normal(u*, σ_v/√n).

---

## The Models

### M1: Effort-only (κ)

**Theoretical claim:** People avoid effort. Threat is irrelevant to patch selection.

**Choice:** ΔV = ΔR − κ_i × Δeffort(D). No survival function.
**Vigor:** Not modeled from W (baseline predictor only).
**Per-subject:** κ only.
**What it tests:** Does threat add anything beyond effort cost?

### M2: Threat-only (ω)

**Theoretical claim:** Only survival probability matters. Individual differences arise only in capture cost.

**Choice:** V_j = S_j·R_j − (1−S_j)·ω_i·(R_j+C). No per-subject effort term.
**Vigor:** u* from W with ω_i, population κ.
**Per-subject:** ω only.
**What it tests:** Do people differ in effort sensitivity, or only in threat sensitivity?

### M3: Single-parameter (θ)

**Theoretical claim:** One trait governs both avoidance and motor intensity.

**Choice + Vigor:** W(u) with θ_i entering as both ω and κ (θ = ω = κ).
**Per-subject:** θ only.
**What it tests:** Can one number serve both roles?

### M4: Joint W(u) (ω + κ)

**Theoretical claim:** One fitness function governs both channels. Individual differences in capture cost (ω) and effort cost (κ) jointly determine patch selection and motor vigor.

**Choice + Vigor:** Both from W(u) = S(u)·R − (1−S(u))·ω_i·(R_j+C) − κ_i·(u−req)²·D.
**Per-subject:** ω and κ (both enter both channels).
**What it tests:** Is the full joint optimization the best description?

---

## Sub-hypotheses

### H3a: M5 outperforms M1 — threat shapes choice beyond effort

**Prediction:** ΔBIC(M1 − M5) > 0.

### H3b: M5 outperforms M2 — individual effort differences matter

**Prediction:** ΔBIC(M2 − M5) > 0.

### H3c: M5 outperforms M3 — capture cost and effort cost are separable

**Prediction:** ΔBIC(M3 − M5) > 0.

---

## Vigor Data

Vigor observations are per-subject condition cell means (subject × threat × distance × cookie type). Each cell averages ~4.4 trials, denoising motor variability while preserving condition structure. Cell means are weighted by √n in the likelihood (σ_v/√n). Total: ~5,215 cell-mean observations across 290 subjects.

## Survival Function

S(u, T, D) = exp(−h·T^γ·D / speed(u)), where speed(u) = sigmoid((u − 0.25·req) / σ_sp).

Speed saturates above the required pressing rate, reflecting the task mechanic: pressing above full-speed threshold doesn't increase movement speed. This saturation gives κ leverage — the effort cost is the primary determinant of pressing above req.

## Exploratory Benchmarks (Discovery Sample, N=290)

| Metric | M5 (joint) |
|--------|-----------|
| Choice acc | 0.779 |
| Choice r² | 0.894 |
| Vigor r² (cell means) | 0.386 |
| ω recovery | r = 0.90 |
| κ recovery | r = 0.78 |
| ω↔κ | r = 0.27 |

Parameter dissociation:
- ω → choice: r = −0.68 (primary avoidance)
- κ → choice: r = −0.36 (effort-based avoidance)
- κ → mean vigor: r = −0.38 (pressing intensity)
- ω → mean vigor: r = +0.08 (null)
