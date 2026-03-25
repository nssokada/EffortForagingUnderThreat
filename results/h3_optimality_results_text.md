# H3 Results: Optimality of Reallocation

## Overview

H3 tests whether the threat-driven reallocation (shifting choice toward safety while increasing motor effort) approximates a rational policy. N = 293.

---

## H3a: Reallocation predicts earnings

Reallocation index = |Δchoice|_z + |Δvigor|_z (sum of z-scored absolute shifts).

Pearson r(reallocation_index, total_earnings) = **0.577**, p < 0.001 (one-tailed).

| Measure | Mean | SD |
|---|---|---|
| Reallocation index | 0.000 | 1.738 |
| Total earnings (pts) | 6.9 | 88.6 |

Criterion: r > 0, p < 0.01 (one-tailed). **H3a: SUPPORTED.**

---

## H3b: Dominant deviation is excessive caution

Expected values computed using empirical escape rates from 11844 attack trials, conditioned on threat and chosen distance. EV = (1−T)R + T[P_esc·R − (1−P_esc)·C], where C = 5 (capture cost). This reflects actual task dynamics including effort/speed effects on survival, rather than model-derived S.

| Category | N trials | % |
|---|---|---|
| Optimal | 7,982 | 60.5% |
| Too cautious | 4,886 | 37.1% |
| Too risky | 317 | 2.4% |

Among suboptimal trials: **93.9%** too cautious vs **6.1%** too risky.

Per-subject mean % cautious errors: 91.0% (SD = 16.6%).
One-sample t-test vs 50%: t = 42.25, p < 0.001 (one-tailed).

Criterion: > 50%, p < 0.05. **H3b: SUPPORTED.**

---

## Summary

| Sub-hypothesis | Statistic | p (one-tailed) | Criterion | Result |
|---|---|---|---|---|
| H3a (reallocation → earnings) | r = 0.577 | p < 0.001 | r > 0, p < 0.01 | PASS |
| H3b (cautious > 50%) | 91.0%, t = 42.25 | p < 0.001 | > 50%, p < 0.05 | PASS |

**H3 overall: SUPPORTED.**
