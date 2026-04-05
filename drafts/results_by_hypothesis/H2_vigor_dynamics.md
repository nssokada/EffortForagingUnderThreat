# H2: Vigor Dynamics Follow the Predatory Imminence Continuum

## Preregistered prediction

Motor vigor will show distinct anticipatory (threat-modulated, graded) and reactive (encounter-triggered, all-or-nothing) dynamics, consistent with the predatory imminence continuum (Fanselow 1994; Mobbs et al. 2020).

## Tests and thresholds

| Test | Model | Threshold |
|------|-------|-----------|
| H2a: Encounter spike | One-sample t vs 0 (attack − non-attack reactive epoch) | P < .001, d > 0.20 |
| H2b: Encounter temporal signature | GAM LRT (spline × encounter interaction) | P < .01 |
| H2b: Threat temporal signature | GAM LRT (spline × threat interaction) | P < .01 |

## Results

### H2a: Encounter spike

Per-subject mean reactive-epoch pressing rate on attack minus non-attack trials:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| Mean spike | +0.036 | +0.036 |
| d | +0.56 | +0.65 |
| t | 9.57 | 11.01 |
| P | < 0.001 | < 0.001 |
| % subjects positive | 83% | 83% |

The spike is large (d > 0.55), reliable (>80% of participants), and nearly identical across samples. **Confirmed in both samples.**

**Threat independence:** The spike magnitude did not differ between T = 0.9 and T = 0.1 (confirmatory: t = −1.42, P = 0.16; exploratory: P = 0.21). This is consistent with a reflexive post-encounter defense response that does not scale with prior threat probability — once the predator appears, the motor system responds maximally regardless of expectation.

### H2b: GAM temporal signatures

Generalized additive models with natural cubic regression splines (K = 10), fitted via MixedLM with cookie covariate and random intercepts. Likelihood ratio tests for smooth-by-condition interactions:

| Interaction | Exploratory χ² | Exploratory P | Confirmatory χ² | Confirmatory P |
|-------------|---------------|---------------|-----------------|----------------|
| Encounter (attack vs non-attack) | 760 | < 0.001 | 1,025 | < 0.001 |
| Threat (T = 0.9 vs T = 0.1) | 292 | < 0.001 | 115 | < 0.001 |

The vigor timecourse differs qualitatively by encounter status and by threat level. The encounter effect is larger (χ² = 760–1,025) than the threat effect (χ² = 115–292), consistent with the imminence continuum's prediction that reactive defense is a stronger motor driver than anticipatory threat assessment. **Confirmed in both samples.**

## Summary

| Test | Exploratory | Confirmatory |
|------|-------------|--------------|
| H2a: Encounter spike | PASS (d = 0.56) | PASS (d = 0.65) |
| H2b: Encounter GAM | PASS (χ² = 760) | PASS (χ² = 1,025) |
| H2b: Threat GAM | PASS (χ² = 292) | PASS (χ² = 115) |
| **Total** | **3/3** | **3/3** |

## Interpretation

Vigor dynamics follow the predatory imminence continuum: anticipatory vigor is graded by threat probability (pre-encounter defense), while the encounter spike is a threshold response that fires regardless of prior threat level (post-encounter defense). This dissociation — strategic modulation before encounter, reflexive activation after — mirrors the freezing-to-flight transition observed in animal models (Fanselow 1994) but expressed through motor vigor rather than locomotive behavior. The encounter spike represents a defensive mobilization response that is robust across individuals (83% positive) and stable across independent samples.
