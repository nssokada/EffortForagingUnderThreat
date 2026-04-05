# H4: Foraging Profiles and Optimality

## Preregistered prediction

The two model parameters define ecologically meaningful foraging profiles: avoidance sensitivity (omega) predicts who survives and what errors they make, activation intensity (kappa) predicts motor output, and the balance between them (omega-kappa angle) predicts decision quality.

## Tests and thresholds

| Test | Model | Threshold |
|------|-------|-----------|
| H4a: ω → escape rate | escape_rate ~ omega_z + kappa_z | 95% HDI excludes zero, positive |
| H4b: ω → overcaution ratio | overcaution_ratio ~ omega_z | 95% HDI excludes zero, positive |
| H4c: κ → mean vigor | mean_vigor ~ kappa_z | 95% HDI excludes zero, negative |
| H4d: angle → optimality | pct_optimal ~ angle_z | 95% HDI excludes zero, negative |
| H4e: consistency → earnings | earnings ~ choice_consistency_z + intensity_deviation_z | Both 95% HDIs exclude zero |

All regressions: Bayesian linear models (bambi; 4 chains × 2,000 draws + 1,000 tuning).

## Results

### H4a: Avoidance sensitivity predicts survival

Escape rate on attack trials regressed on omega_z and kappa_z:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(omega) | +0.060 [+0.029, +0.093] | +0.046 [+0.017, +0.075] |
| β(kappa) | −0.003 [−0.033, +0.029] | +0.003 [−0.028, +0.030] |

Omega predicts escape; kappa does not. People who perceive capture as more costly adopt strategies (more cautious choices, faster pressing) that increase survival on attack trials. **Confirmed in both samples.**

### H4b: Overcaution is the dominant error, driven by omega

Among suboptimal choices, overcautious errors (choosing light when heavy has higher expected reward) dominate:
- Exploratory: 79% overcautious
- Confirmatory: 90% overcautious

Omega predicts the overcaution ratio:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(omega → OC) | +0.177 [+0.163, +0.193] | +0.123 [+0.109, +0.137] |

High-omega individuals systematically avoid the risky option even when it is optimal for reward. The error is directional — they are too cautious, not too reckless. **Confirmed in both samples.**

### H4c: Effort cost governs motor output

Mean vigor (average normalized press rate across conditions) regressed on kappa_z:

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(kappa) | −0.194 [−0.215, −0.173] | −0.196 [−0.217, −0.176] |

The effect is large, precise, and nearly identical across samples. Kappa governs motor output — the activation side of the avoid-activate decomposition. **Confirmed in both samples.**

### H4d: Effort-driven avoidance is less optimal

Percent optimal choices regressed on the omega-kappa angle (atan2(kappa_z, omega_z)):

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(angle) | −0.041 [−0.055, −0.026] | −0.054 [−0.072, −0.036] |

Higher angle (more effort-driven relative to threat-driven avoidance) predicts worse decision quality. Threat-sensitive avoidance is context-appropriate — you avoid when it is dangerous. Effort-sensitive avoidance is indiscriminate — you avoid the hard option regardless of threat level. **Confirmed in both samples.**

### H4e: Model consistency → earnings (NOT CONFIRMED)

Earnings regressed on choice consistency (fraction matching model prediction) and intensity deviation (RMSE from model-predicted vigor):

|  | Exploratory | Confirmatory |
|--|-------------|--------------|
| β(choice consistency) | +14.3 [+5.0, +23.2] | +8.4 [−2.3, +19.0] |
| β(intensity deviation) | −19.3 [−28.8, −9.4] | −4.1 [−14.6, +7.4] |

Both effects were significant in the exploratory sample but neither replicated. The indirect link between computational consistency and total earnings is weaker than the direct parameter-outcome relationships tested in H4a–d. **Not confirmed in confirmatory sample.**

## Summary

| Test | Exploratory | Confirmatory |
|------|-------------|--------------|
| H4a: ω → escape | PASS | PASS |
| H4b: overcaution % | 79% | 90% |
| H4b: ω → OC ratio | PASS | PASS |
| H4c: κ → vigor | PASS | PASS |
| H4d: angle → optimality | PASS | PASS |
| H4e: choice cons → earnings | PASS | FAIL |
| H4e: intensity dev → earnings | PASS | FAIL |
| **Total** | **7/7** | **5/7** |

## Interpretation

The core avoid-activate decomposition replicates cleanly: omega governs avoidance (who escapes, who is overcautious) and kappa governs activation (how hard you press). Their balance predicts decision quality — people who avoid for the wrong reason (effort aversion rather than threat sensitivity) make worse choices. The only failure is H4e, which tested whether being more consistent with one's own fitness function predicts earnings. This is the most indirect prediction — it requires the model to be correct at the individual level for each person, not just at the group level — and the effect was too small to replicate at N = 281.
