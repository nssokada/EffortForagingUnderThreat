# Editorial Decision

**Manuscript:** "Metacognitive bias, not threat computation, bridges foraging decisions to clinical anxiety"
**Decision: Major Revisions**

---

## Summary of Reviews

All three reviewers agree that the paper addresses a genuine gap and makes a potentially meaningful contribution. The joint choice-vigor model, the Simpson's paradox demonstration, and the calibration-discrepancy decomposition are recognized as strengths. However, all three recommend Major Revisions with substantial concerns about framing, mathematical claims, and terminology.

---

## Critical Issues Requiring Resolution

### 1. The cd "cancellation" claim is mathematically imprecise (R1-MC1)
R1 demonstrates that cd does not algebraically cancel from the choice comparison — it is collinear with the reward differential but not zero. The paper omits cd from the choice equation and calls this "cancellation," but it is more accurately described as a deliberate modeling choice to exclude a collinear term. **The paper must rewrite this as an identifiability argument, not a mathematical cancellation.**

### 2. "Metacognition" terminology is overloaded (R2-M1)
R2 argues convincingly that calibration and discrepancy measure affective accuracy and affective bias, not metacognition in the Fleming & Dolan or Wells sense. The paper should either (a) reframe as "affective calibration" throughout, or (b) provide a rigorous justification for why affect-computation alignment constitutes metacognition. The current Wells citation is strained.

### 3. The "LQR" label is aspirational (R1-MC2)
R1 notes there are no state dynamics, feedback law, or Riccati equation. The model has two different quadratic cost terms, not an LQR controller. **Either drop the LQR terminology or explicitly acknowledge it as an analogy, not a formal implementation.**

### 4. Ecological framing is overstated (R3-MC1,2,3)
R3 identifies fundamental departures from foraging ecology: no patch dynamics, no state dependence, no energy budget, explicit threat probabilities. **Acknowledge these limitations more prominently and moderate the ecological claims.**

### 5. Causality and shared method variance (R2-M4)
Discrepancy is computed from anxiety ratings; clinical outcomes are also anxiety self-report. The correlation may reflect shared method variance. **Discuss this explicitly and note that the direction of causality is ambiguous.**

### 6. "Double dissociation" claim needs qualification (R1-MC4, R2-M2)
The leakage (calibration→STAI, discrepancy→survival) is acknowledged but not formally tested. **Either run a formal interaction test or replace "predominant double dissociation" with "differential prediction pattern."**

---

## Additional Issues to Address

7. SVI vs MCMC — acknowledge the approximation and its implications for HDI/ROPE (R1-MC3)
8. Population ε weakens the survival interpretation (R1-MC5) — discuss what is lost
9. Non-clinical sample limits clinical generalizability (R2-M6, R3) — moderate claims
10. Discrepancy may index general negative affectivity (R2-M3) — test against a general factor
11. Calibration reliability with n=18 trials (R2-minor) — report reliability estimates
12. Missing confirmatory sample (R1-MC6, R2) — note status clearly
13. Reference 23 vs 33 (R1, R2) — fix citation
14. Behavioral profiles section disrupts narrative (R1, R3) — consider supplementary

---

## Instructions to Authors

Please revise the manuscript addressing all critical issues (1-6) and as many additional issues (7-14) as feasible. The revision should:

1. Replace "cancellation" language with identifiability/collinearity argument
2. Reframe metacognition terminology (consider "affective calibration/discrepancy")
3. Moderate the LQR terminology to "LQR-inspired" or similar
4. Temper ecological framing and acknowledge departures from foraging theory
5. Add explicit discussion of causality and shared method variance
6. Qualify the double dissociation claim with appropriate statistical tests or softer language

The core finding — that the computational parameters predict behavior while the affective mismatch (discrepancy) predicts clinical symptoms — is robust and novel. The revision should ensure the framing matches the evidence.
