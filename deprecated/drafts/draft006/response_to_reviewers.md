# Point-by-Point Response to Reviewers

**Manuscript:** "Metacognitive bias, not threat computation, bridges foraging decisions to clinical anxiety"
**Revision:** Draft 005 → Draft 006

---

## Editor — Critical Issues

### Issue 1: c_d "cancellation" → collinearity/identifiability argument

**Editor/R1-MC1:** R1 demonstrated that c_d does not algebraically cancel from the choice comparison. The paper must rewrite this as an identifiability argument.

**Response:** We agree completely with R1's derivation. The fixed penalty C cancels between options, but the residual c_d term scales with (R_H - R_L) and is therefore collinear with the reward term — not zero. We have rewritten all mentions of "cancellation" throughout the manuscript. The Results now include the full derivation:

> "c_d is excluded from the choice equation because its contribution to the option differential is collinear with the reward term, making it empirically unidentifiable from choice data alone. In the full expected utility, the capture-loss term for option i is -(1-S) × c_d × (R_i + C); because the fixed penalty C cancels in the difference between options, the residual c_d term scales with the same (R_H - R_L) factor as the reward term, rendering the two inseparable."

The Discussion now explicitly acknowledges this as "a deliberate simplifying assumption" and notes that "this exclusion means the model may misattribute some c_d-driven choice variation to c_e."

**Location:** Results (paragraph on three features), Discussion (identification paragraph), Methods (choice model).

---

### Issue 2: "Metacognition" → justification added (Option B)

**Editor/R2-M1:** R2 argues this is affective accuracy/bias, not metacognition in the Fleming/Dolan or Wells sense.

**Response:** We retain "metacognitive" but add a substantial justification paragraph in the Discussion. We explicitly acknowledge that this is not metacognition in the narrow sense of second-order confidence judgments (Fleming & Dolan, 2012) or meta-worry (Wells, 2009), but argue that calibration and discrepancy capture a second-order property: the person's *relationship to* their computation, not the computation itself. We note that "affective calibration" and "affective bias" are equally valid descriptors and that our usage is closer to Paulus and Stein's interoceptive prediction error framework. We also acknowledge R2's point about the Paulus and Stein connection being analogical rather than literal: our measure is a residual between self-reported affect and model-derived danger, not a mismatch between predicted and actual body states.

The Results section heading has been changed from "Metacognitive calibration predicts performance; discrepancy predicts clinical symptoms" to "Affective calibration predicts performance; discrepancy predicts anxiety symptoms" to signal that the constructs are primarily affective. The metacognitive interpretation is developed in the Discussion.

**Location:** Discussion ("Affective calibration as the bridge..."), Results heading, throughout.

---

### Issue 3: "LQR" → "LQR-inspired"

**Editor/R1-MC2:** R1 notes no state dynamics or Riccati equation.

**Response:** We have replaced "LQR cost structure" with "LQR-inspired cost structure" or "a cost structure motivated by linear-quadratic optimal control" throughout. We added an explicit acknowledgment: "We note that this is an analogy to LQR optimal control, not a formal implementation: our model has no state dynamics, no feedback law, and no Riccati equation. The quadratic cost structure is motivated by LQR theory but the optimization is static."

**Location:** Abstract, Introduction, Results, Discussion, Methods, Figure legends.

---

### Issue 4: Ecological framing moderated

**Editor/R3-MC1:** R3 identifies fundamental departures from foraging ecology.

**Response:** We added a scoping paragraph in the Introduction:

> "Our task captures the core foraging-under-threat trade-off — choosing between options that differ in reward, effort, and risk — but abstracts away several features of natural foraging ecology. There are no patch dynamics or marginal value calculations (cf. Charnov), no energy state or metabolic urgency (cf. McNamara & Houston), and threat probabilities are stated rather than learned. We view this as a reductionist probe of the effort-threat integration problem — isolating the computational structure that governs choice and vigor under parametric threat — rather than a complete model of ecological foraging."

The Discussion heading has been changed from "An EVC framework for physical effort under ecological threat" to "An EVC framework for physical effort under threat."

**Location:** Introduction (new paragraph after EVC framework description), Discussion heading.

---

### Issue 5: Causality and shared method variance

**Editor/R2-M4:** Discrepancy is computed from anxiety ratings; clinical outcomes are also self-report anxiety. Shared method variance concern.

**Response:** We added a new constraint paragraph in the "Associations with symptom severity" section:

> "The correlation between discrepancy and clinical measures may be partly inflated by shared method variance, as both involve self-reported anxiety... However, the specificity of the association argues against pure method artifact: calibration — also computed from anxiety ratings — does not predict symptom severity (6 of 7 measures p > .10), and discrepancy predicts not only anxiety measures but also depression (DASS-Depression, PHQ-9) and apathy (AMI), which are conceptually and methodologically distinct from the task anxiety ratings."

We also explicitly note that "we cannot rule out the possibility that discrepancy partly indexes a general negative affectivity or neuroticism factor."

**Location:** Discussion, third constraint paragraph.

---

### Issue 6: "Double dissociation" → "differential prediction pattern"

**Editor/R1-MC4, R2-M2:** The leakage (calibration→STAI, discrepancy→survival) precludes a true double dissociation.

**Response:** We replaced "predominant double dissociation" with "differential prediction pattern" throughout. The Discussion now states: "We characterize this as a differential prediction pattern rather than a double dissociation in the strict neuropsychological sense, because the separation is not absolute" and explicitly notes the leakage that "a true double dissociation would preclude."

**Location:** Abstract, Introduction, Results, Discussion.

---

## Additional Issues

### Issue 7: SVI approximation acknowledged

**R1-MC3, R2-m1, R3-4.4:** SVI with mean-field guide underestimates posterior uncertainty.

**Response:** Added to Methods:

> "We used SVI with a mean-field approximation, which provides point estimates of posterior means but may underestimate posterior uncertainty due to the assumption of posterior independence between parameters. HDIs and ROPE analyses reported for the clinical regressions (which use full MCMC via bambi/PyMC) are not affected by this approximation, but the population-level parameter estimates (γ, ε, τ) from the main model should be interpreted as approximate posterior modes rather than full posteriors."

We also added a note about the non-standard BIC computation: "Using the ELBO loss in place of the log-likelihood for BIC computation is non-standard, as the ELBO is a lower bound on the marginal likelihood. This may affect model comparison... Future work should supplement BIC with WAIC or LOO-CV."

**Location:** Methods, Model fitting.

---

### Issue 8: Population ε limitation expanded

**R1-MC5:** Population ε masks individual differences in effort-efficacy beliefs.

**Response:** Expanded the Limitations paragraph on ε:

> "Population-level ε means our model assumes everyone shares the same (very low, ε = 0.098) belief about effort's survival value. If some participants actually believe effort helps survival more than others, their 'excess anxiety' on high-effort trials might be rational rather than biased, and individual differences in effort-efficacy beliefs may be an important source of variation in both vigor and discrepancy that our model cannot capture."

**Location:** Limitations.

---

### Issue 9: "Clinical anxiety" → "anxiety symptoms"; non-clinical sample noted

**R2-M6:** Non-clinical sample limits clinical generalizability.

**Response:** We replaced "clinical symptoms" with "anxiety symptoms" or "symptom severity" throughout where appropriate. The Limitations now state: "All references to 'anxiety symptoms' throughout this paper refer to dimensional variation on validated symptom scales, not clinical diagnoses." We also added: "Effect sizes observed in non-clinical samples may not generalize to clinical populations due to floor/ceiling effects, medication, or qualitative differences in threat processing." The section heading was changed from "Clinical implications" to "Associations with symptom severity and effect-size honesty."

**Location:** Throughout (Abstract, Results, Discussion, Limitations).

---

### Issue 10: Reference 23 vs 33

**R1-m7, R2-m5, R3-minor1:** Reference 23 (COVID risk perception) is cited where reference 33 (interactive threat) is intended.

**Response:** The erroneous citation in the Discussion has been corrected to reference 33.

**Location:** Discussion, calibration-performance paragraph.

---

### Issue 11: Behavioral profiles moved to supplementary

**R1, R3:** Behavioral profiles section disrupts narrative flow.

**Response:** The behavioral profiles section has been condensed and the detailed median-split typology moved to Supplementary Table S2. The key finding (Vigilant profile earns most points) is integrated into a shorter Discussion paragraph ("The dissociation of choice and vigor").

**Location:** Discussion (condensed), Supplementary Table S2 (new).

---

### Issue 12: ROPE interpretation refined

**R2-m8:** ROPE containment of 53-56% is ambiguous, not evidence for the null.

**Response:** We added to the Table 2 note: "High percentage in ROPE (>80%) provides Bayesian evidence for the null; values in the 53–60% range are better characterized as inconclusive." The main text now distinguishes between high-ROPE cases (positive null evidence) and low-ROPE cases (inconclusive).

**Location:** Table 2 note, Results paragraph.

---

### Issue 13: Effort cost derivation spelled out

**R1-m4:** The constants 0.81 and 0.16 are not explained.

**Response:** Added parenthetical: "(specifically, 0.9² × D_H - 0.4² × 1 = 0.81D_H - 0.16)."

**Location:** Results, choice model paragraph.

---

## Issues Noted for Future Work (Not Fully Addressed in This Revision)

The following reviewer concerns are acknowledged as important but require additional data or analyses beyond the scope of this revision:

- **Confirmatory sample (R1-MC6, R2-m6, R3-4.5):** Data collection is underway. Results will be added before final publication.
- **Full MCMC fit (R1-MC3, R3-4.4):** We acknowledge the SVI limitation and plan MCMC supplementary fits for the confirmatory analysis.
- **WAIC/LOO-CV model comparison (R1-MC3):** Will be implemented for the confirmatory analysis.
- **Decomposition of vigor R² into between/within (R1-m3, R1-Q6):** Planned for revision with confirmatory data.
- **Calibration reliability (R2-m3, R3-minor3):** Split-half reliability will be reported in the next revision.
- **Learning analysis (R3-minor9):** Block-level parameter stability analysis planned.
- **Discrepancy vs. raw mean anxiety (R2-Q1):** Will test whether model-based decomposition adds value over raw affect.
- **Confidence-discrepancy analysis (R2-Q5):** Will compute and test analogous confidence-discrepancy measure.

---

## Summary of Changes

| Change | Scope | Sections affected |
|--------|-------|-------------------|
| c_d "cancellation" → identifiability/collinearity | Major rewrite | Results, Discussion, Methods |
| Metacognition justification paragraph added | New paragraph | Discussion |
| LQR → LQR-inspired throughout | Terminology | All sections |
| Ecological scoping paragraph added | New paragraph | Introduction |
| Shared method variance discussed | New paragraph | Discussion |
| "Double dissociation" → "differential prediction" | Terminology | All sections |
| SVI approximation acknowledged | New sentences | Methods |
| Population ε limitation expanded | Expanded paragraph | Limitations |
| "Clinical anxiety" → "anxiety symptoms" | Terminology | Throughout |
| Reference 23 → 33 corrected | Citation fix | Discussion |
| Behavioral profiles → supplementary | Restructured | Discussion, Supplementary |
| ROPE interpretation refined | Clarification | Table 2, Results |
| Effort cost constants explained | Clarification | Results |
| New references added (38, 39) | References | Reference list |
