# Choice-Vigor Dissociation: Analysis Plan

## The Central Claim

Decision-making and action execution are dissociable behavioral domains in effort-threat foraging. People who choose high-effort options do not necessarily exert high motor effort during execution (r = −0.02, p = 0.76). This independence is not noise — it is structured by the computational parameters that govern choice, it is modulated by threat context, and it predicts real survival outcomes. Specifically:

- **k (effort discounting)** drives overall effort willingness across both domains (the diagonal axis)
- **β (threat bias)** selectively suppresses choice without constraining motor execution (the off-diagonal axis)
- **Vigor, not choice, determines who survives predator attacks** (vigor β = +0.80 vs choice β = −0.16 for escape prediction)

---

## Why This Matters

Most computational models of effort-based decision-making assume that valuation and execution are coupled — that the same computation that determines what you choose also determines how vigorously you act. The vigor literature (Manohar et al., Shadmehr et al.) explicitly links subjective value to movement vigor. Our finding challenges this: in a threat-foraging context, the value computation (governed by k, z, β) strongly predicts what people decide (R² = 0.82) but only weakly predicts how hard they press (R² = 0.08). The dissociation has practical consequences — the HL profile (ambitious choice, weak execution) is the worst strategy in the task, yet these individuals are the most confident.

---

## Potential Pitfalls

### 1. The independence could be an artifact of the vigor measure
**Risk:** If our vigor measure (pre-encounter press rate, capacity-normalized, choice-ratio-adjusted) doesn't capture true motor effort, the null correlation with choice could be a measurement problem rather than a real dissociation.
**Mitigation:** Check split-half reliability of the vigor measure. If it's reliable (r > 0.7, which we expect given ICC = 0.71 for onset), then the null choice-vigor correlation is a real property of the data, not noise. Also check with alternative vigor operationalizations (post-encounter rate, total presses, raw rate without normalization) to ensure the independence isn't an artifact of one specific measure.
**Exit criterion:** If vigor split-half < 0.5, the measure is too noisy to support dissociation claims. Abandon this framing.

### 2. The choice-vigor independence is trivially expected
**Risk:** A reviewer might argue: "Of course choice and pressing rate are uncorrelated — choice depends on option values while pressing rate depends on motor ability. This is not a deep finding."
**Mitigation:** Show that the independence is *structured* — it's not just that choice and vigor are unrelated, but that specific computational parameters (β) create the dissociation in a predictable way. Also show it's threat-dependent (r = +0.20 at low threat → r = −0.22 at high threat), which a "trivially separate systems" account doesn't predict. The context-dependence is the key rebuttal.
**Exit criterion:** If the threat reversal doesn't replicate in confirmatory sample, the structured dissociation claim weakens substantially.

### 3. β may be poorly identified or confounded with k
**Risk:** β and k both suppress choice of the high-effort option. If they're not independently identifiable in the model, the "β drives dissociation" claim is on shaky ground.
**Mitigation:** Check parameter recovery — can the model distinguish k from β in simulated data? Check the posterior correlation between k and β (r = +0.14, which is low — good). Verify that β's selective effect on choice (not vigor) holds in partial correlations controlling for k (already confirmed: β|k → choice r = −0.38, β|k → vigor r = +0.13).
**Exit criterion:** If β and k have posterior correlation > 0.5, the dissociation claim about β specifically becomes unreliable.

### 4. The quadrant framing is a median split (statistically weak)
**Risk:** Median splits are widely criticized — they discard continuous information, create arbitrary groups, and inflate Type I error.
**Mitigation:** Use quadrants only for visualization and intuition. All statistical tests should use continuous measures: multiple regression, CCA, moderated mediation. The quadrant plots make the story accessible; the continuous statistics make it rigorous.
**Exit criterion:** Not an exit issue if we use continuous stats. But if the continuous analyses don't support what the quadrants show, abandon the quadrant framing.

### 5. N = 293 may be underpowered for the multivariate analyses
**Risk:** CCA, moderated mediation, and cross-level interactions require adequate power. With 3 predictors and 2 outcomes, N = 293 is reasonable but not large.
**Mitigation:** The confirmatory sample (N = 350) provides a replication. Combined N = 643 is strong. Also, the effect sizes are large (k → choice r = −0.78, escape ANOVA F = 115) — power is not the concern for the primary findings. It's the subtler effects (β → vigor r = +0.11, threat reversal r = ±0.20) that need larger N.
**Exit criterion:** If key effects don't replicate at N = 350, reconsider whether they're real.

### 6. The outcome analysis may be circular
**Risk:** Vigor predicts escape rate — but vigor IS the action that determines escape. Pressing harder → move faster → escape predator. This is mechanics, not psychology.
**Mitigation:** Acknowledge the mechanical link explicitly. The interesting finding is not that pressing harder helps you escape (obvious) but that: (a) choice doesn't help — choosing hard cookies actually hurts escape; (b) the computational model captures choice (R² = 0.82) but not vigor (R² = 0.08), so it's missing the behavioral channel that actually matters; (c) the profiles that look "good" on choice (HL) are the worst at surviving.
**Exit criterion:** This isn't an exit issue — just needs honest framing.

### 7. Psychiatric correlates are weak
**Risk:** The quadrant differences in PHQ-9 and DASS-Depression are significant (p = 0.01-0.02) but the LL group driving the effect is the "disengaged" group, not a clinical profile. Reviewers may not find this compelling.
**Mitigation:** Frame psychiatric findings as exploratory, not primary. The main story is computational (params → choice/vigor dissociation → outcomes). Psychiatric correlates are supplementary and hypothesis-generating.
**Exit criterion:** Don't lean on psychiatric findings as a primary result. If they don't replicate, drop them.

---

## Opportunities

### 1. β as a "decision-specific threat signal" is novel
No prior work (to my knowledge) has shown that a computational parameter governing choice under threat selectively fails to predict motor execution. This challenges the common assumption in vigor/effort models that subjective value → movement vigor. β shows that threat bias enters the decision stage but not the action stage.

### 2. The threat reversal is a strong experimental manipulation
The shift from r = +0.20 to r = −0.22 across threat levels is a within-task, within-subject demonstration that the choice-vigor coupling is context-dependent. This isn't a between-group comparison — it's the same people showing different coupling as threat changes. That's powerful.

### 3. The maladaptive HL profile connects to overconfidence literature
HL people choose ambitiously (high p_high), report high confidence (3.40), but escape only 19% of attacks. This maps onto overconfidence in decision-making — they overestimate their ability to execute. The computational model shows why: low β (not threat-biased) → choose risky, low vigor → can't follow through.

### 4. The LH "adaptive apathy" finding challenges clinical assumptions
LH people report the highest apathy (AMI = 32.2) but have the best escape rate (60%) and earn well (+71.4). In clinical settings, high AMI would flag concern. Here, it reflects adaptive selectivity — these people don't waste effort choosing hard when they don't need to, but they press hard when it matters. This reframes apathy as potentially strategic in threat contexts.

### 5. Replication opportunity is already available
The confirmatory N = 350 is sitting unprocessed. If the dissociation replicates, that's extremely strong for Nature Comms — exploratory + confirmatory in a single paper.

### 6. The dissociation bridges multiple literatures
- **Effort-based decision-making** (Westbrook, Pessiglione): choice under effort
- **Vigor and motor control** (Shadmehr, Manohar): execution energy
- **Threat imminence** (Fanselow, Mobbs): defensive behavior under threat
- **Computational psychiatry** (Huys, Daw): model parameters and clinical dimensions

Few papers span all four. The dissociation is the connective tissue.

---

## Analysis Plan

### Phase 0: Validation checks (go/no-go gates)

**Step 0.1: Vigor measure reliability**
- Split-half reliability of mean pre-encounter pressing rate
- ICC across trial blocks (first half vs second half of experiment)
- **Gate:** Split-half r > 0.5 for vigor measure. If not, the vigor measure is too noisy.

**Step 0.2: Choice-vigor independence is robust to operationalization**
- Test r(choice, vigor) with:
  - (a) Pre-encounter rate (primary)
  - (b) Post-encounter rate
  - (c) Total presses per trial (unnormalized)
  - (d) Raw rate without capacity normalization
  - (e) Without choice-ratio adjustment
- **Gate:** Independence (|r| < 0.10) holds for at least 3 of 5 operationalizations. If choice-vigor are correlated under most measures, the dissociation is an artifact of our normalization.

**Step 0.3: Parameter identifiability**
- Posterior correlation between k and β from MCMC samples
- Parameter recovery simulation: generate data with known k, β, fit model, check recovery
- **Gate:** Posterior k-β correlation < 0.5. Recovery R² > 0.7 for both k and β.

### Phase 1: Characterize the 2D choice-vigor space (descriptive)

**Step 1.1: Establish independence**
- Pearson r(choice_z, vigor_z) with CI
- Quadrant population counts and chi-square test for uniform distribution
- Visualization: scatter plot colored by quadrant, by β, by k

**Step 1.2: What predicts position?**
- Multiple regression: choice_z ~ z + k + β (expect R² ≈ 0.82)
- Multiple regression: vigor_z ~ z + k + β (expect R² ≈ 0.08)
- The asymmetry in R² is a core finding — report it directly

**Step 1.3: Partial correlations**
- Each param → choice, controlling for the other two
- Each param → vigor, controlling for the other two
- Key contrast: β → choice (significant) vs β → vigor (null)

**Step 1.4: Diagonal / off-diagonal decomposition**
- Compute diagonal axis (choice_z + vigor_z) and off-diagonal axis (vigor_z − choice_z)
- Correlate with each param
- Report which param drives which axis

### Phase 2: Formal statistical tests (confirmatory-grade)

**Step 2.1: Canonical Correlation Analysis (CCA)**
- X = {z, k, β}, Y = {choice_z, vigor_z}
- Test: how many significant canonical dimensions?
- If 2 dimensions: the param space maps onto behavior through two independent pathways (one for choice, one for vigor)
- Report canonical correlations, loadings, and Wilks' lambda

**Step 2.2: Multivariate regression with bootstrap**
- Joint DV: [choice_z, vigor_z]
- Predictors: z, k, β
- Bootstrap 10,000 iterations for coefficient CIs
- Key test: β → choice CI excludes zero, β → vigor CI includes zero
- Also: k → choice and k → vigor CIs — does k significantly predict both?

**Step 2.3: Interaction test for β's selective effect**
- Formally: is the β → choice coefficient significantly different from the β → vigor coefficient?
- Test via seemingly unrelated regression (SUR) or multivariate regression with equality constraint
- H0: β_choice = β_vigor. If rejected → β has a significantly stronger effect on choice than vigor

### Phase 3: Threat modulation (the experimental manipulation)

**Step 3.1: Trial-level LMM for threat-dependent coupling**
- Model: `vigor_trial ~ choice_subj_z * threat_z + distance_z + (1|subj)`
- choice_subj_z is the subject's mean choice tendency (between-subject)
- threat_z is trial-level threat (within-subject)
- The cross-level interaction choice_subj_z × threat_z tests whether the choice-vigor coupling changes with threat
- Expect: negative interaction (high-choice people press relatively less under high threat, or low-choice people press more)

**Step 3.2: Per-threat correlations (visualization)**
- r(choice, vigor) at each threat level: 0.1, 0.5, 0.9
- Plot with CIs
- This is the intuitive version of Step 3.1

**Step 3.3: β mediates the threat reversal**
- At high threat, β's effect on choice is amplified (β enters through the survival function, which is more extreme at high threat)
- But β's null effect on vigor doesn't change with threat
- So the choice-vigor coupling reverses because choice becomes more β-driven while vigor doesn't
- Test: does the β → choice path strengthen at high threat? (Moderation analysis)

### Phase 4: Outcome prediction (functional consequences)

**Step 4.1: Trial-level escape prediction**
- LMM: `escaped ~ vigor_z + choice_z + threat_z + distance_z + (1|subj)`
- Report vigor and choice coefficients
- Key: vigor significantly predicts escape; choice does not (or goes negative)

**Step 4.2: Subject-level earnings**
- OLS: earnings ~ choice_z + vigor_z + choice_z × vigor_z
- Report R² and coefficients
- Visualize by quadrant

**Step 4.3: Profile-specific outcomes**
- For each quadrant: mean escape rate, earnings, with CIs
- Pairwise comparisons: HH vs HL (same choice, different vigor), LH vs LL (same choice, different vigor)
- These hold choice constant and vary vigor — cleanest test of vigor's contribution

### Phase 5: Affect and metacognition

**Step 5.1: Confidence miscalibration**
- For each subject: compute confidence bias = mean confidence − actual escape rate
- Correlate with choice_z and vigor_z
- Expect: choice_z → positive confidence bias (choosers are overconfident), vigor_z → negative bias (pressers are underconfident or calibrated)

**Step 5.2: Anxiety calibration by profile**
- Use existing per-subject calibration (Spearman r between threat and anxiety rating)
- Test: does vigor predict calibration accuracy beyond what choice predicts?

**Step 5.3: Affect as a function of the choice-vigor space**
- trait_anx ~ choice_z + vigor_z + choice_z × vigor_z
- trait_conf ~ same
- Report interaction: is affect shaped by the alignment (or misalignment) of choice and vigor?

### Phase 6: Psychiatric correlates (exploratory)

**Step 6.1: Profile-specific psychiatric patterns**
- ANOVA for each psychiatric measure across quadrants
- Focus on AMI (apathy) × LH finding — adaptive apathy
- Focus on PHQ-9/DASS-Depression × LL finding — efficient disengagement

**Step 6.2: Continuous approach**
- Each psychiatric measure ~ choice_z + vigor_z + interaction
- FDR correction across measures

### Phase 7: Confirmatory replication (N = 350)

**Step 7.1: Rerun Phase 0 validation**
- Vigor reliability, choice-vigor independence, parameter identifiability

**Step 7.2: Rerun Phase 1-2 core analyses**
- Same regressions, CCA, bootstrap tests
- Report whether coefficients fall within exploratory CIs

**Step 7.3: Rerun Phase 3 threat modulation**
- Does the threat reversal replicate?

**Step 7.4: Rerun Phase 4 outcomes**
- Same quadrant outcome patterns?

**Step 7.5: Combined sample (N = 643)**
- Pool exploratory + confirmatory for maximum power
- Final parameter estimates and CIs for the paper

---

## Decision points

| After Phase | Gate | Action if fail |
|---|---|---|
| 0 | Vigor reliable + independence robust | Abandon dissociation framing |
| 1 | R² asymmetry confirmed (choice >> vigor) | If vigor R² > 0.3, it's coupled — different story |
| 2 | β → choice ≠ β → vigor formally | If equal, β doesn't dissociate — k story only |
| 3 | Threat reversal significant | If null, drop context-dependence claim |
| 4 | Vigor > choice for escape | If choice matters more, reframe |
| 7 | Replication of core findings | If doesn't replicate, publish as exploratory only |

---

## Notebook plan

| Notebook | Content | Phase |
|---|---|---|
| 14_choice_vigor_dissociation.ipynb | Already built — quadrant visualization + descriptive stats | 1 |
| 15_dissociation_validation.ipynb | Phase 0 gates + robustness checks | 0 |
| 16_dissociation_formal_tests.ipynb | CCA, bootstrap multivariate regression, SUR test | 2 |
| 17_threat_modulation.ipynb | Trial-level LMM, per-threat correlations, β mediation | 3 |
| 18_outcome_prediction.ipynb | Escape LMM, earnings, profile-specific outcomes | 4-5 |
| 19_confirmatory_replication.ipynb | Full pipeline on N=350 | 7 |
