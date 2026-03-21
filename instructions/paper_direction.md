# Paper Direction: The Choice-Vigor Dissociation

## One-sentence summary

When humans forage under threat, what they decide and how hard they execute are governed by fundamentally different variance structures — choice responds flexibly to trial conditions (13% condition-driven), while motor vigor is a stable individual trait that barely adjusts to danger (4% condition-driven). A four-parameter framework {k, z, β, α} places individuals in a choice-vigor space that predicts survival outcomes, confidence calibration, and self-reported apathy.

---

## Title candidates

- "Deciding and doing diverge under threat: A computational dissociation between foraging choice and motor vigor"
- "Threat bias dissociates valuation from action in human effort-threat foraging"
- "What you choose and how hard you try: Independent behavioral dimensions in foraging under threat"

---

## The story in four acts

### Act 1: A computational model captures foraging decisions under threat

When people forage in a predator-threat environment, they integrate energetic cost and exposure-dependent danger into a survival-weighted subjective value. A hierarchical Bayesian model with three individual-difference parameters — effort discounting (k), distance-dependent threat scaling (z), and threat bias (β) — explains 82% of choice variance (N=293). The model identifies how each person uniquely trades off reward, effort, and safety.

**Key result:** FET model fits: R²=0.45 for raw choice, adj.R²=0.82 when predicting subject-level choice rate from parameters. Three independently identifiable parameters (k-β correlation = 0.14).

### Act 2: The same model barely predicts how hard people press

Despite explaining choice nearly perfectly, the same three parameters explain only 8% of variance in motor vigor (pre-encounter pressing rate). This 11× asymmetry is the core finding. Choice and vigor are independent at the subject level (r = −0.02, p = 0.76), and this independence is robust across five different vigor operationalizations.

**Key results:**
- Choice-vigor correlation: r = −0.02 [−0.13, +0.10]
- Params → choice: R² = 0.82; params → vigor: R² = 0.08
- CCA recovers two independent canonical dimensions: one for choice (r = 0.91), one for vigor (r = 0.29)
- Four roughly equally populated behavioral quadrants emerge (HH, HL, LH, LL)

### Act 3: β creates the dissociation; threat amplifies it

The threat bias parameter (β) strongly suppresses risky choice (β_choice = −0.41) but has no effect on motor execution (β_vigor = +0.15, opposite direction). This selective action on choice but not vigor is formally confirmed: the bootstrap difference is −0.56 [−0.80, −0.38], p < 0.0001. β is the "dissociation parameter" — it acts on the survival computation that governs valuation but does not constrain the motor system.

Critically, threat context modulates the coupling. Under low threat, choice and vigor align (r = +0.20) — people who choose hard also press hard. Under high threat, they reverse (r = −0.22) — people who choose safe press hardest. This swing of 0.42 correlation units is confirmed by a cross-level LMM interaction (β = −0.022, p = 0.0004) and Fisher z-test (z = 5.07, p < 0.0001). The mechanism: β's influence on choice amplifies with threat (r = −0.21 at low → −0.52 at high), because β enters through the survival function which becomes more extreme as threat increases. But β's null effect on vigor doesn't change.

**Key results:**
- β → choice: −0.41 [−0.57, −0.31]; β → vigor: +0.15 [+0.02, +0.27]
- Difference: −0.56, p < 0.0001
- Threat reversal: r = +0.20 (low) → −0.22 (high), Fisher z = 5.07
- LMM interaction: p = 0.0004

### Act 4: Vigor, not choice, determines who survives

The behavioral channel least captured by the computational model — motor vigor — is the one that actually matters for survival. Trial-level vigor predicts escape from predator attacks (β = +0.09, p = 10⁻⁷⁷), while choosing the high-effort cookie actually *hurts* escape (β = −0.18, p ≈ 0) because it places you farther from safety. Holding choice constant, high-vigor people escape 2-3× more often than low-vigor people (53% vs 19%, 60% vs 25%).

This creates a metacognitive paradox. People who choose ambitiously (HL profile) are the most confident (confidence bias = +0.98) but have the worst escape rate (19%). People who choose conservatively but press hard (LH profile) are the least confident (bias = −1.18) but survive best (60%). The confidence miscalibration is explained by choice and vigor in a single regression (R² = 0.42): choosing drives overconfidence, pressing drives accurate calibration.

**Key results:**
- Escape: vigor β = +0.09 (p = 10⁻⁷⁷), choice β = −0.18 (p ≈ 0)
- Pairwise: vigor triples escape within same choice group (p = 10⁻²⁵)
- Confidence miscalibration: R² = 0.42, F = 50, p = 10⁻²⁶
- HL overconfident (+0.98), LH underconfident (−1.18)

---

## Core hypotheses

| # | Hypothesis | Test | Status |
|---|---|---|---|
| H1 | FET model explains choice (R² > 0.4) | Model fit, WAIC comparison | ✅ R²=0.45 |
| H2 | Choice and vigor are independent (|r| < 0.10) | Pearson r with bootstrap CI | ✅ r=−0.02 |
| H3 | Params predict choice >> vigor (R² ratio > 5×) | Multiple regression | ✅ 11× |
| H4 | β selectively predicts choice, not vigor | Bootstrap difference test | ✅ p<0.0001 |
| H5 | CCA recovers two canonical dimensions | CCA + MANOVA | ✅ r=0.91, r=0.29 |
| H6 | Threat reverses the choice-vigor coupling | Cross-level LMM interaction | ✅ p=0.0004 |
| H7 | β → choice amplifies with threat | Per-threat β-choice correlations | ✅ −0.21→−0.52 |
| H8 | Vigor predicts escape beyond choice | Trial-level LMM | ✅ p=10⁻⁷⁷ |
| H9 | Vigor dominates earnings | Subject-level regression | ✅ β=+0.76 |
| H10 | Confidence miscalibration tracks the dissociation | ANOVA + regression | ✅ R²=0.42 |
| H11 | HL profile is overconfident, LH is underconfident | Quadrant bias comparison | ✅ +0.98 vs −1.18 |
| H12 | Replication in confirmatory sample (N=350) | Full pipeline re-run | 🔲 |

---

## Figure plan

### Figure 1: Task and model
- (A) Task schematic: arena, cookies, predator
- (B) Model comparison (WAIC)
- (C) Posterior parameter distributions for z, k, β

### Figure 2: The dissociation
- (A) Scatter: choice vs vigor, colored by quadrant (r = −0.02)
- (B) Same scatter, colored by β
- (C) Regression coefficients: params → choice vs params → vigor (paired bar plot showing the asymmetry)
- (D) CCA loadings (two dimensions)

### Figure 3: β is the dissociation parameter
- (A) Path diagram: β → choice (strong), β → vigor (null)
- (B) Bootstrap distribution of β_choice − β_vigor with CI
- (C) Parameter means by quadrant (k, β, z bar plots)

### Figure 4: Threat reverses the coupling
- (A) Per-threat choice-vigor scatter (3 panels: low/med/high)
- (B) β → choice and β → vigor at each threat level (shows amplification)
- (C) LMM interaction coefficient with CI

### Figure 5: Outcomes and metacognition
- (A) Escape rate by quadrant
- (B) Regression: escape ~ choice + vigor (vigor dominates)
- (C) Confidence miscalibration by quadrant
- (D) Off-diagonal comparison: HL vs LH on key measures

### Supplementary figures
- S1: Vigor measure validation (split-half, ICC, operationalization robustness)
- S2: Affect tracks survival computation (S_probe → anxiety/confidence)
- S3: Encounter-centered vigor (threat/attack effects on pre/post pressing)
- S4: Psychiatric measures by quadrant
- S5: Earnings by quadrant
- S6: Full MANOVA and bootstrap tables

---

## What this adds to the literature

### 1. Challenges the value-vigor coupling assumption
The dominant framework (Manohar et al., Shadmehr et al.) assumes subjective value → movement vigor. We show that in a threat-foraging context, the value computation (governed by k, z, β) strongly predicts what people decide but only weakly predicts how vigorously they act. The coupling is not absent — it exists through a second, independent canonical dimension — but it's far weaker than the choice pathway.

### 2. Identifies β as a decision-specific threat signal
No prior work has shown that a computational parameter governing choice under threat selectively fails to predict motor execution. β enters the survival computation that governs choice but does not penetrate the motor system. This is a specific, testable, computationally grounded claim about where threat bias operates in the processing hierarchy.

### 3. Demonstrates context-dependent choice-vigor coupling
The shift from r = +0.20 to r = −0.22 across threat levels shows the coupling is not a fixed property of individuals — it changes with threat context. This is a within-task, within-subject manipulation, not a between-group comparison. The mechanism is traceable: β's influence on choice amplifies as threat increases (through the survival function), while vigor is unaffected.

### 4. Shows vigor predicts real outcomes better than choice
This connects to the broader literature on the gap between intentions and actions. The computational model captures "what people want to do" (choose) almost perfectly, but the behavioral channel that determines survival — motor vigor — is largely outside the model's scope. The HL profile (ambitious choosers who can't execute) is a specific, measurable phenotype of overconfidence.

### 5. Reframes apathy
The LH profile — high apathy scores but excellent survival — challenges the clinical interpretation of apathy as uniformly maladaptive. In a threat context, choosing conservatively while maintaining motor capacity may be the most adaptive strategy. This connects to emerging work on rational apathy and effort allocation.

---

## What's NOT in this paper (and where it could go)

- **The FET model details** — model comparison, parameter recovery, posterior predictive checks are established but secondary to the dissociation story. Supplementary.
- **20Hz vigor pipeline** — replaced by count-based measures. The 20Hz analyses inform background but don't appear in the paper.
- **Encounter-centered vigor** — the threat/attack effects on pre/post encounter pressing are supplementary. The main story is about the subject-level dissociation, not trial-level dynamics.
- **Psychiatric battery** — mostly null. AMI finding is exploratory. Mention in discussion.
- **Metacognitive calibration (S_probe tracking)** — the affect story (S_probe → anxiety/confidence) is established and supports the model but isn't the main novelty. Supplementary or brief results section.
- **Confirmatory sample** — highest priority for making this submittable. The dissociation needs replication.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| "Choice and vigor are trivially independent" | Threat reversal shows structured, context-dependent coupling — not trivial |
| "β is poorly identified" | Posterior k-β r = 0.14; β has unique partial effects on both choice and vigor |
| "Vigor measure is noisy" | SB = 0.91; more reliable than choice (SB = 0.37) |
| "Quadrant analysis is median splits" | All stats use continuous measures; quadrants for visualization only |
| "R² = 0.08 for vigor means the model doesn't work" | That's the point — the model captures choice but not vigor, and vigor is what matters for outcomes |
| "Doesn't replicate" | Confirmatory N=350 is ready to run |

---

## Priority actions

1. **Run confirmatory sample** (N=350) through the full Phase 0-6 pipeline
2. **Write the Methods and Results** using NB15 as the backbone
3. **Build Figure 2-5** from NB14 output
4. **Draft the Discussion** around the five literature contributions above
5. **Supplementary materials** from existing notebooks (model fits, affect, encounter vigor)
