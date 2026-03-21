# Paper Proposal: Nature Communications

## Title

**Three computational dimensions of effort under threat predict survival, confidence, and apathy**

Alternative: *How you decide, how hard you try, and how threat reshapes both: a computational framework for effort-threat foraging*

---

## The pitch (1 paragraph)

When humans forage under predation threat, their behavior decomposes into three independent computational dimensions: how much they avoid effort (*k*), how much threat biases their decisions (*β*), and how hard they physically execute (*α*). In a unified Bayesian model where *α* enters both the survival computation governing choice (as effective threat exposure: faster pressers traverse distance more quickly) and the vigor measurement directly, *k* and *α* remain completely independent (*r* = 0.006). People do not align their choices with their motor capability — even when the model gives them the opportunity. Each dimension independently predicts a different domain: *k* determines what you attempt, *β* determines how threat reshapes those attempts, and *α* determines whether you survive. High-vigor individuals escape predators 3.6 times more often than low-vigor individuals making identical choices. This creates a metacognitive paradox: ambitious-but-weak foragers are massively overconfident (+1.05 SD bias), while cautious-but-vigorous foragers are the most underconfident yet survive best — and paradoxically report the highest clinical apathy (*α* → apathy *R²* = 0.155). Psychiatric self-report can predict who will be a high-vigor forager (AUC = 0.675) but is blind to who will make ambitious choices. The dissociation between deciding and doing has consequences for survival, self-knowledge, and clinical assessment.

---

## Introduction Framing

**Opening:** Foraging under threat requires coordinating two behavioral systems: a decision system that evaluates costs, rewards, and danger, and a motor system that executes the chosen action. When a predator lurks nearby, do you pursue the high-reward resource, and if so, how vigorously? Normative theories typically model this as a single valuation problem — compute expected utility, then act accordingly (Shadmehr et al., 2019; Manohar et al., 2015). But people differ enormously in how they balance these demands, and the nature of these individual differences is poorly understood.

**Gap:** Most computational models of effort-based decision-making focus exclusively on choice — what people decide — and treat motor execution as a downstream consequence of value. This assumption has rarely been tested in threatening environments where survival depends not just on making good decisions but on acting effectively. Three questions remain open: (1) Do the same computational processes that govern foraging decisions also govern motor vigor? (2) How does threat modulate the relationship between choosing and doing? (3) What are the consequences — for survival, for self-knowledge, and for mental health — when these systems are misaligned?

**What we do:** We developed an effort-based foraging task under predator threat and deployed it in a large online sample (N = 293). Using hierarchical Bayesian modeling, we identify three individual-difference parameters that capture the meaningful structure of behavior: effort discounting (*k*), threat bias (*β*), and tonic motor vigor (*α*). We show that these dimensions are independent, predict different outcomes, and connect to different aspects of subjective experience and clinical self-report.

---

## Results

### Result 1: A computational model captures foraging decisions through effort discounting, threat scaling, and threat bias

A survival-weighted value model (FET: SV = R·exp(−k·E)·S − (1−S)·C + β·(1−S), where S = exp(−T·D^z)) explains 82% of between-subject choice variance with three individual-difference parameters. *k* governs how steeply effort reduces value. *z* governs how distance scales threat. *β* captures residual threat aversion beyond the survival-weighted computation. Parameters are independently identifiable (max inter-parameter r = 0.14).

**Key stats:** WAIC comparison vs 5 alternative models, adj.R² = 0.82, AUC = 0.91, N = 293.

### Result 2: A unified model reveals α as an independent behavioral dimension

We incorporate tonic vigor (*α*, sustained pressing rate = 52% of motor capacity, split-half = 0.93) directly into the choice model's survival function: S_i = exp(−λ · T · (D/α_i)^z). This gives α a mechanistic role in choice — faster pressers have lower effective threat exposure — while λ and z remain population-level. A Bayesian hierarchical model separately estimates α from pre-encounter pressing and phasic sprint *ρ* (universal defensive boost, P(μ_ρ > 0) = 1.0) from terminal attack contrast.

**Despite sharing the survival computation, k and α are completely independent (r = 0.006).** People do not incorporate their motor capability into their foraging decisions. The unified model with 3 per-subject params {k, β, α} matches or exceeds the original 4-param model on choice prediction (R² = 0.88 vs 0.83) while being substantially more parsimonious (saves 293 per-subject z parameters).

Vigor is trait-like: 26% of variance is between-person, only 4% is condition-driven. Choice is the opposite. People adjust their choices flexibly to each trial's conditions. They bring the same motor engagement to every trial regardless.

**Key stats:** Unified SVI fit (NumPyro), vigor HBM (0 divergences). k-α r = 0.006.

### Result 3: Threat bias (β) selectively reshapes choice without affecting vigor — and threat amplifies the effect

β is the parameter that most sharply distinguishes choice from vigor. β suppresses risky choice (standardized β = −0.41) but has no effect on motor vigor (+0.15, opposite direction). Bootstrap difference: −0.56, p < 0.0001.

The mechanism is traceable: β enters the value computation through (1−S), which grows with threat. At low threat, S ≈ 1 and β has little influence, so choice and vigor naturally align (r = +0.20). At high threat, S drops, β's suppression amplifies (choice-β correlation: −0.21 → −0.52), but vigor is unaffected — so the two systems decouple (r = −0.22). Cross-level LMM confirms: choice × threat interaction on vigor, p = 0.0004. Fisher z = 5.07.

This means threat doesn't uniformly suppress behavior. It selectively reshapes the decision channel while leaving the motor channel intact. People who become most cautious in their choices (high β) maintain full motor engagement.

**Key stats:** 10K bootstrap, cross-level LMM with random slopes.

### Result 4: The three dimensions predict different outcomes

Each parameter independently predicts a different domain of real-world performance:

| | *k* (effort discounting) | *β* (threat bias) | *α* (tonic vigor) |
|---|---|---|---|
| **Choice** | β = −0.69*** | β = −0.41*** | n.s. |
| **Escape** | n.s. | β = +0.03*** | β = +0.19*** |
| **Earnings** | β = −16.6*** | n.s. | β = +65.2*** |

Vigor determines survival: within the same choice group, high-α individuals escape 3.6 times more often than low-α individuals (62% vs 17%, p = 10⁻²⁵). Choosing hard actually hurts escape (β = −0.18) because it places the forager farther from safety during attack. Effects are purely additive — no k×α interaction on escape (p = 0.82) or earnings (p = 0.47).

**Key stats:** Trial-level LMM (N = 10,257 attack trials), vigor β = +0.09 (p = 10⁻⁷⁷).

### Result 5: The parameter space predicts confidence and anxiety calibration

Subjective ratings of anxiety and confidence track the model's survival computation at the trial level (anxiety ~ S_probe: β = −0.60, p < 10⁻¹³⁰). People experience threat as the model computes it. But metacognitive accuracy — whether confidence matches actual survival — depends on the k×α profile.

Confidence miscalibration (confidence_z − escape_z) is predicted by the parameters (R² = 0.45): α drives accurate calibration (β = −0.79), while k drives overconfidence (β = −0.24). Critically, the k×α interaction is significant (p = 0.006, ΔR² = 0.03): people who are both effort-averse AND vigorous show a specific metacognitive blind spot — they underestimate their own capability because they associate effort avoidance with poor performance, even though their motor system is fully engaged.

The sharpest contrast: ambitious-but-weak individuals (high choice, low α) are the most overconfident (+0.78 bias) yet have the worst escape rate (17%). Cautious-but-vigorous individuals (low choice, high α) are the least confident (−0.65 bias) yet survive best (55%).

**Key stats:** OLS with interactions, F-test for interaction improvement p = 0.002.

### Result 6: Tonic vigor selectively predicts apathy — not distress, not fatigue

Factor analysis of 14 psychiatric subscales yields three orthogonal factors: general distress (37%), fatigue (20%), and apathy/amotivation (12%). The five behavioral parameters jointly predict apathy (R² = 0.155, p = 3×10⁻⁹) but not distress (R² = 0.015, n.s.) or fatigue (R² = 0.030, n.s.). α alone drives the effect (β = −0.34); no interactions improve prediction (p = 0.69).

The direction is paradoxical: higher α → more self-reported apathy. People who deploy the most sustained motor effort, escape the most, and earn the most also report the highest apathy scores (AMI = 30 vs 26 for low-α individuals). This suggests that clinical apathy instruments may partially index a conservative-but-capable behavioral strategy — choosing easy, trying hard, surviving well — rather than a global motivational deficit.

PLS confirms a multivariate mapping (permutation p < 0.0001, CV R² = 0.04): the α-k axis maps jointly onto anxiety calibration and apathy, with the engaged-effort profile (high α, low k) associated with better threat calibration and higher self-reported apathy.

**Key stats:** EFA (KMO = 0.93), OLS, PLS with 5K permutations.

### Result 7: Psychiatric self-report predicts vigor profiles but not choice profiles

Flipping the predictive direction: can mental health features predict behavioral profile membership? Logistic regression with 14 psychiatric subscales as features:

- **MH → high/low vigor:** 62% accuracy, AUC = 0.675 (chance = 50%). AMI subscales drive prediction.
- **MH → HL vs LH (critical off-diagonal):** 61% accuracy, AUC = 0.645. AMI Social (+0.64) predicts LH; STAI Trait (−0.44) predicts HL.
- **MH → high/low choice:** 49% accuracy (chance). Completely blind.
- **MH → 4 quadrants:** 32-35% accuracy (chance = 25%). Modest but above chance.

Clinical instruments see the motor channel (α) but are blind to the decision channel (k, β). The person who will be an overconfident underperformer (HL) is the one with high trait anxiety and low apathy. The person who will be an underconfident survivor (LH) is the one with high self-reported apathy.

**Key stats:** 10-fold stratified CV, logistic regression and random forest.

---

## Discussion Framing

### 1. Three dimensions, not a single value

Rather than showing that value-vigor coupling "breaks down," we demonstrate that effort-threat behavior has a richer structure than single-value models capture. Three parameters — effort discounting, threat bias, and motor engagement — define a behavioral space where individuals occupy distinct positions with distinct consequences. This connects to emerging multi-dimensional frameworks for understanding individual differences in motivated behavior (Husain & Roiser, 2018; Le Heron et al., 2019).

### 2. Threat reshapes decisions, not actions

β's selective action on choice but not vigor provides a computational account of how threat information propagates through the processing hierarchy. The survival computation S = exp(−T·D^z) governs valuation, and through β, threat amplifies conservative choice. But this signal does not reach the motor system: tonic vigor α is set independently, and phasic sprint ρ is a universal defensive reflex. This is consistent with dual-process accounts where deliberative evaluation and habitual motor programs are governed by distinct systems (Daw et al., 2005), but here instantiated in a naturalistic foraging context with clear survival contingencies.

### 3. Motor engagement as the survival-relevant channel

The finding that vigor — not choice — determines escape connects to the broader intention-action gap literature. Computational models of choice capture what people *intend* to do with high fidelity (R² = 0.82). But intentions don't determine survival; execution does. The ambitious-but-weak profile (high choice, low α) is a computationally precise characterization of overconfidence with measurable survival consequences. This has implications for domains beyond foraging — clinical settings where treatment engagement depends on execution, not just intention.

### 4. Confidence tracks the wrong channel

People's confidence tracks their choice ambition rather than their motor capability. Since choice is the deliberatively accessible channel (you know what you decided) while vigor is automatic (you don't monitor your pressing rate), metacognitive access is biased toward the less survival-relevant signal. The k×α interaction on miscalibration suggests that self-knowledge about effort capacity is specifically impaired when decisional avoidance and motor engagement diverge.

### 5. Reframing apathy as adaptive effort allocation

The α → apathy finding challenges the assumption that apathy is uniformly maladaptive. In our task, the most "apathetic" individuals (by self-report) deploy the most sustained motor effort, escape predators most often, and earn the most. This connects to rational effort allocation theories (Kurzban et al., 2013): apathy may reflect a strategy of conserving decisional ambition while maintaining motor readiness — choosing easy but executing fully. This is the most adaptive profile in a predator-threat environment, where ambitious choices increase exposure. Standard apathy scales may conflate choice conservatism (adaptive) with motor disengagement (maladaptive).

---

## Figures (6 main)

### Figure 1: Task and computational model
- (A) Task schematic with effort/threat mechanics
- (B) Model comparison (WAIC)
- (C) Three parameter distributions {k, β, α} with individual variation
- (D) Model fit: predicted vs observed choice

### Figure 2: Three independent behavioral dimensions
- (A) Choice rate vs α scatter (r = −0.02), showing independence
- (B) Variance budget: person vs condition for choice and vigor
- (C) Regression asymmetry: {k, z, β} → choice (R² = 0.82) vs → vigor (R² = 0.08)
- (D) 5-parameter correlation matrix (showing independence structure)

### Figure 3: Threat bias reshapes choice, not vigor
- (A) β → choice (−0.41) vs β → vigor (+0.15) with bootstrap CI
- (B) Choice-vigor coupling across threat levels (r = +0.20 → −0.22)
- (C) β amplification: β-choice correlation by threat level
- (D) Schematic: survival computation → choice (high gain) vs vigor (low gain)

### Figure 4: Each dimension predicts different outcomes
- (A) Escape rate by α quartile within choice groups (3.6× effect)
- (B) Three-panel: k → earnings, β → choice suppression, α → escape
- (C) Additive model: no interactions on survival

### Figure 5: Confidence miscalibration
- (A) Confidence bias by choice × vigor position
- (B) k×α interaction on miscalibration
- (C) Trial-level: anxiety/confidence track S_probe
- (D) The paradox: most confident ≠ most capable

### Figure 6: Tonic vigor predicts apathy, not distress
- (A) Three psychiatric factors (loadings heatmap)
- (B) 5 params → 3 factors (only α → apathy)
- (C) α vs apathy scatter
- (D) The adaptive apathy profile: highest α, highest AMI, best survival

---

## Unified 3-parameter framework

The key innovation: **α enters the survival function directly**, making the model genuinely unified:

```
S_i = exp(-λ · T · (D / α_i)^z)
SV = R · exp(-k_i · E) · S - (1-S) · C + β_i · (1-S)
```

α represents effective exposure — faster pressers traverse distance more quickly, reducing time exposed to the predator. This gives α a mechanistic role in BOTH choice (through S) and vigor (directly as pressing rate). λ and z are population-level structural parameters.

| Parameter | What it captures | Key prediction |
|-----------|-----------------|----------------|
| **k** (effort discounting) | How much you avoid effortful options | Earnings (β=−14.4), choice (R²=0.88) |
| **β** (threat bias) | Residual threat aversion after accounting for motor capability | Conf miscalibration (β=+0.18), distinguishes HH from HL |
| **α** (tonic vigor) | Motor engagement / effective threat exposure | Escape (R²=0.73), apathy (R²=0.155) |

**Critical result:** k and α are completely independent (r = 0.006) even within the unified model. People do not align their choices with their motor capability — the dissociation is in the people, not an artifact of separate estimation.

*z* (distance-threat scaling) → population-level, supplementary.
*ρ* (phasic sprint) → universal defensive response (P(μ_ρ > 0) = 1.0), supplementary.

---

## Why Nature Communications

1. **Unified framework:** Three computational parameters from a single foraging task predict survival, confidence, and apathy — bridging decision science, motor control, metacognition, and clinical psychiatry.
2. **Large, preregistered:** N = 293 exploratory + N = 350 confirmatory with preregistered replication targets.
3. **Robust effects:** 3.6× survival difference, R² = 0.82 for choice, R² = 0.45 for miscalibration, R² = 0.155 for apathy. Core results survive bootstrap, permutation, and FDR.
4. **Novel mechanism:** β as a decision-specific threat signal that leaves motor execution intact — new to the foraging, effort, and threat literatures.
5. **Clinical reframe:** Apathy as adaptive effort allocation, grounded in computational parameters, not clinical speculation.

---

## What the confirmatory sample must replicate

1. k-α independence within unified model (|r| < 0.10)
2. Unified model fits comparably to or better than per-subject z model
3. Three-parameter profiles: k → choice, β → threat modulation, α → escape
4. Threat reversal of choice-vigor coupling
5. Vigor determines survival (>2× escape difference within same choice group)
6. Confidence miscalibration from choice × vigor position (R² > 0.30)
7. α → apathy factor (R² > 0.08)
8. MH predicts vigor but not choice profiles
9. Distress and fatigue remain null
