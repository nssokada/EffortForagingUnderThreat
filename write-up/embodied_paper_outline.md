# Embodied paper — outline (LOCKED 2026-06-07)

**TITLE:** *Threat and effort jointly govern human defensive decisions in a virtually embodied foraging task*

**Status:** Outline locked. Five Results sections with H1–H5 hypothesis labels. Apathy / clinical findings out of scope for this paper.

---

## SIGNIFICANCE STATEMENT

Surviving in a dangerous world requires deciding not only what to do, but how hard and how fast to do it—choices that pit the cost of effort against the risk of harm. Yet threat and effort-based decision-making have largely been studied in isolation, and rarely while people are actively moving, leaving open how the mind coordinates choice with bodily exertion under real danger. Using a video-game-like task in which people forage for rewards while evading a predator, we measured both their choices and the vigor of their actions at once. A single account—weighing the cost of capture against the cost of effort—explained behavior better than treating threat and effort separately. These individual tendencies predicted who survived, the errors people made, and how confident they felt.

---

## INTRODUCTION

- **Threat:** defensive behavior is organized along a predatory-imminence continuum, graded from anticipatory caution to stereotyped reactive responses, yet this has been characterized largely in Pavlovian or non-instrumental settings that omit the cost–benefit structure of real choice.
  - Cite:
  - Cite:
  - Cite:

- **Effort:** optimal-foraging and effort-based decision frameworks treat action selection as a tradeoff between reward and the cost of acting, but they rarely incorporate acute threat or the motor execution (vigor) of the chosen action.
  - Cite:
  - Cite:

- **Embodiment:** because embodied-decision research shows that choice and movement unfold jointly and that motor dynamics carry decision variables, we unify these literatures by casting defense as a two-axis embodied computation—trading subjective capture cost against effort cost—in a virtually embodied foraging-under-predation task that jointly measures choice and motor vigor.
  - Cite:
  - Cite:

---

## RESULTS

### 1. Threat and Effort Jointly Structure Choice and Motor Vigor, Coordinated Within Individuals

**H1.** Threat reduces high-effort choices, increases motor vigor.

- **H1a.** High-effort choices decrease with threat probability and distance.
- **H1c.** Within each chosen effort level, pressing intensity increases with threat.

### 2. Defensive Vigor Unfolds as a Threat-Graded Anticipatory Ramp and a Stereotyped Reactive Surge at Predator Detection

**H2.** The within-trial vigor trajectory shows two qualitatively distinct components mapping onto the predatory imminence continuum.

- **H2a.** Predator encounter triggers a rapid spike in pressing rate.
- **H2b.** The temporal shape of the vigor timecourse differs by encounter status and by threat level.

### 3. A Joint Two-Axis Fitness Function Explains Behavior Better Than Separable Alternatives

**H3.** A joint fitness model with two per-subject parameters (capture cost, effort cost) outperforms simpler alternatives.

- **H3a.** The joint model outperforms an effort-only model that ignores threat.
- **H3b.** The joint model outperforms a threat-only model lacking individual effort sensitivity.
- **H3c.** The joint model outperforms a single-parameter model, showing capture cost and effort cost are separable traits.

### 4. Vigilance and Mobilization Parameters Predict Survival, Error Structure, and Deviation from the Foraging Optimum

**H4.** The model parameters predict survival, error patterns, and decision quality.

- **H4a.** Higher capture cost predicts higher escape rates on attack trials.
- **H4b.** Higher capture cost predicts a greater share of overcautious errors (choosing light when heavy has higher expected reward).
- **H4c.** Higher effort cost predicts lower pressing intensity.
- **H4d.** The balance of capture and effort cost predicts decision quality: effort-driven avoidance is less optimal than threat-driven avoidance.
- **H4e.** Consistency with the joint fitness function predicts foraging earnings, with choice consistency and intensity-pattern match contributing independently.

### 5. Confidence, but Not Anxiety, Registers Vigilance–Mobilization State and Predicts Choice Across Samples

**H5.** Anxiety and confidence independently monitor the foraging computation and predict efficiency beyond the model parameters.

- **H5a.** Anxiety calibration (tracking of threat) predicts foraging optimality beyond the model parameters.
- **H5b.** Anxiety reactivity (slope on threat) predicts adaptive choice shifting across threat levels.
- **H5c.** Capture cost predicts subjective confidence but not anxiety.
- **H5d.** Confidence predicts error type—fewer overcautious, more reckless errors—without changing the overall error rate.

---

## DISCUSSION

- Threat and effort jointly and continuously govern both what people choose and how vigorously they act, and a single two-axis fitness function trading capture cost against effort cost outperforms every separable model—suggesting defense is computed in a common, integrated currency rather than by independent threat and effort systems.

- The per-subject vigilance and mobilization parameters are behaviorally consequential, predicting survival, the structure of errors (overcautious vs. reckless), and departures from foraging optimality—while confidence, not anxiety, emerges as the affective register of this defensive state.

- By studying defense in a virtually embodied, survival-relevant foraging task, we recover graded anticipatory-to-reactive vigor dynamics that nonembodied or Pavlovian paradigms cannot resolve, underscoring the value of ecologically valid embodied settings for understanding adaptive decision-making under threat.

---

## Methods (appendix to outline — for drafting reference)

**Participants:** Two pre-registered Prolific samples (N = 290 exploratory, N = 281 confirmatory).

**Experimental Setup:** Virtually embodied foraging task with binary choice (heavy R=5 vs light R=1) at three threat probabilities (T ∈ {0.1, 0.5, 0.9}) and three distances (D ∈ {1, 2, 3}), with continuous keypress-driven transport and Gaussian-timed predator attacks on subset of trials, plus per-trial anxiety and confidence probes.

**Computational Model:** W(u) = S(u)·R − (1−S(u))·ω·(R+C) − κ·(u−req)²·D, fitted via NumPyro HMC with per-subject (ω, κ).

**Data analysis:** Mixed-effects models for behavioral and dynamic outcomes; GAMMs for temporal vigor structure; OLS for per-subject dispositional regressions; replication threshold = p<0.05 in both samples, same sign.

---

## Naming convention

- ω (mathematical parameter) ↔ **capture cost / vigilance** (defensive intensity scaling)
- κ (mathematical parameter) ↔ **effort cost / mobilization** (motor amplitude cost)

In prose: use "capture cost" and "effort cost" when describing the mathematical model components in Methods and §3; use "vigilance" and "mobilization" as the dispositional individual-differences labels in §4 and §5 and Discussion.

---

## Figures (proposed)

| # | Content |
|---|---|
| 1 | Task design schematic + W(u) formula + (ω, κ) plane diagram |
| 2 | Behavioral coordination: P(heavy) × (T, D); vigor × T within cookie; within-subject cross-channel correlation scatter (§1) |
| 3 | Temporal vigor dynamics: encounter-aligned GAMM showing anticipatory ramp gradedness on T + reactive surge prevalence + reactive surge null on prior T (§2) |
| 4 | Model comparison (M4 vs M1/M2/M3/M3b) + recovery + cross-channel hold-out prediction (§3) |
| 5 | Parameter consequences: escape rate × ω; overcautious errors × ω; pressing intensity × κ; foraging optimality decomposition; earnings × model consistency (§4) |
| 6 | Confidence as appraisal register: confidence intercept/slope_reward → (ω, κ); same features → P(heavy); confidence → error type composition; anxiety calibration/reactivity → optimality (§5) |

Six figures total.

---

## What's out of scope for this paper

- **Apathy / clinical findings** (§4.61 in discoveries.md) — out of main and supplementary; saved for future work.
- **Confidence_intercept → lower reactive peak** (§4.59) — exploratory-only finding; included only if confirmatory `smoothed_vigor_ts.parquet` replicates it.
- **All MMR/CCA clinical analyses** (§4.6) — explicitly out of scope; not mentioned beyond a brief limitations note.

---

## Outline status

**LOCKED 2026-06-07** by user. Final canonical outline.

**Pre-drafting checklist:**
1. **Verify H4 and H5 sub-hypotheses** against existing analyses — several H5 claims (anxiety calibration → optimality, anxiety reactivity → adaptive shifting, confidence → error type) need to be confirmed-or-tested before drafting. Existing memory shows anxiety features are null for cross-sample prediction of (ω, κ) and P(high); H5a/H5b may require new analyses or may rest on results not yet checked against the replication threshold.
2. **Process confirmatory `smoothed_vigor_ts.parquet`** through the timecourse pipeline so §2 GAMM analyses and any reactive acceleration claims replicate cross-sample.
3. **Senior-author sign-off** on the locked outline.
4. **Begin prose drafting** starting from Introduction and §1.
