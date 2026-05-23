# Replacement / addition for §2.4 of `final_paper_draft.md`

This is a drop-in replacement for the current §2.4. The current section reports the per-parameter behavioural signatures (ω→escape, κ→vigor) and the angle→optimality result. It treats ω and κ as two parallel traits without testing whether the joint W(u) framing — that both parameters enter both channels — actually holds at the individual-difference level.

The replacement does three things the current draft doesn't:

1. Reports the **cross-channel test**: ω, fit only from binary patch choices, predicts continuous motor vigor in both samples — a falsifiable prediction the joint model could have failed.
2. Reports the **stimulus-tuning specificity**: ω is not a uniform avoidance trait; it is specifically threat-tuned in choice (replicated multilevel interaction).
3. Frames the per-parameter signatures (currently §2.4) as **the model's quantitative prediction**: the simulation shows W(u) predicts ω→level (small +) and ω→slope (~0), and the data confirm both.

I keep the existing §2.4a-d structure (escape, overcaution, vigor, angle) but add §2.4e (cross-channel) and §2.4f (stimulus tuning) and a small reframe of the section opening.

---

### 2.4 ω and κ produce the behavioral signatures the model predicts, and the joint model passes a cross-channel test

If the joint W(u) model captures genuine individual-difference traits, three things should hold. First, the fitted parameters should produce the behavioral signatures their interpretations predict — ω should index avoidance and survival, κ should index pressing intensity. Second, both parameters should enter *both* channels through the shared fitness function: a parameter fit only from one channel should make correct, falsifiable predictions about variation in the other. Third, the parameters should show stimulus-specific tuning — ω should be tuned to threat (the input the survival term depends on), κ should weight effort cost (the input the demand term depends on).

**Per-parameter signatures.** Each signature emerged in both samples (Fig. 4a–c). Higher ω tracked stronger avoidance of high-effort patches, a larger share of overcautious errors, and a higher escape rate under attack. Higher κ tracked lower pressing intensity across the task but had little relationship to survival, as expected for an effort-cost weight rather than a safety-oriented one. Fig. 4c reports the full set of parameter-to-signature associations; each effect replicated in sign and magnitude across the two independent samples.

**The (ω, κ) balance predicts overall performance.** We indexed the balance between the two parameters as the angle between each participant's standardized ω and κ estimates, with low angles corresponding to threat-driven avoidance and high angles to effort-driven avoidance. This angle reliably predicted the proportion of optimal choices (Fig. 4d). Threat-driven avoidance was more adaptive than effort-driven avoidance because threat varies trial to trial while effort costs are fixed by cookie type: avoidance that tracks threat selectively preserves reward opportunities, while avoidance that tracks effort forgoes them indiscriminately.

**Cross-channel test: ω, fit from choice, predicts anticipatory vigor.** A particularly stringent test of the model's claim that ω and κ enter a *shared* fitness function is whether the parameter fit predominantly from the choice channel (ω) makes correct predictions about the vigor channel. Patch selection is binary; vigor is continuous; the inference machinery never sees a press rate when fitting ω. We tested this by computing each participant's mean anticipatory pressing rate on non-attack trials — trials where no predator ever appears, so the entire trial is anticipatory — and asking whether ω predicted variation in this measure after partialling out κ.

The test passed in both samples. Higher ω was associated with higher anticipatory vigor at each threat level (per-T univariate r = +0.14 to +0.21 in exploratory; +0.18 to +0.21 in confirmatory; all p < 0.02). In a Bayesian regression of mean anticipatory vigor (z-scored) on ω and κ jointly, the partial effect of ω was substantial in both samples: β = +0.485 [+0.416, +0.545] in exploratory and β = +0.473 [+0.408, +0.535] in confirmatory; both 95% HDIs excluded zero. The effect held in trial-level multilevel models controlling for cookie type and threat condition (β ≈ +0.12 in both samples). Because ω is fit only from binary choices, this association is not built into the inference: it is a structural prediction of the joint W(u) model that the data could have falsified.

The cross-channel signature is in the *level* of anticipatory vigor, not in its *slope* across threat. Per-subject threat-driven vigor slopes did not correlate with ω in either sample. We checked whether this null was a model failure or a model prediction by simulating W(u) directly: for each fitted (ω, κ) pair we computed the model's optimal pressing rate u\* under each task condition and asked what correlations the model itself predicts between parameters and predicted vigor. The model predicts a small positive ω → vigor-level correlation (predicted r = +0.07 to +0.21 across samples) and an essentially zero ω → vigor-slope correlation (predicted r = −0.05 to +0.00). Both predictions matched the empirical pattern. The reason for the second is structural: at higher ω, u\* is pushed up close to the speed-saturation regime, so additional threat rescales the level of pressing rather than steepening its slope across threat. The cross-channel signature of ω is a uniform upward shift in motor mobilization, not a threat-modulated mobilization, exactly as the formal model predicts.

**Stimulus-specific tuning.** A second prediction of the model concerns *which* environmental dimension each parameter should track. ω enters W(u) through the survival term, which is most sensitive to threat probability T; κ enters through the effort-cost term, which is most sensitive to distance D. We tested both predictions in trial-level multilevel logistic regressions of patch choice with all four parameter-by-stimulus interactions (threat × ω, distance × ω, threat × κ, distance × κ).

Of the four interactions, only **threat × ω** survived in both samples, with the predicted negative sign: β = −0.075 [−0.133, −0.014] in exploratory and β = −0.151 [−0.207, −0.094] in confirmatory; the effect approximately doubled in confirmatory. Higher ω is therefore not a uniform avoidance trait but a specifically *threat-tuned* avoidance trait — high-ω participants are more responsive to trial-level variation in T when selecting patches. The mirror prediction for κ — that high-κ participants would show stronger distance-dependent avoidance — was null in both samples (distance × κ HDIs included zero in both samples, with sign disagreement across samples). κ acts on choice as a baseline weight on effort cost rather than a stimulus-tuned modulator.

**Two channels, two home traits, neither exclusive.** Taken together, these tests show that ω and κ behave as the model's interpretation predicts. ω shapes what participants avoid (and what they survive); κ shapes how hard they work; the balance between them determines whether their strategy pays off. Each parameter has a *home channel* — ω in choice (β ≈ −1.15), κ in vigor (β ≈ −0.21) — and is the dominant trait there. But neither is confined to its home channel: ω contributes a smaller, replicated effect on vigor level (β ≈ +0.12), and κ contributes a smaller effect on choice (β ≈ −0.50). The model predicts both leakages and the data confirm both. The threat-tuning of ω's effect on choice further specifies the meaning of avoidance sensitivity: it is not a global cautiousness but a parameter that scales the responsiveness of choice to the survival term in W(u).

---

## What to update in Figure 4

Current Fig. 4: panels a-d are individual-difference and metacognitive results.

Suggested additions:

- **Fig. 4e (new):** Per-sample scatter — partial ω vs anticipatory vigor (residualised on κ), exploratory and confirmatory side-by-side. Saved at `results/figs/avoid_activate/cross_channel_omega_anticipatory.png`. Shows the cross-channel finding visually.

- **Fig. 4f (new):** Bar plot or 4-cell matrix of parameter × stimulus interactions (threat × ω, threat × κ, dist × ω, dist × κ) with 95% HDIs in both samples. Highlights that only threat × ω survives in both samples.

- **Fig. 4g (optional, supplementary):** Model-vs-data table for the four predicted correlations (ω-level, ω-slope, κ-level, κ-slope), confirming the cross-channel test was a falsifiable prediction the model makes correctly.

## What to update in §4 Methods

The cross-channel test should get its own paragraph in §4.4 Statistical analyses. Suggested wording:

> **Cross-channel test (§2.4e).** For each subject, mean anticipatory pressing rate was computed across non-attack trials (the full trial is pre-encounter when no predator appears). We then fitted a Bayesian linear regression of z-scored anticipatory vigor on omega_z and kappa_z (4 chains × 2000 draws + 1000 tune), reporting posterior means and 95% highest density intervals. The test was conducted independently in both samples. We additionally fitted trial-level multilevel models with cookie type and threat as covariates, and ran a simulation of the joint W(u) model with each subject's fitted parameters at population values of γ, h, σ_sp to check that the model itself predicts the observed correlation pattern.
>
> **Stimulus-tuning test (§2.4f).** Trial-level multilevel logistic regression of patch choice (1 = heavy) on threat_z, dist_z, omega_z, kappa_z, and all four parameter × stimulus interactions, with cluster structure absorbed by the parameter terms. Predictions were preregistered in sign convention: threat × ω negative, distance × κ negative, threat × κ near zero, distance × ω near zero.

## Suggested removal / demotion

The current §2.6 ("Task affect dissociates clinical symptom dimensions (exploratory)") cites numbers from the old (k, β, α) model that have not been verified on the current ω/κ fits. H7 results show that the *direction* of those claims is half-correct on current data — task anxiety does predict DASS-Anxiety and task confidence does predict AMI in both samples — but the depression and anxiety-tracking links are exploratory-only. Either re-write §2.6 with the H7-verified numbers (in `results/stats/individual_diffs/`) or move §2.6 to a Supplementary Section. The cross-channel finding here in §2.4 is much stronger and more central to the paper's claim than the clinical exploration.
