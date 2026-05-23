# Pressure-testing the affect/metacognition story

A working document. The goal is to be honest about whether the "anxiety = primary appraisal, confidence = secondary appraisal" claim actually holds, or whether we're stretching modest findings into a tighter story than the data supports.

---

## 1. The claim, stated plainly

> Trial-level affect dissociates into two computationally distinct channels. Anxiety reads the perceptual threat input (T, D, cookie) but not the subject's cost weights (ω, κ). Confidence reads both. This dissociation maps onto Lazarus's distinction between primary appraisal (threat assessment) and secondary appraisal (coping assessment), and provides the connection between subjective experience and the W(u) value computation that drives behavior.

This is a cognitive architecture claim. It says where in the foraging decision pipeline the two affective signals live.

## 2. The chain of reasoning the claim depends on

For the claim to hold, all of the following have to be true:

| Step | Statement | Where the evidence sits |
|---|---|---|
| A | Trial-level confidence depends on (ω, κ) beyond stimulus features | ΔELPD test, both samples |
| B | Trial-level anxiety does NOT depend on (ω, κ) beyond stimulus features | ΔELPD test, both samples |
| C | The dependence in A is mechanistic, not a between-subject confound | Untested |
| D | The non-dependence in B is real, not a power issue | Untested |
| E | The split maps onto Lazarus's distinction specifically | Post-hoc interpretation |
| F | Confidence is reading the W(u) computation, not just "how heavy this trial is" | Indirectly tested via stimulus partialling |
| G | The dissociation replicates cleanly across both samples | Trial-level: yes. Trait-level: borderline. |
| H | Trait-level concordance | ω→mean_anxiety null both, ω→mean_confidence sig exp / marginal conf |

If any of A–G is wrong, the claim is in trouble.

## 3. Pressure-testing each step

### 3A — Trial-level confidence depends on (ω, κ) beyond stimulus

**Evidence:** ΔELPD +45.9 (exp) / +29.4 (conf) when adding (ω, κ, T:ω, T:κ) to a base model of T + D + is_heavy. log(ω) β = −0.097 / −0.059 ★, log(κ) β = −0.059 / −0.068 ★, both replicated.

**Critical weakness — no subject random intercepts.** The bambi models I ran do NOT include `(1|subj)`. Each subject's ~18 confidence trials are treated as independent. The model is using within-subject pooling to detect *between-subject* mean differences in confidence that correlate with (ω, κ).

In other words: the trial-level test is mathematically equivalent to a trait-level test on `mean_confidence ~ ω + κ` with extra power from pooling. It is not telling us about a within-trial computation. **The "trial-level finding" I have been describing is actually a trait-level finding in disguise.**

The trait-level test (which I ran separately) gives:
- exp: ω β = −0.164 ★ [−0.296, −0.052]
- conf: ω β = −0.107 [−0.228, +0.016]

The conf result HDI grazes zero. This is the actual evidence strength. The trial-level ΔELPD of +29 is impressive only because it pools 5,000 trials worth of the same between-subject signal.

**Verdict:** The "trial-level dependence" framing overclaims. The honest version is "trait-level confidence loads on ω, replicated weakly." That's a much more modest finding.

### 3B — Trial-level anxiety does NOT depend on (ω, κ)

**Evidence:** ΔELPD +2.6 / −0.8 from adding (ω, κ) to anxiety models. log(ω) sign disagreement across samples (+0.04 exp / −0.03 conf).

**Same weakness as 3A.** This is also a trait-level test in disguise. The trait-level result is `mean_anxiety ~ ω`: β = +0.07 (exp) / −0.06 (conf), both null with sign disagreement. Anxiety as a trait doesn't load on ω. The "trial-level null" is just the trait-level null with more nominal observations.

**Verdict:** This part of the claim is fine — anxiety really does not correlate with ω at the trait level, in both samples, with replicated nulls. But the trial-level framing is overclaiming.

### 3C — The dependence in A is mechanistic, not a confound

**The alternative explanation I have to rule out:** confidence is rating "how easy will this trial be for me," and high κ subjects know they hate effort while high ω subjects know they tend to fail. Their confidence ratings reflect their general task disposition, not a readout of the W(u) computation. The correlation with (ω, κ) is an introspective rating of preference, not a computational signature.

**Why this is hard to rule out:** the confidence question literally is "how confident are you about completing this trial." A subject who weights effort cost heavily WILL feel less confident about a high-effort trial because they don't want to expend the effort, regardless of whether they're "reading W(u)."

**Untested:** I have not shown that confidence reads the *integrated* output of W(u) (i.e., u* or W_max or S(u*)) rather than the *constituent inputs* the subject already knows about themselves (ω, κ as personality traits).

**Verdict:** This is the biggest gap in the story. The dissociation is real, but the mechanistic interpretation is one of several plausible readings.

### 3D — The non-dependence in B is real, not power

**Power calculation needed but not done.** Both samples have ~290 subjects. The trait-level β for ω → mean_anxiety is +0.07 / −0.06. With N=290 and unit-z covariates, a β of ~0.12 would be detectable at ~80% power. The observed effects are smaller, so we can't rule out a real effect of magnitude 0.05–0.10.

**Verdict:** The null is consistent with a small (~|β|<0.10) effect that we don't have the power to detect. The "anxiety is independent of ω" claim is bounded by power, not proven.

### 3E — The split maps onto Lazarus specifically

**Post-hoc interpretation.** We did not preregister "primary vs secondary appraisal" as the test. Lazarus's distinction is one of several possible framings of why anxiety and confidence might dissociate. Other framings:

- **Stimulus vs disposition:** Anxiety asks about the world; confidence asks about the self. The dissociation reflects the *referent* of each rating, not a stage in the cognitive pipeline.
- **Outcome valence:** Anxiety = negative outcome estimate, confidence = positive outcome estimate. They are mirror-image questions, not different cognitive computations.
- **Trait-state mixing:** Confidence ratings on a 1–10 scale collapse into "how I usually feel about effortful tasks" for many subjects. Anxiety is more state-bound.

**Verdict:** Lazarus is a plausible post-hoc interpretation, not a tested mechanism. We should be honest about this in the paper.

### 3F — Confidence reads the W(u) computation specifically

**Untested directly.** The trial-level model with (ω, κ) added is consistent with confidence reading either (a) the W(u) value at the optimum, (b) the predicted u*, (c) the predicted survival S, or (d) just the subject's own knowledge of their cost weights as personality traits.

**The right test would be to compare regressions:**
- Confidence ~ stimulus + ω + κ (current)
- Confidence ~ stimulus + W_pred(u*; ω, κ)  (does the value of the optimum predict confidence?)
- Confidence ~ stimulus + u_pred(ω, κ, T, D)  (does the optimal pressing rate predict confidence?)
- Confidence ~ stimulus + S_pred(u*, T, D, ω, κ)  (does the survival prob predict confidence?)

If one of (W_pred, u_pred, S_pred) wins over linear (ω + κ), then confidence is reading the *integrated* model output, not just the parameters. If none win, confidence is just reading "the kind of person I am" via ω, κ as introspectable traits.

**Verdict:** Untested. Without this, the "confidence = readout of W(u)" claim is weaker than it sounds.

### 3G — Cross-sample replication

| Test | Exp | Conf | Status |
|---|---|---|---|
| Trial-level anxiety: stimulus dominates | +109 ELPD | +61 ELPD | Replicates |
| Trial-level confidence: stimulus dominates | +113 ELPD | +129 ELPD | Replicates |
| Trial-level confidence: ω, κ add | +46 ELPD | +29 ELPD | Replicates |
| Trial-level anxiety: ω, κ don't add | +2.6 ELPD | −0.8 ELPD | Replicates |
| log(ω) on confidence | −0.097 ★ | −0.059 ★ | Replicates |
| log(κ) on confidence | −0.059 ★ | −0.068 ★ | Replicates |
| log(ω) on anxiety (after stimulus) | +0.04 ★ exp | **−0.03 ★ conf** | **Sign-disagree** |
| Trait ω → mean_confidence | −0.164 ★ | −0.107 [HDI grazes 0] | Borderline |
| Trait ω → mean_anxiety | +0.07 [null] | −0.06 [null] | **Sign-disagree** |

**Verdict:** The trial-level pooled results replicate cleanly. The trait-level results replicate weakly. The sign-disagreements on anxiety (small but non-zero) are consistent with noise rather than a true effect, but they still mean the anxiety-related claims rest on nulls, which are inherently weaker than positive findings.

### 3H — Trait-level concordance

Already covered. Trait-level ω → confidence is borderline, trait-level ω → anxiety is null. The concordance is in the right direction but not strong.

## 4. Alternative cognitive stories

If the Lazarus framing doesn't hold, what else could we be saying?

**A. Affect doesn't connect to the computation. Report null.**
The paper has its behavioral findings (cross-channel + joint structure). The affect section can be descriptive: anxiety rises with threat, confidence falls with threat, and there's no clean computational signature in either at the individual-difference level. Honest, modest, doesn't overclaim.

**B. Confidence is the introspective rating of subjective competence; anxiety is the introspective rating of the situation.** Drop the Lazarus framing. The two ratings differ in their *referent* (self vs world), not in their cognitive *stage*. This is more defensible because it's about what the questions ask, not about what the brain computes.

**C. Affect carries variance complementary to the parameters (current §2.5).** The anxiety-tracking → outcomes finding from §2.5 is replicated and modest. We could keep the existing framing without trying to make it computationally cleaner.

**D. Affect reads model-derived outcome estimates, not parameters.** Test confidence ~ S_pred and confidence ~ u_pred etc. and see if any of those wins. If yes, we have a real computational claim. If no, the affect data are loosely related to behavior but not to the model's internal quantities.

**E. The interesting affect signal is in trial-by-trial fluctuations, not trait differences.** Test whether trial-by-trial deviations of anxiety or confidence (after partialling stimulus) predict trial-level *behavior* (next-trial choice, vigor variability). This would be a within-subject computational coupling test that doesn't require between-subject claims.

## 5. Honest verdict on the current story

The current "anxiety = primary appraisal, confidence = secondary appraisal" framing has these problems:

1. **The "trial-level" framing is misleading** because no random intercepts were used and (ω, κ) are within-subject constants. The actual finding is trait-level with high pooling power, not a within-trial computational signature.
2. **The trait-level confidence finding is borderline** in the confirmatory sample (HDI grazes zero).
3. **The trait-level anxiety finding is a null with small power**, not a strong dissociation.
4. **Lazarus is a post-hoc interpretation**, not a tested mechanism.
5. **The "confidence reads W(u)" claim is unsupported** without comparing to model-derived outcome estimates (W_pred, u_pred, S_pred).

The honest version of the affect story is much more modest:

> Confidence ratings are weakly negatively related to ω at the trait level (replicated borderline, exp β=−0.16, conf β=−0.11). Anxiety ratings are not related to ω at the trait level (replicated null in both samples). The two affective channels differ in what they introspect about (self-coping for confidence, situational threat for anxiety), and this difference is reflected in their relationship to a model parameter that captures avoidance disposition.

That's about what the data say. It's not the architectural claim I was building.

## 6. What would actually make the affect story strong

To get from the modest version to a real cognitive contribution, we would need ONE of the following:

1. **Show that confidence reads a model-derived integrated quantity** (W_pred, u_pred, S_pred) better than it reads the linear (ω + κ). This would establish that confidence is reading the *output* of the value computation, not just the parameters as introspected traits.

2. **Show within-subject trial-by-trial coupling between affect deviations and behavior**, with the coupling itself being a stable individual difference. This is the "metacognitive sensitivity" framing and it doesn't require claims about between-subject parameter loading.

3. **Show that confidence and anxiety differentially predict different downstream outcomes** in a way that aligns with their proposed computational roles. For example: anxiety should predict reactive defensive responses (encounter spike?) while confidence should predict anticipatory effort allocation. If the dissociation is on the *behavioral output* side too, the cognitive claim has more teeth.

4. **Show that affect dynamics across trials reveal learning or updating** of the W(u) representation. If subjects' anxiety on trial t+1 depends on what happened on trial t in a way that suggests an internal model update, that's evidence the affective system is computing something.

None of these have been tested. Each is doable in the existing data.

## 7. Recommendation

The current affect story as I've been telling it does not survive pressure testing. Three options:

**Option 1 — Demote the affect section to descriptive findings.** Keep §2.5 as a list of replicated correlations (anxiety rises with threat, confidence falls with threat, anxiety-tracking adds outcome prediction beyond ω, κ — the existing findings) without the architectural claim. This is what an honest paper looks like given current evidence.

**Option 2 — Run one of the four real tests above** (model-derived quantities, within-subject coupling, differential downstream prediction, trial-by-trial updating) and let the result determine the story. If one of them works, we have a real cognitive claim. If none does, fall back to Option 1.

**Option 3 — Pivot the affect story entirely.** Maybe the interesting thing about affect in this task isn't its relationship to (ω, κ). Maybe it's something else — anxiety calibration as a metacognitive sensitivity index, affect-behavior coupling within trials, affect dynamics across blocks. Brainstorm what we have and pick the strongest angle.

My vote: **Option 2, specifically test (1)**. The "does confidence read W_pred / u_pred / S_pred better than linear (ω + κ)" test is the cleanest single experiment that would distinguish the strong cognitive claim from the modest one. It takes the same kind of bambi model comparison I've been running. If confidence reads an integrated model quantity, the architecture claim is real. If it doesn't, fall back to Option 1.

Either way, **stop selling the current ΔELPD result as a trial-level computational dissociation**. It's a trait-level result with high pooling power, and saying otherwise is overclaiming.
