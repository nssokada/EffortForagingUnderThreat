# Vigor reactivity report: model parameters and affect dissociate across anticipatory and reactive defensive phases

## Summary

Two temporal phases of motor vigor — anticipatory ramp-up and the reactive encounter spike — are governed by different combinations of model parameters and affective signals. The model parameters and affect contribute independently, and which layer dominates depends on the phase of defense.

---

## 1. Anticipatory reactivity (threat-graded pre-encounter vigor)

Anticipatory reactivity is defined as the within-subject difference in mean pre-encounter pressing rate between high-threat (T = 0.9) and low-threat (T = 0.1) trials.

### Model parameters

Avoidance sensitivity (omega) predicted anticipatory reactivity in both samples (exploratory: beta = +0.215, 95% HDI [+0.094, +0.337]; confirmatory: beta = +0.188, 95% HDI [+0.066, +0.318]). Activation intensity (kappa) did not (both HDIs including zero). Higher omega — greater weighting of capture cost — produced larger threat-graded increases in pre-encounter effort. This effect survived the addition of all affect betas to the model.

### Affect

Affect betas added negligibly to anticipatory reactivity beyond omega and kappa (ΔELPD = +0.2 exploratory, -1.8 confirmatory). In the exploratory sample, higher baseline anxiety (anxiety intercept) predicted *less* anticipatory reactivity (beta = -0.144, 95% HDI [-0.260, -0.032]), but this did not replicate in the confirmatory sample. In confirmatory, higher baseline confidence predicted *more* anticipatory reactivity (beta = +0.144, 95% HDI [+0.021, +0.259]), but this did not replicate in exploratory. No affect reactivity betas (slopes on threat or distance) predicted anticipatory vigor reactivity in either sample.

### Interpretation

Anticipatory vigor adjustment is primarily driven by the computational model's avoidance parameter. Omega determines how much a participant ramps up pressing before the predator appears, consistent with a value-based anticipatory preparation governed by the cost-weighting computation. Affect contributes little beyond this — the anticipatory phase is a computational output, not an affective one.

---

## 2. Encounter spike (reactive motor acceleration at predator appearance)

The encounter spike is defined as the within-subject difference between mean post-encounter and mean pre-encounter pressing rate on attack trials.

### Model parameters

Activation intensity (kappa) predicted the encounter spike in both samples (exploratory: beta = -0.320, 95% HDI [-0.442, -0.201]; confirmatory: beta = -0.195, 95% HDI [-0.320, -0.069]). Avoidance sensitivity (omega) did not (both HDIs including zero). Higher kappa — greater effort cost sensitivity — constrained the reactive motor response: participants who find effort costly mobilized less even under immediate threat. This is the opposite pattern from anticipatory reactivity, where omega drove the effect and kappa was null.

### Affect

Affect betas added substantially to encounter spike prediction beyond omega and kappa (ΔELPD = +10.3 exploratory, +20.8 confirmatory). Three affect signals survived in the joint model controlling for omega and kappa, and all three replicated:

**Anxiety intercept (tonic anxiety) amplified the spike.** Higher baseline anxiety predicted a larger encounter spike (exploratory: beta = +0.126, 95% HDI [+0.019, +0.233]; confirmatory: beta = +0.163, 95% HDI [+0.051, +0.269]). Participants with elevated tonic anxiety responded more strongly when the predator appeared.

**Anxiety reactivity to threat (phasic anxiety) reduced the spike.** Participants whose anxiety adjusted more steeply to threat probability had a *smaller* encounter spike (exploratory: beta = -0.182, 95% HDI [-0.295, -0.062]; confirmatory: beta = -0.159, 95% HDI [-0.275, -0.038]). This is the key dissociation: tonic and phasic anxiety predict the reactive motor response in opposite directions. The interpretation is that phasic anxiety — condition-sensitive threat tracking — enables anticipatory preparation that reduces the surprise of the encounter. If your anxiety was already elevated because you knew this trial was dangerous, the predator's appearance is less novel and triggers a smaller reactive surge.

**Confidence reactivity to distance amplified the spike.** Participants whose confidence dropped more steeply with distance had a larger encounter spike (exploratory: beta = +0.205, 95% HDI [+0.084, +0.320]; confirmatory: beta = +0.189, 95% HDI [+0.075, +0.299]). This suggests that sensitivity to exposure demands — registering that a distant trial is more dangerous — primes a stronger reactive defense when the predator materializes. These participants are tracking their vulnerability and mobilizing accordingly.

An additional effect emerged in the confirmatory sample only: lower baseline confidence (confidence intercept) predicted a larger spike (beta = -0.141, 95% HDI [-0.249, -0.032]). This did not replicate in exploratory and should be treated as tentative.

### Interpretation

The reactive encounter spike is jointly governed by the motor capacity constraint (kappa) and affective state. Kappa sets the ceiling — how much mobilization is possible given the participant's effort-cost sensitivity. Within that ceiling, the spike is amplified by tonic anxiety (baseline arousal) and by confidence sensitivity to exposure (vulnerability tracking), and attenuated by phasic anxiety (threat-calibrated preparation). The encounter spike is therefore not a reflexive, affect-independent startle. It is shaped by how the participant was feeling in the moments before the encounter, with well-calibrated anxiety reducing the need for reactive mobilization and vulnerability awareness amplifying it.

---

## 3. The full architecture

|                          | Anticipatory ramp-up         | Encounter spike              |
|--------------------------|------------------------------|------------------------------|
| **omega (avoidance)**    | Drives it (beta ~ +0.20)     | Null                         |
| **kappa (effort cost)**  | Null                         | Constrains it (beta ~ -0.26) |
| **Anxiety level**        | Inconsistent                 | Amplifies it (beta ~ +0.14)  |
| **Anxiety reactivity**   | Null                         | Reduces it (beta ~ -0.17)    |
| **Confidence reactivity**| Null                         | Amplifies it (beta ~ +0.20)  |
| **Affect adds beyond params** | No (ΔELPD ~ 0)         | Yes (ΔELPD = +10 to +21)    |
| **Params add beyond affect**  | Yes (ΔELPD = +3 to +4) | Yes (ΔELPD = +6 to +14)     |

The two phases of defensive vigor are governed by different layers of the architecture:

- **Anticipatory vigor** is a value-based, model-governed process. Omega determines how much you prepare. Affect adds nothing. This is the fitness function operating as designed — computing trial-level value and adjusting motor output accordingly.

- **The encounter spike** is a jointly determined process in which the model parameter kappa sets the motor ceiling and affect determines how much of that ceiling is used. Tonic anxiety (high baseline arousal) and confidence-based vulnerability tracking amplify the response; phasic anxiety (good threat calibration) attenuates it. The encounter spike is therefore not a simple reflexive startle but a modulated defensive response shaped by both motor capacity and the participant's affective state.

This dissociation connects the computational model to the predatory imminence framework: the anticipatory phase reflects pre-encounter defense (value-based, graded, model-governed), while the reactive phase reflects circa-strike defense (motor-limited, affect-modulated, partially reflexive). The model parameters and affective signals carve the defensive response along the same temporal boundary that the predatory imminence literature predicts.

---

## 4. Statistical details

All regressions fitted with bambi (Capretto et al., 2022), 4 chains x 2,000 draws + 1,000 tuning, default weakly informative priors. Predictors and outcomes z-scored within sample. Model comparison via LOO-CV (Vehtari et al., 2017). Effects reported only if the 95% HDI excludes zero. Replication across both samples required for all primary claims; single-sample effects noted as tentative.

### Raw numbers

**Encounter spike ~ omega + kappa + affect (joint model):**

| Predictor | Exploratory beta [95% HDI] | Confirmatory beta [95% HDI] |
|---|---|---|
| omega | +0.093 [-0.030, +0.216] | +0.012 [-0.113, +0.138] |
| kappa | -0.336 [-0.453, -0.221] ★ | -0.230 [-0.344, -0.111] ★ |
| anxiety intercept | +0.126 [+0.019, +0.233] ★ | +0.163 [+0.051, +0.269] ★ |
| anxiety slope (T) | -0.182 [-0.295, -0.062] ★ | -0.159 [-0.275, -0.038] ★ |
| anxiety slope (D) | -0.024 [-0.142, +0.094] | -0.066 [-0.180, +0.054] |
| confidence intercept | +0.032 [-0.080, +0.147] | -0.141 [-0.249, -0.032] ★ (conf only) |
| confidence slope (T) | +0.065 [-0.055, +0.180] | +0.103 [-0.012, +0.217] |
| confidence slope (D) | +0.205 [+0.084, +0.320] ★ | +0.189 [+0.075, +0.299] ★ |

**Anticipatory reactivity ~ omega + kappa + affect (joint model):**

| Predictor | Exploratory beta [95% HDI] | Confirmatory beta [95% HDI] |
|---|---|---|
| omega | +0.224 [+0.104, +0.351] ★ | +0.199 [+0.078, +0.324] ★ |
| kappa | +0.044 [-0.077, +0.164] | -0.047 [-0.172, +0.076] |
| anxiety intercept | -0.144 [-0.260, -0.032] ★ (exp only) | -0.066 [-0.183, +0.053] |
| confidence intercept | -0.020 [-0.144, +0.098] | +0.144 [+0.021, +0.259] ★ (conf only) |
| All other affect betas | HDIs include zero | HDIs include zero |
