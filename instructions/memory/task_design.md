---
name: Task Design
description: Full description of the virtual foraging task — structure, mechanics, stimuli, probes, model variables
type: reference
---

# Task Design: Effort-Threat Foraging Paradigm

## Overview

A virtual foraging task implemented in Unity (WebGL), run online via Prolific desktop browser.
Participants forage for rewards (cookies) in a bounded circular arena viewed top-down.
The task simultaneously manipulates energetic cost and probabilistic predation risk.

**Two online samples:** Exploratory N=293, Confirmatory N=350 (same task design).

---

## Arena Layout

- **Shape:** Bounded circular arena, top-down view
- **Safe zone:** Center of the arena
- **Cookies:** Placed at radial distances from center (i.e., from safety)
- **Predator:** Spawns at the perimeter position closest to participant on attack trials

---

## Block and Trial Structure

- **3 blocks × 27 trials = 81 total events per subject**
- Of the 27 trials per block: **15 are regular choice trials**, **12 are probe trials** (6 anxiety + 6 confidence)
- Total: 45 regular behavioral trials + 36 probe events = 81 events (matches `trial` index 0–80)
- Between blocks: brief effort recalibration + embedded psychiatric questionnaires

### Regular trial flow
1. Two cookies presented simultaneously at different positions in the arena
2. Participant **clicks** chosen cookie to select it (commits — cannot abandon)
3. Participant **presses S+D+F repeatedly** to transport cookie back to safe zone (center)
4. Trial ends when cookie returned (escape) or predator intercepts (capture)

### Probe trial flow
1. Both options are **identical** — same reward magnitude, weight, distance, threat probability
2. Participant clicks one (**forced choice** — selection is uninformative by design)
3. One probe question appears **after the choice is selected**:
   - Anxiety: "How anxious are you about being captured on this trial?"
   - Confidence: "How confident are you in your ability to reach safety on this trial?"
4. Probe trial still has live predator dynamics — participant must still escape after rating
5. **Ratings are prospective/concurrent**: about the demands of the CURRENT trial

---

## Effort Calibration

- Pre-task: press S+D+F simultaneously as fast as possible, 3 × 10-second trials
- `calibrationMax` (= `f_max_i`) = maximum presses achieved across trials
- Minimum threshold of 10 presses; failure = exclusion
- All effort demands are defined relative to this individual calibrated maximum

---

## Effort Manipulation

Cookie **weight** determines the required press rate as a fraction of calibrated max:
- Heavy cookie = 100% of calibrated max
- Light cookie = 40% of calibrated max

Cookie **distance** determines how long effort must be sustained (farther = longer pressing needed).

Together, weight × distance determine the **effort level** (`effort_H`):
- Low-effort option (L): always light + close = **40% capacity** (`effort_L = 0.4`)
- High-effort option (H): varies = **60%, 80%, or 100%** of capacity (`effort_H ∈ {0.6, 0.8, 1.0}`)

`distance_H` (1, 2, 3) = discrete distance level of the high-effort cookie.
Maps to the `D` parameter in the survival function: D = distance_H.
Physical distances: D=1 → 5 game units, D=2 → 7 game units, D=3 → 9 game units.
(Only relative magnitudes matter, not absolute values.)

---

## Reward Structure

- **High-effort option:** R_H = **5 points** (fixed across all trials)
- **Low-effort option:** R_L = **1 point** (fixed across all trials)

On capture: participant loses the **current trial's cookie reward + a penalty of 5 points**.
→ Total cost of capture = reward lost + 5 penalty.
Capture penalty C in the SV formula = **5** (matching the high-effort reward).

---

## Movement Dynamics (Vigor Mechanics)

Pressing rate is continuously mapped to movement speed via discrete tiers:

| Press rate (% of required rate for that cookie) | Movement speed |
|----------------------------------------------|----------------|
| ≥ 100% | Full speed |
| ≥ 50%  | Half speed |
| ≥ 25%  | Quarter speed |
| < 25%  | Zero (movement decays to stop) |

**Key implication:** Vigor (pressing rate) directly determines movement speed → determines whether
participant reaches safety before predator. Pressing is the escape behavior, not a proxy.

Participants **cannot abandon a choice** — once a cookie is selected, they must transport it.

---

## Predator Dynamics and Timeline

Each trial has threat probability T ∈ {0.1, 0.5, 0.9} governing whether predator attacks.

### Three-phase timeline (on attack trials)

```
Trial start          encounterTime           Strike time               Trial end
     |                     |                      |                        |
     |←── no predator ────►|←── predator visible ►|←── full attack speed ─►|
     |    visible           |    slow approach     |    4× player max       |
     |    (all trials)      |    (attack only)     |    pursues directly    |
```

**Phase 1 — No predator visible [trial start → encounterTime]:**
- No predator on screen on ANY trial (attack or non-attack)
- This is the anticipatory period — threat is possible but not manifest
- Corresponds to `t ∈ [0, encounterTime]` in vigor analysis

**Phase 2 — Predator appears [encounterTime → strike]:**
- On attack trials: predator spawns at perimeter, becomes visible, **moves slowly (0.5 units/sec)**
- On non-attack trials: predator never appears (encounterTime = scheduled but never fires)
- `encounterTime` is set for ALL trials (attack and non-attack); on non-attack trials it is the
  scheduled appearance time that was randomly drawn but never activated
- Provides an anticipatory threat cue; participant sees predator approaching
- Onset calibrated to ≈ half the expected return time at full speed (adequate warning time regardless of distance)

**Phase 3 — Full attack [strike → trial end]:**
- Predator accelerates to **4× participant's calibrated max movement speed**
- Pursues participant directly — cannot be outrun
- Strike timing drawn from Gaussian centered at 2× encounterTime (introduces temporal uncertainty)
- Participant must reach safe zone before interception

**`startDistance`** (5, 7, 9 game units): Predator starting position distance from participant.
Controls warning time — larger startDistance means more time to detect before strike.

### On non-attack trials
- No predator ever appears on screen
- `encounterTime` still recorded (scheduled but inactive)
- `isAttackTrial = 0`; participant just returns cookie normally

---

## Self-Report Probes

Probe trials assess subjective state for specific factorially-controlled (threat × distance × effort) cells.

**Questions:**
- Anxiety: "How anxious are you about being captured on this trial?"
- Confidence: "How confident are you in your ability to reach safety on this trial?"
- Response scale: 0–7

**Design:** 9 probe types spanning all combinations of threat (0.1, 0.5, 0.9) × distance (near/mid/far).
Per block: 6 anxiety probes + 6 confidence probes = 12 probe trials.
Total: 18 anxiety + 18 confidence = 36 probes per subject.

**Probe timing:** Rating collected AFTER the participant clicks a cookie (choice committed) but
BEFORE pressing begins — the game **pauses** for the rating. The rating is therefore a clean
prospective judgment about the upcoming execution demands, uncontaminated by ongoing motor effort.

**`distanceFromSafety`** in `feelings.csv` (values 4, 5, 7, 9 game units): Physical distance
of the probe trial's cookie from safe zone.
**`distance`** in `feelings.csv` (0, 1, 2 → D = 1, 2, 3): Discrete distance level used in model.

---

## Model Variables: Mapping to Data Columns

| Model symbol | Data column | Values | Meaning |
|---|---|---|---|
| T | `threat` / `attackingProb` | 0.1, 0.5, 0.9 | Threat probability |
| D | `distance_H` / `distance`+1 | 1, 2, 3 | Cookie distance (high-effort option or probe) |
| E | `effort_H` | 0.6, 0.8, 1.0 | Effort fraction of calibrated max |
| R_H | — | 5 (fixed) | High-effort option reward |
| R_L | — | 1 (fixed) | Low-effort option reward |
| C | — | 5 (fixed) | Capture penalty (SV formula) |
| z_i | fitted per subject | — | Hazard sensitivity (distance → danger nonlinearity) |
| k_i / κ_i | fitted per subject | — | Effort discounting rate |
| β_i | fitted per subject | — | Residual threat bias (beyond EV) |
| S_probe | computed | — | `exp(-p_threat × D^z_i)` per probe trial |
| f_max_i | `calibrationMax` | — | Calibrated max press rate (from effort_ts.parquet) |

---

## Vigor Analysis: Phase Definitions

Given the three-phase trial structure, vigor phases map as follows:

| Vigor phase | Window | What it captures |
|---|---|---|
| **Onset** | t ∈ [0, 2s] from trial start | Anticipatory mobilization — before predator appears |
| **Encounter** | t_enc ∈ [−1.5, +1.5s] around encounterTime | Response to predator APPEARING (slow approach visible) |
| **Terminal** | last 2s before trial end | Final escape effort under full attack speed |

**Important:** The encounter spike in vigor captures the response to the predator becoming VISIBLE
(slow approach phase), NOT the response to the full attack strike. The strike happens at an
uncertain time after encounterTime (Gaussian-distributed).

---

## Psychiatric Battery (administered between blocks)

| Measure | Subscales used | Construct |
|---|---|---|
| DASS-21 | Stress, Anxiety, Depression | Affective distress |
| PHQ-9 | Total | Depressive symptoms |
| OASIS | Total | Anxiety severity and impairment |
| STAI-T | Trait only | Stable trait anxiety |
| AMI | Behavioural, Social, Emotional | Apathy/motivational traits |
| MFIS | Physical, Cognitive, Psychosocial | Fatigue impact |
| STICSA | Total | Somatic and cognitive anxiety |

All z-scored across participants before analysis.
Available in `data/exploratory_350/processed/stage5_filtered_data_*/psych.csv`.

---

## Key Design Facts for Interpretation

- **Vigor is survival behavior, not a proxy.** Pressing rate = movement speed = escape probability.
  Faster pressing directly increases survival on attack trials.
- **Low-effort option is always identical across trials** (40% effort, D=1). Only the high-effort
  option varies in weight × distance.
- **R_H = 5, R_L = 1, C = 5**: Capture costs as much as a high-effort reward, making avoidance
  rational at high threat levels.
- **Probe ratings are concurrent, not retrospective.** The participant has committed to a choice
  (click) and knows exactly what demands they face. The rating reflects expected anxiety/confidence
  about the trial they are about to execute.
- **encounterTime is set for all trials**, including non-attack. On non-attack trials it represents
  the time when a predator WOULD have appeared. This enables attack vs. no-attack contrasts
  using the same temporal reference frame.
- **Encounter spike ≠ strike response.** The vigor response around encounterTime reflects
  seeing the predator appear (and begin slow approach), not the rapid-pursuit phase.
- **Distance confounds effort and exposure.** D=distance_H drives BOTH how long pressing is
  needed (effort duration) AND survival probability via D^z. These are inseparable by design —
  farther cookies require more sustained effort AND more exposure to threat.
- **No abandonment allowed.** Once a cookie is clicked, participants must commit to returning it.
  The model's SV computation therefore reflects a committed policy, not an option to retreat.
