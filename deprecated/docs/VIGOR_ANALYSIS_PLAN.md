# Vigor Analysis Plan: Motor Execution Along the Threat Imminence Continuum

## Three Questions

1. **Do people exert more vigor in response to threat?** (controlling for cookie choice)
2. **What happens at encounter?** (the moment threat becomes real)
3. **What happens after encounter?** (terminal persistence and collapse)

## Lessons From Prior Analysis

**Use these metrics:** median IPI (raw, not smoothed), frac_full (fraction at full speed), pause frequency, relative vigor (rate/req).
**Always condition on cookie type** or the Simpson's paradox will mask everything.
**Use 5Hz (200ms bins)** from raw keypresses for timecourses — the native data resolution. Smooth with 3-point moving average (600ms) for display only.
**Encounter effect = attack minus non-attack** at the same timepoint, computed per-subject first.
**Heavy and light cookies tell different stories:** heavy has variance in frac_full (mean 0.60, range 0-1); light is at ceiling (mean 0.97). Threat effects in vigor are visible in both but for different reasons.

---

## Vigor Metrics

Compute ALL of the following per trial from raw keypress timestamps (`alignedEffortRate`):

### M1: Median IPI
```
ipis = diff(timestamps)
ipis = ipis[ipis > 0.01]  # filter artifacts
median_ipi = median(ipis)
```
Lower = faster pressing. Most direct measure of motor output. In seconds.

### M2: Normalized press rate
```
rates = (1 / ipis) / calibrationMax
median_rate = median(rates)
```
Rate in proportion of calibrated maximum. Scale: 0 to ~1.5.

### M3: Relative vigor
```
relative_vigor = median_rate / required_rate
```
Scale: 0 to ~3. 1.0 = pressing at exactly the required rate (full speed threshold). Removes cookie confound by dividing by the cookie's demand.

### M4: Fraction at full speed
```
frac_full = mean(rates >= required_rate)
```
Scale: 0 to 1. Proportion of keypresses at or above the full-speed threshold. Most relevant for survival (r=0.28 with escape). Light cookies at ceiling (~0.97).

### M5: Press rate variability
```
press_sd = std(rates)
press_cv = press_sd / mean(rates)
```
Higher = more inconsistent pressing. Independently predicts capture (OR=0.27 controlling for mean rate). Momentary dips below threshold cause speed drops.

### M6: Pause frequency
```
long_ipis = sum(ipis > threshold)  # e.g., > 0.5s or > 2× median IPI
pause_rate = long_ipis / n_ipis
```
Scale: 0 to 1. How often the person has a significant gap in pressing. Captures "giving up" moments within the trial.

### M7: Press count
```
n_presses = len(timestamps)
```
Simple count. More presses = more vigor. Confounded with trial duration but useful as a sanity check.

---

## Epoch Definitions

Use actual timing from the data, not hardcoded values.

| Epoch | Start | End | What it captures |
|-------|-------|-----|-----------------|
| **Onset** | First keypress | encounterTime | Ramp-up period. How fast do you get to cruising speed? |
| **Anticipatory** | encounterTime − 1s | encounterTime | Steady-state pressing just before the predator could appear |
| **Reactive** | encounterTime | encounterTime + 2s | Immediate response to predator appearance (attack trials) or continued pressing (non-attack) |
| **Terminal** | strike_time − 2s | strike_time | Final pressing before predator strikes. Attack trials only. |

**Important:** For non-attack trials, encounterTime is the scheduled (but never triggered) encounter time. Participants don't know whether it's an attack trial during pressing, so their anticipatory behavior reflects the stated probability.

---

## Part 1: Does Threat Modulate Vigor? (Controlling for Choice)

### Analysis 1.1: Trial-level vigor by threat, within cookie type

For each metric (M1–M7), compute per-trial values. Run two mixed models — one for heavy cookies, one for light:

```
metric ~ threat + distance + trial_number + (1 | participant)
```

Report the threat coefficient for each metric × cookie type combination (14 tests). Apply FDR correction within this family.

**Expected:** Significant threat effect for M2, M3, M4 within heavy cookies. Light cookies may show ceiling effects for M4 (frac_full ≈ 0.97 at all threat levels).

### Analysis 1.2: Within-subject paired comparison

For each participant, compute mean of each metric at T=0.1 vs T=0.9, separately for heavy and light cookies. Paired t-test.

Report: mean difference, t-value, p-value, Cohen's d for each metric × cookie type.

**This is the cleanest test** — no mixed model assumptions, just "does the same person press differently at high vs low threat when they chose the same cookie type?"

### Analysis 1.3: Relative vigor 3×3 surface

Compute mean relative vigor for each of the 9 T×D conditions, within heavy and within light separately. This is the vigor analog of the choice PPC — a 3×3 table showing how vigor varies across the design.

Report as two 3×3 tables (heavy and light) with means ± SE.

---

## Part 2: What Happens at Encounter?

### Analysis 2.1: Encounter-aligned timecourse

Using 200ms bins from raw keypresses, compute relative vigor at each bin from t_enc = -2s to +4s.

**Two versions:**
- (a) By threat level (T=0.1, 0.5, 0.9) — shows whether threat modulates the encounter response
- (b) Attack vs non-attack difference (per-subject) — isolates the encounter effect

Plot both at 5Hz with 600ms moving average. Report peak encounter effect (time and magnitude).

### Analysis 2.2: Per-subject encounter spike

For each participant, compute the encounter spike for each metric:
```
spike = mean(metric, attack trials, t_enc ∈ [0, 2]) − mean(metric, non-attack trials, same window)
```

Report:
- Population mean spike with t-test vs 0
- Between-subject SD
- Whether spike varies with threat level (ANOVA or paired t)
- Correlation with model parameters (ce, cd)
- Correlation with survival rate

**Key prediction:** Spike is positive (people press harder when predator appears), driven by cd not ce, and threat-independent.

### Analysis 2.3: Encounter spike by metric

Compare the encounter spike across all 7 metrics. Which metric shows the cleanest encounter effect? Rank by effect size (Cohen's d).

| Metric | Expected encounter effect |
|--------|------------------------|
| Median IPI | Decrease (faster pressing) |
| Normalized rate | Increase |
| Relative vigor | Increase |
| Frac full | Increase (more time at full speed) |
| Press SD | Decrease (more consistent) |
| Pause frequency | Decrease (fewer gaps) |
| Press count | Increase (more presses per unit time) |

### Analysis 2.4: Onset of the encounter effect

At what time after encounter does the vigor change become significant? For the best metric from 2.3, test each 200ms bin post-encounter against the pre-encounter baseline (paired t-test). Report the first bin that reaches p < .05 (uncorrected) — this is the reaction time of the motor system to predator appearance.

---

## Part 3: Terminal Persistence

### Analysis 3.1: Terminal vigor by threat

On attack trials only, compute each metric in the terminal epoch (strike_time − 2s to strike_time). Compare across threat levels.

**Key question:** Does threat still modulate vigor in the terminal phase, or has it dropped out? (Prior finding: threat drops out of terminal, z=−0.50, p=.618 for normalized press rate.)

Check all 7 metrics — the answer may differ by metric. Frac_full or pause_rate might show terminal threat effects that median rate doesn't.

### Analysis 3.2: Terminal collapse

How many people show a press-rate collapse in the terminal phase? Define collapse as:
```
collapse = (terminal_rate < 0.5 × anticipatory_rate)
```

Report:
- Prevalence of collapse (% of attack trials, % of participants who collapse on ≥1 trial)
- Does collapse vary with threat? With distance? With cookie type?
- Do collapsers differ from non-collapsers on ce, cd, or clinical measures?

### Analysis 3.3: Persistence metric

Define terminal persistence as:
```
persistence = terminal_rate / anticipatory_rate
```

Scale: 1.0 = maintained pressing, <1.0 = slowed down, >1.0 = sped up.

Report:
- Mean persistence by threat × distance
- Does cd predict persistence? (Expected: high cd → higher persistence)
- Does persistence predict survival?

### Analysis 3.4: Terminal acceleration vs deceleration

Split trials into those where the person sped up in terminal (persistence > 1) vs slowed down (persistence < 1).

Report the proportions by threat level. Does threat cause people to speed up (fight response) or slow down (freeze/give-up)?

---

## Part 4: Figures

### Figure A: Trial-level vigor surface
Two 3×3 heatmaps: relative vigor by T×D for heavy and light cookies separately. Shows the main effect of threat on vigor, conditioned on choice.

### Figure B: Encounter-aligned timecourse
3 lines by threat level (relative vigor, 5Hz + 600ms smooth). Mark encounterTime with vertical line. Shows pre-encounter ramp, encounter response, and post-encounter dynamics.

### Figure C: Encounter spike
Left: attack vs non-attack difference timecourse. Right: scatter of per-subject spike vs cd.

### Figure D: Terminal dynamics
Persistence metric by threat × distance. Or: timecourse from encounter to strike showing how vigor evolves in the post-encounter period.

---

## Controlling for Choice

The Simpson's paradox means we must handle cookie type carefully. Three approaches, all run:

**Approach 1: Separate by cookie type.** Run every analysis twice — once for heavy, once for light. Cleanest. May lose power for light (ceiling effects).

**Approach 2: Cookie-type as covariate.** Include cookie_type as a fixed effect (and random slope) in mixed models. The coefficient on threat then represents the within-cookie-type threat effect.

**Approach 3: Forced-choice (probe) trials only.** 36 probe trials per person where cookie is randomly assigned. No selection bias. Smaller N but unconfounded. Run key analyses on probes as a robustness check.

---

## Execution Order

1. Compute all 7 metrics per trial per epoch from raw keypresses (one big computation pass)
2. Part 1: Trial-level threat modulation (the basic finding, should be quick)
3. Part 2: Encounter dynamics (the temporal dynamics)
4. Part 3: Terminal behavior (the least explored)
5. Part 4: Figures

Each part produces a clear output table and at least one figure panel. Stop and assess after Part 1 — if the basic threat effect isn't there in any metric after controlling for choice, the vigor story is weaker than we think.

---

## What We Already Know Will Work

- Relative vigor increases with threat within cookie type (d ≈ 0.42-0.49)
- frac_full predicts survival (r=0.28) and is predicted by cd (r=0.71)
- The encounter spike is real (d=0.56), cd-linked (r=0.25), threat-independent
- Press variability independently predicts capture (OR=0.27)
- Terminal threat drops out in epoch analysis (p=.618)
- The imminence gradient: β → anticipatory, cd → reactive

## What We Don't Know Yet

- Which metric shows the CLEANEST encounter effect
- Whether any metric shows terminal threat modulation (median rate didn't, but frac_full or pauses might)
- The reaction time of the encounter response (first significant 200ms bin)
- Terminal collapse prevalence and what predicts it
- Whether pause frequency captures something the other metrics miss
- Whether the findings hold on probe trials (unconfounded by choice)
