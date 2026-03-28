# Control of Movement Vigor and Decision Making During Foraging

**Source:** Yoon, T., Geary, R. B., Ahmed, A. A., & Shadmehr, R. (2018). Control of movement vigor and decision making during foraging. *PNAS, 115*(44), E10476–E10485.

---

## Core Claim

Both **how long to stay** at a reward site (harvest duration — the decision-making problem) and **how fast to travel** to the next site (movement vigor — the motor control problem) can be derived from a single normative principle: **maximizing the global capture rate** — the sum of all rewards acquired minus all efforts expended, divided by total time. The brain achieves this by comparing a local measure of utility with its history.

---

## The Foraging Framework

### Optimal Foraging Theory

Animals distribute effort across patches of reward. The classic optimization objective is to maximize the **global capture rate** J̄:

```
J̄ = Σ[f(tₕ) - uₘ(d, tₘ)] / Σ[tₕ + tₘ]
```

Where:
- **f(tₕ):** Harvest function — reward accumulated minus effort expended during harvest time tₕ
- **uₘ(d, tₘ):** Energy expended during travel of distance d over duration tₘ
- **tₕ:** Harvest duration (how long to stay)
- **tₘ:** Movement duration (how fast to travel)

### Marginal Value Theorem (MVT)

The classic MVT (Charnov, 1976) provides the decision rule for when to leave a patch: leave when the local capture rate drops to the global average. However, MVT assumes travel time is independent of harvest decisions and does not address movement vigor.

---

## The Generalized Theory

### Two Control Variables

The subject controls:
1. **tₕ** — how long to stay at each patch (decision making)
2. **tₘ** — how fast to move between patches (motor control)

Both are selected to maximize J̄.

### The Harvest Function

During harvest, reward accumulates via a saturating function minus ongoing effort:

```
f(tₕ) = α(1 - 1/(1 + βtₕ)) - k·tₕ
```

Where:
- **α:** Total reward available at the patch
- **β:** Rate of reward accumulation
- **k:** Effort rate (e.g., effort of maintaining gaze at an eccentric location)

Critical assumption: the harvest function is **concave downward** — initially increasing but with diminishing returns.

### The Movement Effort Function

Energy expenditure during movement is a **concave upward** function of movement duration. This is justified empirically: energetic cost during walking, running, and reaching is concave upward with duration.

### Optimality Conditions

**When to leave the patch:**
```
df/dtₕ = J̄   (leave when local harvest rate equals global capture rate)
```

**How fast to move:**
```
duₘ/dtₘ = -J̄   (move so that rate of energy loss equals negative of global capture rate)
```

These two conditions are coupled through J̄ — the same global capture rate governs both decisions and movements.

---

## Predictions

### Effects on Harvest Duration (tₕ)

| Factor | Effect on tₕ | Mechanism |
|---|---|---|
| Current reward ↑ | tₕ ↑ (stay longer) | Higher α shifts harvest function up |
| Past reward ↑ | tₕ ↓ (stay shorter) | Higher J̄ raises the leaving threshold |
| Future reward ↑ | tₕ ↓ (stay shorter) | Higher J̄ raises the leaving threshold |
| Current effort ↑ | tₕ ↓ (stay shorter) | Higher k reduces net harvest |
| Past effort ↑ | tₕ ↑ (stay longer) | Lower J̄ lowers the leaving threshold |
| Future effort ↑ | tₕ ↑ (stay longer) | Lower J̄ lowers the leaving threshold |

### Effects on Movement Vigor (1/tₘ)

| Factor | Effect on vigor | Mechanism |
|---|---|---|
| Future reward ↑ | Vigor ↑ (move faster) | Higher J̄ |
| Past reward ↑ | Vigor ↑ (move faster) | Higher J̄ |
| Future effort ↑ | Vigor ↓ (move slower) | Lower J̄ |
| Past effort ↑ | **Predicted:** Vigor ↓ | Lower J̄ |
| | **Observed:** Vigor ↑ | Possible "justification of effort" effect |

---

## Experimental Design

The experiments use **image viewing** as a foraging analog:
- **Reward patches** = small images on a screen
- **Harvesting** = gazing at the image (accumulating visual information)
- **Travel** = saccadic eye movements between images
- **Reward magnitude** = image content (faces > objects > shapes > noise)
- **Effort** = image eccentricity (more eccentric = more effort to hold gaze)

### Five Experiments

1. **Exp. 1 (n=16):** Controlled harvest duration; showed harvest function is concave downward (shorter gaze → faster saccades; longer gaze → slower saccades)
2. **Exp. 2 (n=19):** History of short harvests produced lasting high vigor in control trials
3. **Exp. 3 (n=17):** Two simultaneous images; confirmed reward and effort effects on both gaze duration and saccade vigor
4. **Exp. 4 (n=22):** Single image, unlimited viewing time; confirmed all predictions for harvest duration and most for vigor
5. **Exp. 5 (n=18):** Manipulated effort history via blocks; demonstrated that past high effort increased both gaze duration and vigor in probe trials

---

## Key Results

### Confirmed Predictions

**Harvest duration:**
- Current reward ↑ → longer gaze ✓
- Past reward ↑ → shorter gaze ✓
- Current effort ↑ → shorter gaze ✓
- Past effort ↑ → longer gaze ✓
- Future effort ↑ → longer gaze ✓

**Movement vigor:**
- Future reward ↑ → faster saccades ✓
- Past reward ↑ → faster saccades ✓
- Future effort ↑ → slower saccades ✓
- History of high reward rates → persistently high vigor ✓

### Failed Prediction: Past Effort and Vigor

The theory predicted that past high effort should **decrease** vigor (because high effort lowers J̄). Instead, subjects consistently **increased** vigor after high effort. This may reflect **justification of effort** — the subjective value of a reward increases with the effort invested in acquiring it. Experiment 5 confirmed this: in probe trials, high-effort history produced both longer gaze (consistent with increased subjective reward value) and faster saccades.

---

## Theoretical Significance

### Unification of Decision Making and Motor Control

The key theoretical contribution is showing that the **same normative principle** can govern both:
- **When to stop harvesting** (a decision-making problem, traditionally studied by behavioral ecologists)
- **How fast to move** (a motor control problem, traditionally studied by motor neuroscientists)

Both are controlled via a comparison between the **local capture rate** (reward and effort at the current action) and the **global capture rate** (history and expectations).

### Contrast with Previous Vigor Theories

Previous approaches to movement vigor proposed that:
- Movement duration discounts reward value, creating a local tradeoff between temporal cost (move fast) and effort cost (move slow).
- These theories cannot account for **history-dependent** changes in vigor because they only consider the immediate movement.

The foraging framework naturally produces history dependence because the global capture rate J̄ is shaped by cumulative experience.

---

## Neural Substrates

### Harvest Duration Decision

- **Frontal eye field (foveal neurons):** Encode effort rate (eccentricity) during fixation
- **Cingulate cortex:** Encodes value of leaving; activity rises to threshold, triggering saccade
  - Rate of rise: slower when travel effort is high, faster when environment has low reward rates
- **Frontal eye field (saccade neurons):** Modulated by expected reward at destination

### Movement Vigor

- **Basal ganglia:** Central to vigor control
  - **SNr:** Deeper pause → more vigorous saccades; reward modulates pause depth
  - **Caudate:** Receives dopamine; fires more before rewarding saccades
  - **GPe:** Fires more strongly before vigorous saccades; bilateral lesion eliminates reward-based vigor modulation
- **Dopamine:** Phasic burst to reward-promising stimuli → more vigorous movement; chronic depletion → ~30% vigor reduction

### Open Neural Questions

- Niv et al. (2007) proposed that tonic dopamine encodes reward history (J̄), but direct evidence is limited.
- Reward history may instead be reflected in tonic serotonergic neuron discharge.
- Recent evidence shows that phasic dopamine before movement onset influences upcoming movement velocity.

---

## Limitations

1. The shape of the harvest function during image gazing is inferred but not directly measured.
2. The assumption that saccade effort is concave upward with duration needs direct testing.
3. Saccades and gaze occur on millisecond timescales vs. minutes for traditional foraging — the scaling of these principles across timescales is assumed but not proven.
4. The "justification of effort" effect was not predicted by the basic theory and requires incorporation of effort-dependent reward valuation.
