# Effort Cost of Harvest Affects Decisions and Movement Vigor During Foraging

**Source:** Hage, Jang, Looi, Fakharian, Orozco, Pi, Sedaghat-Nejad & Shadmehr (2023). *eLife*, 12:RP87238.

---

## Central Thesis

During foraging, decision-making (what to do and for how long) and motor control (how vigorously to move) are coordinated through a single optimization objective: **maximizing the capture rate** (rewards − efforts) / time. Increased effort costs should simultaneously alter decisions (work longer, delay gratification) and movements (reduce vigor to conserve energy). Pupil size tracks these coordinated changes in real time.

---

## The Foraging Task

Marmosets performed a work–harvest cycle:

- **Work period:** Make sequences of visually guided saccades to targets. Each successful trial (3 correct saccades) deposits a small food increment into a tube.
- **Harvest period:** Stop working and lick the accumulated food from the tube.
- **Key manipulation:** Food tube distance from the mouth varied across sessions, changing the effort cost of harvest without changing the effort cost of work.

---

## Theoretical Framework

### The Capture Rate (Equation 1)

$$J = \frac{\alpha \beta_s n_s \left(1 - \frac{1}{1 + \beta_l n_l}\right) - n_l c_l - n_s^2 c_s}{n_l T_l + n_s T_s}$$

Where:

| Symbol | Meaning |
|--------|---------|
| α | Food increment per successful trial |
| β_s | Fraction of successful saccade trials |
| n_s | Number of saccade trials in work period |
| β_l | Fraction of successful licks |
| n_l | Number of licks in harvest period |
| c_l | Effort cost per lick |
| c_s | Effort cost per work trial |
| T_l | Duration of each lick |
| T_s | Duration of each trial |

Key structural features:

- **Reward accumulation** grows linearly with successful trials.
- **Food depletion** during harvest follows a hyperbolic function (early licks get more food than late licks).
- **Effort of work** grows as n_s² — a superlinear cost reflecting that prolonged work periods impose disproportionately greater cost.

### The Metabolic Cost of a Single Lick (Equation 2)

$$c_l(T_l) = \frac{d^2}{T_l} + kT_l$$

This is a **concave-upward** (U-shaped) function of lick duration T_l, parameterized by tube distance d and a rate constant k. The key insight: there exists an energetically optimal lick speed, but the capture-rate-optimal speed is *faster* than the energetically optimal speed — it pays to move vigorously to acquire reward.

---

## Core Predictions

### Prediction 1: Increased effort cost of harvest → work longer

When the tube is farther away (higher c_l):

- The optimal number of work trials n_s* increases.
- Subjects should stockpile more food before commencing harvest.
- This is a form of **delayed gratification** driven by the economics of the task.

### Prediction 2: Increased effort cost of harvest → reduce vigor

- The maximum achievable capture rate drops when harvest is more effortful.
- The optimal lick duration T_l* increases faster than linearly with tube distance — i.e., slow down *more* than biomechanics alone would require.
- This vigor reduction extends to saccades during the work period, even though saccade effort itself hasn't changed. The global reduction in capture rate suppresses vigor across all movements.

### Prediction 3: Hunger dissociates effort and vigor

- Increased subjective reward value (hunger) should also promote longer work periods.
- But unlike effort, hunger should *increase* vigor (move faster to acquire the now-more-valuable reward).
- This creates a double dissociation: effort and hunger both promote more work, but effort reduces vigor while hunger increases it.

---

## Why Additive (Not Multiplicative) Reward–Effort Interaction

The capture rate formulation combines reward and effort **additively** (reward − effort). This is critical because:

- **Multiplicative models** (reward × f(effort)) predict that changing reward magnitude merely scales utility without affecting optimal vigor. They fail to predict that reward invigorates movements.
- **Additive models** correctly predict that both reward and effort independently modulate vigor.

---

## Key Experimental Results

### Decisions

- When the tube was farther, marmosets worked more trials before harvesting (ANOVA p < 10⁻²⁵).
- This delayed gratification was stable throughout recording sessions.
- More food was cached at the start of harvest when tube was farther.

### Saccade Vigor (Work Period)

- Reward-relevant saccades were more vigorous than task-irrelevant saccades.
- Saccade vigor declined trial-by-trial within each work period.
- Saccade vigor was globally lower when the tube was placed farther (p < 10⁻³³).
- Higher vigor saccades were also *more accurate* — no speed-accuracy tradeoff.

### Lick Vigor (Harvest Period)

- Lick vigor showed a rapid increase at harvest onset, then gradual decline.
- Lick vigor was lower when tube was farther (p < 10⁻⁵⁰).
- Lick vigor increased with the amount of food cached (more work trials → more vigorous licks).
- Following a successful lick (food contact), the next lick was invigorated; following a failed lick, vigor stalled or decreased.

### Hunger Effects

- Lower body weight → longer work periods (consistent with higher reward valuation).
- Lower body weight → greater lick vigor (but not consistently greater saccade vigor).
- This confirms the dissociation: effort reduces vigor, hunger increases it.

---

## Pupil Size as a Neural Proxy

Pupil size tracked vigor moment-to-moment across both movement types:

- **Within work periods:** Pupils dilated at onset, then constricted trial-by-trial, paralleling saccade vigor decline.
- **Within harvest periods:** Pupils rapidly dilated (matching the initial vigor ramp-up), then gradually constricted.
- **Across sessions:** Pupils were more constricted when tube was farther (higher effort cost).
- **Correlations:** Pupil size correlated with saccade vigor (r ≈ 0.97–0.99) and lick vigor (r ≈ 0.97–0.99) across both subjects.
- Pupil dilation was associated with shorter work periods (choose to harvest sooner) and longer harvest periods.

### Interpretation

Pupil dilation is a proxy for **locus coeruleus norepinephrine (LC-NE)** activity. The proposed mechanism:

- Increased effort costs → decreased LC-NE activity → pupil constriction.
- Reduced NE release simultaneously:
  - Encourages work and delayed gratification in decision circuits.
  - Promotes sloth and energy conservation in motor circuits.
- NE may thus serve as a **bridge** coordinating decision-making and motor control circuits toward a consistent policy that improves the capture rate.

### Role of Dopamine

- Dopamine in the ventral striatum promotes willingness to expend effort for reward.
- Hunger disinhibits dopamine release via hypothalamic circuits.
- Striatal dopamine drops when effort price increases.
- Pre-movement dopamine release invigorates the upcoming movement.
- Dopamine may complement NE in coordinating decisions (effort expenditure) with movements (vigor).

---

## Limitations and Mismatches with Theory

- The theory predicted that increased effort cost should reduce harvest duration (fewer licks), but this was not observed — subjects did not leave food behind.
- Hunger did not robustly increase saccade vigor (only lick vigor).
- The model considered only a single work-harvest period rather than a long sequence.
- The model did not account for finite tube capacity (food spillage) or a link between lick vigor and success probability.
