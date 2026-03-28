# The Effect of a Change in Foraging Options on Intake Rate and Predation Rate

**McNamara, J. M. & Houston, A. I. (1994). The effect of a change in foraging options on intake rate and predation rate. *The American Naturalist*, 144, 978–1000.**

---

## Central Thesis

When an animal chooses between foraging options that differ in energy gain and predation risk, changes in the environment can produce **paradoxical effects**: an increase in food availability can *decrease* intake rate, and an increase in predation risk can *decrease* the probability of being killed. Whether these paradoxes occur depends on two key factors: (1) how the change differentially affects the available options, and (2) whether the change is temporary or permanent.

---

## The Optimization Framework

### Instantaneous Reproductive Value

An animal in a given state at a given time has a range of foraging options, each characterized by an intake rate γ and a predation rate M. The rate of change of reproductive value if the animal chooses an option (γ, M) is:

**γV − MR**

where:
- **R** = the animal's current reproductive value
- **V** = the rate at which R increases per unit of energy gained (marginal value of energy)
- **γV** = rate of increase in reproductive value from food intake
- **MR** = rate of decrease in reproductive value from predation risk

The optimal choice maximizes this expression. The ratio **V/R** is the **marginal rate of substitution** of predation risk for energy — the rate at which the animal should be willing to trade safety for food.

### Graphical Interpretation

In the (γ, M) plane:
- Each foraging option is a point
- Lines of constant fitness are straight lines with slope V/R
- The optimal option lies where the highest isofitness line touches the set of available options
- **Admissible options** form the lower-right boundary of the option set (increasing γ requires increasing M)
- Within admissible options, M is an increasing and **accelerating** (convex) function of γ

---

## Two Critical Questions

### Question 1: How Does the Change Differentially Affect the Options?

This is captured by the mixed partial derivative **∂²γ/∂M∂α**, where α is the environmental parameter being changed.

**Case A: ∂²γ/∂M∂α > 0** — improvement has more effect on high-gain options

Example: In the **vigilance model** (γ = au − b, where u is proportion of time foraging), increasing food availability *a* stretches the options curve to the right. The improvement is larger for high-u (high-gain, high-risk) options. Result: both γ* and M* **increase**.

**Case B: ∂²γ/∂M∂α < 0** — improvement has more effect on low-gain options

Example: In **habitat choice** with two habitats, increasing food in only the safer (low-gain) habitat can flip the optimal choice. Result: both γ* and M* may **decrease** — the animal switches to the now-improved safe option.

### The Handling Time Example

When encounter rate λ = kα (proportional to availability) and all items have handling time h:

- Gain rate = kα / (1 + kαh)
- The cross-derivative ∂²γ/∂M∂α is negative when kαh > 1

Interpretation: When handling time exceeds mean search time, increasing encounter rates has *more* effect on areas where food is less abundant (because high-abundance areas are already saturated by handling time). This reverses the usual prediction.

### Question 2: Is the Change Temporary or Permanent?

**Temporary change**: Only current options change. V/R remains fixed. Analysis is purely graphical.

**Permanent change**: Both current options *and* future expectations change. V/R itself shifts.

---

## The Critical-State Model

A simple model that cleanly illustrates long-term effects: an animal must reach state x_c from state x₀ to reproduce. Time to reach x_c has no effect on reproductive success.

With fixed γ and M:

**R(x₀) = K · exp(−M(x_c − x₀)/γ)**

and

**V/R = M/γ**

Key implications:
- Increasing **future food** → decreases V/R → animal values food less relative to safety
- Increasing **future mortality** → increases V/R → animal values food more (willing to accept more risk)

In the critical-state model, optimal behavior minimizes **M/γ** — this is **Gilliam's rule**. This instantaneous criterion implicitly assumes the future is identical to the present (i.e., the change is permanent).

---

## Permanent Changes in Food Availability

With γ(u) = au − b and M(u) = m₀u^n:

Optimal u* = bn / [a(n−1)]

Optimal intake rate: **γ* = b/(n−1)**

Striking results:
- γ* is **independent of** food availability *a*
- γ* is **proportional to** metabolic cost *b*
- Increasing *a* (which benefits high-gain options more) produces **no change** in intake rate — the animal takes all the benefit as reduced predation risk
- Decreasing *b* (which benefits all options equally) **decreases** intake rate — the animal takes all the benefit as reduced risk

---

## Permanent Changes in Predation Level

With M(u) = m₀N(u) + μ (where μ is background mortality independent of behavior):

- Increasing **m₀** (behavior-dependent attack rate): u* **decreases**
  - Higher m₀ amplifies the risk of the riskiest options most → animal plays it safe
  
- Increasing **μ** (behavior-independent background mortality): u* **increases**
  - The increase is the same under all options → best response is to grow faster, reducing total exposure time

This produces two opposite strategies for coping with high predation:
1. **Be cautious** (reduce foraging effort) — appropriate when predation risk is behavior-dependent
2. **Forage harder** (increase effort to grow fast and reduce exposure) — appropriate when predation is background/unavoidable

---

## Short-Term vs. Long-Term: Summary Table

For the vigilance model γ = au − b, M = m₀u²:

| Change | Duration | u* | γ* | M* |
|--------|----------|----|----|-----|
| Increase *a* (food) | Short-term | ↑ | ↑ | ↑ |
| Increase *a* (food) | Long-term | ↓ | 0 | ↓ |
| Decrease *b* (metabolic cost) | Short-term | 0 | ↑ | 0 |
| Decrease *b* (metabolic cost) | Long-term | ↓ | ↓ | ↓ |
| Decrease *m₀* (predation) | Short-term | ↑ | ↑ | ↑ |
| Decrease *m₀* (predation) | Long-term | 0 | 0 | ↓ |

(↑ = increase, ↓ = decrease, 0 = no change)

The general pattern: **short-term improvements increase intake; long-term improvements leave intake constant or decrease it**.

---

## Behavior Over a Finite Time Interval

### The Risk-Spreading Theorem

Optimal behavior is **constant over time** if:
1. Energetic gain under each option is deterministic
2. γ and M do not depend on the animal's state or time
3. The foraging process is not subject to interruptions before time T

When these conditions hold, fitness can be written as a function of constant behavior and optimized directly — no dynamic programming needed.

### Example Model

With γ(u) = au and M(u) = −log(1 − u/2), terminal reward R_T(x) = x:

Optimal intake rate: **au* = [1/(T+1)] · (2a − x₀)**

The rate of increase of γ* with food availability *a* is:

**dγ*/da = 2/(T+1)**

Intake increases with food availability, but the rate of increase **declines as the time horizon T increases**. This directly shows how the duration of a change modulates its effect.

### The Marginal Rate of Substitution

**V/R = (T+1) / (x₀ + 2aT)**

V/R decreases with increasing food availability *a* — as the future improves, the marginal value of food declines relative to the value of life. This is the mechanism through which long-term improvements dampen or reverse the intake response.

---

## Open vs. Closed Economies

This framework explains qualitative differences between experimental economies:

- **Open economy** (supplemental feeding after sessions): The terminal reward at session's end is little influenced by food availability in future sessions. Changes in availability act like short-term changes → foraging effort increases with availability.

- **Closed economy** (intake solely from behavior): The terminal reward depends strongly on future food availability. Changes act like long-term changes → foraging effort may decrease with availability.

---

## Key Takeaways

1. **Differential effects matter**: How a change in the environment differentially affects the available options (∂²γ/∂M∂α) is critical for predicting whether intake increases or decreases

2. **Time horizon matters**: Short-term improvements generally increase intake; long-term improvements may leave it constant or decrease it, because improved future prospects reduce the marginal value of food relative to survival

3. **Paradoxical effects are not paradoxes**: They follow logically from the optimization framework once both the differential impact and the time horizon are specified

4. **Gilliam's M/γ rule** emerges as a special case — valid when changes are permanent and conditions of the critical-state model hold

5. **Two strategies for coping with predation**: Reduce foraging effort (when risk is behavior-dependent) or increase it (when risk is background) — the optimal response depends on the structure of the predation function

6. **V/R is the key quantity**: The marginal rate of substitution of predation risk for energy determines optimal behavior, and it changes with the animal's future expectations
