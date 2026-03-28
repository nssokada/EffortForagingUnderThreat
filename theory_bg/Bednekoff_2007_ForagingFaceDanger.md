# Foraging in the Face of Danger

**Bednekoff, P. A. (2007). Foraging in the face of danger. In D. W. Stephens, J. S. Brown, & R. C. Ydenberg (Eds.), *Foraging: Behavior and Ecology* (pp. 305–329). University of Chicago Press.**

---

## Central Thesis

Virtually all animals face a trade-off between acquiring resources and becoming a resource for another. This chapter develops a **life-history framework** for modeling foraging under predation risk, derives key predictions from simple models, and reviews the multiple routes through which foraging and danger become linked.

---

## Why Foraging and Danger Are Linked

There are several distinct mechanisms that create a trade-off between feeding rate and predation risk. Wherever one or more applies, organisms face a foraging–danger trade-off:

### 1. Time Spent Exposed
Animals that feed for longer must extend into more dangerous periods (e.g., twilight, moonlit nights). Restricting feeding to safe periods reduces intake; extending feeding increases danger.

### 2. Habitat Choice
Productive habitats are often exposed. Algae grows on sunny (exposed) rock surfaces; zooplankton thrive in open water. Refuge habitats are safer but poorer. Animals switch between habitats as growth changes both foraging gain and vulnerability.

### 3. Movement
Faster-moving foragers encounter more food *and* more predators (Werner & Anholt 1993). Anaesthetized tadpoles are less likely to be killed by invertebrate predators — immobility is protective.

### 4. Detection Behavior
Scanning for predators competes with foraging attention. Reduced vigilance increases intake but decreases the probability of detecting attacks. Guppies react more slowly to predators when foraging, especially nose-down.

### 5. Depletion and Density Dependence
Central-place foragers (marmots, lizards) deplete food near their refuge, creating a gradient: richer food farther out, but at greater danger. When prey congregate for safety, competition depletes the safe zone, compounding the trade-off.

---

## The Life-History Modeling Framework

### First Principles

Two axioms:
1. **Food is good** — higher foraging success leads to greater future reproductive success
2. **Death is bad for fitness** — being killed eliminates all future reproduction

Because costs (death) and benefits (food) are in different units, both must be translated into **fitness**. A life-history perspective provides the conversion.

### The Fitness Equation

For a non-reproducing animal (b = 0), fitness is:

**W(u) = S(u) · V(u)**

where:
- **u** = foraging effort (fraction of maximum, 0 to 1)
- **S(u)** = survival probability = exp[−M(u)] = exp[−ku^z]
- **V(u)** = future reproductive value = κu
- **k** = mortality constant
- **z** = mortality exponent (shape of the trade-off)

### Optimal Foraging Effort (Fixed Time)

Differentiating W(u) and solving:

**u* = 1 / (kz)^(1/z)**

so long as u* ≥ R (the minimum requirement to avoid starvation).

Key results:
- Optimal foraging effort **decreases** as danger constant *k* increases
- The decline is **less steep** as the mortality exponent *z* increases
- When the requirement R exceeds the otherwise-optimal effort, the animal effectively **maximizes survival** — a life-history approach converges on survival-maximization models

### Automatic State Dependence

The costs and benefits of foraging are linked through the animal's state. As an animal accumulates resources:
- Its future reproductive value (the potential loss from death) **increases**
- The *relative* value of further foraging is therefore **lower**

This creates an inherent tendency toward greater caution with greater accumulated success. Juvenile coho salmon are more cautious when larger, because larger individuals expect greater reproduction if they survive.

---

## Two Biological Scenarios for the Mortality Function

### Scenario 1: Mobile Forager, Sit-and-Wait Predators
(e.g., tadpoles vs. dragonfly larvae)

Faster movement encounters more food and more predators. The exponent *z* reflects metabolic cost per distance; the constant *k* combines relative encounter rate with predators and kill probability per encounter.

### Scenario 2: Relatively Immobile Forager, Mobile Predators
(e.g., birds hunted by Accipiter hawks)

Greater foraging effort doesn't increase encounter rate (hawks seek you out regardless), but it increases vulnerability per attack. The constant *k* includes the attack rate α; the exponent *z* reflects how foraging effort increases kill probability per attack.

---

## Gathering Resources with No Time Limit (Growth)

For growing animals accumulating a fixed amount of resources K:

**W(u) = exp[−M(u) · T(u)] · V**

where T(u) = K/(u − R) is the time to reach the required size.

Optimal foraging effort:

**u* = zR / (z − 1)**

Key differences from the fixed-time model:
- u* = 2R when z = 2; u* = 4R/3 when z = 4
- Optimal effort **does not depend on k** (absolute danger level is irrelevant)
- Only the **shape** of the trade-off (z) matters
- Animals at different absolute danger levels should behave identically if their trade-off shapes are the same
- This produces **Gilliam's M/g rule** — minimize mortality per unit gain

---

## Danger and State

Whether danger depends directly on the animal's state (e.g., body mass) is theoretically important but empirically difficult to demonstrate. We expect behavior to depend on state whenever *future reproductive value* depends on state — this alone doesn't prove that *danger* depends on state.

### Fat Reserves in Small Birds

Winter birds face dual risks: starvation and predation. Reserve levels are far below what migrants carry, suggesting costs to carrying reserves. The puzzle: do mass-dependent predation costs exist?

- Aerodynamic theory predicts mass affects escape flight
- Large fat reserves slow escape flights in startled birds
- But daily mass changes (dawn-to-dusk) have little effect on escape performance
- Results suggest either a threshold effect or continuous but nonlinear costs
- Reproductive states (carrying eggs, depleted wing muscles) clearly impair escape

---

## Danger Changes Over Time

When danger fluctuates (e.g., moonlit vs. dark nights), animals should **allocate** their foraging effort to match the temporal pattern.

### The Allocation Model (Box 9.1)

With two situations differing in attack rate (α₁, α₂), the optimal feeding rates satisfy:

**α₁u₁ = α₂u₂**

The ratio of feeding rates is the **inverse** of the ratio of attack rates. This means:
- Foragers change behavior **more** in response to *variations* in danger than to *average* rates
- Under some conditions, foragers respond **only** to variation, not to mean danger level
- Different individuals at different overall danger levels may behave similarly

This produces equal **M/g ratios** across situations — Gilliam's rule emerges from the allocation problem.

---

## Danger and Group Size

### Why Groups Are Safer

Groups decrease danger through:
1. **Dilution**: per capita attack rate often increases less than proportionally with group size
2. **Collective detection**: detection by one member can benefit others
3. **Combined effect**: these interact depending on information flow quality

### Three Models of Information Flow (Box 9.2)

Given individual failure-to-detect probability *f* and group size *n*:

**Perfect collective detection**: risk/attack = f^n / n

**Two-to-go rule** (collective detection requires ≥2 detectors): risk/attack = f^n/n + (1−f)f^(n−1)

**No collective detection**: risk/attack = [1 − (1−f)^n] / n

All show danger declining with group size, but the shape differs. The two-to-go rule behaves like no collective detection for small groups and approaches perfect detection for large groups.

### Group Size Effect on Vigilance

The literature overwhelmingly shows individual vigilance decreasing and feeding rates increasing with group size. If group size fluctuates, animals should forage intensely in large groups because they will soon be in smaller, more dangerous groups.

---

## How Foragers Assess Danger

Assessment occurs at three stages of the predation sequence:

### Encounter Probability
- Direct detection of predator signs (sights, sounds, smells)
- Eavesdropping on predator territorial behavior
- Chemical cues in aquatic systems

### Attack Probability
- Moonlight levels (affects both detection and exposure)
- Habitat structure and cover

### Escape Probability
- Distance and habitat structure between forager and safety
- Ground squirrels flee more slowly through shrubs
- Fox squirrels react to escape substratum

Escape probability may be the hardest component to learn from experience.

---

## Should Foragers Overestimate Danger?

Intuition suggests yes — the costs of underestimating (death) seem to exceed the costs of overestimating (reduced intake). But the mathematics disagree in many cases.

For the model W(u) = [exp(−ku^z)][κu]:

**Foragers should overestimate danger only if z > 3** — i.e., only if mortality costs accelerate *very* steeply with foraging effort.

Empirical evidence from anuran larvae suggests z ≈ 2, implying they might actually underestimate danger. Field examples show both over- and underestimation:
- Moose without predator experience are highly vulnerable (underestimation)
- New England cottontails rarely feed away from cover even when predators are now rare (possible overestimation)

---

## Key Takeaways

1. The foraging–danger trade-off is **ubiquitous** — animals without such trade-offs are exceptional
2. A **life-history framework** (W = S · V) naturally unifies the costs of death and benefits of food
3. The **shape** of the mortality function (exponent z) is often more important than its absolute level
4. **State dependence** is automatic: accumulated success increases the cost of death
5. **Temporal variation** in danger drives larger behavioral changes than average danger levels
6. **Group size** reduces danger through dilution and collective detection, with the details depending on information flow
7. Whether foragers should over- or underestimate danger depends on the curvature of the mortality function — not on general intuition about the "asymmetry of death"
