# The Expected Value of Control (EVC)

**Source:** Shenhav, A., Botvinick, M. M., & Cohen, J. D. (2013). The expected value of control: An integrative theory of anterior cingulate cortex function. *Neuron, 79*(2), 217–240.

---

## Core Claim

The dorsal anterior cingulate cortex (dACC) performs a single integrative function: it estimates the **expected value of control** (EVC) — the net value of allocating cognitive control to a given task — and uses this to **specify** the optimal control signal. This signal determines both **what** task to pursue (control signal identity) and **how hard** to pursue it (control signal intensity).

---

## What is Cognitive Control?

Cognitive control is the set of mechanisms required to pursue a goal, especially when distraction or strong competing responses must be overcome. The classic illustration is the Stroop task: naming the ink color of a color word (e.g., saying "green" when the word RED is printed in green ink) requires control because word reading is automatic while color naming is not.

---

## Three Component Functions of Cognitive Control

### 1. Regulation
The capacity to govern lower-level information processing. A control signal has two characteristics:
- **Identity:** Which parameter is targeted (e.g., "attend to color," "attend to word")
- **Intensity:** How strongly the parameter is displaced from its default value

Regulation is implemented by lateral prefrontal cortex (lPFC) and associated structures (basal ganglia, brainstem dopaminergic nuclei).

### 2. Specification
The decision process that determines which control signal should be selected and how intensely it should be engaged. This is the **decision-making** component of control.

**Specification is the function ascribed to dACC by the EVC theory.**

### 3. Monitoring
Access to information about current circumstances and how well the system is serving task demands. Includes detection of conflict, errors, feedback, pain, and changes in the environment that indicate the need to adjust control.

dACC performs monitoring as an input to specification, but monitoring should be distinguished from the primary **valuation** processes (in OFC, vmPFC, insula, amygdala) that represent the information being monitored.

---

## The EVC Equation

```
EVC(signal, state) = Σᵢ Pr(outcomeᵢ | signal, state) · Value(outcomeᵢ) - Cost(signal)
```

Where:
- **Signal:** A specific control signal (identity + intensity)
- **State:** Current situation (environmental conditions + internal factors like motivation, difficulty)
- **Outcomes:** Future states resulting from applying a control signal, each with a probability
- **Value(outcome):** Immediate reward + discounted future expected value
- **Cost(signal):** Intrinsic cost of engaging control, monotonically increasing with intensity

### Optimal Control Signal Selection

```
signal* ← maxᵢ [EVC(signalᵢ, state)]
```

The brain selects the control signal that maximizes EVC. Once specified, this signal is implemented by regulatory structures (lPFC, basal ganglia, etc.) and maintained until monitoring detects that conditions have changed.

---

## The Cost of Control

A critical component of the EVC framework is that cognitive control carries **intrinsic subjective cost** (mental effort):

- People spontaneously seek to minimize cognitive effort.
- They will delay goals or forgo rewards to avoid control-demanding tasks.
- This cost scales with the intensity of control required.

The cost function creates a tradeoff: stronger control improves performance but is more costly. The EVC-optimal intensity is the point where the marginal benefit of additional control equals its marginal cost.

### Willingness to Pay

The output of dACC can be interpreted as a "willingness to pay" signal in the currency of cognitive control — how much effort is worth investing given the expected payoff. This explains why dACC damage produces apathy (inability to specify the effort investment needed to initiate willed actions).

---

## Monitoring Functions of dACC

### State Information for Control Signal Intensity

**Conflict monitoring:** When competing responses are coactivated (e.g., in the Stroop task), conflict signals indicate inadequate control and the need to increase intensity. dACC activity consistently tracks conflict across diverse tasks — perceptual discriminations, value-based decisions, moral judgments, memory retrieval, and strategy selection.

**Task difficulty:** dACC activity tracks cognitive demands generally — complex vs. simple rules, novel vs. familiar responses, larger vs. smaller option sets.

### State Information for Control Signal Identity

dACC receives inputs from cortical areas associated with perception, motivation (amygdala, insula), and high-level processing, giving it access to information relevant to selecting **which** task to pursue. dACC differentiates representations of response rules, task sets, and specific actions.

### Outcome Information

dACC is responsive to both **negative** outcomes (pain, errors, monetary loss, social rejection) and **positive** outcomes (reward magnitude, probability). Critically, dACC is selectively responsive to outcomes relevant to control allocation (tied to actions or control-demanding tasks), not just any valenced event.

### Reward Prediction Errors

dACC signals both signed and unsigned prediction errors:
- **Signed PEs:** The feedback-related negativity (FRN) and error-related negativity (ERN) reflect negative prediction errors, context-dependent and useful for learning.
- **Unsigned PEs:** dACC responds to surprising outcomes regardless of valence, consistent with "attention for learning" accounts (Pearce-Hall model).

---

## Specification Functions of dACC

### Specifying Control Signal Identity (What to Control)

dACC encodes both the value and identity of potential actions/tasks and anticipates switches between them. Evidence includes:
- dACC neurons multiplex value and direction of saccades
- Local field potentials discriminate which task rule will be used before stimulus onset
- Microstimulation of dACC facilitates antisaccade performance (controlled task)
- Lesions impair set-shifting performance

### Specifying Control Signal Intensity (How Much to Control)

dACC adaptively adjusts control intensity based on task demands:

- **Conflict adaptation:** After a high-conflict trial, dACC increases control, improving performance on the next trial.
- **Speed–accuracy tradeoff adjustments:** dACC interacts with subthalamic nucleus (STN) to adjust decision thresholds and with dorsal striatum to adjust response biases.
- **Incentive-driven intensity:** dACC activity increases with both task difficulty and incentive level (Kouneiher et al., 2009), consistent with EVC maximization.

### Default Override

Many control-demanding situations involve overriding automatic or default behavior:
- **Exploration** (overriding exploitation of known rewards)
- **Foraging** (overriding current reward pursuit for potentially better alternatives)
- **Intertemporal choice** (overriding immediacy bias for larger delayed rewards)

In all cases, dACC tracks the value of the control-demanding alternative and is associated with choosing it over the default.

---

## dACC in the Broader Control Network

### Division of Labor: Specification vs. Regulation

| Structure | Function | Role |
|---|---|---|
| dACC | Monitoring + Specification | Evaluates EVC; specifies optimal control signal identity and intensity |
| lPFC | Regulation | Implements and maintains the specified control signal via top-down biasing |
| Basal ganglia | Gating | Gates action implementation and updating of control representations |
| STN | Threshold regulation | Implements decision thresholds specified by dACC |
| Locus coeruleus | Global modulation | Implements exploration/exploitation balance |
| Insula | Affective salience | Represents motivational significance; feeds into dACC monitoring |
| vmPFC/OFC | Valuation | Represents option values; provides input to dACC |

### Key Evidence for dACC–lPFC Dissociation

- dACC is more sensitive to conflict; lPFC is more sensitive to task-set implementation (MacDonald et al., 2000).
- dACC activity precedes and predicts subsequent lPFC engagement and behavioral adaptation.
- After task switching, dACC selectivity emerges earlier than lPFC; with practice, lPFC selectivity strengthens while dACC selectivity fades (Johnston et al., 2007).
- High-gamma LFP in dACC signals salient events, followed shortly by sustained lPFC responses (Rothé et al., 2011).

---

## Predictions of the EVC Model

1. dACC should be engaged by **any** control-demanding task, regardless of abstraction level.
2. dACC activity should increase with both **task difficulty** and **incentive magnitude**.
3. dACC should track the **value of the control-demanding alternative** relative to the default.
4. dACC should register the **cost of control** and signal when costs outweigh benefits (leading to task disengagement or avoidance).
5. dACC output should predict subsequent changes in **both** lPFC activity and behavioral performance.
6. dACC damage should produce deficits in specifying control signals (e.g., apathy, perseveration, conflict adaptation failure) rather than in implementing them.

---

## Open Questions (as of 2013)

- How is a set of candidate control signals initially learned?
- How might EVC be feasibly estimated by neural mechanisms?
- What exact form does the cost function assume?
- What costs attach to the estimation of EVC itself?
- How are monitoring, specification, and regulation organized within dACC's neural architecture?
