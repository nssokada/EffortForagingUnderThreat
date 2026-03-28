# Metacognition in Wise et al. (2023): A Walkthrough

## A caveat up front

Wise et al. don't use the term "metacognition" directly. Their paper is about **interactive cognitive maps** — how humans build internal models of other agents' goals and use those models to plan flexible behavior under threat. But metacognitive processes run quietly through the entire framework. This walkthrough pulls those threads out.

---

## 1. Confidence as a metacognitive signal

The most explicit metacognitive measure in the paper is **confidence ratings**. In Experiment 1, after participants predicted where the threatening agent would move, they rated how confident they were in that prediction.

Two key findings:

- Confidence increased across trials within a game (β = 0.10, 95% HPDI = [0.08, 0.12])
- Confidence increased across games (β = 0.20, 95% HPDI = [0.14, 0.26])

This matters because confidence here tracks participants' **second-order awareness of their own learning**. They aren't just getting better at predicting the agent — they *know* they're getting better. The confidence trajectory mirrors the accuracy trajectory, suggesting reasonably well-calibrated metacognitive monitoring of an ongoing inference process.

---

## 2. Knowing what someone else wants: metacognition about mental models

The preference rating task at the end of each game is inherently metacognitive. Participants are asked: *What do you think the blob monster likes and dislikes?*

This requires participants to:

1. Represent their own belief about the agent's reward weights
2. Translate that internal representation into an explicit report
3. Do so on a 9-point scale that forces them to commit to magnitudes, not just categories

The finding that 99.3% of participants produced more accurate ratings than a no-learning baseline tells us that people can **introspect on and report the contents of their social inference process** with remarkable consistency. They aren't just acting on implicit knowledge — they can articulate it.

---

## 3. The hypothesis-testing model as formalized metacognition

The winning computational model for reward-weight inference — **HypTest** — has a structure that maps naturally onto metacognitive operations:

- **Generate candidate hypotheses** about the agent's preferences (sampling from a prior over reward weights)
- **Simulate what the agent would do** if those hypotheses were true (using the successor representation to derive action probabilities)
- **Evaluate the hypotheses** against observed behavior (Bayesian updating via likelihood comparison)
- **Maintain a posterior distribution** — not a point estimate, but a graded belief about how likely each hypothesis is

This is essentially a formalization of *thinking about what you think the other agent is thinking*. The Bayesian posterior is a metacognitive object: it represents the participant's uncertainty about their own model of the agent.

---

## 4. Switching between strategies: knowing when your model works

The combined model results from Experiment 1 reveal that participants blend **goal inference** (model-based) and **policy learning** (model-free) strategies, with a weighting parameter W averaging 0.87 (favoring goal inference).

The authors connect this to prior work showing that humans **adaptively switch** between goal inference and simpler strategies depending on expected success (citing Wu, Vélez & Cushman, 2021; Charpentier, Iigaya & O'Doherty, 2020). This adaptive switching requires a metacognitive capacity: you need to monitor how well each strategy is performing and allocate cognitive resources accordingly.

The fact that the combined model fits best, rather than pure goal inference, suggests participants maintain some awareness of when their model-based predictions might be unreliable and hedge with simpler heuristics.

---

## 5. Uncertainty awareness shapes behavior

Experiment 3 manipulates uncertainty along two dimensions:

- **Irreducible uncertainty**: the agent selects actions stochastically
- **Reducible uncertainty**: participants must infer preferences rather than being told

The key behavioral finding is that participants became more avoidant when the agent was unpredictable (stochastic action selection), spending less time in rich reward zones. This behavioral adjustment requires participants to **recognize their own uncertainty** about the agent's future behavior and act on that recognition.

Interestingly, when preferences had to be inferred (reducible uncertainty), participants did *not* become more avoidant. The authors suggest participants learned the preferences quickly enough that this manipulation didn't create meaningful subjective uncertainty. In other words, participants' metacognitive assessment of their own inference accuracy was good enough that they didn't feel uncertain even when they technically had less information.

---

## 6. Individual differences: miscalibrated models of the other

The computational modeling of individual differences in avoidance is where metacognition gets most interesting. Some participants were overly avoidant even when the agent behaved entirely predictably. The modeling reveals this was driven by **threat sensitivity** (how much they weighted the cost of being caught), not by assumptions about the agent's unpredictability.

The softmax temperature parameter — which captures how predictable participants assumed the agent to be — was significantly above zero even in the predictable condition (mean = 2.71), meaning participants didn't fully trust that the agent would behave deterministically. This is a metacognitive prior: *I'm not sure my model of this agent is reliable enough to bet on.*

But critically, this "epistemic caution" didn't predict avoidant behavior. What predicted avoidance was how much participants *cared* about being wrong (threat sensitivity), not how uncertain they thought they were. This dissociation suggests two distinct metacognitive dimensions operating in parallel during threat avoidance.

---

## 7. The cognitive map itself as a metacognitive structure

The paper's central claim is that participants exploit an **interactive cognitive map** — an internal model that represents the environment, the agent's preferences, and the consequences of both their own and the agent's actions. This map is inherently metacognitive in the sense that it is a representation *about* representations:

- It doesn't just encode where things are; it encodes what the agent *wants* and what it will *do*
- It supports counterfactual simulation: *if I go here, and the agent goes there, then...*
- It generalizes across environments, meaning participants understand their model is abstract enough to transfer

The MCTS-RW planning model — which simulates both the participant's and the agent's future trajectories — is the computational instantiation of this. Planning via tree search is, at its core, a metacognitive operation: you're running your own decision process on a simulated version of reality and using the results to guide actual behavior.

---

## Summary

| Metacognitive process | Where it appears in the paper |
|---|---|
| Monitoring one's own learning | Confidence ratings tracking accuracy over time |
| Reporting mental model contents | Preference ratings at end of each game |
| Hypothesis evaluation | HypTest IRL model with Bayesian posterior |
| Strategy selection monitoring | Adaptive weighting between goal inference and policy learning |
| Uncertainty recognition | Behavioral adjustment to stochastic vs. deterministic agents |
| Epistemic caution vs. threat sensitivity | Dissociation in individual difference modeling |
| Counterfactual simulation | MCTS-RW planning incorporating other agent's goals |

The paper doesn't frame itself as being about metacognition, but nearly every major finding depends on participants having access to — and acting on — representations of their own knowledge, uncertainty, and inferential processes.
