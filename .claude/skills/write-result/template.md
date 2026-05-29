---
result_id: ###
class: behavioral_effects        # one of: behavioral_effects | computational_model | vigor_dynamics | choice_vigor_coupling | metacognition | individual_differences
title: <short descriptive title — one line, sentence case>
status: untested                 # supported | supported_exploratory | partial | refuted | refuted_exploratory | dead-end | retracted | untested
prereg_h: []                     # e.g. [H1a, H1b]; empty if exploratory-only
internal_h: []                   # e.g. [H30, H34]; empty if no internal H# applies
samples: []                      # e.g. [exploratory_293, confirmatory_281]; subset if only one was tested
notebooks: []                    # paths to notebooks that produced the headline numbers
scripts: []                      # paths to .py scripts involved
outputs: []                      # paths to .csv/.pkl/etc files that contain the result data
figures: []                      # paths to figure files, or [TODO] if not yet produced
created: YYYY-MM-DD
last_run: YYYY-MM-DD
---

# Result ### — <title>

## Overview

<2–4 sentences. State the question, the finding, and a one-line interpretation. No statistics — those live in Result. Written so a reader could read just this paragraph and understand what was done and what was found.>

## Hypothesis

**Statement.** <Verbatim preregistered statement when applicable, otherwise the post-hoc claim being tested.>

**Predicted direction.** <e.g. β(threat) < 0 and β(distance) < 0>

**Preregistered criterion.** <e.g. both P < .01 for prereg results; "exploratory — no a priori criterion" for exploratory results>

**Source of the hypothesis.** <Preregistration section / theoretical motivation / observed pattern that motivated this analysis>

## Data Source

- **Sample(s):** <name, N pre-exclusion → N post-exclusion, date collected>
- **Input files:**
  - `<path/to/behavior.csv>` — what it contains
  - `<path/to/feelings.csv>` — what it contains
- **Inclusion / exclusion applied for this result:** <if different from project-wide; otherwise "project-default exclusions only">
- **Unit of analysis:** <trial / subject / cell / probe>
- **N entering the model:** <e.g. 13,185 trials from 293 subjects>

## Method

<One paragraph in plain English describing the procedure end-to-end. What is computed from what, in what order. No formulas yet.>

**Model / test specification:**

```
<formula in math or code — e.g.>
choice ~ threat_z + dist_z + threat_z:dist_z
random structure: clustered by subject (cluster-robust SE)
link: logistic
```

**Software / packages:**
- `<package> <version>`
- Environment: `effort_foraging_threat` (Python 3.11)

**Inference criterion:** <e.g. p < .01 with directional sign; 95% HDI excludes zero; ΔWAIC > 0>

**Notebook(s) that produce this result:**
- `<path/to/notebook.ipynb>` — cells `<cell range>`

## Result

<One sentence stating the result in words.>

| Metric | Exploratory (N=293) | Confirmatory (N=281) |
|--------|---------------------|----------------------|
| <param> | <est (test stat, p)> | <est (test stat, p)> |
| <param> | <est (test stat, p)> | <est (test stat, p)> |

**Figure:** `write-up/results/figures/result_###_<slug>.png` <or `[TODO — figure not yet produced]` with a one-line description of what the figure should show>

**Verdict on prereg criterion:** PASS / PARTIAL / FAIL — <one line of justification>

## Interpretation

<This is the load-bearing section. Write at paper-grade density: the goal is that concatenating Interpretations across results would form a publishable Results section.>

<Paragraph 1 — what the numbers mean in task terms. Translate "β = −1.02" into "for every 1-SD increase in threat probability, the log-odds of choosing the heavy cookie fall by 1.02 — a substantial deterrent effect.">

<Paragraph 2 — mechanism. How does this connect to the joint fitness model W(u)? Which parameter (ω, κ, β) is implicated? What does the prereg theory predict, and does this match?>

<Paragraph 3 — connection to other results. Cross-link to related findings: [[result_###]] shows that…, which together with this result suggests…>

## Caveats & Limitations

- <Confound or alternative interpretation a reviewer would raise>
- <Sample-size constraint or trial-count limit>
- <Known conflict with another analysis, e.g. earlier wrong-frame result that this corrects>
- <What this analysis can't tell us>

## Replication

**To regenerate this result:**

```bash
<one command or notebook cell — e.g.>
jupyter nbconvert --to notebook --execute notebooks/02_choice_modeling/01_fit_compare_ppc.ipynb
```

**Expected runtime:** <e.g. 5 min on CPU, 30 s on GPU>

**Expected outputs:**
- `<path/to/output.csv>` — contains the headline numbers in this Result table
- `<path/to/figure.png>` — if generated

## References

**Related results:**
- [[result_###]] — <one-line relevance>
- [[result_###]] — <one-line relevance>

**Notebooks:**
- `<path/to/notebook.ipynb>`

**Literature (if applicable):**
- <citation>

## Revision notes

<Only include this section when a material correction has been made. Otherwise delete the section entirely.>

- **YYYY-MM-DD:** <one-line description of the correction, what changed, and why — e.g. "corrected encounter-time alignment from wrong-frame to trial-start-relative; F changed from 0.91 → 7.84.">
