# Memory Index

All memory files for the EffortForagingUnderThreat project.

## Core project files

| File | Type | Description |
|------|------|-------------|
| [project_goal.md](project_goal.md) | project | Nature Comms submission: title, core claim, 4-Results structure, what remains |
| [paper_framing.md](paper_framing.md) | project | Current framing: effort reallocation, evidence pyramid, key numbers, λ caveat |
| [optimal_control_plan.md](optimal_control_plan.md) | project | **NEW DIRECTION:** Reframe as stochastic optimal control — single cost function (c_effort, c_death) governs choice + vigor mechanistically |
| [active_issues.md](active_issues.md) | issues | MCMC not run, confirmatory not started, missing figures, old file cleanup |
| [next_steps.md](next_steps.md) | project | Priority: (1) MCMC on GPU, (2) remaining figures, (3) confirmatory sample, (4) polish |

## Analysis state

| File | Type | Description |
|------|------|-------------|
| [joint_model_development.md](joint_model_development.md) | project | Independent Bayesian pipeline + joint LKJ model — architecture, λ sensitivity, development history |
| [pipeline_state.md](pipeline_state.md) | state | Execution status of every notebook/script, all output files listed |
| [discoveries.md](discoveries.md) | findings | All empirical results by domain — NOW OUTDATED, see drafts/discovery_results.md for current |
| [allocation_analysis.md](allocation_analysis.md) | findings | β-δ coupling dimension, angle vs magnitude, E-scaling test (failed), psych correlates (mostly null except F3 apathy) |

## Reference

| File | Type | Description |
|------|------|-------------|
| [task_design.md](task_design.md) | reference | Full task mechanics: arena, effort tiers, predator dynamics, probe structure |
| [open_questions.md](open_questions.md) | questions | Remaining questions: vigor vs distance, α-δ independence, MCMC resolution |
| [hypotheses.md](hypotheses.md) | hypotheses | All hypotheses with status + numbering concordance (prereg H1-H7 ↔ internal H#s, updated 2026-03-24) |

## Session logs

| File | Type | Note |
|------|------|------|
| [session_2024_03_24.md](session_2024_03_24.md) | session | Major session: H1-H3 tests, binary-E model, allocation dimension, psych correlates |
| [session_history.md](session_history.md) | history | Older session log — not updated since 2026-03-20 |

## Key external documents

- `drafts/paper.md` — current paper draft (4 Results sections, tightened 2026-03-22)
- `drafts/preregistration.md` — confirmatory preregistration (H1-H7 in 4 sections, updated 2026-03-24 to match simple_prereg numbering)
- `drafts/discovery_results.md` — clean summary of all findings (updated 2026-03-22)
- `scripts/run_mcmc_pipeline.py` — full MCMC pipeline for GPU execution
