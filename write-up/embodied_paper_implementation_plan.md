# Implementation plan — Embodied dynamics paper

Last updated: 2026-06-07 (after baseline-confound diagnostic round)
Companion to: `embodied_paper_outline.md` (the structural outline)
Status: Outline revised, prose drafting not yet started.

---

## Current state — where we are

**Done:**
- Outline drafted and revised at `write-up/embodied_paper_outline.md`
- All empirical analyses for main text complete:
  - Joint W(u) model comparison (§3.3) — results 201–207
  - Parameter recovery + cross-sample replication (§3.4) — result 205
  - (ω, κ) behavioral signatures (§3.5) — result 208
  - Vigor dynamics — anticipatory + reactive components (§3.2) — encounter-aligned analysis
  - Parameters predict vigor dynamics (§3.6) — `scripts/analysis/parameters_predict_vigor_dynamics.py`
  - Spike measurement diagnostic + baseline confound (§3.6 revision) — `scripts/analysis/spike_measurement_diagnostic.py`
  - Reactive acceleration from timecourse (§3.6d, exploratory) — `scripts/analysis/reactive_dynamics_from_timecourse.py`
  - Anxiety modulation tests — null (§3.7 revision) — `scripts/analysis/anxiety_modulates_reactive_dynamics.py`
  - Baseline disentanglement (§3.7 final) — `scripts/analysis/anxiety_peak_disentangle.py`
- Memory updated through §4.59 (discoveries.md)
- Title locked

**Not done:**
- Prose drafting for any section
- Figure production (specs exist; panels need rendering)
- Confirmatory smoothed_vigor_ts.parquet — not yet processed; needed for §3.6d acceleration replication and §3.7 confidence replication
- Supplementary section text
- Senior-author sign-off

## Pending decisions (need before drafting)

1. ~~**Title.**~~ **LOCKED 2026-06-07:** *Embodied defensive computation: parameters and affect dissociate strategic preparation from reactive response*

2. **Affect framing in discussion.** The counterintuitive anxiety_slope_T → smaller spike finding has 3 candidate framings:
   - **Defensive budget:** total preparation has a fixed budget; anxiety reactivity redistributes it anticipatorily, leaving less for reactive surge. Most parsimonious.
   - **Pavlovian inhibition:** strong anticipatory preparation pre-engages defensive circuits and dampens reactive amplification on encounter.
   - **Self-calibration:** subjects with steeper anxiety reactivity calibrate their motor output to match perceived threat earlier, requiring less abrupt reactive correction.

   These are different mechanistic claims; pick one or present as complementary.

3. **PNAS vs Cell Reports vs NHB primary target.** Outline assumes PNAS push. Editor allocation matters — if PNAS rejects, the same manuscript can submit to NHB or Cell Reports with minor restructure.

4. **Manipulation follow-up?** This is the single highest-leverage extension (capture-penalty manipulation, ~6 months, ~150 subjects online). Decide whether to:
   - Submit current data alone to PNAS (50-60% odds)
   - Run manipulation first → submit as integrated paper (60-70% odds, +6-9 months)
   - Run in parallel as follow-up paper for journal-of-rejection

## Order of operations for drafting

### Phase 1 — Setup (1 week)
- Senior-author sign-off on outline and title
- Confirm authorship order
- Confirm submission target (default: PNAS first)

### Phase 2 — Figure production (2-3 weeks)
- Figure 1: task + W(u) schematic (mostly exists in old draft)
- Figure 2: behavioral adaptation (mostly exists)
- Figure 3: vigor dynamics (encounter-aligned; partial code exists in result 305 area)
- Figure 4: model comparison + recovery (exists)
- Figure 5: (ω, κ) dissociation (exists)
- Figure 6: parameter dynamics — NEW (need code)
- Figure 7: affect modulates dynamics — NEW (need code)

Figures 6 and 7 are the new ones. Their underlying data exist in
`results/stats/joint_optimal/parameters_predict_vigor_dynamics.csv` and
`results/stats/affect_analysis/affect_modulates_dynamics.csv`.

### Phase 3 — Prose drafting (4-6 weeks)
Order matters because earlier sections set framing for later ones:

1. **§3.6 (parameter dynamics) first** — load-bearing empirical centerpiece. Once this is drafted clearly, §3.7 and §1 follow naturally.
2. **§3.7 (affect modulates dynamics)** — handle the counterintuitive framing carefully.
3. **§3.2 (vigor dynamics descriptive)** — sets up §3.6.
4. **§1 (Introduction)** — now we know what the empirical center is.
5. **§4 (Discussion)** — comes after all results, since it integrates.
6. **§3.1, §3.3, §3.4, §3.5** — these largely exist in current draft; need refreshing for new framing.
7. **§2 (Methods)** — mostly reuse from current draft, add dynamics-feature descriptions.
8. **Abstract + Significance** — last, after all sections are stable.
9. **Supplementary** — last.

### Phase 4 — Senior review + revision (3-4 weeks)
- Internal review by senior authors
- Revisions
- Polish

### Phase 5 — Submission prep (1-2 weeks)
- Cover letter
- Suggested reviewers
- Data + code repository preparation
- OSF preregistration link

**Total estimated time from sign-off to submission: 12-16 weeks.**

## Per-section drafting guidance

### §1 — Introduction
- Open: classical theory separates choice and action; embodied perspectives challenge
- Cite Cisek & Pastor-Bernier 2014; Pezzulo & Cisek 2016; Gallivan et al. 2018; Nuzzi-Cisek-Pezzulo 2026; Mobbs 2020
- Frame foraging-under-threat as natural test
- Preview the three findings: (1) joint model + dissociation, (2) parameters predict dynamics, (3) affect modulates non-obviously
- Be honest that §3.7's anxiety finding is counterintuitive and replicated

### §2 — Methods
- Reuse most of current draft
- Add subsection on dynamics-feature computation (encounter-aligned trial decomposition)
- Add subsection on per-subject affect-feature computation (reuse from result 510 pipeline)

### §3.1 — Coordinated behavioral adaptation
- Use existing draft §2.1 prose
- Update opening to emphasize "we test whether choice and motor execution adapt to the same input"

### §3.2 — Vigor unfolds in time (REINSTATED)
- Use existing draft §2.2 prose
- Emphasize this is descriptive — sets up §3.6
- ~83% of participants show the spike; large effect size

### §3.3 — Joint W(u) wins
- Use existing draft §2.3 prose
- ΔWAIC table
- Frame as foundational

### §3.4 — Parameters recover and replicate
- New compact section
- Recovery scatter, posterior overlap
- Brief — credibility anchor

### §3.5 — (ω, κ) dissociation
- Use existing draft §2.4 prose
- Per-parameter signatures
- Angle → optimality

### §3.6 — Parameters predict strategic dynamics (NEW)
- Three findings: anticipatory steepness (ω); baseline (κ + ω); reactive null (ω)
- Lead with strategic/reactive dissociation as the structural payoff
- Mapping onto imminence continuum
- THIS IS THE LOAD-BEARING SECTION; draft first

### §3.7 — Affect modulates dynamics (NEW)
- Three findings: anxiety_slope_T → front-loading; anxiety_intercept → spike amplification; confidence_slope_D → both phases
- LEAD with the counterintuitive anxiety_slope_T → smaller spike result
- Frame as "defensive budget" or "preparation/reactive trade-off"
- R² gains substantial

### §4 — Discussion
6 paragraphs:
1. What we found (synthesis)
2. Strategic/reactive dissociation maps onto imminence continuum (theoretical anchor)
3. Affect distributes defensive preparation (counterintuitive finding interpretation)
4. Implications for choice-action separation (theoretical opponent)
5. Limitations (no manipulation, no neural data, virtually embodied)
6. Future directions (manipulation, neural, clinical extensions)

## What lives where

| Asset | Location |
|---|---|
| Outline | `write-up/embodied_paper_outline.md` |
| This plan | `write-up/embodied_paper_implementation_plan.md` |
| Old prose draft (reusable for §3.1–§3.5) | `drafts/final_paper_draft.md` |
| Methods code (W(u), MCMC, etc.) | `scripts/run_mcmc_pipeline.py`, `scripts/modeling/` |
| §3.6 dynamics features script | `scripts/analysis/parameters_predict_vigor_dynamics.py` |
| §3.6 results CSV | `results/stats/joint_optimal/parameters_predict_vigor_dynamics.csv` |
| §3.7 affect-modulation script | `scripts/analysis/affect_modulates_dynamics.py` |
| §3.7 results CSV | `results/stats/affect_analysis/affect_modulates_dynamics.csv` |
| Memory of findings | `instructions/memory/discoveries.md` §4.54 + §4.55 |
| Memory of session | `instructions/memory/session_history.md` (recent entries) |
| Pipeline status | `instructions/memory/pipeline_state.md` (recent entries) |

## Critical things to NOT lose track of

1. **The cross-channel prediction (β = +0.47 of ω on anticipatory vigor)** is now part of §3.6 — don't drop it; it predates the full dynamics analysis and is in the existing draft. It belongs in §3.6a or 3.6b.

2. **The choice-fit cost of M4 vs M2 (~10 R² points)** still applies. Defense in §3.3: M4 buys joint identification of κ + dissociation findings; the choice-fit difference is the price paid for joint structure. Reviewers will ask.

3. **The reactive spike finding (~d=0.65, 83% of participants)** was preregistered as H2. The current dynamics findings extend H2 by linking the spike to parameter null. Don't double-claim.

4. **Affect was demoted then re-promoted.** Earlier outline had affect as one paragraph. New §3.7 promotes it because the dynamics-modulation findings are substantial. Be ready to defend this with the R² jumps.

5. **The reactive spike → ω null replicates as null in BOTH samples on all 3 spike metrics.** This is a real dissociation, not just absence of effect. Frame as predicted by predatory imminence theory.

6. **The anxiety_slope_T → smaller reactive spike result is the "wow" finding.** Don't bury it. The paper's substantive contribution rests on this and the strategic/reactive parameter dissociation together.

## When to revisit this plan

- Before drafting starts: confirm pending decisions (title, framing, target journal)
- After §3.6 first draft: check if §3.7 framing needs adjustment
- Before submission: full review of plan against final manuscript

## Quick-reference numbers — REVISED 2026-06-07 (post-diagnostic)

**Foundational findings (preregistered, both samples):**

| Finding | Effect | Replication |
|---|---|---|
| Joint W(u) ΔWAIC vs M2 | +1,621 conf; +1,966 exp | M4 wins both |
| Parameter recovery | r(ω) = 0.94; r(κ) = 0.92 | |
| Choice accuracy M4 vs M2 | 0.773 vs 0.789 conf | M2 wins choice fit (defend) |
| Vigor R² M4 vs M2 | 0.412 vs 0.012 conf | M4 wins vigor by 34× |
| ω → P(heavy) | β ≈ −0.82 | both samples, p < 10⁻⁵⁰ |
| κ → mean vigor | β ≈ −0.75 | both samples, p < 10⁻³⁰ |
| ω → escape rate | β = +0.222 (pooled) | both samples (§4.40) |

**Parameter dynamics across both phases (§3.6, both samples where measurable):**

| Finding | Effect | Replication |
|---|---|---|
| ω → anticipatory steepness on T | β = +0.21 / +0.19 | both samples |
| κ → baseline at low T | β = −0.46 / −0.51 | both samples, R² ≈ 0.20 |
| ω → baseline at low T (positive) | β = +0.26 / +0.26 | both samples |
| ω → absolute peak strike effort | β = +0.13 / +0.22 | both samples (§4.56) |
| κ → absolute peak strike effort | β = −0.49 / −0.56 | both samples, R² ≈ 0.22-0.31 |
| ω → reactive acceleration (slope of vigor 0-500ms post-encounter) | β = +0.18 | exp only — needs confirmation |
| κ → reactive acceleration | β = −0.17 | exp only — needs confirmation |

**Affect-reactive finding (§3.7, exploratory only):**

| Finding | Effect | Replication |
|---|---|---|
| confidence_intercept → peak_post (with baseline as covariate) | β = −0.125, p = 0.001 | exp only — needs confirmation |

**Confirmed nulls (important to mention in discussion):**

| Finding | Verdict |
|---|---|
| anxiety_slope_T → reactive measures (all metrics, all baseline-controlled tests) | Null — was baseline-ceiling artifact |
| anxiety_intercept → reactive measures (with baseline control) | Null — was baseline-mediated |
| anxiety × ω, anxiety × κ interactions on reactive acceleration | All null |
| ω × κ → clinical scales (joint MMR) | F p = 0.60 exp, 0.075 conf — null |
| Cross-sample CCA behavior × clinical | r drops from 0.32 in-sample to 0.06 cross-sample — null |

**Methodological note for paper:**
- Subtractive spike measures (peak_post − pre_mean) correlate strongly with baseline (r = −0.58 in exp). Use absolute peak or acceleration for reactive measures.
- Baseline disentanglement is essential — many "effects" in subtractive analyses are baseline-mediated
