---
result_id: 104
class: behavioral_effects
title: Vigor timecourse splines distinguish encounter and threat effects (H2 family)
status: untested
prereg_h: [H2a, H2b]
internal_h: []
samples: []
notebooks: [notebooks/analysis/H2_vigor_dynamics.ipynb]
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: null
---

# Result 104 — Vigor timecourse splines distinguish encounter and threat effects (H2 family)

> **⚠️ Deferred — blocked by two pipeline bugs uncovered 2026-06-02. Do not interpret the cached numbers below as a validated current result; they reflect the confirmatory sample only and the analysis path that produced them has not been re-run since.**

## Why this is deferred

Three things need to happen before result_104 can ship as a real both-sample lab report. Two are pipeline bugs; one is a notebook structural issue (which has already been fixed).

### 1. `load_data.load_both()` returns sample-shared `vigor_metrics`

The H2a (encounter spike) and H2b (GAM temporal signature) analyses pull `d['vigor_metrics']` from `load_data.load_both()`. On 2026-06-02, executing cells 1, 3, 5 of `H2_vigor_dynamics.ipynb` as a standalone script produced **byte-for-byte identical** H2a paired-t output for the two samples (Heavy: Δ = +0.0349, t = 7.72, p = 1.89e-13, d = +0.454; Light: Δ = +0.0541, t = 12.95, p = 1.45e-30, d = +0.762, in both samples), and identical H2b encounter-spike output (mean spike +0.0358, t = 11.01, p = 8.73e-24, d = +0.647, in both samples). The trial-level data tables in `d['trials']` differ correctly between samples (23,490 expl vs 22,761 conf), but `len(d['vigor_metrics'])` is `93960` for both. The H2b numbers above also match the cached confirmatory CSV (`results/stats/confirmatory_hypothesis_results.csv`) exactly — strongly suggesting `vigor_metrics` is loaded from a single confirmatory-only source regardless of which sample is requested. This is the same shape of bug as the `data/model_input/` mislabel resolved 2026-05-29 — a data-loading layer returning one sample's data under both labels.

**Likely impact beyond result_104:** any result that reads `d['vigor_metrics']` is at the same risk of contamination. Trial-level paths (`d['vigor']`, `d['vigor_valid']`) used by the H1 notebook and H8 are NOT affected.

**Fix needed:** trace where `vigor_metrics` is loaded from inside `notebooks/analysis/load_data.py`, ensure per-sample dispatch.

### 2. H2c GAM cell errors with `LinAlgError: Singular matrix`

The H2c cell (which produces H2b's GAM LRTs) fits a MixedLM with `K = min(K_SPLINE, 4)` cubic spline basis on a `t_epoch` column that takes 4 unique values (`onset`, `anticipatory`, `reactive`, `terminal`). Cubic splines with df = 4 on 4 unique x-values produce a rank-deficient design matrix — hence the singular matrix. The original analysis (whose numbers appear in the cached confirmatory CSV: encounter LRT χ² = 1024.8, threat LRT χ² = 114.8) appears to have used the raw keypress timecourse (continuous `alignedEffortRate`), not 4 discrete epochs. The cell's own comment acknowledges this:

> Uses epoch-level vigor metrics (anticipatory/reactive/terminal) rather than raw keypress timecourse, which requires the raw alignedEffortRate column.

**Fix needed:** either reduce K (e.g., `K = 3` for an over-parameterized linear-quadratic basis on 4 epochs) and accept that the spline degenerates to polynomial regression, or restore the raw-timecourse path used by the original analysis. The latter matches the prereg's "natural cubic regression splines (K=10) via MixedLM" specification.

### 3. Notebook validation fix (✅ done 2026-06-02)

`H2_vigor_dynamics.ipynb`'s trailing "H2 Summary" cell was missing the required `outputs: []` and `execution_count: null` keys, causing `jupyter nbconvert` to refuse saving the executed notebook with `nbformat.validator.NotebookValidationError: 'outputs' is a required property`. Patched in place 2026-06-02 by an agent run; no source code in the cell was altered.

## Cached confirmatory numbers (use for orientation only — not validated against current pipeline)

From `results/stats/confirmatory_hypothesis_results.csv`:

| Test | Statistic | Value | p |
|---|---|---|---|
| H2a — encounter spike (Cohen's d) | d | 0.647 | 8.73 × 10⁻²⁴ |
| H2b — GAM encounter LRT | χ² | 1024.8 | ≈ 0 |
| H2b — GAM threat LRT | χ² | 114.8 | 2.55 × 10⁻¹⁵ |

The H2a confirmatory d = 0.647 matches what our buggy 2026-06-02 standalone run produced for *both* samples. We currently do not know what the true exploratory H2a d, H2b LRTs, or any H2 effect on the corrected pipeline will look like.

## What to do when picking this back up

1. **Fix `load_data.py`'s `vigor_metrics` sample dispatch.** Confirm by running a small probe that prints `len(d['vigor_metrics'])` for each sample and checking the two values differ.
2. **Decide on the H2c GAM specification.** Either (a) restore the raw-timecourse `alignedEffortRate` path with K=10 splines, matching the prereg; or (b) accept a degenerate epoch-level fit with K ≤ 3 and document the deviation explicitly.
3. **Re-execute `H2_vigor_dynamics.ipynb` end-to-end** using the project's `PYTHONPATH=notebooks/analysis ... --ExecutePreprocessor.kernel_name=python3` recipe (see `result_103` Replication block for the canonical command).
4. **Extract H2a + H2b numbers for both samples**, T3-validate confirmatory against the cached CSV, and write up this result file as a full lab report matching `result_103`'s structure and depth.

## References

- `notebooks/analysis/H2_vigor_dynamics.ipynb` — the source notebook (currently broken: GAM cell + summary-cell validation patched on 2026-06-02 but data-loading bug upstream).
- `notebooks/analysis/load_data.py` — source of the `vigor_metrics` sample-dispatch bug.
- `instructions/memory/pipeline_state.md` § 2026-06-02 — investigation log.
- [[result_103]] — H1c vigor under threat (the static, working counterpart).
- [[result_307]] — Phase dissociation by parameters (a deprecated-framework result in the same vigor-dynamics neighborhood).
