---
name: write-result
description: Write a lab-report-style markdown document for one empirical result in write-up/results/, following the project's M5 collaborative workflow (factual scaffolding → prose + grilling → collaborative Interpretation). Validates headline numbers against the canonical notebook (T3 fail-loud) and inlines figures. Use when the user wants to document a finding, write up a result, or invokes /write-result with a free-text description of the result.
---

# write-result — Build one result_###_<slug>.md from a free-text description

You are writing one entry in a growing library of lab-report-style result documents that live in `write-up/results/`. Each document covers exactly one finding (one preregistered sub-test OR one exploratory analysis), is self-contained and replicable, and has an Interpretation section dense enough to be stitched together with siblings to form a publishable Results section.

You collaborate with the user across three stages (M5). Do not skip stages, and do not batch all three. Each stage has a defined handoff.

---

## Step 0 — Read memory and prereg

Before any action, read in parallel:
- `instructions/memory/MEMORY.md` (the index)
- `instructions/memory/hypotheses.md` (internal H1–H39 with status + the prereg ↔ internal mapping table)
- `instructions/memory/pipeline_state.md` (notebook execution states)
- `write-up/preregistration.md` (prereg H1–H7 with formal criteria)
- `write-up/results/INDEX.md` if it exists (so you know what's already written and which result_ids are taken)
- `.claude/skills/write-result/template.md` (the canonical template — copy this as your starting point)

If the user invokes the skill mid-conversation with prior context, re-read these anyway. Memory drifts.

---

## Step 1 — Match the free-text input to a finding

The user invokes the skill with a free-text description like "/write-result H1b affect LMM" or "/write-result the β dissociation between choice and vigor" or "/write-result threat reverses choice-vigor coupling."

Match the description against:
1. Preregistered hypotheses in `write-up/preregistration.md` (H1a, H1b, H1c, H2a, H2b, H3a–c, H4a–e, H5a–d).
2. Internal hypotheses in `instructions/memory/hypotheses.md` (H1–H39).
3. Existing drafts in `drafts/results_by_hypothesis/` and `drafts/discovery_results.md`.

Then propose to the user:
- **Finding title** (one line, sentence case)
- **Class** (one of: behavioral_effects, computational_model, vigor_dynamics, choice_vigor_coupling, metacognition, individual_differences)
- **Proposed result_id** (next free slot in the class's block — 100s/200s/300s/400s/500s/600s, with ~100 headroom each; sequential within block by writing order)
- **Filename** (`result_###_<short-kebab-slug>.md`)
- **Prereg H# and internal H# this maps to** (often a list)
- **Whether this result already exists** as a file in `write-up/results/` (if so, switch to **update mode** — see Step 7)

Wait for user confirmation before doing anything else. The user may rename the title, reassign the class, or redirect the result_id.

---

## Step 2 — Stage 1: Factual scaffolding (skill works alone, user approves numbers)

Copy `.claude/skills/write-result/template.md` to the target path and fill in everything except Overview, Interpretation, and Caveats & Limitations. These are the factual sections:

- **Frontmatter** — every field populated. Use real data, not CLAUDE.md memory:
  - `samples:` — read the actual N from the data directory or notebook output, not from CLAUDE.md (which is sometimes stale). Project data dirs:
    - Exploratory: `data/exploratory_350/processed/stage5_filtered_data_<latest_timestamp>/`
    - Confirmatory: `data/confirmatory_350/processed/stage5_filtered_data_<latest_timestamp>/`
  - `notebooks:` — paths to notebooks that produce headline numbers
  - `outputs:` — paths to cached CSVs/PKLs that hold the numbers
  - `figures:` — see Step 4 below for the F1 figure lookup
  - `status:` — one of: `supported | supported_exploratory | partial | refuted | refuted_exploratory | dead-end | retracted | untested`
  - `created:` and `last_run:` — today's date
- **Hypothesis** — verbatim prereg statement when applicable, predicted direction, criterion, source.
- **Data Source** — sample N pre/post exclusion, input file paths, unit of analysis, N entering the model.
- **Method** — Bar 2 detail: one-paragraph procedure in plain English, model spec in code or math, software/versions, inference criterion, notebook + cell pointer with a validation note. If a statistical choice is non-obvious (e.g., cluster-robust SE, hierarchical priors, LOO vs WAIC), add a short paragraph explaining what it does and what it implies.
- **Result** — table with one row per term × one column per sample; inline figure (see Step 4); verdict line (PASS/PARTIAL/FAIL on the prereg criterion).
- **Replication** — exact command(s) to regenerate (see Step 3 for the nbconvert quirk), expected runtime, expected outputs.
- **References** — cross-links to related result files using `[[result_###]]` syntax (forward-link freely; resolve later), notebook paths, literature citations if relevant.

When done, write the file and present a summary to the user: what's populated, what's still `<TODO>`, and any flags worth their review (e.g., "data dir says N=290 but CLAUDE.md says 293" — surface conflicts, don't silently pick).

**Do not proceed to Stage 2 until the user explicitly approves Stage 1.**

---

## Step 3 — T3 source-of-truth validation (within Stage 1, before user review)

Every headline number in the Result table MUST be validated against the notebook that produces it. The procedure:

1. **Find the notebook cell.** Use grep across `notebooks/` for the formula or test name. Inspect the cell source to confirm it's the right one.
2. **Check the cell's execution state.** Use:
   ```python
   import json
   nb = json.load(open(path))
   for cell in nb['cells']:
       if cell['cell_type']=='code':
           print(cell.get('execution_count'), len(cell.get('outputs', [])))
   ```
   If `execution_count` is None or outputs are empty, the notebook hasn't been executed since save — you cannot validate.
3. **If outputs exist:** extract the headline numbers from the cell's outputs and compare to the draft summary (`drafts/results_by_hypothesis/`, `instructions/memory/discoveries.md`, `instructions/memory/hypotheses.md`, etc.). If they match within rounding, ship.
4. **If outputs are missing OR numbers conflict:** execute the notebook. Project-specific quirks:
   - The notebook may declare kernel `aversive` which isn't installed. Override with `--ExecutePreprocessor.kernel_name=python3`.
   - Notebooks in `notebooks/analysis/` import local modules `config` and `load_data` from their own directory. When `jupyter nbconvert` runs from the project root, those imports fail unless `PYTHONPATH` includes that directory. The working command from project root is:
     ```bash
     PYTHONPATH=/Users/nokada/Desktop/EffortForagingUnderThreat/notebooks/analysis \
       /opt/anaconda3/envs/effort_foraging_threat/bin/jupyter nbconvert \
       --to notebook --execute notebooks/analysis/<NB>.ipynb \
       --inplace --ExecutePreprocessor.kernel_name=python3 \
       --ExecutePreprocessor.timeout=600
     ```
   - For portability across devices, also document the relative-path form in the result's Replication section: replace the absolute `PYTHONPATH` with `notebooks/analysis` (relative to project root). Inform the user that the absolute form is what reliably executed, and they should adapt for their setup.
5. **After execution:** extract numbers from cell outputs and validate against draft.
6. **If still cannot find the notebook cell:** fail loud. Stop. Tell the user exactly what number you couldn't validate and where you looked. Do not write the result file with unverified numbers. Per project policy (decided 2026-05-23): the user will help provide the data or fix the pipeline.

---

## Step 4 — Figure lookup (F1 with missing-figure nudging)

For each result, look for figures in this order:
1. `results/figs/paper/` (paper-grade PDFs and PNGs)
2. `results/figs/<h_number>/` (organized by H number)
3. `results/stats/<class>/` (notebook-generated PNGs)
4. `data/figures/` (legacy)
5. The notebook itself (if it generates inline figures, they're saved somewhere — check for `plt.savefig` calls)

Conventions:
- `figures:` frontmatter field lists ALL found figures (PDFs first for paper-grade, PNGs second for inline preview).
- In the Result section, inline the PNG using markdown image syntax with a **relative path from the result file**:
  ```markdown
  ![alt text](../../results/stats/<class>/<figure>.png)
  ```
  (`write-up/results/result_*.md` → `../../results/...` reaches the project root.)
- After the inline PNG, add a one-sentence description of what the figure shows.
- Reference the paper-grade PDF separately (it can't be inlined in markdown, but the path is in `figures:`).
- If no figure is found, leave `figures: [TODO]` in frontmatter AND add a paragraph in the Result section like "**Figure:** TODO — needs <description of what the figure should show>." Surface this as a nudge to the user.

---

## Step 5 — Stage 2: Prose + grilling questions (skill drafts, user answers)

After Stage 1 is approved, draft:

- **Overview** — 2–4 sentences. State the question, the finding, one-line interpretation. **No statistics** (those are in Result). Tone: academic, clear, no first-person.
- **Interpretation** — 3 paragraphs, ~300 words total, paper-grade density. Default structure:
  - ¶1: Numbers in task terms (translate β to odds ratios, probabilities, effect sizes, whatever makes the magnitude legible).
  - ¶2: Theoretical backbone. Connect to the broader literature (risk-sensitive foraging, EVC, etc.) NOT to the project's specific model — that comes later in computational_model results. The Interpretation should motivate model-building, not preview the model.
  - ¶3: What this result establishes and what it does NOT establish. Forward-link to results that take the next step (`[[result_###]]`). Do not overreach — the claim made in this paragraph must be supported by the data shown in Result.
- **Caveats & Limitations** — 4–6 bullets. Cover: confounds, post-hoc tests vs preregistered, statistical-choice limitations, design constraints, generalization caveats.

Then present a numbered list of **grilling questions** about framing decisions you had to make. Examples of decisions worth surfacing:
- How heavily this result references the joint model
- Whether to interpret non-preregistered terms (interactions, etc.)
- Forward-link policy when other results don't exist yet
- Linkage to legacy draft files
- Voice/tone calibration on the first result of a class

Wait for the user to answer before Stage 3.

---

## Step 6 — Stage 3: Collaborative Interpretation refinement

The Interpretation section is load-bearing. After Stage 2, present it verbatim and ask the user for direct edits, additions, or pushback on each paragraph individually. Apply the user's revisions, recheck word count (~300), and confirm tone matches prior results in the same class.

Then mark complete.

---

## Step 7 — Update mode (when the result file already exists)

If Step 1 finds an existing `result_###_<slug>.md` matching the description, switch to update mode:

1. Read the existing file.
2. Re-run T3 validation on the headline numbers in the Result table.
3. If numbers have changed, propose a diff: what changed, by how much, in which sample, why (notebook re-ran with new data? bug fix? different preprocessing?).
4. If the change is material (e.g., a sign flip, a significance flip, an effect size change > 30%), add a one-line entry to a "Revision notes" section at the bottom of the file in the V2 format:
   ```markdown
   ## Revision notes
   - **YYYY-MM-DD:** <one-line description of what changed and why>
   ```
5. If the change is trivial (rounding, minor decimal shifts), just update the numbers and bump `last_run:`.
6. Update `last_run:` to today's date.
7. Present the diff to the user; await approval before saving.

---

## Step 8 — Update INDEX.md

After saving the result file, regenerate `write-up/results/INDEX.md` to include the new entry. INDEX.md is a sorted table with columns:
- `result_id`
- `class`
- `title` (linked to file)
- `status`
- `prereg_h`
- `last_run`

Sort by class block, then by result_id within block. INDEX.md has no frontmatter — it's a plain table.

If `INDEX.md` doesn't exist yet, create it from scratch by globbing all `write-up/results/result_*.md` files and reading each file's frontmatter.

---

## Frontmatter field reference

```yaml
result_id: 101                              # integer in [100, 699] per class block
class: behavioral_effects                   # one of 6 slugs
title: <short descriptive title>
status: supported                           # see vocab below
prereg_h: [H1a]                             # list of prereg H# strings
internal_h: [H30]                           # list of internal H# strings (from hypotheses.md)
samples: [exploratory_290, confirmatory_281]  # subset if only one was tested
notebooks: [notebooks/analysis/H1_adaptive_shifts.ipynb]
scripts: []                                 # .py scripts involved
outputs: [results/stats/confirmatory_hypothesis_results.csv]
figures: [results/figs/paper/fig_h1a_choice.pdf, results/stats/avoidance_activation/H1a_choice_surface.png]
created: 2026-05-23
last_run: 2026-05-23
```

### Class block assignments
| Block | Class slug |
|---|---|
| 100–199 | `behavioral_effects` |
| 200–299 | `computational_model` |
| 300–399 | `vigor_dynamics` |
| 400–499 | `choice_vigor_coupling` |
| 500–599 | `metacognition` |
| 600–699 | `individual_differences` |

### Status vocabulary
- `supported` — both samples agree
- `supported_exploratory` — exploratory only, confirmatory not run
- `partial` — mixed across samples or sub-tests
- `refuted` — failed in confirmatory or both
- `refuted_exploratory` — failed in exploratory, never made it to confirmatory
- `dead-end` — methodological dead-end
- `retracted` — corrected/withdrawn
- `untested` — planned but not run

---

## Style discipline

- **Voice:** academic, clear, no first-person ("we" is fine in Method paragraphs; avoid in Interpretation).
- **Tense:** present for findings ("threat reduces P(heavy)"), past for methods ("we fit").
- **Numbers:** report β to 2-3 sig figs, p-values in scientific notation if very small.
- **Length targets:** Overview ~80w, Interpretation ~300w, Caveats 4–6 bullets, Method 1–2 paragraphs + code block.
- **No hedging stack-ups:** "may possibly be associated with" → "is associated with" (with proper caveats elsewhere).
- **Cross-links:** `[[result_###]]` syntax. Forward-link freely.

---

## Things that go wrong and how to handle them

- **Notebook execution fails on import:** check the PYTHONPATH fix (Step 3). If still fails, check the conda env path (`/opt/anaconda3/envs/effort_foraging_threat/bin/python3.11`).
- **Notebook kernel name mismatch:** the project's notebooks often declare kernel `aversive`. Always override with `--ExecutePreprocessor.kernel_name=python3`.
- **CLAUDE.md says N=293 but data says N=290:** trust the data. The user has confirmed (2026-05-23) that 290 is the current exploratory N. Use what the notebook prints, not what memory says.
- **The drafts disagree with each other:** prefer `drafts/results_by_hypothesis/` over `instructions/memory/discoveries.md` (which is flagged as outdated in MEMORY.md). Always cross-check against the notebook.
- **A claimed exploratory finding has no notebook to validate it against:** fail loud per Step 3.6. Do not invent a notebook path. Ask the user.
- **The user's free-text description matches multiple findings:** propose all matches, ask which one to write up.

---

## Calibration anchor: result_101

Use `write-up/results/result_101_choice_threat_distance.md` as a reference for tone, length, structure, and cluster-robust-SE style of explanation. It is the hand-validated template for the `behavioral_effects` class.
