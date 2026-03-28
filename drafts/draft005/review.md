# Reviewer Report — Draft 005

**Manuscript:** "Metacognitive bias, not threat computation, bridges foraging decisions to clinical anxiety"
**Recommendation:** Minor Revisions

---

## Overall Assessment

This is a strong manuscript that makes a genuine contribution to computational psychiatry by showing that the metacognitive relationship between computed danger and experienced affect — not the computational parameters themselves — predicts clinical anxiety. The EVC-LQR model is well-identified, the parameter recovery is excellent, and the calibration-discrepancy decomposition is both novel and well-motivated theoretically. The narrative is cleaner than earlier drafts and the theoretical engagement is substantially improved.

However, several issues need attention before the paper is suitable for Nature Communications.

---

## Major Issues

### 1. The Wise (2023) reference (Ref 23) is wrong

Reference 23 is cited as "Wise et al. — interactive cognitive maps" in the introduction (paragraph 4), but the actual reference listed is "Wise, T., Zbozinek, T. D., Michelini, G., Hagan, C. C. & Mobbs, D. Changes in risk perception... COVID-19 pandemic... R. Soc. Open Sci." This is the wrong paper. The COVID risk perception paper is not about interactive cognitive maps or confidence tracking. The correct reference should be: Wise, T. et al. Interactive cognitive maps support flexible behaviour under threat. Cell Reports 42, 113400 (2023). This IS listed as Ref 33. The introduction cites Ref 23 where it should cite Ref 33. Fix the citation numbering.

### 2. The H3 confidence result has shifted — needs clearer handling

The abstract mentions "confidence (β = 0.575, t = 13.5)" as a within-subject finding, which is correct. But the paper no longer includes a between-subject confidence analysis (the original H3 about confidence tracking choice quality). The earlier discovery documents showed r = 0.230 for confidence × choice quality, but verification against the actual results file shows r = -0.08 (not significant). The paper handles this by focusing entirely on H4 (calibration-discrepancy), which is the right call, but the transition is slightly awkward — the reader expects a between-subject confidence finding after the strong within-subject result and never gets one.

**Fix:** Add a brief paragraph after the within-subject affect results noting that between-subject mean confidence did not reliably predict task performance, motivating the more nuanced calibration-discrepancy decomposition that follows.

### 3. The model comparison table uses M2 BIC=34,227 but earlier in the agent output M2 BIC was reported as 42,767

The model comparison table in the paper (Table 1) lists M2 at BIC=34,227, but the earlier model comparison output from the fix agent reported M2 at BIC=42,767. These are different runs. The plan document Noah edited uses BIC=42,767. The paper should use the values from the most authoritative source. Verify which M2 result is correct and ensure consistency across the paper, plan, and supplementary table.

### 4. No explicit engagement with the Affective Gradient Hypothesis

The discussion cites Shenhav (2024) AGH (Ref 32) but only in passing ("This aligns with functional accounts of anxiety"). The theory review identified AGH as a potential tension — our model implies a cold computation separate from affect, while AGH says affect IS the computation. This deserves a sentence or two acknowledging the tension and proposing reconciliation. The paper doesn't need to resolve it, but ignoring it entirely is a missed opportunity.

---

## Minor Issues

### 5. Abstract length

The abstract is ~250 words but reads as dense. The sentence about the model comparison (M1-M6) results could be cut — the abstract should focus on the key finding (calibration-discrepancy), not the model validation.

### 6. "Predominant double dissociation"

The paper correctly uses "predominant" rather than "clean" dissociation, which is honest. But it should briefly state what the leakage is (calibration weakly predicts STAI, discrepancy weakly predicts survival) rather than just calling it "predominant." A reviewer will want to know the exceptions.

### 7. The behavioral profiles section feels tacked on

R3 (behavioral profiles) sits between the model results (R2) and the metacognition results (R4-R5). It breaks the narrative flow. Consider moving it to supplementary or merging it with the discussion. The profiles are interesting but not essential to the calibration-discrepancy story.

### 8. Convergent validity paragraph

The task-clinical convergent validity (task anxiety × STAI r=0.31, task confidence × AMI r=-0.25) is mentioned briefly but the actual correlations are not all reported. This should be a supplementary table.

### 9. Block stability section is thin

The "discrepancy is stable across blocks" section (r=0.48-0.68) is only two sentences. Either expand it with the within-threat-level analysis or move to supplementary methods.

### 10. Missing Wise (2023) interactive cognitive maps reference

The correct Wise paper (Cell Reports 2023 on interactive cognitive maps) should be cited in the introduction when discussing confidence as a metacognitive signal. Currently Ref 33 exists but is only cited in the discussion. Promote it to the introduction.

---

## Statistics Verification

- BIC = 32,133 ✓ (matches evc_model_comparison_final.csv)
- Choice r² = 0.951 ✓
- Vigor r² = 0.511 ✓
- γ = 0.209 ✓ (0.210 in some files — minor rounding difference)
- ε = 0.098 ✓
- Anxiety LMM β = -0.557, t = -14.04 ✓
- Confidence LMM β = 0.575, t = 13.48 ✓
- Calibration × choice quality r = 0.230 ✓
- Calibration × survival r = 0.179 ✓
- Discrepancy × STAI β = 0.338, HDI [0.23, 0.45] ✓
- Orthogonality r = 0.019 ✓
- Recovery ce r = 0.92, cd r = 0.94 ✓
- Parameter independence r = -0.14 ✓

All statistics verified against results files.

---

## Verdict

The paper is suitable for Nature Communications after minor revisions:
1. Fix the Wise reference numbering
2. Add a transition paragraph for the between-subject confidence null
3. Verify M2 BIC consistency
4. Brief AGH engagement in discussion
5. Consider moving behavioral profiles to supplementary

The core findings — EVC-LQR model with recoverable parameters, calibration-discrepancy dissociation, metacognition as the clinical bridge — are robust and novel.
