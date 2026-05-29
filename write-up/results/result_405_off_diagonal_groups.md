---
result_id: 405
class: choice_vigor_coupling
title: HL and LH subgroups differ in confidence, calibration, and self-reported apathy
status: supported_exploratory
prereg_h: []
internal_h: [H33]
samples: [exploratory_290]
notebooks: []
scripts: []
outputs: []
figures: [TODO]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 405 — HL and LH subgroups differ in confidence, calibration, and self-reported apathy

> **⚠️ Exploratory. Numbers from `instructions/memory/hypotheses.md` § H33.**

## Result

t-tests between HL (chose hard, pressed easy) and LH (chose easy, pressed hard) subgroups in the exploratory sample:

| Measure | HL | LH | p |
|---|---|---|---|
| Mean confidence | 3.40 | 2.76 | **0.003** |
| Anxiety calibration (within-subject r(anxiety, threat)) | 0.22 | **0.36** | **0.007** |
| AMI total apathy | 25.4 | **32.2** | **< 0.001** |

## Interpretation

The two off-diagonal subgroups differ in revealing ways. HL subjects (chose hard despite low vigor capability) are *overconfident* — they have higher mean confidence ratings than their actual behavior warrants — and have *worse* anxiety calibration. LH subjects (chose easy despite being high-vigor capable) report *more apathy* on the AMI but are *better calibrated* and execute well when they do engage.

The pattern reframes "apathy" in this task: LH subjects look apathetic by self-report but actually perform better than HL subjects. The disconnect is consistent with a metacognitive deficit interpretation of HL (overconfidence + miscalibration drives them to take options they cannot handle) and a "knowing-but-disengaging" interpretation of LH (accurate self-assessment + low engagement). This finding directly anchors the clinical framing in [[result_602]] (apathy → vigor, not choice).

## References

- `instructions/memory/hypotheses.md` § H33.
- [[result_404]] — Source of the quadrants.
- [[result_506]] — Continuous version (confidence miscalibration ~ choice + vigor).
- [[result_602]] — AMI apathy → vigor.
