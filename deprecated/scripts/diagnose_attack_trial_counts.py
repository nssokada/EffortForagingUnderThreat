#!/usr/bin/env python3
"""
Diagnose attack trial counts per subject, per threat level, and per block.

Motivation: H2b.2 (threat independence of encounter spike) and H2e.2 (cross-block
spike stability) return NaN. This script checks whether there are enough attack
trials per cell (subject x threat x block) to compute encounter-epoch vigor.

Key facts:
- 81 trials per subject (0-80), 3 blocks of 27
- Block 0: trials 0-26, Block 1: trials 27-53, Block 2: trials 54-80
- Attack trials have is_attack=1
- Encounter epoch vigor requires is_attack=1 (predator actually appears)
- Threat levels: T=0.1, T=0.5, T=0.9
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
EXCLUDE = [154, 197, 208]

# ── Load data ──
print("=" * 70)
print("ATTACK TRIAL COUNT DIAGNOSTIC")
print("=" * 70)

beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False,
                   usecols=['subj', 'trial', 'threat', 'isAttackTrial', 'type'])
beh = beh[~beh['subj'].isin(EXCLUDE)].copy()
beh['T_round'] = beh['threat'].round(1)
beh['is_attack'] = beh['isAttackTrial'].astype(int)
beh['block'] = beh['trial'] // 27  # 0, 1, 2

n_subj = beh['subj'].nunique()
print(f"\nTotal subjects: {n_subj}")
print(f"Total trials: {len(beh)}")
print(f"Total attack trials: {beh['is_attack'].sum()}")
print(f"Total non-attack trials: {(beh['is_attack'] == 0).sum()}")

# ── Trial type breakdown ──
print(f"\nTrial types: {beh['type'].value_counts().to_dict()}")
print("  type=1: regular choice, type=5: anxiety probe, type=6: confidence probe")

# ── Also check vigor_metrics for comparison ──
vm = pd.read_csv("/workspace/results/stats/vigor_analysis/vigor_metrics.csv")
vm_subj = vm['subj'].nunique()
print(f"\nSubjects in vigor_metrics.csv: {vm_subj}")
print(f"  (vs {n_subj} in behavior_rich — diff of {n_subj - vm_subj})")

# ════════════════════════════════════════════════════════════════════
# SECTION 1: Distribution of attack trial counts per subject per threat level
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: Attack trials per subject per threat level")
print("=" * 70)

attack_by_threat = (beh[beh['is_attack'] == 1]
                    .groupby(['subj', 'T_round'])
                    .size()
                    .reset_index(name='n_attack'))

# Make sure all subject x threat combos are represented
all_combos = pd.MultiIndex.from_product(
    [beh['subj'].unique(), [0.1, 0.5, 0.9]], names=['subj', 'T_round'])
attack_full = (attack_by_threat.set_index(['subj', 'T_round'])
               .reindex(all_combos, fill_value=0)
               .reset_index())

for t in [0.1, 0.5, 0.9]:
    subset = attack_full[attack_full['T_round'] == t]['n_attack']
    print(f"\n  T={t}:")
    print(f"    Mean:   {subset.mean():.2f}")
    print(f"    Median: {subset.median():.1f}")
    print(f"    Min:    {subset.min()}")
    print(f"    Max:    {subset.max()}")
    print(f"    SD:     {subset.std():.2f}")
    print(f"    Distribution: {subset.value_counts().sort_index().to_dict()}")
    print(f"    Subjects with 0 attacks: {(subset == 0).sum()}")
    print(f"    Subjects with <=2 attacks: {(subset <= 2).sum()}")

# Expected number of attack trials per threat level per subject
# 81 trials, T proportion of them are attacks
# But wait — T is the probability of predator appearing. Each trial has its own T.
# How many trials at each T?
print("\n  Trials per threat level (all subjects):")
trials_per_t = beh.groupby(['subj', 'T_round']).size().groupby('T_round').describe()
print(trials_per_t.to_string())

print("\n  Attack RATE by threat level:")
for t in [0.1, 0.5, 0.9]:
    sub = beh[beh['T_round'] == t]
    rate = sub['is_attack'].mean()
    print(f"    T={t}: actual attack rate = {rate:.3f} (expected: {t})")

# ════════════════════════════════════════════════════════════════════
# SECTION 2: Attack trials per subject per threat level per block
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Attack trials per subject per threat level per block")
print("=" * 70)

attack_by_threat_block = (beh[beh['is_attack'] == 1]
                          .groupby(['subj', 'T_round', 'block'])
                          .size()
                          .reset_index(name='n_attack'))

all_combos_block = pd.MultiIndex.from_product(
    [beh['subj'].unique(), [0.1, 0.5, 0.9], [0, 1, 2]],
    names=['subj', 'T_round', 'block'])
attack_block_full = (attack_by_threat_block.set_index(['subj', 'T_round', 'block'])
                     .reindex(all_combos_block, fill_value=0)
                     .reset_index())

print("\n  Summary by threat x block:")
for t in [0.1, 0.5, 0.9]:
    for b in [0, 1, 2]:
        subset = attack_block_full[(attack_block_full['T_round'] == t) &
                                    (attack_block_full['block'] == b)]['n_attack']
        print(f"    T={t}, Block={b}: mean={subset.mean():.2f}, "
              f"min={subset.min()}, max={subset.max()}, "
              f"n_zero={int((subset == 0).sum())}, "
              f"n_le1={int((subset <= 1).sum())}, "
              f"n_le2={int((subset <= 2).sum())}")

# Distribution table for T=0.1 specifically (most likely to have issues)
print("\n  Detailed distribution for T=0.1 per block:")
for b in [0, 1, 2]:
    subset = attack_block_full[(attack_block_full['T_round'] == 0.1) &
                                (attack_block_full['block'] == b)]['n_attack']
    print(f"    Block {b}: {subset.value_counts().sort_index().to_dict()}")

# ════════════════════════════════════════════════════════════════════
# SECTION 3: Subjects with 0 attack trials at T=0.1 in any block
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Subjects with 0 attack trials at T=0.1 in any block")
print("=" * 70)

t01_block = attack_block_full[attack_block_full['T_round'] == 0.1].copy()
zero_in_block = t01_block[t01_block['n_attack'] == 0]
subjs_with_zero = zero_in_block['subj'].unique()

print(f"\n  Total subjects with 0 attack trials at T=0.1 in ANY block: {len(subjs_with_zero)}")
print(f"  Breakdown by block:")
for b in [0, 1, 2]:
    n = int((t01_block[(t01_block['block'] == b) & (t01_block['n_attack'] == 0)].shape[0]))
    print(f"    Block {b}: {n} subjects have 0 attack trials at T=0.1")

# Show how many blocks each of these subjects is missing
n_zero_blocks = zero_in_block.groupby('subj').size()
print(f"\n  Among these {len(subjs_with_zero)} subjects:")
print(f"    Missing 1 block: {(n_zero_blocks == 1).sum()}")
print(f"    Missing 2 blocks: {(n_zero_blocks == 2).sum()}")
print(f"    Missing 3 blocks (all): {(n_zero_blocks == 3).sum()}")

# ════════════════════════════════════════════════════════════════════
# SECTION 4: How many trials AT EACH T level exist per block?
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Total trials per T per block (to understand design)")
print("=" * 70)

trials_per_cell = beh.groupby(['subj', 'T_round', 'block']).size().reset_index(name='n_trials')
for t in [0.1, 0.5, 0.9]:
    for b in [0, 1, 2]:
        subset = trials_per_cell[(trials_per_cell['T_round'] == t) &
                                  (trials_per_cell['block'] == b)]['n_trials']
        print(f"  T={t}, Block={b}: mean={subset.mean():.2f}, min={subset.min()}, max={subset.max()}")

print("\n  Expected: Each block has 27 trials. With 3 threat levels,")
print("  each T gets ~9 trials per block per subject.")

# ════════════════════════════════════════════════════════════════════
# SECTION 5: Attack trial counts for ENCOUNTER EPOCH specifically
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Encounter epoch availability in vigor_metrics.csv")
print("=" * 70)

# The encounter epoch from vigor_metrics - only exists for attack trials
vm_enc = vm[vm['epoch'] == 'reactive'].copy()  # reactive = encounter epoch in the pipeline
vm_enc['block'] = vm_enc['trial'] // 27

print(f"  Reactive-epoch rows in vigor_metrics: {len(vm_enc)}")
print(f"  Subjects: {vm_enc['subj'].nunique()}")
print(f"  Attack trials only: {vm_enc[vm_enc['is_attack']==1].shape[0]}")

# Check what epochs are available for attack vs non-attack
for epoch in vm['epoch'].unique():
    e_sub = vm[vm['epoch'] == epoch]
    n_attack = e_sub[e_sub['is_attack'] == 1].shape[0]
    n_noattack = e_sub[e_sub['is_attack'] == 0].shape[0]
    print(f"  {epoch}: {n_attack} attack, {n_noattack} non-attack rows")

# For encounter spike analysis, we need reactive epoch + is_attack
enc_attack = vm_enc[vm_enc['is_attack'] == 1].copy()
print(f"\n  Reactive epoch + attack trials: {len(enc_attack)}")

enc_by_t = enc_attack.groupby(['subj', 'T_round']).size().reset_index(name='n')
all_combos_vm = pd.MultiIndex.from_product(
    [vm['subj'].unique(), [0.1, 0.5, 0.9]], names=['subj', 'T_round'])
enc_by_t_full = (enc_by_t.set_index(['subj', 'T_round'])
                 .reindex(all_combos_vm, fill_value=0)
                 .reset_index())

print("\n  Attack trials with reactive vigor per threat level:")
for t in [0.1, 0.5, 0.9]:
    subset = enc_by_t_full[enc_by_t_full['T_round'] == t]['n']
    print(f"    T={t}: mean={subset.mean():.2f}, min={subset.min()}, max={subset.max()}, "
          f"n_zero={int((subset == 0).sum())}, n_le2={int((subset <= 2).sum())}")

# Per block
enc_by_t_block = enc_attack.groupby(['subj', 'T_round', 'block']).size().reset_index(name='n')
all_combos_vm_block = pd.MultiIndex.from_product(
    [vm['subj'].unique(), [0.1, 0.5, 0.9], [0, 1, 2]],
    names=['subj', 'T_round', 'block'])
enc_by_t_block_full = (enc_by_t_block.set_index(['subj', 'T_round', 'block'])
                       .reindex(all_combos_vm_block, fill_value=0)
                       .reset_index())

print("\n  Attack trials with reactive vigor per threat x block:")
for t in [0.1, 0.5, 0.9]:
    for b in [0, 1, 2]:
        subset = enc_by_t_block_full[(enc_by_t_block_full['T_round'] == t) &
                                      (enc_by_t_block_full['block'] == b)]['n']
        print(f"    T={t}, Block={b}: mean={subset.mean():.2f}, "
              f"n_zero={int((subset == 0).sum())}, n_le1={int((subset <= 1).sum())}")

# ════════════════════════════════════════════════════════════════════
# SECTION 6: NaN diagnosis — what percentage of cells are empty?
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: NaN diagnosis — empty cells that cause NaN in H2b.2 / H2e.2")
print("=" * 70)

# H2b.2: Encounter spike ~ threat (needs attack trials at each T, possibly per-subject means)
# If a subject has 0 attack trials at any T, their mean encounter vigor at that T is NaN
print("\n  For H2b.2 (encounter spike ~ threat):")
print("  Cells with 0 attack trials (subject x threat):")
n_empty = int((enc_by_t_full['n'] == 0).sum())
n_total = len(enc_by_t_full)
print(f"    {n_empty} / {n_total} cells are empty ({n_empty/n_total*100:.1f}%)")
for t in [0.1, 0.5, 0.9]:
    subset = enc_by_t_full[enc_by_t_full['T_round'] == t]
    n_empty_t = int((subset['n'] == 0).sum())
    print(f"    T={t}: {n_empty_t} subjects have 0 attack trials with encounter vigor")

print("\n  For H2e.2 (cross-block encounter spike stability):")
print("  Cells with 0 attack trials (subject x threat x block):")
n_empty_block = int((enc_by_t_block_full['n'] == 0).sum())
n_total_block = len(enc_by_t_block_full)
print(f"    {n_empty_block} / {n_total_block} cells are empty ({n_empty_block/n_total_block*100:.1f}%)")

# Specifically for T=0.1
t01_enc = enc_by_t_block_full[enc_by_t_block_full['T_round'] == 0.1]
n_empty_t01 = int((t01_enc['n'] == 0).sum())
n_total_t01 = len(t01_enc)
print(f"    At T=0.1: {n_empty_t01} / {n_total_t01} cells empty ({n_empty_t01/n_total_t01*100:.1f}%)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
