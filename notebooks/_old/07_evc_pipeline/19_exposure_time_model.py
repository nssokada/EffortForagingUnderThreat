#!/usr/bin/env python3
"""
19_exposure_time_model.py — Exposure-time survival model with frac_full as vigor variable
=========================================================================================

Step 1: Compute frac_full for all 81 trials per subject
Step 2: Estimate game speed parameters and λ from attack trial data
Step 3: Fit vigor model: EU(f) with exposure-time S(f, T, D)
Step 4: Compare to original model (mean press rate + sigmoid S)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ast
from scipy.stats import pearsonr, norm
from scipy.optimize import minimize_scalar
from scipy.special import expit
import statsmodels.api as sm
from pathlib import Path

DATA_DIR = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
PARAMS_FILE = Path("/workspace/results/stats/oc_evc_3param_v2_params.csv")
OUT_DIR = Path("/workspace/results/stats")

print("=" * 70)
print("19. EXPOSURE-TIME SURVIVAL MODEL")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: COMPUTE FRAC_FULL FOR ALL TRIALS
# ═════════════════════════════════════════���═════════════════════════════
print("\nStep 1: Computing frac_full for all trials...")

beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
beh['actual_req'] = np.where(beh['trialCookie_weight'] == 3.0, 0.9, 0.4)
beh['actual_dist'] = beh['startDistance'].map({5: 1, 7: 2, 9: 3})
beh['actual_R'] = np.where(beh['trialCookie_weight'] == 3.0, 5.0, 1.0)
beh['is_heavy'] = (beh['trialCookie_weight'] == 3.0).astype(int)
beh['survived'] = (beh['trialEndState'] == 'escaped').astype(int)
beh['T_round'] = beh['threat'].round(1)

frac_full_list = []
mean_rate_list = []
press_sd_list = []
n_samples_list = []

for idx, row in beh.iterrows():
    try:
        pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
        ipis = np.diff(pt)
        ipis = ipis[ipis > 0.01]
        if len(ipis) < 5:
            frac_full_list.append(np.nan)
            mean_rate_list.append(np.nan)
            press_sd_list.append(np.nan)
            n_samples_list.append(0)
            continue

        rates = (1.0 / ipis) / row['calibrationMax']
        req = row['actual_req']

        frac_full_list.append(np.mean(rates >= req))
        mean_rate_list.append(np.mean(rates))
        press_sd_list.append(np.std(rates))
        n_samples_list.append(len(rates))
    except Exception:
        frac_full_list.append(np.nan)
        mean_rate_list.append(np.nan)
        press_sd_list.append(np.nan)
        n_samples_list.append(0)

beh['frac_full'] = frac_full_list
beh['mean_rate'] = mean_rate_list
beh['press_sd'] = press_sd_list
beh['n_samples'] = n_samples_list

valid = beh.dropna(subset=['frac_full']).copy()
print(f"Valid trials: {len(valid)}/{len(beh)}")
print(f"  Heavy: frac_full mean={valid[valid['is_heavy']==1]['frac_full'].mean():.3f}")
print(f"  Light: frac_full mean={valid[valid['is_heavy']==0]['frac_full'].mean():.3f}")

# ══════════════════════════════════════════════════════════════════════���
# STEP 2: ESTIMATE GAME SPEED AND λ FROM ATTACK TRIAL DATA
# ═════════��═════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 2: Estimating exposure-time parameters")
print("=" * 70)

# We need to estimate:
# 1. v_full (full movement speed in game units/sec)
# 2. t_safe (grace period after encounter before strikes possible)
# 3. λ (hazard rate during exposure window)

# Estimate v_full from trial durations
# trialDuration should be available; combined with distance gives speed
# For trials where participant maintained full speed throughout

atk = valid[valid['isAttackTrial'] == 1].copy()
print(f"\nAttack trials: {len(atk)}")

# Try to get timing info
time_cols = [c for c in beh.columns if 'time' in c.lower() or 'duration' in c.lower()]
print(f"Timing columns: {time_cols}")

# Use available timing columns
atk['enc_time'] = pd.to_numeric(atk['encounterTime'], errors='coerce')
atk['esc_time'] = pd.to_numeric(atk['trialEscapeTime'], errors='coerce')
atk['cap_time'] = pd.to_numeric(atk['trialCaptureTime'], errors='coerce')
atk['start_time'] = pd.to_numeric(atk['playerEffortStartTime'], errors='coerce')
atk['end_time'] = np.where(atk['survived'] == 1, atk['esc_time'], atk['cap_time'])
atk['end_time'] = pd.to_numeric(atk['end_time'], errors='coerce')

# Post-encounter duration = end_time - encounterTime
atk['post_enc_dur'] = atk['end_time'] - atk['enc_time']
# Total effort duration
atk['effort_dur'] = atk['end_time'] - atk['start_time']

valid_timing = atk.dropna(subset=['post_enc_dur', 'effort_dur'])
valid_timing = valid_timing[valid_timing['post_enc_dur'] > 0]
print(f"\nTrials with valid timing: {len(valid_timing)}")
print(f"Effort duration: mean={valid_timing['effort_dur'].mean():.2f}s, med={valid_timing['effort_dur'].median():.2f}s")
print(f"Encounter time: mean={valid_timing['enc_time'].mean():.2f}s, med={valid_timing['enc_time'].median():.2f}s")
print(f"Post-encounter: mean={valid_timing['post_enc_dur'].mean():.2f}s, med={valid_timing['post_enc_dur'].median():.2f}s")

for d in [1, 2, 3]:
    sub = valid_timing[valid_timing['actual_dist'] == d]
    d_game = {1: 5, 2: 7, 3: 9}[d]
    print(f"\n  D={d} (D_game={d_game}):")
    print(f"    Effort dur: mean={sub['effort_dur'].mean():.2f}s, med={sub['effort_dur'].median():.2f}s")
    print(f"    Post-enc: mean={sub['post_enc_dur'].mean():.2f}s, med={sub['post_enc_dur'].median():.2f}s")

    esc_full = sub[(sub['survived'] == 1) & (sub['frac_full'] > 0.95)]
    if len(esc_full) > 10:
        med_post = esc_full['post_enc_dur'].median()
        v_est = (d_game / 2) / med_post if med_post > 0 else np.nan
        print(f"    Escaped at full speed: N={len(esc_full)}, median_post_enc={med_post:.2f}s")
        print(f"    Estimated v_full ≈ {v_est:.2f} game units/sec")

# ── Estimate λ from escape rates by frac_full and distance ──
print("\n--- Estimating λ (hazard rate) ---")

# For each attack trial, approximate exposure time
# exposure ≈ remaining_distance / effective_speed - t_safe
# effective_speed = v_full × (0.5 + 0.5 × f)
# remaining_distance ≈ D_game / 2 (encounter at midpoint)

# First pass: estimate v_full from the fastest escaped trials
# Using the median post-encounter duration at each distance
v_full_estimates = []
for d in [1, 2, 3]:
    d_game = {1: 5, 2: 7, 3: 9}[d]
    esc_full = valid_timing[(valid_timing['actual_dist'] == d) & (valid_timing['survived'] == 1) & (valid_timing['frac_full'] > 0.95)]
    if len(esc_full) > 10:
        fast_time = esc_full['post_enc_dur'].quantile(0.25)
        if fast_time > 0:
            v_full_estimates.append(d_game / 2 / fast_time)

if v_full_estimates:
    v_full = np.median(v_full_estimates)
else:
    v_full = 3.0  # fallback estimate

print(f"Estimated v_full: {v_full:.2f} game units/sec")

# Now compute approximate exposure for each attack trial
t_safe = 1.5  # seconds grace period (from plan)
D_game_map = {1: 5, 2: 7, 3: 9}

def compute_exposure(row, v_full, t_safe):
    f = row['frac_full']
    d = row['actual_dist']
    d_game = D_game_map.get(d, 5)
    remaining = d_game / 2  # approximate: encounter at midpoint
    eff_speed = v_full * (0.5 + 0.5 * f)
    travel_time = remaining / eff_speed
    return max(0, travel_time - t_safe)

atk['exposure'] = atk.apply(lambda r: compute_exposure(r, v_full, t_safe), axis=1)

print(f"\nExposure times:")
for d in [1, 2, 3]:
    sub = atk[atk['actual_dist'] == d]
    print(f"  D={d}: mean={sub['exposure'].mean():.2f}s, median={sub['exposure'].median():.2f}s")

# Fit λ via MLE: P(escape) = exp(-λ × exposure)
# Log-likelihood: survived × (-λ × exposure) + (1-survived) × log(1 - exp(-λ × exposure))
def neg_loglik(lam, exposure, survived):
    if lam <= 0:
        return 1e10
    p_esc = np.exp(-lam * exposure)
    p_esc = np.clip(p_esc, 1e-10, 1 - 1e-10)
    ll = survived * np.log(p_esc) + (1 - survived) * np.log(1 - p_esc)
    return -np.sum(ll)

atk_valid = atk.dropna(subset=['exposure', 'survived'])
atk_valid = atk_valid[atk_valid['exposure'] > 0]  # only trials with positive exposure

result = minimize_scalar(neg_loglik, bounds=(0.01, 5.0), method='bounded',
                        args=(atk_valid['exposure'].values, atk_valid['survived'].values))
lam_hat = result.x
print(f"\nFitted λ = {lam_hat:.3f} /sec")

# Verify: predicted vs observed escape rates
for d in [1, 2, 3]:
    sub = atk_valid[atk_valid['actual_dist'] == d]
    pred_esc = np.exp(-lam_hat * sub['exposure']).mean()
    obs_esc = sub['survived'].mean()
    print(f"  D={d}: predicted esc={pred_esc:.3f}, observed esc={obs_esc:.3f}")

# By speed tier
for tier_name, lo, hi in [('<50%', 0, 0.5), ('50-80%', 0.5, 0.8),
                           ('80-95%', 0.8, 0.95), ('>95%', 0.95, 1.01)]:
    sub = atk_valid[(atk_valid['frac_full'] >= lo) & (atk_valid['frac_full'] < hi)]
    if len(sub) > 10:
        pred = np.exp(-lam_hat * sub['exposure']).mean()
        obs = sub['survived'].mean()
        print(f"  {tier_name}: predicted={pred:.3f}, observed={obs:.3f}, N={len(sub)}")

# ══��═══════════════════════���════════════════════════════════════════════
# STEP 3: FIT VIGOR MODEL WITH EXPOSURE-TIME S
# ═══════════════════════════════════════════════════════════���═══════════
print("\n" + "=" * 70)
print("Step 3: Fitting exposure-time vigor model")
print("=" * 70)

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from jax import random

jax.config.update('jax_enable_x64', True)

# Prepare vigor data (all 81 trials, using frac_full)
vigor_df = valid.copy()

# Cookie-type centering of frac_full
choice_vigor = vigor_df[vigor_df['type'] == 1]
heavy_ff_mean = choice_vigor[choice_vigor['is_heavy'] == 1]['frac_full'].mean()
light_ff_mean = choice_vigor[choice_vigor['is_heavy'] == 0]['frac_full'].mean()
vigor_df['frac_full_cc'] = vigor_df['frac_full'] - np.where(
    vigor_df['is_heavy'] == 1, heavy_ff_mean, light_ff_mean)

print(f"Cookie-type centering: heavy_mean={heavy_ff_mean:.3f}, light_mean={light_ff_mean:.3f}")

# Subject indexing
subjects = sorted(vigor_df['subj'].unique())
subj_to_idx = {s: i for i, s in enumerate(subjects)}
N_S = len(subjects)

# Choice data (same as 3-param v2)
choice_df = beh[beh['type'] == 1].dropna(subset=['frac_full']).copy()
choice_subjects = sorted(set(choice_df['subj'].unique()) & set(subjects))
N_S = len(choice_subjects)
subj_to_idx = {s: i for i, s in enumerate(choice_subjects)}

# Filter to shared subjects
choice_df = choice_df[choice_df['subj'].isin(choice_subjects)]
vigor_df = vigor_df[vigor_df['subj'].isin(choice_subjects)]

# Arrays
ch_subj = jnp.array([subj_to_idx[s] for s in choice_df['subj']])
ch_T = jnp.array(choice_df['threat'].values)
ch_dist_H = jnp.array(choice_df['distance_H'].values, dtype=jnp.float64)
ch_choice = jnp.array(choice_df['choice'].values)

vig_subj = jnp.array([subj_to_idx[s] for s in vigor_df['subj']])
vig_T = jnp.array(vigor_df['threat'].values)
vig_R = jnp.array(vigor_df['actual_R'].values)
vig_req = jnp.array(vigor_df['actual_req'].values)
vig_dist = jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64)
vig_frac_full = jnp.array(vigor_df['frac_full_cc'].values)
vig_ff_offset = jnp.array(np.where(
    vigor_df['is_heavy'].values == 1, heavy_ff_mean, light_ff_mean))

N_choice = len(choice_df)
N_vigor = len(vigor_df)
print(f"N_subjects={N_S}, N_choice={N_choice}, N_vigor={N_vigor}")

# Fixed parameters from empirical estimation
V_FULL = float(v_full)
T_SAFE = float(t_safe)
D_GAME = jnp.array([5.0, 7.0, 9.0])  # indexed by dist-1

def make_exposure_model(N_S, N_choice, N_vigor):
    """3-param model with exposure-time survival in vigor."""

    def model(ch_subj, ch_T, ch_dist_H, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist,
              vig_frac_full, vig_ff_offset):

        # ── Hierarchical priors ──
        mu_k = numpyro.sample('mu_k', dist.Normal(0.0, 1.0))
        mu_beta = numpyro.sample('mu_beta', dist.Normal(1.0, 1.0))
        mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
        sigma_k = numpyro.sample('sigma_k', dist.HalfNormal(0.5))
        sigma_beta = numpyro.sample('sigma_beta', dist.HalfNormal(0.5))
        sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))

        # ── Population params ──
        tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)

        # Lambda (hazard rate) — estimate or fix
        lam_raw = numpyro.sample('lam_raw', dist.Normal(0.0, 1.0))
        lam = numpyro.deterministic('lam', jnp.clip(jnp.exp(lam_raw), 0.01, 5.0))

        # Sigma for within-trial press rate variability (determines f from buffer)
        sigma_motor_raw = numpyro.sample('sigma_motor_raw', dist.Normal(-1.0, 0.5))
        sigma_motor = jnp.clip(jnp.exp(sigma_motor_raw), 0.01, 2.0)

        # Vigor effort cost (population)
        ce_vigor_raw = numpyro.sample('ce_vigor_raw', dist.Normal(0.0, 1.0))
        ce_vigor = numpyro.deterministic('ce_vigor', jnp.clip(jnp.exp(ce_vigor_raw), 0.001, 50.0))

        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.3))

        # ── Subject-level ──
        with numpyro.plate('subjects', N_S):
            k_raw = numpyro.sample('k_raw', dist.Normal(0.0, 1.0))
            beta_raw = numpyro.sample('beta_raw', dist.Normal(0.0, 1.0))
            cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))

        k = jnp.exp(mu_k + sigma_k * k_raw)
        beta = jnp.exp(mu_beta + sigma_beta * beta_raw)
        c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
        numpyro.deterministic('k', k)
        numpyro.deterministic('beta', beta)
        numpyro.deterministic('c_death', c_death)

        # ══════════════════════════════════════════════════════════════
        # CHOICE: same as 3-param v2 (k + β + T)
        # ═════════════════════════���════════════════════════════════════
        k_ch = k[ch_subj]
        beta_ch = beta[ch_subj]
        effort_cost = 0.81 * ch_dist_H - 0.16
        delta_eu = 4.0 - k_ch * effort_cost - beta_ch * ch_T
        logit = jnp.clip(delta_eu / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)

        with numpyro.plate('choice_trials', N_choice):
            numpyro.sample('obs_choice',
                dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                obs=ch_choice)

        # ���════════���════════════════════════════════════════════════════
        # VIGOR: Exposure-time survival with frac_full
        # ═══════════��══════════════════════════════════════════════════
        cd_v = c_death[vig_subj]

        # Grid of frac_full values
        f_grid = jnp.linspace(0.01, 0.99, 30)
        f_g = f_grid[None, :]  # (1, 30)

        # Exposure time: remaining_dist / effective_speed - t_safe
        # D_game values: D=1→5, D=2→7, D=3→9
        d_game_v = jnp.where(vig_dist == 1, 5.0,
                   jnp.where(vig_dist == 2, 7.0, 9.0))
        remaining = d_game_v / 2.0  # encounter at midpoint

        eff_speed = V_FULL * (0.5 + 0.5 * f_g)  # (1, 30)
        travel_time = remaining[:, None] / eff_speed  # (N, 30)
        exposure = jnp.clip(travel_time - T_SAFE, 0.0, 20.0)  # (N, 30)

        # Survival: S(f, T, D) = (1-T) + T × exp(-λ × exposure)
        p_escape = jnp.exp(-lam * exposure)  # (N, 30)
        S_f = (1.0 - vig_T[:, None]) + vig_T[:, None] * p_escape  # (N, 30)

        # Effort cost of maintaining fraction f
        # f = Φ((u - req) / σ_motor) → u - req = σ_motor × Φ⁻¹(f)
        # Effort = ce_vigor × (u - req)² × D = ce_vigor × σ² × [Φ⁻¹(f)]² × D
        # Use probit approximation: Φ⁻¹(f) ≈ logit(f) × √(π/8) ... or just use log-odds
        # Actually jax has no probit. Use approximation via logit:
        # Φ⁻¹(f) ≈ (π/√3) × logit(f) ... not great. Let's use a different approach.
        #
        # Alternative: effort cost as a function of f directly
        # The cost of maintaining f increases as f → 1 (need to press MUCH harder to never dip)
        # Simple convex cost: cost(f) = ce_vigor × (-log(1-f))^2 × D
        # This goes to 0 at f=0 and infinity at f=1
        # Or simpler: cost(f) = ce_vigor × (f / (1-f))^2 × D (also convex, goes to inf at f=1)
        #
        # Actually, the most principled: if press rate noise is Gaussian with SD σ_motor,
        # to achieve fraction f, mean must be req + σ × Φ⁻¹(f).
        # Cost = ce × σ�� × Φ⁻¹(f)² × D
        #
        # In JAX, we can compute Φ⁻¹ via the normal quantile.
        # jax.scipy.stats.norm.ppf exists!

        probit_f = jax.scipy.stats.norm.ppf(jnp.clip(f_g, 0.001, 0.999))  # (1, 30)
        effort_f = ce_vigor * sigma_motor**2 * probit_f**2 * vig_dist[:, None]  # (N, 30)

        # EU(f) = S(f) × R - (1-S(f)) × cd × (R+C) - effort(f)
        eu_grid = (S_f * vig_R[:, None]
                   - (1.0 - S_f) * cd_v[:, None] * (vig_R[:, None] + 5.0)
                   - effort_f)

        # Softmax to get optimal f*
        weights = jax.nn.softmax(eu_grid * 5.0, axis=1)
        f_star = jnp.sum(weights * f_g, axis=1)

        # Predicted frac_full (cookie-type centered)
        f_pred = f_star - vig_ff_offset
        numpyro.deterministic('f_pred', f_pred)

        with numpyro.plate('vigor_trials', N_vigor):
            numpyro.sample('obs_vigor',
                dist.Normal(f_pred, sigma_v),
                obs=vig_frac_full)

    return model

# ── Fit ──
print("\nFitting exposure-time model...")
model = make_exposure_model(N_S, N_choice, N_vigor)

kwargs = {
    'ch_subj': ch_subj, 'ch_T': ch_T, 'ch_dist_H': ch_dist_H, 'ch_choice': ch_choice,
    'vig_subj': vig_subj, 'vig_T': vig_T, 'vig_R': vig_R, 'vig_req': vig_req,
    'vig_dist': vig_dist, 'vig_frac_full': vig_frac_full, 'vig_ff_offset': vig_ff_offset,
}

guide = AutoNormal(model)
optimizer = numpyro.optim.ClippedAdam(step_size=0.001, clip_norm=10.0)
svi = SVI(model, guide, optimizer, Trace_ELBO())
state = svi.init(random.PRNGKey(42), **kwargs)
update_fn = jax.jit(svi.update)

n_steps = 40000
for i in range(n_steps):
    state, loss = update_fn(state, **kwargs)
    if (i + 1) % 5000 == 0:
        print(f"  Step {i+1}: loss={loss:.1f}")

params_fit = svi.get_params(state)

# ── Evaluate ──
print("\nExtracting parameters...")
pred = Predictive(model, guide=guide, params=params_fit, num_samples=300,
                  return_sites=['k', 'beta', 'c_death', 'lam', 'ce_vigor',
                                'f_pred', 'tau_raw', 'sigma_v', 'sigma_motor_raw'])
samples = pred(random.PRNGKey(44), **kwargs)

k_vals = np.array(samples['k']).mean(0)
beta_vals = np.array(samples['beta']).mean(0)
cd_vals = np.array(samples['c_death']).mean(0)
lam_val = float(np.array(samples['lam']).mean())
ce_vigor_val = float(np.array(samples['ce_vigor']).mean())
tau_val = float(np.exp(np.array(samples['tau_raw']).mean()))
sigma_v_val = float(np.array(samples['sigma_v']).mean())
sigma_motor_val = float(np.exp(np.array(samples['sigma_motor_raw']).mean()))
f_pred = np.array(samples['f_pred']).mean(0)

# Vigor fit (frac_full)
f_obs = np.array(vig_frac_full)
r_vigor, _ = pearsonr(f_pred, f_obs)

# Per-subject vigor
vig_subj_np = np.array(vig_subj)
vig_df_eval = pd.DataFrame({'subj': vig_subj_np, 'obs': f_obs, 'pred': f_pred})
subj_vig = vig_df_eval.groupby('subj').agg(obs_m=('obs', 'mean'), pred_m=('pred', 'mean')).reset_index()
r_vigor_subj, _ = pearsonr(subj_vig['obs_m'], subj_vig['pred_m'])

# Choice fit
ch_subj_np = np.array(ch_subj)
ch_T_np = np.array(ch_T)
ch_dist_np = np.array(ch_dist_H)
ch_choice_np = np.array(ch_choice)

effort_cost = 0.81 * ch_dist_np - 0.16
delta_eu = 4.0 - k_vals[ch_subj_np] * effort_cost - beta_vals[ch_subj_np] * ch_T_np
p_H = expit(np.clip(delta_eu / tau_val, -20, 20))
choice_acc = ((p_H >= 0.5).astype(int) == ch_choice_np).mean()

ch_df = pd.DataFrame({'subj': ch_subj_np, 'choice': ch_choice_np, 'p_H': p_H})
subj_ch = ch_df.groupby('subj').agg(obs=('choice', 'mean'), pred=('p_H', 'mean')).reset_index()
r_choice_subj, _ = pearsonr(subj_ch['obs'], subj_ch['pred'])

print(f"\n{'=' * 60}")
print(f"EXPOSURE-TIME MODEL RESULTS")
print(f"{'=' * 60}")
print(f"Choice accuracy: {choice_acc:.3f}")
print(f"Per-subject choice r: {r_choice_subj:.3f} (r²={r_choice_subj**2:.3f})")
print(f"Trial-level vigor (frac_full) r: {r_vigor:.3f} (r²={r_vigor**2:.3f})")
print(f"Per-subject vigor r: {r_vigor_subj:.3f} (r²={r_vigor_subj**2:.3f})")
print(f"\nPopulation params:")
print(f"  λ (hazard rate) = {lam_val:.3f} /sec")
print(f"  σ_motor = {sigma_motor_val:.3f}")
print(f"  ce_vigor = {ce_vigor_val:.4f}")
print(f"  τ = {tau_val:.3f}")
print(f"  σ_v = {sigma_v_val:.3f}")
print(f"\nPer-subject params:")
print(f"  k: median={np.median(k_vals):.3f}, range=[{k_vals.min():.3f}, {k_vals.max():.3f}]")
print(f"  β: median={np.median(beta_vals):.3f}, range=[{beta_vals.min():.3f}, {beta_vals.max():.3f}]")
print(f"  cd: median={np.median(cd_vals):.3f}, range=[{cd_vals.min():.3f}, {cd_vals.max():.3f}]")

lk, lb, lcd = np.log(k_vals), np.log(beta_vals), np.log(cd_vals)
r_kb, p_kb = pearsonr(lk, lb)
r_kcd, _ = pearsonr(lk, lcd)
r_bcd, _ = pearsonr(lb, lcd)
print(f"\n  k × β: r={r_kb:.3f}")
print(f"  k × cd: r={r_kcd:.3f}")
print(f"  β × cd: r={r_bcd:.3f}")

# Compare to original model
print(f"\n{'=' * 60}")
print(f"COMPARISON TO ORIGINAL (mean press rate + sigmoid S)")
print(f"{'=' * 60}")
print(f"                  Original    Exposure-time")
print(f"Choice r²:         0.981       {r_choice_subj**2:.3f}")
print(f"Vigor r² (trial):  0.424       {r_vigor**2:.3f}")
print(f"Vigor r² (subj):   0.669       {r_vigor_subj**2:.3f}")

# Save
param_df = pd.DataFrame({
    'subj': choice_subjects,
    'k': k_vals, 'beta': beta_vals, 'c_death': cd_vals,
})
param_df.to_csv(OUT_DIR / 'exposure_time_params.csv', index=False)

pop_df = pd.DataFrame([{
    'lam': lam_val, 'sigma_motor': sigma_motor_val,
    'ce_vigor': ce_vigor_val, 'tau': tau_val, 'sigma_v': sigma_v_val,
    'v_full': V_FULL, 't_safe': T_SAFE,
}])
pop_df.to_csv(OUT_DIR / 'exposure_time_population.csv', index=False)

print(f"\nSaved params and population to {OUT_DIR}")
print("\nDone.")
