"""
Comprehensive parameter recovery for V8c (M4) joint optimal control model.

Improvements over `param_recovery_v8c.py`:
  1. Uses ACTUAL fitted μ_log, σ_log from `mcmc_m4_params.csv` rather than
     hand-picked σ=0.5 (which was ~3x narrower than the empirical κ spread).
  2. N = 100 subjects × 5 datasets = 500 simulated subjects.
  3. Per-subject trial counts match empirical (~45 choice + ~81 vigor events).
  4. Includes empirical r(log ω, log κ) ≈ 0.37 correlation when generating.
  5. Reports a full metric suite:
        - Pearson r, Spearman ρ on log-transformed posterior means
        - RMSE / MAE on log scale
        - 80% and 95% posterior credible interval coverage
        - Rank-correlation at extremes (top/bottom 10%, 20%)
        - Cross-talk (true κ ↔ recovered ω, and vice versa)
        - Population-level μ, σ recovery
  6. Saves per-subject true/recovered/CI csv for downstream diagnostics.

Outputs:
    results/stats/joint_optimal/param_recovery_v8c_full.csv          — per-subject
    results/stats/joint_optimal/param_recovery_v8c_full_summary.csv  — metrics
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from jax import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit
from pathlib import Path

OUT_DIR = Path("results/stats/joint_optimal")
C_PENALTY = 5.0

# Population parameters from V8c fit (same as the original recovery script)
POP_PARAMS = {
    'gamma': 0.86,
    'hazard': 0.832,
    'tau': 2.01,
    'sigma_v': 0.12,
    'b_cookie': -0.03,
}

# Trial design (matches real subjects' free-choice trial count)
THREATS = [0.1, 0.5, 0.9]
DIST_H = [1, 2, 3]
N_TRIALS_PER_CELL = 5  # 9 cells × 5 = 45 choice trials/subject
N_VIGOR_PER_SUBJ = 81  # matches actual event-count per subject


def load_empirical_distribution(sample='exploratory'):
    """Read fitted M4 per-subject params; return empirical (μ, σ) in log-space."""
    p = Path(f'results/stats/joint_optimal/{sample}/mcmc_m4_params.csv')
    df = pd.read_csv(p)
    log_om = np.log(df['omega'].dropna().clip(lower=1e-9))
    log_kp = np.log(df['kappa'].dropna().clip(lower=1e-9))
    valid = df[['omega', 'kappa']].dropna()
    log_pair = np.log(valid.clip(lower=1e-9))
    r_logok = log_pair.corr().iloc[0, 1]
    print(f"[load_empirical_distribution {sample}]")
    print(f"  N = {len(df)}")
    print(f"  ω: μ_log = {log_om.mean():+.4f},  σ_log = {log_om.std():.4f}")
    print(f"  κ: μ_log = {log_kp.mean():+.4f},  σ_log = {log_kp.std():.4f}")
    print(f"  r(log ω, log κ) = {r_logok:.4f}")
    return {
        'mu_log_om':    float(log_om.mean()),
        'sig_log_om':   float(log_om.std()),
        'mu_log_kap':   float(log_kp.mean()),
        'sig_log_kap':  float(log_kp.std()),
        'r_logok':      float(r_logok),
    }


# ============================================================
# Generative model
# ============================================================

def exp_survival(u, T, D, gamma, hazard):
    T_w = T ** gamma
    return np.exp(-hazard * T_w * D / np.clip(u, 0.1, None))


def compute_W(u, omega, kappa, T, D, R, req, gamma, hazard):
    S = exp_survival(u, T, D, gamma, hazard)
    return S * R - (1 - S) * omega * (R + C_PENALTY) - kappa * (u - req)**2 * D


def find_ustar(omega, kappa, T, D, R, req, gamma, hazard):
    u_grid = np.linspace(0.1, 1.5, 100)
    W_vals = np.array([compute_W(u, omega, kappa, T, D, R, req, gamma, hazard)
                       for u in u_grid])
    best_idx = np.argmax(W_vals)
    return u_grid[best_idx], W_vals[best_idx]


def sample_correlated_params(N, emp, rng):
    """Sample (log ω, log κ) from a bivariate normal matching empirical μ, σ, r."""
    mu = np.array([emp['mu_log_om'], emp['mu_log_kap']])
    sig_o = emp['sig_log_om']
    sig_k = emp['sig_log_kap']
    r = emp['r_logok']
    cov = np.array([
        [sig_o**2, r * sig_o * sig_k],
        [r * sig_o * sig_k, sig_k**2],
    ])
    log_pair = rng.multivariate_normal(mu, cov, size=N)
    omega = np.exp(log_pair[:, 0])
    kappa = np.exp(log_pair[:, 1])
    return omega, kappa


def simulate_data(true_omega, true_kappa, true_baseline, rng):
    """Simulate choice + vigor data for a set of subjects."""
    N_S = len(true_omega)
    gamma = POP_PARAMS['gamma']; hazard = POP_PARAMS['hazard']
    tau = POP_PARAMS['tau']; sigma_v = POP_PARAMS['sigma_v']
    b_cookie = POP_PARAMS['b_cookie']

    choice_records = []
    vigor_records = []

    for s in range(N_S):
        omega = true_omega[s]; kappa = true_kappa[s]; base = true_baseline[s]

        # Choice trials: 9 (T,D) cells × N_TRIALS_PER_CELL each
        for T in THREATS:
            for D_H in DIST_H:
                for trial in range(N_TRIALS_PER_CELL):
                    u_H, V_H = find_ustar(omega, kappa, T, D_H, 5.0, 0.9, gamma, hazard)
                    u_L, V_L = find_ustar(omega, kappa, T, 1.0, 1.0, 0.4, gamma, hazard)
                    p_heavy = expit((V_H - V_L) / tau)
                    chose_heavy = rng.random() < p_heavy

                    choice_records.append({
                        'subj': s, 'threat': T, 'distance_H': D_H,
                        'choice': int(chose_heavy), 'p_heavy': p_heavy,
                    })

                    if chose_heavy:
                        u_star, R, req, D, is_h = u_H, 5.0, 0.9, D_H, 1
                    else:
                        u_star, R, req, D, is_h = u_L, 1.0, 0.4, 1.0, 0
                    rate = u_star + base + b_cookie * is_h + rng.standard_normal() * sigma_v
                    vigor_records.append({
                        'subj': s, 'threat': T, 'actual_dist': D,
                        'actual_R': R, 'actual_req': req,
                        'median_rate': rate, 'is_heavy': is_h,
                    })

        # Extra vigor events to reach N_VIGOR_PER_SUBJ (probes etc.)
        n_extra = N_VIGOR_PER_SUBJ - len(THREATS) * len(DIST_H) * N_TRIALS_PER_CELL
        for _ in range(max(0, n_extra)):
            T = THREATS[rng.integers(3)]
            is_h = int(rng.random() < 0.5)
            R, req, D = (5.0, 0.9, float(DIST_H[rng.integers(3)])) if is_h else (1.0, 0.4, 1.0)
            u_star, _ = find_ustar(omega, kappa, T, D, R, req, gamma, hazard)
            rate = u_star + base + b_cookie * is_h + rng.standard_normal() * sigma_v
            vigor_records.append({
                'subj': s, 'threat': T, 'actual_dist': D,
                'actual_R': R, 'actual_req': req,
                'median_rate': rate, 'is_heavy': is_h,
            })

    choice_df = pd.DataFrame(choice_records)
    vigor_df = pd.DataFrame(vigor_records)
    return choice_df, vigor_df


def prepare_sim_data(choice_df, vigor_df):
    N_S = choice_df['subj'].nunique()
    return {
        'ch_subj': jnp.array(choice_df['subj'].values),
        'ch_T': jnp.array(choice_df['threat'].values),
        'ch_D_H': jnp.array(choice_df['distance_H'].values, dtype=jnp.float64),
        'ch_D_L': jnp.ones(len(choice_df)),
        'ch_choice': jnp.array(choice_df['choice'].values),
        'vig_subj': jnp.array(vigor_df['subj'].values),
        'vig_T': jnp.array(vigor_df['threat'].values),
        'vig_R': jnp.array(vigor_df['actual_R'].values),
        'vig_req': jnp.array(vigor_df['actual_req'].values),
        'vig_dist': jnp.array(vigor_df['actual_dist'].values, dtype=jnp.float64),
        'vig_rate': jnp.array(vigor_df['median_rate'].values),
        'vig_cookie': jnp.array(vigor_df['is_heavy'].values, dtype=jnp.float64),
        'N_S': N_S, 'N_choice': len(choice_df), 'N_vigor': len(vigor_df),
    }


# ============================================================
# Inference model (V8c) — identical to original
# ============================================================

def exp_survival_jax(u, T, D, gamma, hazard):
    T_w = jnp.power(T, gamma)
    return jnp.exp(-hazard * T_w * D / jnp.clip(u, 0.1, None))


def vigor_eu_exp(omega, kappa, T, D, R, req, gamma, hazard, u_grid):
    u_g = u_grid[None, :]
    S = exp_survival_jax(u_g, T[:, None], D[:, None], gamma, hazard)
    W = (S * R[:, None]
         - (1.0 - S) * omega[:, None] * (R[:, None] + C_PENALTY)
         - kappa[:, None] * (u_g - req[:, None]) ** 2 * D[:, None])
    weights = jax.nn.softmax(W * 20.0, axis=1)
    u_star = jnp.sum(weights * u_g, axis=1)
    V_star = jnp.sum(weights * W, axis=1)
    return u_star, V_star


def make_v8c(N_S, N_ch, N_vig):
    def model(ch_subj, ch_T, ch_D_H, ch_D_L, ch_choice,
              vig_subj, vig_T, vig_R, vig_req, vig_dist, vig_rate, vig_cookie):
        gamma_raw = numpyro.sample('gamma_raw', dist.Normal(0.0, 0.5))
        gamma = numpyro.deterministic('gamma', jnp.clip(jnp.exp(gamma_raw), 0.1, 3.0))
        hz_raw = numpyro.sample('hazard_raw', dist.Normal(-1.0, 1.0))
        hazard = numpyro.deterministic('hazard', jnp.exp(hz_raw))
        tau_raw = numpyro.sample('tau_raw', dist.Normal(0.0, 1.0))
        tau = jnp.clip(jnp.exp(tau_raw), 0.01, 50.0)
        sigma_v = numpyro.sample('sigma_v', dist.HalfNormal(0.5))
        b_cookie = numpyro.sample('b_cookie', dist.Normal(0.0, 0.5))

        mu_om = numpyro.sample('mu_om', dist.Normal(1.0, 1.0))
        sigma_om = numpyro.sample('sigma_om', dist.HalfNormal(1.0))
        mu_kap = numpyro.sample('mu_kap', dist.Normal(-2.0, 1.0))
        sigma_kap = numpyro.sample('sigma_kap', dist.HalfNormal(1.5))  # widened to allow σ ≈ 1.4
        mu_base = numpyro.sample('mu_base', dist.Normal(0.0, 0.3))
        sigma_base = numpyro.sample('sigma_base', dist.HalfNormal(0.2))

        with numpyro.plate('subjects', N_S):
            om_raw = numpyro.sample('om_raw', dist.Normal(0.0, 1.0))
            kap_raw = numpyro.sample('kap_raw', dist.Normal(0.0, 1.0))
            base_raw = numpyro.sample('base_raw', dist.Normal(0.0, 1.0))
        omega = jnp.exp(mu_om + sigma_om * om_raw)
        kappa = jnp.exp(mu_kap + sigma_kap * kap_raw)
        baseline = mu_base + sigma_base * base_raw
        numpyro.deterministic('omega', omega)
        numpyro.deterministic('kappa', kappa)
        numpyro.deterministic('baseline', baseline)
        numpyro.deterministic('log_omega', jnp.log(omega))
        numpyro.deterministic('log_kappa', jnp.log(kappa))

        u_grid = jnp.linspace(0.1, 1.5, 30)

        _, V_H = vigor_eu_exp(omega[ch_subj], kappa[ch_subj], ch_T, ch_D_H,
                              jnp.full(N_ch, 5.0), jnp.full(N_ch, 0.9),
                              gamma, hazard, u_grid)
        _, V_L = vigor_eu_exp(omega[ch_subj], kappa[ch_subj], ch_T, ch_D_L,
                              jnp.full(N_ch, 1.0), jnp.full(N_ch, 0.4),
                              gamma, hazard, u_grid)
        logit = jnp.clip((V_H - V_L) / tau, -20, 20)
        p_H = jax.nn.sigmoid(logit)
        with numpyro.plate('choice', N_ch):
            numpyro.sample('obs_ch', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1-1e-6)),
                           obs=ch_choice)

        u_star, _ = vigor_eu_exp(omega[vig_subj], kappa[vig_subj], vig_T, vig_dist,
                                 vig_R, vig_req, gamma, hazard, u_grid)
        rate_pred = u_star + baseline[vig_subj] + b_cookie * vig_cookie
        with numpyro.plate('vigor', N_vig):
            numpyro.sample('obs_vig', dist.Normal(rate_pred, sigma_v), obs=vig_rate)
    return model


KWARGS_KEYS = ['ch_subj', 'ch_T', 'ch_D_H', 'ch_D_L', 'ch_choice',
               'vig_subj', 'vig_T', 'vig_R', 'vig_req', 'vig_dist',
               'vig_rate', 'vig_cookie']


# ============================================================
# Recovery on one dataset
# ============================================================

def fit_and_recover(true_omega, true_kappa, true_baseline, dataset_id, seed=0):
    """Simulate, fit, and return per-subject true/recovered (with CIs)."""
    N_S = len(true_omega)
    print(f"\n  [dataset {dataset_id}] Simulating N={N_S} subjects ...")
    rng = np.random.default_rng(seed)
    choice_df, vigor_df = simulate_data(true_omega, true_kappa, true_baseline, rng)
    data = prepare_sim_data(choice_df, vigor_df)
    print(f"    {data['N_choice']} choice trials, {data['N_vigor']} vigor trials")
    print(f"    Choice base rate (heavy): {choice_df['choice'].mean():.3f}")

    # Fit via SVI
    kwargs = {k: data[k] for k in KWARGS_KEYS}
    model_fn = make_v8c(N_S, data['N_choice'], data['N_vigor'])
    guide = AutoNormal(model_fn)
    opt = numpyro.optim.ClippedAdam(step_size=0.001, clip_norm=10.0)
    svi = SVI(model_fn, guide, opt, Trace_ELBO())
    state = svi.init(random.PRNGKey(seed + 100), **kwargs)
    update = jax.jit(svi.update)

    best_loss, best_params = float('inf'), None
    n_steps = 30000
    for i in range(n_steps):
        state, loss = update(state, **kwargs)
        l = float(loss)
        if l < best_loss and not np.isnan(l):
            best_loss = l; best_params = svi.get_params(state)
        if (i + 1) % 10000 == 0:
            print(f"    step {i+1}: loss={l:.1f} (best={best_loss:.1f})")

    # Draw posterior samples (for credible intervals + coverage)
    pred = Predictive(model_fn, guide=guide, params=best_params,
                      num_samples=1000,
                      return_sites=['omega', 'kappa', 'log_omega', 'log_kappa',
                                    'gamma', 'hazard', 'mu_om', 'sigma_om',
                                    'mu_kap', 'sigma_kap'])
    samples = pred(random.PRNGKey(seed + 200), **kwargs)

    log_om_samp = np.array(samples['log_omega'])  # (1000, N_S)
    log_kp_samp = np.array(samples['log_kappa'])

    rec_log_om_mean = log_om_samp.mean(0)
    rec_log_kp_mean = log_kp_samp.mean(0)
    rec_log_om_lo80, rec_log_om_hi80 = np.percentile(log_om_samp, [10, 90], axis=0)
    rec_log_om_lo95, rec_log_om_hi95 = np.percentile(log_om_samp,  [2.5, 97.5], axis=0)
    rec_log_kp_lo80, rec_log_kp_hi80 = np.percentile(log_kp_samp, [10, 90], axis=0)
    rec_log_kp_lo95, rec_log_kp_hi95 = np.percentile(log_kp_samp,  [2.5, 97.5], axis=0)

    # Population-level recovered
    mu_om_rec = float(np.array(samples['mu_om']).mean())
    sigma_om_rec = float(np.array(samples['sigma_om']).mean())
    mu_kp_rec = float(np.array(samples['mu_kap']).mean())
    sigma_kp_rec = float(np.array(samples['sigma_kap']).mean())
    gamma_rec = float(np.array(samples['gamma']).mean())
    hazard_rec = float(np.array(samples['hazard']).mean())

    # Per-subject DF
    df = pd.DataFrame({
        'dataset': dataset_id, 'subj': np.arange(N_S),
        'log_omega_true': np.log(true_omega),
        'log_omega_rec':  rec_log_om_mean,
        'log_omega_lo80': rec_log_om_lo80, 'log_omega_hi80': rec_log_om_hi80,
        'log_omega_lo95': rec_log_om_lo95, 'log_omega_hi95': rec_log_om_hi95,
        'log_kappa_true': np.log(true_kappa),
        'log_kappa_rec':  rec_log_kp_mean,
        'log_kappa_lo80': rec_log_kp_lo80, 'log_kappa_hi80': rec_log_kp_hi80,
        'log_kappa_lo95': rec_log_kp_lo95, 'log_kappa_hi95': rec_log_kp_hi95,
    })

    pop_rec = {
        'dataset': dataset_id,
        'mu_om_rec': mu_om_rec, 'sigma_om_rec': sigma_om_rec,
        'mu_kp_rec': mu_kp_rec, 'sigma_kp_rec': sigma_kp_rec,
        'gamma_rec': gamma_rec, 'hazard_rec': hazard_rec,
    }
    return df, pop_rec


# ============================================================
# Metrics
# ============================================================

def compute_metrics(df):
    """Compute the full metric suite from per-subject true/recovered DataFrame."""
    out = {}
    for p in ['omega', 'kappa']:
        t = df[f'log_{p}_true'].values
        r = df[f'log_{p}_rec'].values
        lo80, hi80 = df[f'log_{p}_lo80'].values, df[f'log_{p}_hi80'].values
        lo95, hi95 = df[f'log_{p}_lo95'].values, df[f'log_{p}_hi95'].values

        out[f'r_{p}'], _ = pearsonr(t, r)
        out[f'rho_{p}'], _ = spearmanr(t, r)
        out[f'rmse_{p}'] = float(np.sqrt(np.mean((t - r) ** 2)))
        out[f'mae_{p}']  = float(np.mean(np.abs(t - r)))
        out[f'cov80_{p}'] = float(np.mean((lo80 <= t) & (t <= hi80)))
        out[f'cov95_{p}'] = float(np.mean((lo95 <= t) & (t <= hi95)))

        # Rank correlation at extremes (top vs bottom decile by true)
        n = len(t)
        k = max(1, n // 10)
        order_true = np.argsort(t)
        bot = order_true[:k]; top = order_true[-k:]
        # Among the union of true extremes, does rec preserve order?
        idx = np.concatenate([bot, top])
        rho_ext, _ = spearmanr(t[idx], r[idx])
        out[f'rho_extremes_{p}'] = float(rho_ext)
        # Fraction of true-top-decile that the model also places in its top decile
        order_rec = np.argsort(r)
        rec_top = set(order_rec[-k:].tolist())
        out[f'top_decile_overlap_{p}'] = float(len(set(top.tolist()) & rec_top) / k)

    # Cross-talk: true-ω vs recovered-κ, and true-κ vs recovered-ω
    out['cross_talk_om_to_kp'], _ = pearsonr(df['log_omega_true'], df['log_kappa_rec'])
    out['cross_talk_kp_to_om'], _ = pearsonr(df['log_kappa_true'], df['log_omega_rec'])
    return out


def print_metrics(m, label='', emp=None):
    print(f"\n--- Metrics ({label}) ---")
    for p, true_mu, true_sig in [
        ('omega',  emp['mu_log_om']  if emp else None, emp['sig_log_om']  if emp else None),
        ('kappa',  emp['mu_log_kap'] if emp else None, emp['sig_log_kap'] if emp else None),
    ]:
        print(f"  {p}:")
        print(f"    r = {m[f'r_{p}']:+.3f},  ρ = {m[f'rho_{p}']:+.3f}")
        print(f"    RMSE = {m[f'rmse_{p}']:.3f},  MAE = {m[f'mae_{p}']:.3f}  (log scale)")
        print(f"    80% coverage = {m[f'cov80_{p}']:.2%},  95% coverage = {m[f'cov95_{p}']:.2%}")
        print(f"    rho_extremes (top/bot 10%) = {m[f'rho_extremes_{p}']:+.3f}")
        print(f"    top-decile overlap = {m[f'top_decile_overlap_{p}']:.2%}")
    print(f"  Cross-talk:")
    print(f"    r(true ω, rec κ) = {m['cross_talk_om_to_kp']:+.3f}")
    print(f"    r(true κ, rec ω) = {m['cross_talk_kp_to_om']:+.3f}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    t0 = time.time()
    print("=" * 78)
    print("PARAMETER RECOVERY (FULL): V8c / M4 Joint Optimal Control Model")
    print("  Using empirical fitted (μ, σ, r) from `exploratory/mcmc_m4_params.csv`")
    print("  5 datasets × 100 subjects = 500 simulated subjects")
    print("=" * 78)

    emp = load_empirical_distribution(sample='exploratory')

    rng_master = np.random.default_rng(42)
    # One large dataset of 500 subjects — matches the M5 recovery protocol's actual
    # run (n_subj=290, n_datasets=1) and avoids "mean across datasets vs pooled"
    # framing. At N=500, simulation noise from the random true-parameter draw is
    # already negligible.
    N_PER = 500
    N_DATASETS = 1

    all_subj_dfs = []
    pop_records = []
    per_ds_metrics = []
    for ds in range(N_DATASETS):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        true_om, true_kp = sample_correlated_params(N_PER, emp, rng)
        true_base = rng.standard_normal(N_PER) * 0.1
        df_ds, pop_rec = fit_and_recover(
            true_om, true_kp, true_base, dataset_id=ds, seed=ds + 1
        )
        m = compute_metrics(df_ds)
        m['dataset'] = ds
        per_ds_metrics.append(m)
        print_metrics(m, label=f'dataset {ds}', emp=emp)
        all_subj_dfs.append(df_ds)
        pop_records.append(pop_rec)

    # Pooled across all datasets
    pooled = pd.concat(all_subj_dfs, ignore_index=True)
    m_pool = compute_metrics(pooled)
    print("\n" + "=" * 78)
    print(f"POOLED RECOVERY (N_total = {len(pooled)})")
    print("=" * 78)
    print_metrics(m_pool, label='pooled', emp=emp)

    # Population recovery summary
    print("\n--- Population-level recovery ---")
    pop_df = pd.DataFrame(pop_records)
    print(f"  μ_log(ω)  : true = {emp['mu_log_om']:+.3f},  recovered (mean of {N_DATASETS}) = {pop_df['mu_om_rec'].mean():+.3f}")
    print(f"  σ_log(ω)  : true = {emp['sig_log_om']:.3f},  recovered = {pop_df['sigma_om_rec'].mean():.3f}")
    print(f"  μ_log(κ)  : true = {emp['mu_log_kap']:+.3f},  recovered = {pop_df['mu_kp_rec'].mean():+.3f}")
    print(f"  σ_log(κ)  : true = {emp['sig_log_kap']:.3f},  recovered = {pop_df['sigma_kp_rec'].mean():.3f}")
    print(f"  γ          : true = {POP_PARAMS['gamma']:.3f},  recovered = {pop_df['gamma_rec'].mean():.3f}")
    print(f"  h (hazard) : true = {POP_PARAMS['hazard']:.3f},  recovered = {pop_df['hazard_rec'].mean():.3f}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(OUT_DIR / 'param_recovery_v8c_full.csv', index=False)
    summary = pd.DataFrame(per_ds_metrics + [{**m_pool, 'dataset': 'pooled'}])
    summary.to_csv(OUT_DIR / 'param_recovery_v8c_full_summary.csv', index=False)
    pop_df.to_csv(OUT_DIR / 'param_recovery_v8c_full_population.csv', index=False)

    print(f"\nSaved:")
    print(f"  per-subject:    {OUT_DIR / 'param_recovery_v8c_full.csv'}")
    print(f"  summary:        {OUT_DIR / 'param_recovery_v8c_full_summary.csv'}")
    print(f"  population:     {OUT_DIR / 'param_recovery_v8c_full_population.csv'}")
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
