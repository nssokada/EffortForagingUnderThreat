"""
MCMC Model Comparison: M1-M5 on Cell-Mean Vigor + Saturating Survival

Fits all 5 models with identical NUTS inference:
  4 chains x 2000 warmup + 4000 samples, target_accept=0.95, max_tree_depth=10

Model comparison via WAIC (primary) + PSIS-LOO (robustness), computed from
posterior samples using ArviZ.

Models (from model_comparison_cm.py):
  M1: Effort-only (kappa per-subject, no threat, no vigor likelihood)
  M2: Threat-only (omega per-subject, population kappa)
  M3: Single-parameter (theta = omega = kappa)
  M4: Separate equations (lambda choice + omega vigor, no shared W)
  M5: Joint W(u) (omega + kappa, both enter both channels)

Usage:
  python scripts/mcmc/run_model_comparison_mcmc.py [--num_warmup 2000] [--num_samples 4000]
"""

import sys
import os
import time
import argparse

import importlib.util

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood
from jax import random
import numpy as np
import pandas as pd
import arviz as az
from scipy.stats import pearsonr
from scipy.special import expit
from pathlib import Path

# Direct import to bypass scripts.modeling.__init__ (which imports deprecated models)
_cm_path = os.path.join(os.path.dirname(__file__), '..', 'modeling', 'joint_optimal', 'model_comparison_cm.py')
_spec = importlib.util.spec_from_file_location('model_comparison_cm', os.path.abspath(_cm_path))
_cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cm)

prepare_data = _cm.prepare_data
make_m1 = _cm.make_m1
make_m2 = _cm.make_m2
make_m3 = _cm.make_m3
make_m4 = _cm.make_m4
make_m5 = _cm.make_m5
eu_sat = _cm.eu_sat
C = _cm.C
KK = _cm.KK
PARAM_COUNTS = _cm.PARAM_COUNTS

OUT_DIR = Path("results/stats/joint_optimal")


# ============================================================
# MCMC fitting
# ============================================================

def fit_mcmc(name, model_fn, data, num_warmup=2000, num_samples=4000,
             num_chains=4, target_accept_prob=0.95, max_tree_depth=10,
             seed=42):
    """Fit a single model via NUTS and return MCMC object + timing."""

    kw = {k: data[k] for k in KK}

    kernel = NUTS(
        model_fn,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    print(f"\n{'=' * 70}")
    print(f"Fitting {name} via MCMC (NUTS)")
    print(f"  {num_chains} chains x {num_warmup} warmup + {num_samples} samples")
    print(f"  target_accept = {target_accept_prob}, max_tree_depth = {max_tree_depth}")

    t0 = time.time()
    mcmc.run(random.PRNGKey(seed), **kw)
    elapsed = time.time() - t0
    print(f"  {name} completed in {elapsed / 60:.1f} min")

    return mcmc, kw, elapsed


# ============================================================
# Convergence diagnostics
# ============================================================

def check_convergence(mcmc, name, data, num_chains=4):
    """Check R-hat and ESS for all parameters. Returns (passed, summary_df)."""

    samples = mcmc.get_samples(group_by_chain=True)  # (chains, samples, ...)
    flat_samples = mcmc.get_samples()

    rows = []
    max_rhat = 0.0
    min_ess = float('inf')

    for param_name, vals in samples.items():
        # vals shape: (num_chains, num_samples, ...) or (num_chains, num_samples)
        if vals.ndim == 2:
            # Scalar or 1-D: (chains, samples)
            rhat = _split_rhat(np.array(vals))
            ess = _bulk_ess(np.array(vals))
            rows.append({
                'model': name, 'parameter': param_name,
                'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                'rhat': rhat, 'ess': ess,
            })
            max_rhat = max(max_rhat, rhat)
            min_ess = min(min_ess, ess)
        elif vals.ndim == 3:
            # Per-subject: (chains, samples, N_S)
            rhats = []
            esses = []
            for j in range(vals.shape[2]):
                r = _split_rhat(np.array(vals[:, :, j]))
                e = _bulk_ess(np.array(vals[:, :, j]))
                rhats.append(r)
                esses.append(e)
            rhats = np.array(rhats)
            esses = np.array(esses)
            rows.append({
                'model': name, 'parameter': f'{param_name} (max across {vals.shape[2]} subjects)',
                'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                'rhat': float(np.max(rhats)), 'ess': float(np.min(esses)),
            })
            max_rhat = max(max_rhat, float(np.max(rhats)))
            min_ess = min(min_ess, float(np.min(esses)))

    passed = (max_rhat < 1.01) and (min_ess > 400)

    print(f"\n  {name} convergence: max R-hat = {max_rhat:.4f}, min ESS = {min_ess:.0f}"
          f"  {'PASS' if passed else 'FAIL'}")

    return passed, pd.DataFrame(rows)


def _split_rhat(chains):
    """Split R-hat. Input: (n_chains, n_samples)."""
    chains = np.array(chains, dtype=np.float64)
    if chains.ndim != 2 or chains.shape[1] < 4:
        return np.nan
    n_chains, n_samples = chains.shape
    mid = n_samples // 2
    split = np.concatenate([chains[:, :mid], chains[:, mid:2*mid]], axis=0)
    m = split.shape[0]
    n = split.shape[1]
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    B = n * np.var(chain_means, ddof=1)
    W = np.mean(chain_vars)
    if W < 1e-10:
        return 1.0
    var_hat = ((n - 1) / n) * W + (1.0 / n) * B
    return float(np.sqrt(var_hat / W))


def _bulk_ess(chains):
    """Approximate bulk ESS. Input: (n_chains, n_samples)."""
    flat = np.array(chains, dtype=np.float64).flatten()
    n = len(flat)
    var = np.var(flat)
    if var < 1e-10:
        return float(n)
    mean = np.mean(flat)
    max_lag = min(200, n // 4)
    acf_sum = 0.0
    for lag in range(1, max_lag + 1):
        acf = np.mean((flat[:-lag] - mean) * (flat[lag:] - mean)) / var
        if acf < 0.05:
            break
        acf_sum += acf
    return float(max(1.0, n / (1 + 2 * acf_sum)))


# ============================================================
# WAIC and LOO via ArviZ
# ============================================================

def compute_waic_loo(mcmc, model_fn, data, name):
    """Compute WAIC and PSIS-LOO from MCMC posterior samples.

    Returns dict with waic, loo, p_waic, p_loo, and warning counts.
    """
    kw = {k: data[k] for k in KK}

    # Compute pointwise log-likelihoods
    posterior_samples = mcmc.get_samples()

    # Determine which observed sites this model has
    # M1 has only 'oc' (choice), M2-M5 have 'oc' + 'ov' (choice + vigor)
    ll = log_likelihood(model_fn, posterior_samples, **kw)

    # Build ArviZ InferenceData
    num_chains = 4
    n_samples_per_chain = len(list(posterior_samples.values())[0]) // num_chains

    # Reshape log-likelihoods for ArviZ: (chains, draws, obs)
    ll_dict = {}
    for site_name, ll_vals in ll.items():
        arr = np.array(ll_vals)  # (total_samples, n_obs)
        arr = arr.reshape(num_chains, n_samples_per_chain, -1)
        ll_dict[site_name] = arr

    idata = az.from_dict(
        log_likelihood=ll_dict,
    )

    # WAIC
    try:
        waic_result = az.waic(idata)
        waic_val = float(waic_result.elpd_waic) * -2  # Convert to deviance scale
        p_waic = float(waic_result.p_waic)
        se_waic = float(waic_result.se) * 2
    except Exception as e:
        print(f"  WARNING: WAIC computation failed for {name}: {e}")
        waic_val = np.nan
        p_waic = np.nan
        se_waic = np.nan

    # PSIS-LOO
    try:
        loo_result = az.loo(idata)
        loo_val = float(loo_result.elpd_loo) * -2  # Convert to deviance scale
        p_loo = float(loo_result.p_loo)
        se_loo = float(loo_result.se) * 2
        # Count Pareto k warnings
        k_vals = np.array(loo_result.pareto_k)
        n_bad_k = int(np.sum(k_vals > 0.7))
        pct_bad_k = 100 * n_bad_k / len(k_vals)
    except Exception as e:
        print(f"  WARNING: LOO computation failed for {name}: {e}")
        loo_val = np.nan
        p_loo = np.nan
        se_loo = np.nan
        n_bad_k = np.nan
        pct_bad_k = np.nan

    result = {
        'WAIC': waic_val, 'p_WAIC': p_waic, 'SE_WAIC': se_waic,
        'LOO': loo_val, 'p_LOO': p_loo, 'SE_LOO': se_loo,
        'n_pareto_k_bad': n_bad_k, 'pct_pareto_k_bad': pct_bad_k,
    }

    print(f"  {name}: WAIC = {waic_val:.1f} (p_WAIC = {p_waic:.1f}, SE = {se_waic:.1f})")
    print(f"  {name}: LOO  = {loo_val:.1f} (p_LOO = {p_loo:.1f}, SE = {se_loo:.1f})")
    if not np.isnan(pct_bad_k):
        print(f"  {name}: Pareto k > 0.7: {n_bad_k} ({pct_bad_k:.1f}%)")

    return result


# ============================================================
# Choice/vigor evaluation (posterior predictive)
# ============================================================

def evaluate_fit(mcmc, data, name):
    """Compute choice accuracy/r-squared and vigor r-squared from posterior means."""

    samples = mcmc.get_samples()
    cs = np.array(data['cs']); cT = np.array(data['cT'])
    cDH = np.array(data['cDH']); cc = np.array(data['cc'])
    vr = np.array(data['vr'])
    NC = len(cs)

    # Vigor r-squared
    if 'rp' in samples:
        rp = np.array(samples['rp']).mean(0)
        r_vig = pearsonr(rp, vr)[0]
    else:
        r_vig = np.nan

    # Choice reconstruction
    tau_v = float(np.exp(np.mean(np.array(samples['tr'])))) if 'tr' in samples else 1.0

    if name == 'M1':
        kap = np.exp(
            float(np.mean(np.array(samples['mk'])))
            + float(np.mean(np.array(samples['sk'])))
            * np.mean(np.array(samples['kr']), axis=0)
        )
        delta = 4.0 - kap[cs] * (0.81 * cDH - 0.16)
        pH = expit(np.clip(delta / tau_v, -20, 20))
    elif name == 'M4':
        ml = float(np.mean(np.array(samples['ml'])))
        sl = float(np.mean(np.array(samples['sl'])))
        lr_mean = np.mean(np.array(samples['lr']), axis=0)
        lam = np.exp(ml + sl * lr_mean)
        beta = float(np.mean(np.array(samples['beta_pop'])))
        delta = 4.0 - lam[cs] * (0.81 * cDH - 0.16) - beta * cT
        pH = expit(np.clip(delta / tau_v, -20, 20))
    else:
        # M2, M3, M5: reconstruct from W grid search
        gamma_v = float(np.mean(np.array(samples.get('gamma', samples.get('gr')))))
        if 'gamma' not in samples:
            gamma_v = float(np.clip(np.exp(gamma_v), 0.1, 3.0))
        hazard_v = float(np.mean(np.array(samples.get('hazard', samples.get('hr')))))
        if 'hazard' not in samples:
            hazard_v = float(np.exp(hazard_v))
        sp_v = 0.25

        if name == 'M2':
            mo = float(np.mean(np.array(samples['mo'])))
            so = float(np.mean(np.array(samples['so'])))
            or_mean = np.mean(np.array(samples['or']), axis=0)
            om = np.exp(mo + so * or_mean)
            mk = float(np.mean(np.array(samples['mk'])))
            kap_arr = np.full(len(om), np.exp(mk))
        elif name == 'M3':
            mt = float(np.mean(np.array(samples['mt'])))
            st = float(np.mean(np.array(samples['st'])))
            tr_mean = np.mean(np.array(samples['tr_']), axis=0)
            theta = np.exp(mt + st * tr_mean)
            om = theta; kap_arr = theta
        else:  # M5
            mo = float(np.mean(np.array(samples['mo'])))
            so = float(np.mean(np.array(samples['so'])))
            or_mean = np.mean(np.array(samples['or']), axis=0)
            om = np.exp(mo + so * or_mean)
            mk = float(np.mean(np.array(samples['mk'])))
            sk = float(np.mean(np.array(samples['sk'])))
            kr_mean = np.mean(np.array(samples['kr']), axis=0)
            kap_arr = np.exp(mk + sk * kr_mean)

        ug = np.linspace(0.1, 1.5, 40)
        VH = np.zeros(NC); VL = np.zeros(NC)
        for idx in range(NC):
            s = cs[idx]; T = cT[idx]; DH = cDH[idx]
            for R, req, D, store in [(5., .9, DH, VH), (1., .4, 1., VL)]:
                speed = expit((ug - 0.25 * req) / sp_v)
                S = np.exp(-hazard_v * T**gamma_v * D / np.clip(speed, .01, None))
                W = S * R - (1 - S) * om[s] * (R + C) - kap_arr[s] * (ug - req)**2 * D
                store[idx] = W.max()
        pH = expit(np.clip((VH - VL) / tau_v, -20, 20))

    acc = ((pH >= 0.5).astype(int) == cc).mean()
    ch_df = pd.DataFrame({'s': cs, 'c': cc, 'p': pH})
    sc = ch_df.groupby('s').agg(o=('c', 'mean'), p=('p', 'mean'))
    try:
        r_ch = pearsonr(sc['o'], sc['p'])[0]
    except Exception:
        r_ch = np.nan

    return {
        'choice_acc': float(acc),
        'choice_r2': float(r_ch**2) if not np.isnan(r_ch) else np.nan,
        'vigor_r2': float(r_vig**2) if not np.isnan(r_vig) else np.nan,
    }


# ============================================================
# Main
# ============================================================

MODEL_SPECS = [
    ('M1', make_m1, 1, 'Effort-only (kappa)'),
    ('M2', make_m2, 1, 'Threat-only (omega)'),
    ('M3', make_m3, 1, 'Single-param (theta=omega=kappa)'),
    ('M4', make_m4, 2, 'Separate (lambda+omega)'),
    ('M5', make_m5, 2, 'Joint W(u) (omega+kappa)'),
]


def main():
    parser = argparse.ArgumentParser(description='MCMC Model Comparison')
    parser.add_argument('--num_warmup', type=int, default=2000)
    parser.add_argument('--num_samples', type=int, default=4000)
    parser.add_argument('--num_chains', type=int, default=4)
    parser.add_argument('--target_accept', type=float, default=0.95)
    parser.add_argument('--max_tree_depth', type=int, default=10)
    parser.add_argument('--models', type=str, default='M1,M2,M3,M4,M5',
                        help='Comma-separated list of models to fit')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    numpyro.set_host_device_count(args.num_chains)

    models_to_fit = [m.strip() for m in args.models.split(',')]

    t_start = time.time()
    print("=" * 70)
    print("MCMC MODEL COMPARISON")
    print(f"  {args.num_chains} chains x {args.num_warmup} warmup + {args.num_samples} samples")
    print(f"  target_accept = {args.target_accept}, max_tree_depth = {args.max_tree_depth}")
    print(f"  Models: {', '.join(models_to_fit)}")
    print("=" * 70)

    data = prepare_data()
    NS = data['N_S']

    results = []
    all_diags = []
    convergence_failures = []

    for name, make_fn, nps, desc in MODEL_SPECS:
        if name not in models_to_fit:
            continue

        print(f"\n{'=' * 70}")
        print(f"--- {name}: {desc} ---")
        print(f"{'=' * 70}")

        model_fn = make_fn(NS, data['N_choice'], data['N_vigor'])

        # Fit
        mcmc, kw, elapsed = fit_mcmc(
            name, model_fn, data,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            target_accept_prob=args.target_accept,
            max_tree_depth=args.max_tree_depth,
            seed=args.seed,
        )

        # Convergence
        passed, diag_df = check_convergence(mcmc, name, data, args.num_chains)
        all_diags.append(diag_df)

        if not passed:
            convergence_failures.append(name)
            print(f"\n  WARNING: {name} failed convergence criteria (R-hat < 1.01, ESS > 400)")

        # WAIC + LOO
        print(f"\n  Computing WAIC and LOO for {name}...")
        ic_results = compute_waic_loo(mcmc, model_fn, data, name)

        # Predictive evaluation
        metrics = evaluate_fit(mcmc, data, name)

        n_params = PARAM_COUNTS[name](NS)

        row = {
            'Model': name, 'Description': desc,
            'n_per_subj': nps, 'n_params': n_params,
            'WAIC': ic_results['WAIC'],
            'p_WAIC': ic_results['p_WAIC'],
            'SE_WAIC': ic_results['SE_WAIC'],
            'LOO': ic_results['LOO'],
            'p_LOO': ic_results['p_LOO'],
            'SE_LOO': ic_results['SE_LOO'],
            'n_pareto_k_bad': ic_results['n_pareto_k_bad'],
            'pct_pareto_k_bad': ic_results['pct_pareto_k_bad'],
            'choice_acc': metrics['choice_acc'],
            'choice_r2': metrics['choice_r2'],
            'vigor_r2': metrics['vigor_r2'],
            'converged': passed,
            'elapsed_min': elapsed / 60,
        }
        results.append(row)

        print(f"\n  WAIC = {ic_results['WAIC']:.1f}, LOO = {ic_results['LOO']:.1f}")
        print(f"  Choice: acc = {metrics['choice_acc']:.3f}, r2 = {metrics['choice_r2']:.3f}")
        if not np.isnan(metrics.get('vigor_r2', np.nan)):
            print(f"  Vigor:  r2 = {metrics['vigor_r2']:.3f}")
        else:
            print(f"  Vigor:  not modeled")

    # ── Summary ──
    if results:
        df = pd.DataFrame(results)

        # Delta-WAIC relative to M5
        if 'M5' in df['Model'].values:
            waic_m5 = df.loc[df['Model'] == 'M5', 'WAIC'].values[0]
            loo_m5 = df.loc[df['Model'] == 'M5', 'LOO'].values[0]
            df['dWAIC'] = df['WAIC'] - waic_m5
            df['dLOO'] = df['LOO'] - loo_m5

        print("\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)
        cols = ['Model', 'Description', 'n_per_subj', 'WAIC', 'dWAIC', 'LOO', 'dLOO',
                'choice_acc', 'choice_r2', 'vigor_r2', 'converged']
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))

        # Hypothesis tests
        if 'M5' in df['Model'].values:
            print("\n" + "=" * 70)
            print("HYPOTHESIS TESTS (preregistered)")
            print("=" * 70)

            m5_waic = df.loc[df['Model'] == 'M5', 'WAIC'].values[0]
            m5_loo = df.loc[df['Model'] == 'M5', 'LOO'].values[0]

            for alt, h_name in [('M1', 'H3a'), ('M2', 'H3b'), ('M3', 'H3c')]:
                row_alt = df[df['Model'] == alt]
                if len(row_alt) == 0:
                    continue
                alt_waic = row_alt['WAIC'].values[0]
                alt_loo = row_alt['LOO'].values[0]
                dw = alt_waic - m5_waic
                dl = alt_loo - m5_loo

                waic_wins = dw > 0
                loo_wins = dl > 0
                if waic_wins and loo_wins:
                    verdict = "CONFIRMED (WAIC + LOO agree)"
                elif waic_wins or loo_wins:
                    verdict = "EQUIVOCAL (WAIC and LOO disagree)"
                else:
                    verdict = "FAILED"

                print(f"  {h_name}: M5 vs {alt}")
                print(f"    dWAIC = {dw:+.1f}, dLOO = {dl:+.1f} -> {verdict}")

        # Save
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "mcmc_model_comparison.csv", index=False)
        print(f"\nSaved: {OUT_DIR / 'mcmc_model_comparison.csv'}")

        diag_all = pd.concat(all_diags, ignore_index=True)
        diag_all.to_csv(OUT_DIR / "mcmc_convergence_diagnostics.csv", index=False)
        print(f"Saved: {OUT_DIR / 'mcmc_convergence_diagnostics.csv'}")

        if convergence_failures:
            print(f"\n  WARNING: The following models failed convergence: {convergence_failures}")
            print(f"  Consider re-running with --num_samples {args.num_samples * 2}")

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {elapsed_total / 60:.1f} min ({elapsed_total / 3600:.1f} hrs)")


if __name__ == '__main__':
    main()
