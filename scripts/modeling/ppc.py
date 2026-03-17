"""
Posterior Predictive Check (PPC) Utilities for FET Models.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from numpyro.infer import Predictive
from typing import Dict, Tuple, Optional, Union
import warnings


class PosteriorPredictive:
    """Posterior predictive analysis for fitted FET models."""
    
    def __init__(self, fitter):
        self.fitter = fitter
        self.model = fitter.model
        
        if self.model.posterior_samples is None:
            raise ValueError("Model must be fitted first")
    
    def _build_inputs(self, data: pd.DataFrame) -> Dict[str, jnp.ndarray]:
        subj_to_idx = self.model.subj_to_idx
        subjects = data['subj'].values
        
        known_subjects = set(subj_to_idx.keys())
        unknown = set(subjects) - known_subjects
        if unknown:
            warnings.warn(f"Unknown subjects: {unknown}. Using index 0.")
        
        subj_idx = jnp.array([subj_to_idx.get(s, 0) for s in subjects])
        
        inputs = {
            'subj_idx': subj_idx,
            'threat': jnp.array(data['threat'].values),
            'distance_H': jnp.array(data.get('distance_H', np.zeros(len(data))).values),
            'distance_L': jnp.array(data.get('distance_L', np.zeros(len(data))).values),
            'effort_H': jnp.array(data['effort_H'].values),
            'effort_L': jnp.array(data['effort_L'].values),
            'n_subjects': self.model.n_subjects,
        }
        
        fp = self.model.fit_params or {}
        inputs['R_H'] = fp.get('R_H', 5.0)
        inputs['R_L'] = fp.get('R_L', 1.0)
        inputs['C'] = fp.get('C', 5.0)
        
        return inputs
    
    def _subsample_posterior(self, samples: Dict, n: Optional[int], seed: int) -> Dict:
        if n is None:
            return samples
        
        total_draws = next(iter(samples.values())).shape[0]
        n = min(n, total_draws)
        
        rng = np.random.default_rng(seed)
        idx = rng.choice(total_draws, size=n, replace=False)
        
        return {k: v[idx] for k, v in samples.items()}
    
    def predict(
        self,
        data: pd.DataFrame,
        n_draws: Optional[int] = 500,
        seed: int = 0,
        chunk_size: int = 100,
        return_draws: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """Generate posterior predictions aligned to data."""
        
        samples = {k: jnp.array(v) for k, v in self.model.posterior_samples.items()}
        inputs = self._build_inputs(data)
        samples = self._subsample_posterior(samples, n_draws, seed)
        
        total_draws = next(iter(samples.values())).shape[0]
        n_trials = len(data)
        
        print(f"Generating predictions: {total_draws} draws x {n_trials} trials")
        
        return_sites = ['p_high', 'S_u_H', 'S_u_L', 'SV_H', 'SV_L', 'discount_H', 'discount_L']
        
        all_predictions = {site: [] for site in return_sites}
        n_chunks = max(1, (total_draws + chunk_size - 1) // chunk_size)
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_draws)
            
            chunk_samples = {k: v[start_idx:end_idx] for k, v in samples.items()}
            rng_key = jax.random.PRNGKey(seed + i)
            
            try:
                pred = Predictive(
                    self.model.model,
                    posterior_samples=chunk_samples,
                    return_sites=return_sites
                )(rng_key, choice=None, include_ppc_sites=True, **inputs)
                
                for site in return_sites:
                    if site in pred:
                        all_predictions[site].append(np.asarray(pred[site]))
                        
            except Exception as e:
                warnings.warn(f"Prediction fallback: {e}")
                pred = Predictive(
                    self.model.model,
                    posterior_samples=chunk_samples,
                    return_sites=['obs', 'S_u_H', 'S_u_L']
                )(rng_key, choice=None, include_ppc_sites=True, **inputs)
                
                all_predictions['p_high'].append(np.asarray(pred['obs'], dtype=np.float64))
                for site in ['S_u_H', 'S_u_L']:
                    if site in pred:
                        all_predictions[site].append(np.asarray(pred[site]))
            
            jax.clear_caches()
            
            if (i + 1) % 5 == 0 or i == n_chunks - 1:
                print(f"  Chunk {i+1}/{n_chunks} complete")
        
        draws = {}
        for site in return_sites:
            if all_predictions[site]:
                draws[site] = np.concatenate(all_predictions[site], axis=0)
        
        p_high = draws.get('p_high', np.full(n_trials, np.nan)).mean(axis=0)
        S_u_H = draws.get('S_u_H', np.full(n_trials, np.nan)).mean(axis=0)
        S_u_L = draws.get('S_u_L', np.full(n_trials, np.nan)).mean(axis=0)
        SV_H = draws.get('SV_H', np.full(n_trials, np.nan)).mean(axis=0)
        SV_L = draws.get('SV_L', np.full(n_trials, np.nan)).mean(axis=0)
        
        if 'choice' in data.columns:
            y = data['choice'].values.astype(int)
            choice_likelihood = np.where(y == 1, p_high, 1.0 - p_high)
        else:
            choice_likelihood = np.full(n_trials, np.nan)
        
        keep_cols = [c for c in ['subj', 'trial', 'threat', 'distance_H', 'distance_L',
                                 'effort_H', 'effort_L', 'choice'] if c in data.columns]
        out = data[keep_cols].copy()
        out['p_high'] = p_high
        out['S_u_H'] = S_u_H
        out['S_u_L'] = S_u_L
        out['SV_H'] = SV_H
        out['SV_L'] = SV_L
        out['choice_likelihood'] = choice_likelihood
        
        if 'discount_H' in draws:
            out['discount_H'] = draws['discount_H'].mean(axis=0)
        if 'discount_L' in draws:
            out['discount_L'] = draws['discount_L'].mean(axis=0)
        
        if 'p_high' in draws:
            out['p_high_std'] = draws['p_high'].std(axis=0)
            out['p_high_ci_lower'] = np.percentile(draws['p_high'], 2.5, axis=0)
            out['p_high_ci_upper'] = np.percentile(draws['p_high'], 97.5, axis=0)
        
        print("Predictions complete")
        
        if return_draws:
            return out, draws
        return out
    
    def compute_fit_metrics(
        self,
        data: pd.DataFrame,
        pred_df: Optional[pd.DataFrame] = None,
        n_bins: int = 10
    ) -> Dict:
        """Compute comprehensive fit metrics."""
        if pred_df is None:
            pred_df = self.predict(data)
        
        p = pred_df['p_high'].values
        y = data['choice'].values.astype(int)
        eps = 1e-12
        
        brier = np.mean((p - y) ** 2)
        
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        calibration_data = []
        N = len(p)
        
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
            n_bin = mask.sum()
            if n_bin > 0:
                p_bin = p[mask].mean()
                o_bin = y[mask].mean()
                gap = abs(p_bin - o_bin)
                ece += (n_bin / N) * gap
                mce = max(mce, gap)
                calibration_data.append({
                    'bin_lower': lo, 'bin_upper': hi,
                    'predicted_mean': p_bin, 'observed_mean': o_bin,
                    'n_trials': n_bin, 'gap': gap
                })
        
        ll_model = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        base_rate = y.mean()
        ll_null = np.sum(y * np.log(base_rate + eps) + (1 - y) * np.log(1 - base_rate + eps))
        
        r2_mcf = 1.0 - (ll_model / ll_null) if ll_null != 0 else 0.0
        accuracy = np.mean((p > 0.5).astype(int) == y)
        
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, p)
        except ImportError:
            sorted_idx = np.argsort(p)
            sorted_y = y[sorted_idx]
            n_pos, n_neg = y.sum(), len(y) - y.sum()
            if n_pos > 0 and n_neg > 0:
                tpr_sum = np.cumsum(sorted_y[::-1])
                auc = tpr_sum[sorted_y[::-1] == 0].sum() / (n_pos * n_neg)
            else:
                auc = 0.5
        
        return {
            'Brier': brier, 'ECE': ece, 'MCE': mce, 'McFadden_R2': r2_mcf,
            'LogLik_model': ll_model, 'LogLik_null': ll_null,
            'Accuracy': accuracy, 'AUC': auc, 'BaseRate': base_rate, 'N_trials': N,
            'calibration_data': pd.DataFrame(calibration_data)
        }
    
    def compute_subject_metrics(
        self,
        data: pd.DataFrame,
        pred_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Compute per-subject fit metrics."""
        if pred_df is None:
            pred_df = self.predict(data)
        
        merged = data.copy()
        merged['p_high'] = pred_df['p_high'].values
        
        def subj_metrics(df):
            p = df['p_high'].values
            y = df['choice'].values.astype(int)
            eps = 1e-12
            
            return pd.Series({
                'accuracy': np.mean((p > 0.5).astype(int) == y),
                'brier': np.mean((p - y) ** 2),
                'log_lik': np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)),
                'mean_p_high': p.mean(),
                'mean_choice': y.mean(),
                'n_trials': len(df)
            })
        
        return merged.groupby('subj').apply(subj_metrics).reset_index()
    
    @staticmethod
    def print_metrics(metrics: Dict, model_name: str = "Model") -> None:
        """Print formatted metrics table."""
        print(f"\n{'='*60}")
        print(f"FIT METRICS: {model_name}")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'Value':>15}")
        print("-" * 45)
        print(f"{'Brier Score':<25} {metrics['Brier']:>15.4f}")
        print(f"{'ECE':<25} {metrics['ECE']:>15.4f}")
        print(f"{'MCE':<25} {metrics['MCE']:>15.4f}")
        print(f"{'McFadden R2':<25} {metrics['McFadden_R2']:>15.4f}")
        print(f"{'Accuracy':<25} {metrics['Accuracy']:>15.4f}")
        print(f"{'AUC-ROC':<25} {metrics['AUC']:>15.4f}")
        print(f"{'Log-Lik (Model)':<25} {metrics['LogLik_model']:>15.2f}")
        print(f"{'Log-Lik (Null)':<25} {metrics['LogLik_null']:>15.2f}")
        print(f"{'Base Rate':<25} {metrics['BaseRate']:>15.4f}")
        print(f"{'N Trials':<25} {metrics['N_trials']:>15d}")
        print("=" * 60)


def posterior_trial_predictions_df(
    fitter_or_model,
    data: pd.DataFrame,
    n_draws: int = 500,
    seed: int = 0,
    return_draws: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """Convenience function for backward compatibility."""
    # Import here to avoid circular imports
    from .fitter import ModelFitter
    
    if hasattr(fitter_or_model, 'model'):
        ppc = PosteriorPredictive(fitter_or_model)
    else:
        fitter = ModelFitter(fitter_or_model)
        fitter.fit_result = {}
        ppc = PosteriorPredictive(fitter)
    
    return ppc.predict(data, n_draws=n_draws, seed=seed, return_draws=return_draws)


def compute_calibration_metrics(
    p_high: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """Compute calibration metrics from raw arrays."""
    p = np.asarray(p_high, dtype=float)
    y = np.asarray(observed, dtype=int)
    eps = 1e-12
    
    brier = np.mean((p - y) ** 2)
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    N = len(p)
    
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if mask.any():
            gap = abs(p[mask].mean() - y[mask].mean())
            ece += (mask.sum() / N) * gap
            mce = max(mce, gap)
    
    ll_model = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    base = y.mean()
    ll_null = np.sum(y * np.log(base + eps) + (1 - y) * np.log(1 - base + eps))
    r2_mcf = 1.0 - (ll_model / ll_null) if ll_null != 0 else 0.0
    
    return {
        'Brier': brier, 'ECE': ece, 'MCE': mce, 'McFadden_R2': r2_mcf,
        'LogLik_model': ll_model, 'LogLik_null': ll_null,
        'BaseRate': base, 'N': N
    }


def compute_waic(
    fitter,
    data: pd.DataFrame,
    n_draws: Optional[int] = None,
    seed: int = 0
) -> Dict:
    """
    Compute WAIC (Widely Applicable Information Criterion) for model comparison.
    
    WAIC = -2 * (lppd - p_waic)
    
    Where:
    - lppd: log pointwise predictive density
    - p_waic: effective number of parameters
    
    Lower WAIC = better model (penalizes complexity)
    
    Parameters
    ----------
    fitter : ModelFitter
        Fitted model
    data : pd.DataFrame
        Data used for fitting
    n_draws : int, optional
        Number of posterior draws to use (None = all)
    seed : int
        Random seed for subsampling
        
    Returns
    -------
    dict
        WAIC, lppd, p_waic, se (standard error)
    """
    model = fitter.model
    samples = {k: jnp.array(v) for k, v in model.posterior_samples.items()}
    
    # Subsample if requested
    if n_draws is not None:
        total = next(iter(samples.values())).shape[0]
        n_draws = min(n_draws, total)
        rng = np.random.default_rng(seed)
        idx = rng.choice(total, size=n_draws, replace=False)
        samples = {k: v[idx] for k, v in samples.items()}
    
    n_samples = next(iter(samples.values())).shape[0]
    n_trials = len(data)
    
    # Build inputs
    subj_to_idx = model.subj_to_idx
    subj_idx = jnp.array([subj_to_idx.get(s, 0) for s in data['subj'].values])
    
    threat = jnp.array(data['threat'].values)
    distance_H = jnp.array(data.get('distance_H', np.zeros(n_trials)).values)
    distance_L = jnp.array(data.get('distance_L', np.zeros(n_trials)).values)
    effort_H = jnp.array(data['effort_H'].values)
    effort_L = jnp.array(data['effort_L'].values)
    choice = data['choice'].values.astype(int)
    
    fp = model.fit_params or {}
    R_H = fp.get('R_H', 5.0)
    R_L = fp.get('R_L', 1.0)
    C = fp.get('C', 5.0)
    
    # Compute log-likelihood for each draw and each trial
    # Shape: (n_samples, n_trials)
    log_lik = np.zeros((n_samples, n_trials))
    
    eps = 1e-12
    
    for s in range(n_samples):
        # Get parameters for this draw
        tau = float(samples['tau'][s])
        
        # Get subject-level k and z for this draw
        if 'k' in samples:
            k = samples['k'][s]  # Shape: (n_subjects,)
            k_i = k[subj_idx]
        else:
            k_i = None
            
        if 'z' in samples:
            z = samples['z'][s]  # Shape: (n_subjects,)
            z_i = z[subj_idx]
        else:
            z_i = None
        
        # Get beta if present
        if 'beta' in samples:
            beta = samples['beta'][s]
            beta_i = beta[subj_idx]
        else:
            beta_i = None
        
        # Compute SV_H, SV_L
        if model.use_effort_component and k_i is not None:
            discount_H = model.discount_function(effort_H, k_i)
            discount_L = model.discount_function(effort_L, k_i)
        else:
            discount_H = jnp.ones(n_trials)
            discount_L = jnp.ones(n_trials)
        
        if model.use_threat_component and z_i is not None:
            S_u_H = model.survival_function(distance_H, threat, z_i)
            S_u_L = model.survival_function(distance_L, threat, z_i)
        else:
            S_u_H = jnp.ones(n_trials)
            S_u_L = jnp.ones(n_trials)
        
        if model.use_effort_component and model.use_threat_component:
            SV_H = R_H * discount_H * S_u_H - (1 - S_u_H) * C
            SV_L = R_L * discount_L * S_u_L - (1 - S_u_L) * C
        elif model.use_effort_component:
            SV_H = R_H * discount_H
            SV_L = R_L * discount_L
        else:
            SV_H = R_H * S_u_H - (1 - S_u_H) * C
            SV_L = R_L * S_u_L - (1 - S_u_L) * C
        
        # Compute p_high
        SV_diff = np.clip(np.array(SV_H - SV_L), -20, 20)
        logit_p = SV_diff / tau
        
        # Apply bias if present
        if beta_i is not None:
            logit_p = logit_p - np.array(beta_i * threat)
        
        logit_p = np.clip(logit_p, -20, 20)
        p_high = 1 / (1 + np.exp(-logit_p))
        
        # Log-likelihood for each trial
        p_high = np.clip(p_high, eps, 1 - eps)
        log_lik[s, :] = choice * np.log(p_high) + (1 - choice) * np.log(1 - p_high)
    
    # WAIC computation
    # lppd: log pointwise predictive density
    # For each trial, compute log(mean(exp(log_lik))) = log(mean(lik))
    # Use log-sum-exp trick for numerical stability
    max_ll = log_lik.max(axis=0)
    lppd_i = max_ll + np.log(np.mean(np.exp(log_lik - max_ll), axis=0))
    lppd = np.sum(lppd_i)
    
    # p_waic: effective number of parameters (variance of log-lik)
    p_waic_i = np.var(log_lik, axis=0)
    p_waic = np.sum(p_waic_i)
    
    # WAIC
    waic_i = -2 * (lppd_i - p_waic_i)
    waic = np.sum(waic_i)
    
    # Standard error
    se = np.sqrt(n_trials * np.var(waic_i))
    
    return {
        'WAIC': waic,
        'lppd': lppd,
        'p_waic': p_waic,
        'se': se,
        'pointwise': waic_i  # For LOO-style comparisons
    }


def compute_loo(
    fitter,
    data: pd.DataFrame,
    n_draws: Optional[int] = None,
    seed: int = 0
) -> Dict:
    """
    Compute LOO-CV (Leave-One-Out Cross-Validation) using PSIS 
    (Pareto Smoothed Importance Sampling).
    
    This is often preferred over WAIC for model comparison.
    
    Parameters
    ----------
    fitter : ModelFitter
        Fitted model
    data : pd.DataFrame
        Data used for fitting
    n_draws : int, optional
        Number of posterior draws to use
    seed : int
        Random seed
        
    Returns
    -------
    dict
        loo, p_loo, se, k_pareto (diagnostic)
    """
    # First compute WAIC to get pointwise log-likelihoods
    waic_result = compute_waic(fitter, data, n_draws, seed)
    
    # For a proper LOO, we'd need PSIS, but as an approximation
    # we can use WAIC pointwise values
    # This is a simplified version - for full PSIS-LOO, use arviz
    
    loo = waic_result['WAIC']  # Approximation
    p_loo = waic_result['p_waic']
    se = waic_result['se']
    
    return {
        'LOO': loo,
        'p_loo': p_loo,
        'se': se,
        'warning': 'Approximate LOO (WAIC-based). For exact PSIS-LOO, use arviz.'
    }


def compare_models(
    fitted_models: Dict,
    data: pd.DataFrame,
    n_draws: int = 500,
    seed: int = 0
) -> pd.DataFrame:
    """
    Compare multiple fitted models using WAIC and fit metrics.
    
    Parameters
    ----------
    fitted_models : dict
        Dictionary of {name: ModelFitter}
    data : pd.DataFrame
        Data used for fitting
    n_draws : int
        Number of posterior draws for WAIC
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Comparison table sorted by WAIC (lower = better)
    """
    results = []
    
    print("Computing model comparison metrics...")
    
    for name, fitter in fitted_models.items():
        print(f"  Processing: {name}")
        
        # Compute WAIC
        waic_result = compute_waic(fitter, data, n_draws, seed)
        
        # Compute fit metrics
        ppc = PosteriorPredictive(fitter)
        pred_df = ppc.predict(data, n_draws=n_draws, seed=seed)
        metrics = ppc.compute_fit_metrics(data, pred_df)
        
        results.append({
            'Model': name,
            'WAIC': waic_result['WAIC'],
            'p_waic': waic_result['p_waic'],
            'lppd': waic_result['lppd'],
            'WAIC_se': waic_result['se'],
            'Brier': metrics['Brier'],
            'ECE': metrics['ECE'],
            'McFadden_R2': metrics['McFadden_R2'],
            'Accuracy': metrics['Accuracy'],
            'AUC': metrics['AUC'],
        })
    
    df = pd.DataFrame(results)
    
    # Sort by WAIC (lower = better)
    df = df.sort_values('WAIC').reset_index(drop=True)
    
    # Add delta WAIC (difference from best model)
    df['dWAIC'] = df['WAIC'] - df['WAIC'].min()
    
    # Reorder columns
    cols = ['Model', 'WAIC', 'dWAIC', 'WAIC_se', 'p_waic', 'lppd', 
            'Brier', 'ECE', 'McFadden_R2', 'Accuracy', 'AUC']
    df = df[cols]
    
    print("Done!")
    
    return df
