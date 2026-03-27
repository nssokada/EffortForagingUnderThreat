"""
Model Fitting Utilities for FET Choice Models.
"""

import os
import time
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import multiprocessing as mp

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from numpyro.infer import Predictive

# CORRECT IMPORT
from .base_model import BaseEffortThreatModel, configure_device


class ModelFitter:
    """Unified interface for fitting, saving, and loading FET models."""
    
    def __init__(self, model: BaseEffortThreatModel):
        self.model = model
        self.data = None
        self.fit_result = None
    
    @staticmethod
    def configure_device(use_gpu: bool = True, gpu_memory_fraction: float = 0.9):
        return configure_device(use_gpu, gpu_memory_fraction)
    
    def fit(self, data: pd.DataFrame, 
            R_H: float = 5.0, R_L: float = 1.0, C: float = 5.0,
            num_warmup: int = 1000, num_samples: int = 1000,
            num_chains: int = 4, target_accept_prob: float = 0.95,
            max_tree_depth: int = 12, seed: int = 42,
            progress_bar: bool = True) -> 'ModelFitter':
        """Fit the model to data."""
        self.data = data.copy()
        start_time = time.time()
        
        self.model.fit(
            data=data, R_H=R_H, R_L=R_L, C=C,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth, seed=seed, progress_bar=progress_bar
        )
        
        fit_time = (time.time() - start_time) / 60
        self.fit_result = self._package_results(fit_time)
        return self
    
    def _package_results(self, fit_time: float) -> Dict[str, Any]:
        subject_params = {}
        population_params = {}
        
        for param, values in self.model.posterior_samples.items():
            if len(values.shape) == 2 and values.shape[1] == self.model.n_subjects:
                subject_params[param] = values
            else:
                population_params[param] = values
        
        return {
            'model_class': self.model.__class__,
            'model_class_name': self.model.__class__.__name__,
            'posterior_samples': self.model.posterior_samples,
            'subject_level_params': subject_params,
            'population_params': population_params,
            'subjects': self.model.subjects,
            'subj_to_idx': self.model.subj_to_idx,
            'n_subjects': self.model.n_subjects,
            'fit_params': self.model.fit_params,
            'fit_time_minutes': fit_time,
            'model_kwargs': self._get_model_kwargs(),
            'data_summary': {
                'n_trials': len(self.data),
                'n_subjects': self.model.n_subjects
            }
        }
    
    def _get_model_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        if hasattr(self.model, 'discount_type'):
            kwargs['discount'] = self.model.discount_type
        return kwargs
    
    def save(self, filepath: Union[str, Path], save_data: bool = True) -> None:
        """Save fitted model to disk."""
        if self.fit_result is None:
            raise ValueError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = self.fit_result.copy()
        if save_data and self.data is not None:
            save_dict['data'] = self.data
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ModelFitter':
        """Load a fitted model from disk."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
        
        model_class = result['model_class']
        model_kwargs = result.get('model_kwargs', {})
        model = model_class(**model_kwargs)
        
        model.posterior_samples = result['posterior_samples']
        model.subjects = result['subjects']
        model.subj_to_idx = result['subj_to_idx']
        model.n_subjects = result['n_subjects']
        model.fit_params = result['fit_params']
        
        fitter = cls(model)
        fitter.fit_result = result
        fitter.data = result.get('data', None)
        
        print(f"Loaded: {result['model_class_name']}")
        print(f"  Subjects: {result['n_subjects']}, Trials: {result['data_summary']['n_trials']}")
        
        return fitter
    
    def get_subject_params(self, param: str) -> pd.DataFrame:
        return self.model.get_subject_params(param)
    
    def get_population_params(self) -> pd.DataFrame:
        return self.model.get_population_params()


def _fit_single_model_worker(args):
    """Worker function for parallel model fitting."""
    model_name, model_class, model_kwargs, data_path, gpu_id, gpu_fraction, fit_params = args
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(gpu_fraction)
    
    print(f"[{model_name}] Starting on GPU {gpu_id}")
    start_time = time.time()
    
    data = pd.read_pickle(data_path)
    model = model_class(**model_kwargs)
    
    model.fit(
        data=data,
        num_warmup=fit_params.get('num_warmup', 1000),
        num_samples=fit_params.get('num_samples', 1000),
        num_chains=fit_params.get('num_chains', 4),
        target_accept_prob=fit_params.get('target_accept_prob', 0.85),
        R_H=fit_params.get('R_H', 5.0),
        R_L=fit_params.get('R_L', 1.0),
        C=fit_params.get('C', 5.0),
        progress_bar=False
    )
    
    fit_time = (time.time() - start_time) / 60
    print(f"[{model_name}] Complete in {fit_time:.1f} min")
    
    subject_params = {}
    population_params = {}
    for param, values in model.posterior_samples.items():
        if len(values.shape) == 2 and values.shape[1] == model.n_subjects:
            subject_params[param] = values
        else:
            population_params[param] = values
    
    return {
        'name': model_name,
        'model_class': model_class,
        'model_class_name': model_class.__name__,
        'model_kwargs': model_kwargs,
        'posterior_samples': model.posterior_samples,
        'subject_level_params': subject_params,
        'population_params': population_params,
        'subjects': model.subjects,
        'subj_to_idx': model.subj_to_idx,
        'n_subjects': model.n_subjects,
        'fit_params': model.fit_params,
        'fit_time_minutes': fit_time,
        'data_summary': {'n_trials': len(data), 'n_subjects': model.n_subjects}
    }


def fit_models_parallel(
    models_dict: Dict[str, BaseEffortThreatModel],
    data: pd.DataFrame,
    n_parallel: int = 2,
    save_dir: str = './results/model_fits',
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    target_accept_prob: float = 0.85,
    R_H: float = 5.0,
    R_L: float = 1.0,
    C: float = 5.0,
    use_gpu: bool = True
) -> Dict[str, ModelFitter]:
    """Fit multiple models in parallel."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = save_dir / 'temp_data.pkl'
    data.to_pickle(data_path)
    data.to_pickle(save_dir / 'original_data.pkl')
    
    fit_params = {
        'num_warmup': num_warmup, 'num_samples': num_samples,
        'num_chains': num_chains, 'target_accept_prob': target_accept_prob,
        'R_H': R_H, 'R_L': R_L, 'C': C
    }
    
    with open(save_dir / 'fit_params.json', 'w') as f:
        json.dump(fit_params, f, indent=2)
    
    model_items = []
    for name, model in models_dict.items():
        model_kwargs = {}
        if hasattr(model, 'discount_type'):
            model_kwargs['discount'] = model.discount_type
        model_items.append((name, model.__class__, model_kwargs))
    
    n_models = len(model_items)
    gpu_fraction = {1: 0.90, 2: 0.45, 3: 0.30}.get(n_parallel, 0.40) if use_gpu else 0.0
    
    print(f"Fitting {n_models} models, {n_parallel} parallel")
    
    start_time = time.time()
    fitted_models = {}
    
    mp_context = mp.get_context('spawn')
    n_batches = (n_models + n_parallel - 1) // n_parallel
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * n_parallel
        batch_end = min(batch_start + n_parallel, n_models)
        batch_models = model_items[batch_start:batch_end]
        
        print(f"\nBatch {batch_idx + 1}/{n_batches}: {[m[0] for m in batch_models]}")
        
        worker_args = [
            (name, cls, kwargs, str(data_path), 0, gpu_fraction, fit_params)
            for name, cls, kwargs in batch_models
        ]
        
        with mp_context.Pool(len(batch_models)) as pool:
            batch_results = pool.map(_fit_single_model_worker, worker_args)
        
        for result in batch_results:
            model_path = save_dir / f"{result['name']}_fit.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result, f)
            
            model = result['model_class'](**result.get('model_kwargs', {}))
            model.posterior_samples = result['posterior_samples']
            model.subjects = result['subjects']
            model.subj_to_idx = result['subj_to_idx']
            model.n_subjects = result['n_subjects']
            model.fit_params = result['fit_params']
            
            fitter = ModelFitter(model)
            fitter.fit_result = result
            fitter.data = data
            
            fitted_models[result['name']] = fitter
            print(f"  Saved: {model_path.name}")
    
    data_path.unlink()
    
    total_time = (time.time() - start_time) / 60
    print(f"\nAll models fitted in {total_time:.1f} min")
    
    return fitted_models


def load_fitted_models(
    save_dir: str = './results/model_fits',
    load_data: bool = True
) -> Union[Dict[str, ModelFitter], tuple]:
    """Load previously fitted models from disk."""
    save_dir = Path(save_dir)
    
    if not save_dir.exists():
        raise FileNotFoundError(f"Directory not found: {save_dir}")
    
    fitted_models = {}
    pkl_files = list(save_dir.glob('*_fit.pkl'))
    
    if not pkl_files:
        raise FileNotFoundError(f"No fitted models found in {save_dir}")
    
    print(f"Loading {len(pkl_files)} models from {save_dir}...")
    
    data = None
    if load_data:
        data_path = save_dir / 'original_data.pkl'
        if data_path.exists():
            data = pd.read_pickle(data_path)
            print(f"Data: {len(data)} trials, {len(data['subj'].unique())} subjects")
    
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
        
        model_class = result['model_class']
        model_kwargs = result.get('model_kwargs', {})
        model = model_class(**model_kwargs)
        
        model.posterior_samples = result['posterior_samples']
        model.subjects = result['subjects']
        model.subj_to_idx = result['subj_to_idx']
        model.n_subjects = result['n_subjects']
        model.fit_params = result['fit_params']
        
        fitter = ModelFitter(model)
        fitter.fit_result = result
        fitter.data = data
        
        # ---- Robust name handling (parallel + serial compatibility) ----
        model_name = result.get("name")
        if model_name is None:
            model_name = pkl_path.stem.replace("_fit", "")
            result["name"] = model_name  # optional but helpful

        fitted_models[model_name] = fitter
        print(f"  {model_name}: {list(result['subject_level_params'].keys())}")

    
    if load_data:
        return fitted_models, data
    return fitted_models
