"""
FET Models: Factorized Effort-Threat Choice Models
"""

__version__ = '2.0.0'

from .base_model import BaseEffortThreatModel, configure_device

from .models import (
    FETExponential,
    FETHyperbolic,
    FETLinear,
    FETQuadratic,
    FETExponentialBias,
    ThreatOnly,
    EffortOnly,
    MODEL_REGISTRY,
    get_model,
    list_models,
)

from .fitter import (
    ModelFitter,
    fit_models_parallel,
    load_fitted_models,
)

from .ppc import (
    PosteriorPredictive,
    posterior_trial_predictions_df,
    compute_calibration_metrics,
    compute_waic,
    compute_loo,
    compare_models,
)

__all__ = [
    '__version__',
    'configure_device',
    'BaseEffortThreatModel',
    'FETExponential',
    'FETHyperbolic', 
    'FETLinear',
    'FETQuadratic',
    'FETExponentialBias',
    'ThreatOnly',
    'EffortOnly',
    'MODEL_REGISTRY',
    'get_model',
    'list_models',
    'ModelFitter',
    'fit_models_parallel',
    'load_fitted_models',
    'PosteriorPredictive',
    'posterior_trial_predictions_df',
    'compute_calibration_metrics',
    'compute_waic',
    'compute_loo',
    'compare_models',
]
