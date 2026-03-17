"""
Effort Foraging Under Threat — Preprocessing Pipeline
======================================================

A modular preprocessing pipeline for the experiment data.

Modules:
    - config: Configuration settings
    - utils: Common utility functions
    - stage1_raw_processing: JSON to DataFrame conversion
    - stage2_trial_processing: Trial data processing and vigor extraction
    - stage3_subjective_processing: Anxiety/confidence probe processing
    - stage4_mental_health: Questionnaire scoring (DASS-21, PHQ-9, OASIS, STAI-T, AMI, MFIS)
    - stage5_filtering: Quality control exclusions and final merge
    - pipeline: Main pipeline orchestration

Usage:
    from preprocessing import run_full_pipeline

    results = run_full_pipeline(
        raw_data_dir="./raw_data",
        output_base_dir="./processed_data"
    )
"""

from .config import PipelineConfig, default_config
from .pipeline import run_full_pipeline, run_single_stage
from .stage1_raw_processing import run_stage1
from .stage2_trial_processing import run_stage2
from .stage3_subjective_processing import run_stage3
from .stage4_mental_health import run_stage4
from .stage5_filtering import run_stage5

__version__ = "1.0.0"
__author__ = "Okada, Garg, Wise, Mobbs"

__all__ = [
    "PipelineConfig",
    "default_config",
    "run_full_pipeline",
    "run_single_stage",
    "run_stage1",
    "run_stage2",
    "run_stage3",
    "run_stage4",
    "run_stage5",
]
