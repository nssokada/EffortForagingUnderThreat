"""
Effort Foraging Under Threat — Preprocessing Pipeline
======================================================
Orchestrates all preprocessing stages for the experiment data.

Pipeline Stages:
1. Raw Data Processing: JSON → DataFrames
2. Trial Data Processing: Effort alignment and vigor metrics
3. Subjective Reports Processing: Anxiety/confidence probe classification
4. Mental Health Processing: Questionnaire item scoring (DASS-21, PHQ-9, OASIS, STAI-T, AMI, MFIS)
5. Data Filtering: Quality control exclusions and final merge

Usage:
    from pipeline import run_full_pipeline

    results = run_full_pipeline(
        raw_data_dir="./raw_data",
        output_dir="./processed_data"
    )
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from config import PipelineConfig
from utils import PipelineLogger, create_processing_report
from stage1_raw_processing import run_stage1
from stage2_trial_processing import run_stage2
from stage3_subjective_processing import run_stage3
from stage4_mental_health import run_stage4
from stage5_filtering import run_stage5


def run_full_pipeline(
    raw_data_dir: str = None,
    output_base_dir: str = None,
    config: PipelineConfig = None,
    stages: Optional[list] = None
) -> Dict[str, Any]:
    """
    Run the complete LIMA preprocessing pipeline.
    
    Parameters:
        raw_data_dir: Path to directory containing raw JSON files
        output_base_dir: Base path for all output directories
        config: Pipeline configuration (uses defaults if not provided)
        stages: List of stages to run (default: all stages 1-6)
        
    Returns:
        Dictionary containing output paths for each stage
    """
    # Initialize configuration
    if config is None:
        config = PipelineConfig()
    
    if raw_data_dir:
        config.raw_data_dir = raw_data_dir
    if output_base_dir:
        config.output_base_dir = output_base_dir
    
    # Default to all stages
    if stages is None:
        stages = [1, 2, 3, 4, 5, 6, 7]
    
    # Create base output directory
    base_output = Path(config.output_base_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Initialize results
    results = {
        'config': config,
        'timestamp': config.timestamp,
        'stages': {}
    }
    
    # Store intermediate paths
    stage_outputs = {}
    
    print("=" * 60)
    print("EFFORT FORAGING UNDER THREAT — PREPROCESSING PIPELINE")
    print(f"Timestamp: {config.timestamp}")
    print(f"Raw data: {config.raw_data_dir}")
    print(f"Output: {config.output_base_dir}")
    print("=" * 60)
    
    # ==========================================================================
    # Stage 1: Raw Data Processing
    # ==========================================================================
    if 1 in stages:
        print("\n" + "=" * 60)
        print("STAGE 1: Raw Data Processing")
        print("=" * 60)
        
        stage1_outputs = run_stage1(
            config=config,
            raw_data_dir=config.raw_data_dir,
            output_dir=config.stage1_dir
        )
        
        stage_outputs['stage1'] = stage1_outputs
        results['stages']['stage1'] = {
            'output_dir': str(config.stage1_dir),
            'outputs': {k: {kk: str(vv) for kk, vv in v.items()} for k, v in stage1_outputs.items()}
        }
    
    # ==========================================================================
    # Stage 2: Trial Data Processing
    # ==========================================================================
    if 2 in stages:
        print("\n" + "=" * 60)
        print("STAGE 2: Trial Data Processing")
        print("=" * 60)
        
        # Get paths from stage 1 or use defaults
        if 'stage1' in stage_outputs:
            trial_data_path = stage_outputs['stage1']['trial_data']['pickle']
            experiment_info_path = stage_outputs['stage1']['experiment_info']['pickle']
        else:
            trial_data_path = config.stage1_dir / 'trial_data.pkl'
            experiment_info_path = config.stage1_dir / 'experiment_info.pkl'
        
        stage2_outputs = run_stage2(
            trial_data_path=trial_data_path,
            experiment_info_path=experiment_info_path,
            output_dir=config.stage2_dir,
            config=config
        )
        
        stage_outputs['stage2'] = stage2_outputs
        results['stages']['stage2'] = {
            'output_dir': str(config.stage2_dir),
            'outputs': {k: {kk: str(vv) for kk, vv in v.items()} for k, v in stage2_outputs.items()}
        }
    
    # ==========================================================================
    # Stage 3: Subjective Reports Processing
    # ==========================================================================
    if 3 in stages:
        print("\n" + "=" * 60)
        print("STAGE 3: Subjective Reports Processing")
        print("=" * 60)
        
        if 'stage1' in stage_outputs:
            subjective_reports_path = stage_outputs['stage1']['subjective_reports']['pickle']
        else:
            subjective_reports_path = config.stage1_dir / 'subjective_reports.pkl'
        
        stage3_outputs = run_stage3(
            subjective_reports_path=subjective_reports_path,
            output_dir=config.stage3_dir,
            config=config
        )
        
        stage_outputs['stage3'] = stage3_outputs
        results['stages']['stage3'] = {
            'output_dir': str(config.stage3_dir),
            'outputs': {k: {kk: str(vv) for kk, vv in v.items()} for k, v in stage3_outputs.items()}
        }
    
    # ==========================================================================
    # Stage 4: Mental Health Processing
    # ==========================================================================
    if 4 in stages:
        print("\n" + "=" * 60)
        print("STAGE 4: Mental Health Data Processing")
        print("=" * 60)
        
        if 'stage1' in stage_outputs:
            surveys_path = stage_outputs['stage1']['surveys']['pickle']
        else:
            surveys_path = config.stage1_dir / 'surveys.pkl'
        
        stage4_outputs = run_stage4(
            surveys_path=surveys_path,
            output_dir=config.stage4_dir,
            config=config
        )
        
        stage_outputs['stage4'] = stage4_outputs
        results['stages']['stage4'] = {
            'output_dir': str(config.stage4_dir),
            'outputs': {k: {kk: str(vv) for kk, vv in v.items()} for k, v in stage4_outputs.items()}
        }
    
    # ==========================================================================
    # Stage 5: Participant Filtering & Harmonization
    # ==========================================================================
    if 5 in stages:
        print("\n" + "=" * 60)
        print("STAGE 5: Participant Filtering & Harmonization")
        print("=" * 60)

        # Get paths from previous stages (prefer stage outputs; fallback to default filenames)
        if 'stage2' in stage_outputs:
            s2 = stage_outputs['stage2']
            # behavior (analysis-ready / compact)
            if 'behavior' in s2:
                behavior_path = s2['behavior']['pickle']
            else:
                behavior_path = config.stage2_dir / 'behavior.pkl'

            # behavior_rich (trial-level / richer; used for outcome-based filter)
            if 'behavior_rich' in s2:
                behavior_rich_path = s2['behavior_rich']['pickle']
            elif 'processed' in s2:
                # Backward compatibility: older Stage 2 used 'processed' as the main trials table
                behavior_rich_path = s2['processed']['pickle']
            else:
                behavior_rich_path = config.stage2_dir / 'processed_trials.pkl'
        else:
            behavior_path = config.stage2_dir / 'behavior.pkl'
            behavior_rich_path = config.stage2_dir / 'processed_trials.pkl'

        if 'stage3' in stage_outputs:
            subjective_path = stage_outputs['stage3']['processed']['pickle']
        else:
            subjective_path = config.stage3_dir / 'subjective_reports_processed.pkl'

        if 'stage4' in stage_outputs:
            mental_health_path = stage_outputs['stage4']['scores']['pickle']
        else:
            mental_health_path = config.stage4_dir / 'mental_health_scores.pkl'

        stage5_outputs = run_stage5(
            behavior_path=behavior_path,
            behavior_rich_path=behavior_rich_path,
            subjective_reports_path=subjective_path,
            mental_health_path=mental_health_path,
            output_dir=config.stage5_dir,
            config=config
        )

        stage_outputs['stage5'] = stage5_outputs
        results['stages']['stage5'] = {
            'output_dir': str(config.stage5_dir),
            'outputs': {k: {kk: str(vv) for kk, vv in v.items()} for k, v in stage5_outputs.items()}
        }

    # ==========================================================================
    # Stage 6: Vigor Computation
    # ==========================================================================
    if 6 in stages:
        print("\n" + "=" * 60)
        print("STAGE 6: Vigor Computation")
        print("=" * 60)

        from compute_vigor import process_trial_vigor, process_epoch_metrics, compute_cell_means

        # Get stage5 output dir
        if 'stage5' in stage_outputs:
            s5_dir = Path(stage_outputs['stage5']['behavior_rich']['csv']).parent
        else:
            s5_dir = config.stage5_dir

        import pandas as pd
        beh_rich = pd.read_csv(s5_dir / "behavior_rich.csv", low_memory=False)

        # Exclude calibration outliers
        exclude = getattr(config, 'exclude_subjects', [])
        if exclude:
            beh_rich = beh_rich[~beh_rich['subj'].isin(exclude)]

        print(f"  {len(beh_rich)} trials, {beh_rich['subj'].nunique()} subjects")

        # Vigor output goes to results/stats/vigor_analysis/
        vigor_dir = Path("results/stats/vigor_analysis")
        vigor_dir.mkdir(parents=True, exist_ok=True)

        print("  Computing trial-level vigor...")
        trial_vigor = process_trial_vigor(beh_rich)
        trial_vigor.to_csv(s5_dir / "trial_vigor.csv", index=False)

        print("  Computing epoch metrics...")
        epoch_metrics = process_epoch_metrics(beh_rich)
        epoch_metrics.to_csv(vigor_dir / "vigor_metrics.csv", index=False)

        print("  Computing cell means...")
        cell_means = compute_cell_means(epoch_metrics)
        cell_means.to_csv(vigor_dir / "cell_means.csv", index=False)

        results['stages']['stage6'] = {
            'output_dir': str(vigor_dir),
            'outputs': {
                'trial_vigor': str(s5_dir / "trial_vigor.csv"),
                'vigor_metrics': str(vigor_dir / "vigor_metrics.csv"),
                'cell_means': str(vigor_dir / "cell_means.csv"),
            }
        }

        print(f"  trial_vigor: {len(trial_vigor)} trials → {s5_dir / 'trial_vigor.csv'}")
        print(f"  vigor_metrics: {len(epoch_metrics)} rows → {vigor_dir / 'vigor_metrics.csv'}")
        print(f"  cell_means: {len(cell_means)} cells → {vigor_dir / 'cell_means.csv'}")

    # ==========================================================================
    # Stage 7: Prepare Model Input
    # ==========================================================================
    if 7 in stages:
        print("\n" + "=" * 60)
        print("STAGE 7: Prepare Model Input")
        print("=" * 60)

        from prepare_model_input import main as prepare_model_input_main
        import sys

        # Get stage5 dir
        if 'stage5' in stage_outputs:
            s5_dir = Path(stage_outputs['stage5']['behavior_rich']['csv']).parent
        else:
            s5_dir = config.stage5_dir

        vigor_dir = Path("results/stats/vigor_analysis")
        model_input_dir = Path("data/model_input")

        exclude = getattr(config, 'exclude_subjects', [])
        exclude_args = []
        if exclude:
            exclude_args = ['--exclude'] + [str(e) for e in exclude]

        # Call with args
        old_argv = sys.argv
        sys.argv = ['prepare_model_input',
                     '--stage5_dir', str(s5_dir),
                     '--vigor_dir', str(vigor_dir),
                     '--output_dir', str(model_input_dir)] + exclude_args
        prepare_model_input_main()
        sys.argv = old_argv

        results['stages']['stage7'] = {
            'output_dir': str(model_input_dir),
            'outputs': {
                'choice_trials': str(model_input_dir / 'choice_trials.csv'),
                'vigor_cell_means': str(model_input_dir / 'vigor_cell_means.csv'),
                'subject_mapping': str(model_input_dir / 'subject_mapping.csv'),
            }
        }

    # ==========================================================================
    # Create Final Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    print("\nOutput Directories:")
    for stage_name, stage_data in results['stages'].items():
        print(f"  {stage_name}: {stage_data['output_dir']}")
    
    return results


def run_single_stage(
    stage: int,
    input_paths: Dict[str, Path] = None,
    output_dir: Path = None,
    config: PipelineConfig = None
) -> Dict[str, Any]:
    """
    Run a single pipeline stage.
    
    Parameters:
        stage: Stage number (1-5)
        input_paths: Dictionary of input file paths
        output_dir: Output directory
        config: Pipeline configuration

    Returns:
        Dictionary of output paths
    """
    config = config or PipelineConfig()
    input_paths = input_paths or {}

    stage_functions = {
        1: lambda: run_stage1(config, input_paths.get('raw_data_dir'), output_dir),
        2: lambda: run_stage2(
            input_paths.get('trial_data'),
            input_paths.get('experiment_info'),
            output_dir,
            config
        ),
        3: lambda: run_stage3(input_paths.get('subjective_reports'), output_dir, config),
        4: lambda: run_stage4(input_paths.get('surveys'), output_dir, config),
        5: lambda: run_stage5(
            input_paths.get('trial_data'),
            input_paths.get('subjective_reports'),
            input_paths.get('mental_health'),
            output_dir,
            config
        ),
    }

    if stage not in stage_functions:
        raise ValueError(f"Invalid stage number: {stage}. Must be 1-5.")
    
    return stage_functions[stage]()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Effort Foraging Under Threat Preprocessing Pipeline")
    parser.add_argument("--raw-data", "-r", required=True, help="Path to raw data directory")
    parser.add_argument("--output", "-o", help="Path to output base directory")
    parser.add_argument("--stages", "-s", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7],
                        help="Stages to run (default: all)")
    
    args = parser.parse_args()
    
    results = run_full_pipeline(
        raw_data_dir=args.raw_data,
        output_base_dir=args.output,
        stages=args.stages
    )
