"""
LIMA Preprocessing Pipeline Utilities
======================================
Common utility functions used across all pipeline stages.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime


class PipelineLogger:
    """Logger for pipeline stages with file and console output."""
    
    def __init__(self, stage_name: str, output_dir: Path):
        self.stage_name = stage_name
        self.output_dir = Path(output_dir)
        self.messages: List[str] = []
        
        # Set up logging
        self.logger = logging.getLogger(stage_name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
    
    def log(self, msg: str, level: str = "info"):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        self.messages.append(full_msg)
        
        if level == "info":
            self.logger.info(msg)
        elif level == "warning":
            self.logger.warning(msg)
        elif level == "error":
            self.logger.error(msg)
    
    def save_log(self, filename: str = None):
        """Save log messages to file."""
        if filename is None:
            filename = f"{self.stage_name}_log.txt"
        
        log_path = self.output_dir / filename
        with open(log_path, 'w') as f:
            f.write("\n".join(self.messages))
        self.log(f"Log saved to {log_path}")


def load_pickle(filepath: Union[str, Path], logger: Optional[PipelineLogger] = None) -> pd.DataFrame:
    """
    Load a pickle file and return the DataFrame.
    
    Parameters:
        filepath: Path to the pickle file
        logger: Optional logger instance
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    try:
        df = pd.read_pickle(filepath)
        if logger:
            logger.log(f"Loaded pickle file: {filepath}")
        return df
    except Exception as e:
        msg = f"Error loading pickle file at {filepath}: {e}"
        if logger:
            logger.log(msg, "error")
        raise FileNotFoundError(msg)


def save_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    base_name: str,
    save_csv: bool = True,
    save_pickle: bool = True,
    logger: Optional[PipelineLogger] = None,
    csv_kwargs: Optional[Dict] = None
) -> Dict[str, Path]:
    """
    Save DataFrame to both CSV and pickle formats.
    
    Parameters:
        df: DataFrame to save
        output_dir: Output directory
        base_name: Base name for output files
        save_csv: Whether to save CSV
        save_pickle: Whether to save pickle
        logger: Optional logger instance
        csv_kwargs: Additional arguments for to_csv
        
    Returns:
        Dictionary with paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    csv_kwargs = csv_kwargs or {}
    
    if save_csv:
        csv_path = output_dir / f"{base_name}.csv"
        # Handle complex columns for CSV export
        df_csv = prepare_for_csv(df)
        df_csv.to_csv(csv_path, index=False, **csv_kwargs)
        outputs['csv'] = csv_path
        if logger:
            logger.log(f"Saved CSV: {csv_path}")
    
    if save_pickle:
        pkl_path = output_dir / f"{base_name}.pkl"
        df.to_pickle(pkl_path)
        outputs['pickle'] = pkl_path
        if logger:
            logger.log(f"Saved pickle: {pkl_path}")
    
    return outputs


def prepare_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for CSV export by converting complex types to strings.
    
    Parameters:
        df: Input DataFrame
        
    Returns:
        DataFrame suitable for CSV export
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Check if column contains lists or dicts
        sample = df_copy[col].dropna().head(1)
        if len(sample) > 0:
            first_val = sample.iloc[0]
            if isinstance(first_val, (list, dict, np.ndarray)):
                df_copy[col] = df_copy[col].apply(
                    lambda x: str(x) if x is not None else None
                )
    
    return df_copy


def generate_summary_stats(df: pd.DataFrame, groupby_col: str = None) -> pd.DataFrame:
    """
    Generate summary statistics for a DataFrame.
    
    Parameters:
        df: Input DataFrame
        groupby_col: Optional column to group by
        
    Returns:
        DataFrame with summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if groupby_col and groupby_col in df.columns:
        stats = df.groupby(groupby_col)[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max'])
    else:
        stats = df[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max'])
    
    return stats


def validate_participant_data(
    df: pd.DataFrame, 
    participant_col: str = 'participantID',
    required_trials: int = 81
) -> Dict[str, Any]:
    """
    Validate participant data and return summary.
    
    Parameters:
        df: Input DataFrame
        participant_col: Column containing participant IDs
        required_trials: Expected number of trials per participant
        
    Returns:
        Dictionary with validation results
    """
    trial_counts = df.groupby(participant_col).size()
    
    validation = {
        'total_participants': len(trial_counts),
        'valid_participants': len(trial_counts[trial_counts == required_trials]),
        'invalid_participants': len(trial_counts[trial_counts != required_trials]),
        'trial_count_summary': {
            'min': trial_counts.min(),
            'max': trial_counts.max(),
            'mean': trial_counts.mean()
        },
        'participants_not_meeting_criteria': trial_counts[trial_counts != required_trials].to_dict()
    }
    
    return validation


def create_processing_report(
    output_dir: Path,
    stage_name: str,
    input_info: Dict[str, Any],
    output_info: Dict[str, Any],
    processing_stats: Dict[str, Any],
    logger: Optional[PipelineLogger] = None
) -> Path:
    """
    Create a processing report for a pipeline stage.
    
    Parameters:
        output_dir: Output directory
        stage_name: Name of the processing stage
        input_info: Information about inputs
        output_info: Information about outputs
        processing_stats: Processing statistics
        logger: Optional logger instance
        
    Returns:
        Path to the report file
    """
    report_path = Path(output_dir) / f"{stage_name}_report.txt"
    
    lines = [
        f"{'='*60}",
        f"LIMA PREPROCESSING PIPELINE - {stage_name.upper()}",
        f"{'='*60}",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "INPUT INFORMATION:",
        "-" * 40
    ]
    
    for key, value in input_info.items():
        lines.append(f"  {key}: {value}")
    
    lines.extend([
        "",
        "OUTPUT INFORMATION:",
        "-" * 40
    ])
    
    for key, value in output_info.items():
        lines.append(f"  {key}: {value}")
    
    lines.extend([
        "",
        "PROCESSING STATISTICS:",
        "-" * 40
    ])
    
    for key, value in processing_stats.items():
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            for k, v in value.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append(f"  {key}: {value}")
    
    lines.append(f"\n{'='*60}")
    
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    
    if logger:
        logger.log(f"Processing report saved: {report_path}")
    
    return report_path
