"""
LIMA Preprocessing Pipeline - Stage 3: Subjective Reports Processing
=====================================================================
Processes subjective reports (anxiety, confidence ratings) and adds:
- Distance from safety calculations
- Threat level classifications
- Distance categorizations
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from config import PipelineConfig, default_config
from utils import PipelineLogger, load_pickle, save_outputs, create_processing_report


def classify_threat(prob: float) -> str:
    """Classify threat level based on attacking probability."""
    if prob <= 0.1:
        return 'low'
    elif 0.3 <= prob <= 0.7:
        return 'medium'
    return 'high'


def calculate_distance_from_safety(x: float, y: float) -> float:
    """Calculate Euclidean distance from safety."""
    return np.sqrt(x**2 + y**2)


def run_stage3(
    subjective_reports_path: Path = None,
    output_dir: Path = None,
    config: PipelineConfig = None
) -> Dict[str, Path]:
    """
    Run Stage 3: Subjective reports processing.
    
    Parameters:
        subjective_reports_path: Path to subjective reports pickle
        output_dir: Output directory
        config: Pipeline configuration
        
    Returns:
        Dictionary of output file paths
    """
    config = config or default_config
    output_dir = Path(output_dir or config.stage3_dir)
    
    logger = PipelineLogger("Stage3_SubjectiveReports", output_dir)
    logger.log("Starting Stage 3: Subjective Reports Processing")
    
    # Load data
    data = load_pickle(subjective_reports_path, logger)
    initial_rows = len(data)
    logger.log(f"Loaded {initial_rows} subjective report rows")
    
    # Step 1: Round attacking probability
    if 'attackingProb' in data.columns:
        data['threat'] = data['attackingProb'].round(1)
        logger.log("Rounded attackingProb to nearest 0.1")
    
    # Step 2: Calculate distance from safety
    if 'trialCookie_xPos' in data.columns and 'trialCookie_yPos' in data.columns:
        data['distanceFromSafety'] = data.apply(
            lambda row: calculate_distance_from_safety(
                row['trialCookie_xPos'], 
                row['trialCookie_yPos']
            ),
            axis=1
        ).round(0)
        logger.log("Calculated distanceFromSafety")
    
    
    # Step 3: Categorize distances
    if 'distanceFromSafety' in data.columns:
        data['distance'] = pd.cut(
            data['distanceFromSafety'],
            bins=[0, 5, 7, np.inf],
            labels=[0, 1, 2]
        )
        logger.log("Categorized distances")
    
    # Step 5: Add question type labels
    if 'questionType' in data.columns:
        question_labels = {5: 'anxiety', 6: 'confidence'}
        data['questionLabel'] = data['questionType'].map(question_labels)
        logger.log("Added question type labels")
    
    # Save outputs
    outputs = {}
    
    # Save full processed data
    outputs['processed'] = save_outputs(data, output_dir, 'subjective_reports_processed', logger=logger)
    
    # Create separate DataFrames for anxiety and confidence
    if 'questionType' in data.columns:
        anxiety_df = data[data['questionType'] == 5].copy()
        confidence_df = data[data['questionType'] == 6].copy()
        col_to_drop = ['distanceFromSafety','category','attackingProb','trialCookie_rewardValue','trialCookie_time','trialCookie_weight','trialCookie_xPos','trialCookie_yPos']
        
        if not anxiety_df.empty:
            anxiety_df=anxiety_df.drop(columns=col_to_drop)
            outputs['anxiety'] = save_outputs(anxiety_df, output_dir, 'anxiety_reports', logger=logger)
            logger.log(f"Saved {len(anxiety_df)} anxiety reports")
        
        if not confidence_df.empty:
            confidence_df=confidence_df.drop(columns=col_to_drop)
            outputs['confidence'] = save_outputs(confidence_df, output_dir, 'confidence_reports', logger=logger)
            logger.log(f"Saved {len(confidence_df)} confidence reports")
    
    # Create summary statistics
    summary_stats = []
    
    if all(col in data.columns for col in ['threat_level', 'distance_category', 'questionType', 'response']):
        for q_type in data['questionType'].unique():
            q_data = data[data['questionType'] == q_type]
            for threat in ['low', 'medium', 'high']:
                for dist in ['close', 'med', 'far']:
                    subset = q_data[
                        (q_data['threat_level'] == threat) & 
                        (q_data['distance_category'] == dist)
                    ]
                    if not subset.empty:
                        summary_stats.append({
                            'questionType': q_type,
                            'questionLabel': question_labels.get(q_type, 'unknown'),
                            'threat_level': threat,
                            'distance_category': dist,
                            'n': len(subset),
                            'mean_response': subset['response'].mean(),
                            'std_response': subset['response'].std(),
                            'min_response': subset['response'].min(),
                            'max_response': subset['response'].max()
                        })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            outputs['summary'] = save_outputs(summary_df, output_dir, 'response_summary', logger=logger)
    
    # Create processing report
    input_info = {
        'subjective_reports_path': str(subjective_reports_path),
        'initial_rows': initial_rows
    }
    
    output_info = {
        'final_rows': len(data),
        'participants': data['participantID'].nunique() if 'participantID' in data.columns else 'unknown',
        'output_directory': str(output_dir)
    }
    
    processing_stats = {
        'question_types': data['questionType'].value_counts().to_dict() if 'questionType' in data.columns else {},
        'threat_levels': data['threat_level'].value_counts().to_dict() if 'threat_level' in data.columns else {},
        'distance_categories': data['distance_category'].value_counts().to_dict() if 'distance_category' in data.columns else {},
        'response_range': {
            'min': data['response'].min() if 'response' in data.columns else None,
            'max': data['response'].max() if 'response' in data.columns else None
        }
    }
    
    create_processing_report(output_dir, "stage3", input_info, output_info, processing_stats, logger)
    logger.save_log()
    
    logger.log("Stage 3 complete!")
    
    return outputs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Stage 3: Subjective Reports Processing")
    parser.add_argument("--input", "-i", required=True, help="Path to subjective reports pickle")
    parser.add_argument("--output", "-o", help="Path to output directory")
    
    args = parser.parse_args()
    
    config = PipelineConfig()
    run_stage3(Path(args.input), args.output, config)
