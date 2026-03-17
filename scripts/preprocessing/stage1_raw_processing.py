"""
LIMA Preprocessing Pipeline - Stage 1: Raw Data Processing
===========================================================
Processes raw JSON files from participants and extracts:
- Experiment information
- Subjective reports
- Survey responses
- Trial data
"""

import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from config import PipelineConfig, default_config
from utils import PipelineLogger, save_outputs, create_processing_report


def process_experiment_info(data: Dict, participant_id: str) -> List[Dict]:
    """
    Extract experiment information from JSON data.
    
    Parameters:
        data: Parsed JSON data
        participant_id: Participant identifier
        
    Returns:
        List of experiment info dictionaries
    """
    experiment_info_list = []
    
    for key, value in data.items():
        if isinstance(value, dict) and 'condition' in value and 'experimentDate' in value:
            entry = value.copy()
            entry['experimentInfoKey'] = key
            entry['participantID'] = participant_id
            experiment_info_list.append(entry)
    
    return experiment_info_list


def process_subjective_reports(data: Dict, participant_id: str) -> List[Dict]:
    """
    Extract subjective reports from JSON data.
    
    Parameters:
        data: Parsed JSON data
        participant_id: Participant identifier
        
    Returns:
        List of subjective report dictionaries
    """
    subjective_reports = data.get('SubjectiveReports', {})
    reports_list = []
    
    for category, details in subjective_reports.items():
        entry = details.copy()
        entry['category'] = category
        
        # Flatten trialCookie
        trial_cookie = entry.pop('trialCookie', {})
        for k, v in trial_cookie.items():
            entry[f'trialCookie_{k}'] = v
        
        entry['participantID'] = participant_id
        reports_list.append(entry)
    
    return reports_list


def process_surveys(data: Dict, participant_id: str) -> List[Dict]:
    """
    Extract survey responses from JSON data.
    
    Parameters:
        data: Parsed JSON data
        participant_id: Participant identifier
        
    Returns:
        List of survey response dictionaries
    """
    surveys = data.get('Surveys', {})
    surveys_list = []
    
    for survey_name, survey_details in surveys.items():
        question_responses = survey_details.get('questionResponses', [])
        
        for idx, response in enumerate(question_responses):
            entry = response.copy()
            entry['surveyName'] = survey_name
            entry['itemIndex'] = idx  # Add item index for item-level analysis
            entry['participantID'] = participant_id
            surveys_list.append(entry)
    
    return surveys_list


def process_trial_data(data: Dict, participant_id: str) -> List[Dict]:
    """
    Extract trial data from JSON data.
    
    Parameters:
        data: Parsed JSON data
        participant_id: Participant identifier
        
    Returns:
        List of trial data dictionaries
    """
    trial_data = data.get('TrialData', {})
    trial_list = []
    
    for trial_name, trial_content in trial_data.items():
        trial_info = trial_content.get('trial_data', {})
        if not trial_info:
            continue
        
        entry = {'trialName': trial_name, 'participantID': participant_id}
        
        for key, value in trial_info.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    entry[f"{key}_{subkey}"] = subvalue
            else:
                entry[key] = value
        
        trial_list.append(entry)
    
    return trial_list


def run_stage1(
    config: PipelineConfig = None,
    raw_data_dir: str = None,
    output_dir: Path = None
) -> Dict[str, Path]:
    """
    Run Stage 1: Raw data processing.
    
    Parameters:
        config: Pipeline configuration
        raw_data_dir: Override for raw data directory
        output_dir: Override for output directory
        
    Returns:
        Dictionary of output file paths
    """
    config = config or default_config
    raw_data_dir = Path(raw_data_dir or config.raw_data_dir)
    output_dir = Path(output_dir or config.stage1_dir)
    
    logger = PipelineLogger("Stage1_RawProcessing", output_dir)
    logger.log("Starting Stage 1: Raw Data Processing")
    
    # Initialize collection lists
    experiment_info_list = []
    subjective_reports_list = []
    surveys_list = []
    trial_data_list = []
    
    # Find all JSON files
    json_files = list(raw_data_dir.glob('*.json'))
    logger.log(f"Found {len(json_files)} JSON files to process")
    
    processing_errors = []
    
    for json_file in json_files:
        filename = json_file.stem
        participant_id = filename.replace('participant_', '')
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Process each section
            exp_info = process_experiment_info(data, participant_id)
            if exp_info:
                experiment_info_list.extend(exp_info)
            else:
                logger.log(f"No ExperimentInfo for participant: {participant_id}", "warning")
            
            subjective_reports_list.extend(process_subjective_reports(data, participant_id))
            surveys_list.extend(process_surveys(data, participant_id))
            trial_data_list.extend(process_trial_data(data, participant_id))
            
        except Exception as e:
            logger.log(f"Error processing {json_file}: {e}", "error")
            processing_errors.append({'file': str(json_file), 'error': str(e)})
    
    # Create DataFrames
    experiment_info_df = pd.DataFrame(experiment_info_list) if experiment_info_list else pd.DataFrame()
    subjective_reports_df = pd.DataFrame(subjective_reports_list) if subjective_reports_list else pd.DataFrame()
    surveys_df = pd.DataFrame(surveys_list) if surveys_list else pd.DataFrame()
    trial_data_df = pd.DataFrame(trial_data_list) if trial_data_list else pd.DataFrame()
    
    # Filter trial data for participants with required number of trials
    valid_trial_data_df = trial_data_df.copy()
    if not trial_data_df.empty and 'participantID' in trial_data_df.columns:
        trial_counts = trial_data_df.groupby('participantID').size()
        valid_participants = trial_counts[trial_counts == config.required_trials].index
        valid_trial_data_df = trial_data_df[trial_data_df['participantID'].isin(valid_participants)]
        
        # Log participants not meeting criteria
        invalid_participants = trial_counts[trial_counts != config.required_trials]
        if len(invalid_participants) > 0:
            logger.log(f"Participants not meeting {config.required_trials} trials: {len(invalid_participants)}")
            for pid, count in invalid_participants.items():
                logger.log(f"  {pid}: {count} trials", "warning")
    
    # Save outputs
    outputs = {}
    
    for name, df in [
        ('experiment_info', experiment_info_df),
        ('subjective_reports', subjective_reports_df),
        ('surveys', surveys_df),
        ('trial_data', valid_trial_data_df)
    ]:
        saved = save_outputs(df, output_dir, name, logger=logger)
        outputs[name] = saved
    
    # Create processing report
    input_info = {
        'raw_data_directory': str(raw_data_dir),
        'json_files_found': len(json_files)
    }
    
    output_info = {
        'experiment_info_rows': len(experiment_info_df),
        'subjective_reports_rows': len(subjective_reports_df),
        'surveys_rows': len(surveys_df),
        'trial_data_rows': len(valid_trial_data_df),
        'output_directory': str(output_dir)
    }
    
    processing_stats = {
        'participants_processed': experiment_info_df['participantID'].nunique() if not experiment_info_df.empty else 0,
        'valid_participants': valid_trial_data_df['participantID'].nunique() if not valid_trial_data_df.empty else 0,
        'processing_errors': len(processing_errors),
        'required_trials': config.required_trials
    }
    
    create_processing_report(output_dir, "stage1", input_info, output_info, processing_stats, logger)
    logger.save_log()
    
    logger.log("Stage 1 complete!")
    
    return outputs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Stage 1: Raw Data Processing")
    parser.add_argument("--input", "-i", required=True, help="Path to raw data directory")
    parser.add_argument("--output", "-o", help="Path to output directory")
    
    args = parser.parse_args()
    
    config = PipelineConfig()
    run_stage1(config, args.input, args.output)
