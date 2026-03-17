"""
LIMA Preprocessing Pipeline - Stage 4: Mental Health Data Processing
=====================================================================
Processes mental health survey data to produce:
- Item-level responses for each questionnaire
- Summed subscale and total scores
- Response time metrics

Supported questionnaires:
- DASS-21 (Depression Anxiety Stress Scales)
- AMI (Apathy Motivation Index)
- MFIS (Modified Fatigue Impact Scale)
- OASIS (Overall Anxiety Severity and Impairment Scale)
- PHQ-9 (Patient Health Questionnaire)
- STICSA (State-Trait Inventory for Cognitive and Somatic Anxiety)
- STAI (State-Trait Anxiety Inventory)
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from functools import reduce
import pandas as pd
import numpy as np

from config import PipelineConfig, default_config
from utils import PipelineLogger, load_pickle, save_outputs, create_processing_report


# =============================================================================
# Item-Level Processing Functions
# =============================================================================

def create_item_level_df(
    survey_df: pd.DataFrame,
    survey_name: str,
    logger: Optional[PipelineLogger] = None
) -> pd.DataFrame:
    """
    Create item-level DataFrame for a specific survey.
    
    Parameters:
        survey_df: Full surveys DataFrame
        survey_name: Name of the survey to extract
        logger: Optional logger
        
    Returns:
        DataFrame with one row per participant-item combination
    """
    # Filter for specific survey
    df = survey_df[survey_df['surveyName'] == survey_name].copy()
    
    if df.empty:
        if logger:
            logger.log(f"No data found for survey: {survey_name}", "warning")
        return pd.DataFrame()
    
    # Ensure item index is present
    if 'itemIndex' not in df.columns:
        df['itemIndex'] = df.groupby('participantID').cumcount()
    
    # Calculate response time if possible
    if 'responseTime' in df.columns and 'onsetTime' in df.columns:
        df['responseRT'] = df['responseTime'] - df['onsetTime']
    
    # Select and rename columns
    item_df = df[['participantID', 'itemIndex', 'response', 'onsetTime', 'responseTime']].copy()
    item_df.columns = ['participantID', 'itemIndex', 'response', 'onsetTime', 'responseTime']
    
    if 'responseRT' in df.columns:
        item_df['responseRT'] = df['responseRT'].values
    
    item_df['surveyName'] = survey_name
    
    if logger:
        logger.log(f"Created item-level data for {survey_name}: {len(item_df)} items")
    
    return item_df


def pivot_items_wide(
    item_df: pd.DataFrame,
    survey_name: str,
    prefix: str = None
) -> pd.DataFrame:
    """
    Pivot item-level data to wide format (one row per participant).
    
    Parameters:
        item_df: Item-level DataFrame
        survey_name: Survey name
        prefix: Optional prefix for column names (defaults to survey_name)
        
    Returns:
        Wide-format DataFrame
    """
    if item_df.empty:
        return pd.DataFrame()
    
    prefix = prefix or survey_name
    
    # Pivot responses
    wide_df = item_df.pivot(
        index='participantID',
        columns='itemIndex',
        values='response'
    ).reset_index()
    
    # Rename columns
    response_cols = {col: f"{prefix}_item_{col}" for col in wide_df.columns if col != 'participantID'}
    wide_df = wide_df.rename(columns=response_cols)
    
    return wide_df


# =============================================================================
# Score Calculation Functions
# =============================================================================

def score_dass21(df: pd.DataFrame, logger: Optional[PipelineLogger] = None) -> pd.DataFrame:
    """Calculate DASS-21 subscale scores."""
    survey_df = df[df['surveyName'] == 'DASS21'].copy()
    if survey_df.empty:
        return pd.DataFrame()
    
    survey_df['itemIndex'] = survey_df.groupby('participantID').cumcount()
    
    stress_items = [0, 5, 7, 10, 11, 13, 17]
    anxiety_items = [1, 3, 6, 8, 14, 18, 19]
    depression_items = [2, 4, 9, 12, 15, 16, 20]
    
    results = []
    for pid in survey_df['participantID'].unique():
        pdata = survey_df[survey_df['participantID'] == pid]
        
        stress = pdata[pdata['itemIndex'].isin(stress_items)]['response'].sum() * 2
        anxiety = pdata[pdata['itemIndex'].isin(anxiety_items)]['response'].sum() * 2
        depression = pdata[pdata['itemIndex'].isin(depression_items)]['response'].sum() * 2
        
        avg_rt = (pdata['responseTime'] - pdata['onsetTime']).mean() if 'responseTime' in pdata.columns else None
        
        results.append({
            'participantID': pid,
            'DASS21_Stress': stress,
            'DASS21_Anxiety': anxiety,
            'DASS21_Depression': depression,
            'DASS21_Total': stress + anxiety + depression,
            'DASS21_RT': avg_rt
        })
    
    if logger:
        logger.log(f"Calculated DASS-21 scores for {len(results)} participants")
    
    return pd.DataFrame(results)


def score_ami(df: pd.DataFrame, logger: Optional[PipelineLogger] = None) -> pd.DataFrame:
    """Calculate AMI subscale scores."""
    survey_df = df[df['surveyName'] == 'AMI'].copy()
    if survey_df.empty:
        return pd.DataFrame()
    
    survey_df['itemIndex'] = survey_df.groupby('participantID').cumcount()
    # Reverse score all items (0->4, 1->3, 2->2, 3->1, 4->0)
    survey_df['reversed_response'] = 4 - survey_df['response']
    
    behavioural_items = [4, 8, 9, 10, 11, 14]
    social_items = [1, 2, 3, 7, 13, 16]
    emotional_items = [0, 5, 6, 12, 15, 17]
    
    results = []
    for pid in survey_df['participantID'].unique():
        pdata = survey_df[survey_df['participantID'] == pid]
        
        behavioural = pdata[pdata['itemIndex'].isin(behavioural_items)]['reversed_response'].sum()
        social = pdata[pdata['itemIndex'].isin(social_items)]['reversed_response'].sum()
        emotional = pdata[pdata['itemIndex'].isin(emotional_items)]['reversed_response'].sum()
        
        avg_rt = (pdata['responseTime'] - pdata['onsetTime']).mean() if 'responseTime' in pdata.columns else None
        
        results.append({
            'participantID': pid,
            'AMI_Behavioural': behavioural,
            'AMI_Social': social,
            'AMI_Emotional': emotional,
            'AMI_Total': behavioural + social + emotional,
            'AMI_RT': avg_rt
        })
    
    if logger:
        logger.log(f"Calculated AMI scores for {len(results)} participants")
    
    return pd.DataFrame(results)


def score_mfis(df: pd.DataFrame, logger: Optional[PipelineLogger] = None) -> pd.DataFrame:
    """Calculate MFIS subscale scores."""
    survey_df = df[df['surveyName'] == 'MFIS'].copy()
    if survey_df.empty:
        return pd.DataFrame()
    
    survey_df['itemIndex'] = survey_df.groupby('participantID').cumcount()
    
    physical_items = [3, 5, 6, 9, 12, 13, 16, 19, 20]
    cognitive_items = [0, 1, 2, 4, 10, 11, 14, 15, 17, 18]
    psychosocial_items = [7, 8]
    
    results = []
    for pid in survey_df['participantID'].unique():
        pdata = survey_df[survey_df['participantID'] == pid]
        
        physical = pdata[pdata['itemIndex'].isin(physical_items)]['response'].sum()
        cognitive = pdata[pdata['itemIndex'].isin(cognitive_items)]['response'].sum()
        psychosocial = pdata[pdata['itemIndex'].isin(psychosocial_items)]['response'].sum()
        
        avg_rt = (pdata['responseTime'] - pdata['onsetTime']).mean() if 'responseTime' in pdata.columns else None
        
        results.append({
            'participantID': pid,
            'MFIS_Physical': physical,
            'MFIS_Cognitive': cognitive,
            'MFIS_Psychosocial': psychosocial,
            'MFIS_Total': physical + cognitive + psychosocial,
            'MFIS_RT': avg_rt
        })
    
    if logger:
        logger.log(f"Calculated MFIS scores for {len(results)} participants")
    
    return pd.DataFrame(results)


def score_oasis(df: pd.DataFrame, logger: Optional[PipelineLogger] = None) -> pd.DataFrame:
    """Calculate OASIS total score."""
    survey_df = df[df['surveyName'] == 'OASIS'].copy()
    if survey_df.empty:
        return pd.DataFrame()
    
    results = []
    for pid in survey_df['participantID'].unique():
        pdata = survey_df[survey_df['participantID'] == pid]
        
        total = pdata['response'].sum()
        avg_rt = (pdata['responseTime'] - pdata['onsetTime']).mean() if 'responseTime' in pdata.columns else None
        
        results.append({
            'participantID': pid,
            'OASIS_Total': total,
            'OASIS_RT': avg_rt
        })
    
    if logger:
        logger.log(f"Calculated OASIS scores for {len(results)} participants")
    
    return pd.DataFrame(results)


def score_phq9(df: pd.DataFrame, logger: Optional[PipelineLogger] = None) -> pd.DataFrame:
    """Calculate PHQ-9 total score."""
    survey_df = df[df['surveyName'] == 'PHQ9'].copy()
    if survey_df.empty:
        return pd.DataFrame()
    
    results = []
    for pid in survey_df['participantID'].unique():
        pdata = survey_df[survey_df['participantID'] == pid]
        
        total = pdata['response'].sum()
        avg_rt = (pdata['responseTime'] - pdata['onsetTime']).mean() if 'responseTime' in pdata.columns else None
        
        results.append({
            'participantID': pid,
            'PHQ9_Total': total,
            'PHQ9_RT': avg_rt
        })
    
    if logger:
        logger.log(f"Calculated PHQ-9 scores for {len(results)} participants")
    
    return pd.DataFrame(results)


def score_sticsa(df: pd.DataFrame, logger: Optional[PipelineLogger] = None) -> pd.DataFrame:
    """Calculate STICSA total score."""
    survey_df = df[df['surveyName'] == 'STICSA'].copy()
    if survey_df.empty:
        return pd.DataFrame()
    
    # Add 1 to responses (scale adjustment)
    survey_df['adjusted_response'] = survey_df['response'] + 1
    
    results = []
    for pid in survey_df['participantID'].unique():
        pdata = survey_df[survey_df['participantID'] == pid]
        
        total = pdata['adjusted_response'].sum()
        avg_rt = (pdata['responseTime'] - pdata['onsetTime']).mean() if 'responseTime' in pdata.columns else None
        
        results.append({
            'participantID': pid,
            'STICSA_Total': total,
            'STICSA_RT': avg_rt
        })
    
    if logger:
        logger.log(f"Calculated STICSA scores for {len(results)} participants")
    
    return pd.DataFrame(results)


def score_stai(df: pd.DataFrame, logger: Optional[PipelineLogger] = None) -> pd.DataFrame:
    """Calculate STAI State and Trait scores."""
    direct_items = [2, 3, 5, 6, 8, 9, 11, 12, 13, 16, 17]
    reverse_items = [0, 1, 4, 7, 10, 14, 15, 18, 19]
    
    def score_response(idx, response):
        # Add 1 to convert 0-3 scale to 1-4
        response = response + 1
        if idx in direct_items:
            return response
        elif idx in reverse_items:
            return 5 - response
        return response
    
    results = []
    
    for scale_name, survey_name in [('State', 'STAI'), ('Trait', 'STAI_2')]:
        survey_df = df[df['surveyName'] == survey_name].copy()
        if survey_df.empty:
            continue
        
        survey_df['itemIndex'] = survey_df.groupby('participantID').cumcount()
        survey_df['scored_response'] = survey_df.apply(
            lambda row: score_response(row['itemIndex'], row['response']),
            axis=1
        )
        
        for pid in survey_df['participantID'].unique():
            pdata = survey_df[survey_df['participantID'] == pid]
            
            total = pdata['scored_response'].sum()
            avg_rt = (pdata['responseTime'] - pdata['onsetTime']).mean() if 'responseTime' in pdata.columns else None
            
            # Find or create participant entry
            existing = [r for r in results if r['participantID'] == pid]
            if existing:
                existing[0][f'STAI_{scale_name}'] = total
                existing[0][f'STAI_{scale_name}_RT'] = avg_rt
            else:
                results.append({
                    'participantID': pid,
                    f'STAI_{scale_name}': total,
                    f'STAI_{scale_name}_RT': avg_rt
                })
    
    if logger:
        logger.log(f"Calculated STAI scores for {len(results)} participants")
    
    return pd.DataFrame(results)


# =============================================================================
# Main Processing Function
# =============================================================================

def run_stage4(
    surveys_path: Path = None,
    output_dir: Path = None,
    config: PipelineConfig = None
) -> Dict[str, Path]:
    """
    Run Stage 4: Mental health data processing.
    
    Produces two main outputs:
    - Item-level DataFrame with individual item responses
    - Scored DataFrame with summed subscale/total scores
    
    Parameters:
        surveys_path: Path to surveys pickle
        output_dir: Output directory
        config: Pipeline configuration
        
    Returns:
        Dictionary of output file paths
    """
    config = config or default_config
    output_dir = Path(output_dir or config.stage4_dir)
    
    logger = PipelineLogger("Stage4_MentalHealth", output_dir)
    logger.log("Starting Stage 4: Mental Health Data Processing")
    
    # Load data
    surveys_df = load_pickle(surveys_path, logger)
    logger.log(f"Loaded {len(surveys_df)} survey responses")
    
    # Get list of available surveys
    available_surveys = surveys_df['surveyName'].unique().tolist()
    logger.log(f"Available surveys: {available_surveys}")
    
    outputs = {}
    
    # ==========================================================================
    # Part 1: Create Item-Level DataFrames
    # ==========================================================================
    logger.log("Creating item-level DataFrames...")
    
    item_dfs = []
    item_wide_dfs = []
    
    for survey in available_surveys:
        # Long format (one row per item)
        item_df = create_item_level_df(surveys_df, survey, logger)
        if not item_df.empty:
            item_dfs.append(item_df)
            
            # Wide format (one row per participant)
            wide_df = pivot_items_wide(item_df, survey)
            if not wide_df.empty:
                item_wide_dfs.append(wide_df)
    
    # Combine all item-level data (long format)
    if item_dfs:
        all_items_long = pd.concat(item_dfs, ignore_index=True)
        outputs['items_long'] = save_outputs(all_items_long, output_dir, 'mental_health_items_long', logger=logger)
        logger.log(f"Created combined item-level data (long): {len(all_items_long)} rows")
    
    # Combine all item-level data (wide format)
    if item_wide_dfs:
        all_items_wide = reduce(
            lambda left, right: pd.merge(left, right, on='participantID', how='outer'),
            item_wide_dfs
        )
        outputs['items_wide'] = save_outputs(all_items_wide, output_dir, 'mental_health_items_wide', logger=logger)
        logger.log(f"Created combined item-level data (wide): {len(all_items_wide)} participants")
    
    # Save individual survey item-level files
    for survey in available_surveys:
        item_df = create_item_level_df(surveys_df, survey)
        if not item_df.empty:
            save_outputs(item_df, output_dir, f'{survey}_items', logger=logger)
    
    # ==========================================================================
    # Part 2: Calculate Summed Scores
    # ==========================================================================
    logger.log("Calculating summed scores...")
    
    score_functions = [
        ('DASS21', score_dass21),
        ('AMI', score_ami),
        ('MFIS', score_mfis),
        ('OASIS', score_oasis),
        ('PHQ9', score_phq9),
        ('STICSA', score_sticsa),
        ('STAI', score_stai)
    ]
    
    score_dfs = []
    for name, func in score_functions:
        try:
            score_df = func(surveys_df, logger)
            if not score_df.empty:
                score_dfs.append(score_df)
                # Save individual score file
                save_outputs(score_df, output_dir, f'{name}_scores', logger=logger)
        except Exception as e:
            logger.log(f"Error calculating {name} scores: {e}", "error")
    
    # Merge all scores
    if score_dfs:
        # Convert participantID to string for consistent merging
        for df in score_dfs:
            df['participantID'] = df['participantID'].astype(str)
        
        all_scores = reduce(
            lambda left, right: pd.merge(left, right, on='participantID', how='outer'),
            score_dfs
        )
        
        outputs['scores'] = save_outputs(all_scores, output_dir, 'mental_health_scores', logger=logger)
        logger.log(f"Created combined scores data: {len(all_scores)} participants")
    
    # ==========================================================================
    # Part 3: Create Summary Statistics
    # ==========================================================================
    if score_dfs:
        # Score descriptive statistics
        score_cols = [col for col in all_scores.columns if col != 'participantID' and not col.endswith('_RT')]
        if score_cols:
            score_stats = all_scores[score_cols].describe()
            save_outputs(score_stats.reset_index(), output_dir, 'score_statistics', 
                        save_pickle=False, logger=logger)
    
    # Create processing report
    input_info = {
        'surveys_path': str(surveys_path),
        'total_responses': len(surveys_df)
    }
    
    output_info = {
        'surveys_processed': available_surveys,
        'participants_with_items': all_items_wide['participantID'].nunique() if item_wide_dfs else 0,
        'participants_with_scores': len(all_scores) if score_dfs else 0,
        'output_directory': str(output_dir)
    }
    
    processing_stats = {
        'items_per_survey': {
            survey: len(surveys_df[surveys_df['surveyName'] == survey]['participantID'].unique())
            for survey in available_surveys
        },
        'score_columns': list(all_scores.columns) if score_dfs else []
    }
    
    create_processing_report(output_dir, "stage4", input_info, output_info, processing_stats, logger)
    logger.save_log()
    
    logger.log("Stage 4 complete!")
    
    return outputs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Stage 4: Mental Health Data Processing")
    parser.add_argument("--input", "-i", required=True, help="Path to surveys pickle")
    parser.add_argument("--output", "-o", help="Path to output directory")
    
    args = parser.parse_args()
    
    config = PipelineConfig()
    run_stage4(Path(args.input), args.output, config)
