"""
Effort Foraging Under Threat — Preprocessing Pipeline Configuration
====================================================================
Central configuration for all preprocessing parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class PipelineConfig:
    """Configuration for the LIMA preprocessing pipeline."""
    
    # Directory settings
    raw_data_dir: str = "./raw_data"
    output_base_dir: str = "./processed_data"
    
    # Processing parameters
    required_trials: int = 81
    press_count_threshold: float = 89.0
    max_trial_time: float = 25.0
    max_attacking_prob: float = 0.9
    max_calibration: float = 9.0
    
    # Strike detection parameters
    strike_threshold: float = 1.5
    strike_T: float = 0.75
    strike_min_dt: float = 0.5
    strike_max_dt: float = 1.0
    
    # Survey scoring configuration
    survey_configs: Dict = field(default_factory=lambda: {
        'DASS21': {
            'n_items': 21,
            'subscales': {
                'stress': [0, 5, 7, 10, 11, 13, 17],
                'anxiety': [1, 3, 6, 8, 14, 18, 19],
                'depression': [2, 4, 9, 12, 15, 16, 20]
            },
            'multiplier': 2,
            'reverse_items': []
        },
        'AMI': {
            'n_items': 18,
            'subscales': {
                'behavioural': [4, 8, 9, 10, 11, 14],
                'social': [1, 2, 3, 7, 13, 16],
                'emotional': [0, 5, 6, 12, 15, 17]
            },
            'reverse_all': True,
            'reverse_max': 4
        },
        'MFIS': {
            'n_items': 21,
            'subscales': {
                'physical': [3, 5, 6, 9, 12, 13, 16, 19, 20],
                'cognitive': [0, 1, 2, 4, 10, 11, 14, 15, 17, 18],
                'psychosocial': [7, 8]
            }
        },
        'OASIS': {
            'n_items': 5,
            'total_only': True
        },
        'PHQ9': {
            'n_items': 9,
            'total_only': True
        },
        'STICSA': {
            'n_items': 21,
            'total_only': True,
            'add_one': True
        },
        'STAI': {
            'n_items': 20,
            'scale_name': 'State',
            'direct_items': [2, 3, 5, 6, 8, 9, 11, 12, 13, 16, 17],
            'reverse_items': [0, 1, 4, 7, 10, 14, 15, 18, 19],
            'add_one': True
        },
        'STAI_2': {
            'n_items': 20,
            'scale_name': 'Trait',
            'direct_items': [2, 3, 5, 6, 8, 9, 11, 12, 13, 16, 17],
            'reverse_items': [0, 1, 4, 7, 10, 14, 15, 18, 19],
            'add_one': True
        }
    })
    
    # Timestamp for output directories
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    def get_stage_dir(self, stage: int, name: str) -> Path:
        """Get the output directory for a specific stage."""
        dir_path = Path(self.output_base_dir) / f"stage{stage}_{name}_{self.timestamp}"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @property
    def stage1_dir(self) -> Path:
        return self.get_stage_dir(1, "raw_processing")
    
    @property
    def stage2_dir(self) -> Path:
        return self.get_stage_dir(2, "trial_processing")
    
    @property
    def stage3_dir(self) -> Path:
        return self.get_stage_dir(3, "subjective_reports")
    
    @property
    def stage4_dir(self) -> Path:
        return self.get_stage_dir(4, "mental_health")
    
    @property
    def stage5_dir(self) -> Path:
        return self.get_stage_dir(5, "filtered_data")
    

# Default configuration instance
default_config = PipelineConfig()
