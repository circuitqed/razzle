"""Training components: self-play, trainer, and Vast.ai integration."""

from .selfplay import SelfPlay, GameRecord
from .trainer import Trainer, TrainingConfig
from .vastai import VastAI
from .metrics import (
    PolicyMetrics,
    ValueMetrics,
    CalibrationBucket,
    compute_policy_accuracy,
    compute_policy_metrics,
    compute_value_metrics,
    compute_value_calibration,
    compute_calibration_error,
    compute_pass_stats,
    compute_game_diversity,
    compute_batch_metrics,
)
