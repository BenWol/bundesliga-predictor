"""
Ensemble strategies for combining model predictions.

This module contains various ensemble strategies that have been tested
and validated through backtesting.
"""

# Original ensemble
from ..ensemble import ConsensusEnsemble

# V2 ensembles (validated improvements)
from ..ensemble_v2 import (
    SimpleTendencyEnsemble,
    TendencyExpertEnsemble,
    TendencyConsensusEnsemble,
    HybridEnsembleV2,
    RecommendedEnsemble,
)

# Experimental ensembles
from .experimental import (
    OptimizedConsensusEnsemble,
    HybridEnsemble,
    AdaptiveScorelineEnsemble,
    BayesianOptimalEnsemble,
    AggressiveScorelineEnsemble,
    UltimateTendencyEnsemble,
    SuperConsensusEnsemble,
    MaxPointsEnsemble,
)

__all__ = [
    # Original
    "ConsensusEnsemble",
    # V2 (validated)
    "SimpleTendencyEnsemble",
    "TendencyExpertEnsemble",
    "TendencyConsensusEnsemble",
    "HybridEnsembleV2",
    "RecommendedEnsemble",
    # Experimental
    "OptimizedConsensusEnsemble",
    "HybridEnsemble",
    "AdaptiveScorelineEnsemble",
    "BayesianOptimalEnsemble",
    "AggressiveScorelineEnsemble",
    "UltimateTendencyEnsemble",
    "SuperConsensusEnsemble",
    "MaxPointsEnsemble",
]
