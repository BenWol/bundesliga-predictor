"""
Prediction models for Bundesliga match forecasting.
"""

from .base import BaseModel
from .multi_output import MultiOutputRegressionModel
from .classification import MultiClassClassificationModel
from .poisson import PoissonRegressionModel
from .naive_odds import NaiveOddsModel
from .context_aware import ContextAwareModel, DerbySpecialistModel, RelegationBattleModel

# Experimental models (under testing)
from .experimental import (
    GradientBoostingModel,
    BivariatePoissonModel,
    SmartOddsModel,
    TendencyFirstModel,
    ProbabilityMaxModel,
)

__all__ = [
    # Base
    "BaseModel",
    # Core models (validated)
    "MultiOutputRegressionModel",
    "MultiClassClassificationModel",
    "PoissonRegressionModel",
    "NaiveOddsModel",
    # Context-aware models
    "ContextAwareModel",
    "DerbySpecialistModel",
    "RelegationBattleModel",
    # Experimental models
    "GradientBoostingModel",
    "BivariatePoissonModel",
    "SmartOddsModel",
    "TendencyFirstModel",
    "ProbabilityMaxModel",
]
