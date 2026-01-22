"""
Bundesliga Match Predictor Package

A modular, OOP-based framework for predicting Bundesliga match scores
using multiple ML models and ensemble strategies.

Usage:
    from bundesliga_predictor import BundesligaPredictor

    predictor = BundesligaPredictor()
    predictions = predictor.predict_next_matchday()

Pipeline usage:
    from bundesliga_predictor import fetch_data, run_backtest, run_predict, submit_kicktipp

    fetch_data()
    results = run_backtest()
    predictions = run_predict()
    submit_kicktipp(predictions, use_model=False)
"""

from .predictor import BundesligaPredictor
from .models.base import BaseModel
from .models.multi_output import MultiOutputRegressionModel
from .models.classification import MultiClassClassificationModel
from .models.poisson import PoissonRegressionModel
from .models.naive_odds import NaiveOddsModel
from .ensemble import ConsensusEnsemble
from .pipeline import (
    fetch_data,
    run_backtest,
    run_predict,
    submit_kicktipp,
    print_results_table,
)

__version__ = "1.0.0"
__all__ = [
    "BundesligaPredictor",
    "BaseModel",
    "MultiOutputRegressionModel",
    "MultiClassClassificationModel",
    "PoissonRegressionModel",
    "NaiveOddsModel",
    "ConsensusEnsemble",
    # Pipeline functions
    "fetch_data",
    "run_backtest",
    "run_predict",
    "submit_kicktipp",
    "print_results_table",
]
