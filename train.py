#!/usr/bin/env python3
"""
Train all models and show evaluation metrics.

Usage:
    uv run python train.py
"""

from bundesliga_predictor import BundesligaPredictor


def main():
    predictor = BundesligaPredictor()
    predictor.load_data()
    predictor.train_models()


if __name__ == "__main__":
    main()
