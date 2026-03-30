"""
FreqAI - Machine Learning module for Reco-Trading.
Provides ML-based prediction for trading strategies.
"""

from reco_trading.freqai.freqai_interface import FreqAI, FreqaiWrapper, IFreqaiModel
from reco_trading.freqai.data_kitchen import DataKitchen
from reco_trading.freqai.feature_engineering import FeatureEngineer, FeatureSelector
from reco_trading.freqai.training import FreqAITrainer, FreqAIPredictor, TrainingConfig, TrainingStats

__all__ = [
    "FreqAI",
    "FreqaiWrapper", 
    "IFreqaiModel",
    "DataKitchen",
    "FeatureEngineer",
    "FeatureSelector",
    "FreqAITrainer",
    "FreqAIPredictor",
    "TrainingConfig",
    "TrainingStats",
]
