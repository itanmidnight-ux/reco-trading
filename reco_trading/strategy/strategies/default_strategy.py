"""
Default Strategy for Reco-Trading.
A simple strategy template using RSI and EMA indicators.
"""

import logging
from pandas import DataFrame

from reco_trading.strategy.interface import IStrategy


logger = logging.getLogger(__name__)


class DefaultStrategy(IStrategy):
    """
    Default strategy template.
    
    Buy when:
    - RSI is oversold (< 30)
    - EMA 9 crosses above EMA 21
    
    Sell when:
    - RSI is overbought (> 70)
    - EMA 9 crosses below EMA 21
    """
    
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.02,
    }
    
    stoploss = -0.10
    
    timeframe = "5m"
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for the strategy.
        
        Args:
            dataframe: DataFrame with OHLCV data
            metadata: Dictionary with pair info
            
        Returns:
            DataFrame with indicators
        """
        import pandas as pd
        
        dataframe["rsi"] = self._rsi(dataframe["close"], period=14)
        
        dataframe["ema_9"] = dataframe["close"].ewm(span=9, adjust=False).mean()
        dataframe["ema_21"] = dataframe["close"].ewm(span=21, adjust=False).mean()
        
        dataframe["ema_50"] = dataframe["close"].ewm(span=50, adjust=False).mean()
        dataframe["ema_200"] = dataframe["close"].ewm(span=200, adjust=False).mean()
        
        return dataframe
    
    def _rsi(self, close, period: int = 14) -> DataFrame:
        """Calculate RSI indicator."""
        import pandas as pd
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals.
        
        Args:
            dataframe: DataFrame with indicators
            metadata: Dictionary with pair info
            
        Returns:
            DataFrame with entry signals
        """
        dataframe["enter_long"] = 0
        
        rsi_oversold = dataframe["rsi"] < 30
        
        ema_cross_up = (
            (dataframe["ema_9"] > dataframe["ema_21"]) & 
            (dataframe["ema_9"].shift(1) <= dataframe["ema_21"].shift(1))
        )
        
        strong_uptrend = dataframe["ema_50"] > dataframe["ema_200"]
        
        dataframe.loc[
            (rsi_oversold | ema_cross_up) & strong_uptrend,
            "enter_long"
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals.
        
        Args:
            dataframe: DataFrame with indicators
            metadata: Dictionary with pair info
            
        Returns:
            DataFrame with exit signals
        """
        dataframe["exit_long"] = 0
        
        rsi_overbought = dataframe["rsi"] > 70
        
        ema_cross_down = (
            (dataframe["ema_9"] < dataframe["ema_21"]) & 
            (dataframe["ema_9"].shift(1) >= dataframe["ema_21"].shift(1))
        )
        
        dataframe.loc[
            rsi_overbought | ema_cross_down,
            "exit_long"
        ] = 1
        
        return dataframe
