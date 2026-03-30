"""
IStrategy interface for Reco-Trading.
Defines the base class for all trading strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from pandas import DataFrame

from reco_trading.constants import Config
from reco_trading.enums import SignalType, RunMode


logger = logging.getLogger(__name__)


class IStrategy(ABC):
    """
    Base class for Reco-Trading strategies.
    
    All custom strategies must inherit from this class and implement:
    - populate_indicators()
    - populate_entry_trend()
    - populate_exit_trend()
    
    Attributes:
        minimal_roi: Dictionary of minimal ROI thresholds (minutes -> profit ratio)
        stoploss: Stop loss value as negative ratio (e.g., -0.10 for 10%)
        timeframe: Trading timeframe (e.g., "5m", "1h")
        max_open_trades: Maximum number of concurrent open trades
        trailing_stop: Enable trailing stop
        trailing_stop_positive: Trailing stop offset
        use_exit_signal: Use exit signals
        exit_profit_only: Exit only on profit
        order_types: Dictionary of order types
        order_time_in_force: Dictionary of time in force
    """
    
    INTERFACE_VERSION: int = 1
    
    minimal_roi: dict = {}
    stoploss: float = -0.10
    timeframe: str = "5m"
    max_open_trades: int = 3
    
    trailing_stop: bool = False
    trailing_stop_positive: float | None = None
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached: bool = False
    
    use_exit_signal: bool = True
    exit_profit_only: bool = False
    exit_profit_offset: float = 0.0
    ignore_roi_if_entry_signal: bool = False
    
    order_types: dict = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
    }
    
    order_time_in_force: dict = {
        "entry": "GTC",
        "exit": "GTC",
    }
    
    process_only_new_candles: bool = True
    startup_candle_count: int = 0
    
    protections: list = []
    
    plot_config: dict = {}
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dp = None
        self.wallets = None
        self._last_candle: dict[str, Any] = {}
        
    @property
    def name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__
    
    @property
    def runmode(self) -> RunMode:
        """Get current run mode."""
        mode = self.config.get("runmode", RunMode.DRY_RUN)
        if isinstance(mode, str):
            return RunMode(mode)
        return mode
    
    @abstractmethod
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators that will be used in the strategy.
        
        This method must be implemented by all strategies.
        Add your indicator calculations here.
        
        Args:
            dataframe: DataFrame with OHLCV data
            metadata: Dictionary with pair information
            
        Returns:
            DataFrame with added indicators
        """
        pass
    
    @abstractmethod
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals for the strategy.
        
        This method must be implemented by all strategies.
        Add your entry signal logic here.
        
        Args:
            dataframe: DataFrame with indicators
            metadata: Dictionary with pair information
            
        Returns:
            DataFrame with entry signals (column 'enter_long' or 'enter_tag')
        """
        pass
    
    @abstractmethod
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals for the strategy.
        
        This method must be implemented by all strategies.
        Add your exit signal logic here.
        
        Args:
            dataframe: DataFrame with indicators
            metadata: Dictionary with pair information
            
        Returns:
            DataFrame with exit signals (column 'exit_long')
        """
        pass
    
    def bot_start(self) -> None:
        """
        Called when the bot starts.
        Override this method to perform initialization.
        """
        logger.info(f"Strategy {self.name} started")
    
    def bot_loop_start(self) -> None:
        """
        Called at the start of each bot iteration.
        Override this method for per-iteration logic.
        """
        pass
    
    def check_entry(self, pair: str, dataframe: DataFrame) -> tuple[str, float]:
        """
        Custom entry check logic.
        
        Args:
            pair: Trading pair
            dataframe: DataFrame with signals
            
        Returns:
            Tuple of (entry_reason, confidence)
        """
        return ("", 1.0)
    
    def check_exit(self, pair: str, trade: Any, dataframe: DataFrame) -> tuple[str, float]:
        """
        Custom exit check logic.
        
        Args:
            pair: Trading pair
            trade: Trade object
            dataframe: DataFrame with signals
            
        Returns:
            Tuple of (exit_reason, confidence)
        """
        return ("", 1.0)
    
    def get_entry_price(self, pair: str, side: str, sanity_check: bool = True) -> float | None:
        """
        Custom entry price calculation.
        
        Args:
            pair: Trading pair
            side: Entry side
            sanity_check: Perform sanity check
            
        Returns:
            Entry price or None
        """
        return None
    
    def get_exit_price(self, pair: str, side: str, trade: Any) -> float | None:
        """
        Custom exit price calculation.
        
        Args:
            pair: Trading pair
            side: Exit side
            trade: Trade object
            
        Returns:
            Exit price or None
        """
        return None
    
    def confirm_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        """
        Confirm entry order.
        
        Args:
            pair: Trading pair
            order_type: Order type
            amount: Order amount
            rate: Order rate
            time_in_force: Time in force
            
        Returns:
            True to confirm, False to cancel
        """
        return True
    
    def confirm_exit(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        """
        Confirm exit order.
        
        Args:
            pair: Trading pair
            order_type: Order type
            amount: Order amount
            rate: Order rate
            time_in_force: Time in force
            
        Returns:
            True to confirm, False to cancel
        """
        return True
    
    def version(self) -> str | None:
        """
        Get strategy version.
        
        Returns:
            Version string or None
        """
        return None
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} timeframe={self.timeframe}>"
