"""
Strategy Loader for Reco-Trading.
Dynamically loads and manages strategies.
"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from reco_trading.constants import Config
from reco_trading.strategy.interface import IStrategy


logger = logging.getLogger(__name__)


class StrategyLoader:
    """
    Strategy loader that dynamically loads strategies.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the strategy loader.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
        self._strategies: dict[str, type[IStrategy]] = {}
        self._loaded_strategy: IStrategy | None = None
        
    @property
    def strategy_path(self) -> Path:
        """Get strategy path from config."""
        strategy_path = self._config.get("strategy_path", "")
        if strategy_path:
            return Path(strategy_path)
        
        user_data = self._config.get("user_data_dir", Path.cwd() / "user_data")
        return user_data / "strategies"
    
    def load_strategy(self, strategy_name: str) -> IStrategy:
        """
        Load a strategy by name.
        
        Args:
            strategy_name: Name of the strategy class
            
        Returns:
            Instance of the strategy
            
        Raises:
            ValueError: If strategy not found or invalid
        """
        logger.info(f"Loading strategy: {strategy_name}")
        
        strategy_class = self._resolve_strategy(strategy_name)
        
        try:
            strategy_instance = strategy_class(self._config)
            self._loaded_strategy = strategy_instance
            logger.info(f"Strategy {strategy_name} loaded successfully")
            return strategy_instance
        except Exception as e:
            raise ValueError(f"Failed to instantiate strategy {strategy_name}: {e}")
    
    def _resolve_strategy(self, strategy_name: str) -> type[IStrategy]:
        """
        Resolve strategy class from name.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Strategy class
        """
        if strategy_name in self._strategies:
            return self._strategies[strategy_name]
        
        for path in sys.path:
            strategies_dir = Path(path) / "reco_trading" / "strategy" / "strategies"
            if strategies_dir.exists():
                strategy_file = strategies_dir / f"{strategy_name.lower()}.py"
                if strategy_file.exists():
                    return self._load_from_file(strategy_file, strategy_name)
        
        if self.strategy_path.exists():
            strategy_file = self.strategy_path / f"{strategy_name}.py"
            if strategy_file.exists():
                return self._load_from_file(strategy_file, strategy_name)
        
        raise ValueError(f"Strategy '{strategy_name}' not found")
    
    def _load_from_file(self, file_path: Path, class_name: str) -> type[IStrategy]:
        """
        Load strategy class from file.
        
        Args:
            file_path: Path to strategy file
            class_name: Strategy class name
            
        Returns:
            Strategy class
        """
        module_name = f"strategy_{file_path.stem}"
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load spec from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ValueError(f"Error loading strategy module: {e}")
        
        if not hasattr(module, class_name):
            raise ValueError(f"Class {class_name} not found in {file_path}")
        
        strategy_class = getattr(module, class_name)
        
        if not issubclass(strategy_class, IStrategy):
            raise ValueError(f"Strategy must inherit from IStrategy")
        
        self._strategies[class_name] = strategy_class
        return strategy_class
    
    def list_strategies(self) -> list[str]:
        """
        List available strategies.
        
        Returns:
            List of strategy names
        """
        strategies = []
        
        for path in sys.path:
            strategies_dir = Path(path) / "reco_trading" / "strategy" / "strategies"
            if strategies_dir.exists():
                for f in strategies_dir.glob("*.py"):
                    if not f.name.startswith("_"):
                        strategies.append(f.stem)
        
        if self.strategy_path.exists():
            for f in self.strategy_path.glob("*.py"):
                if not f.name.startswith("_"):
                    strategies.append(f.stem)
        
        return sorted(set(strategies))
    
    def get_current_strategy(self) -> IStrategy | None:
        """
        Get the currently loaded strategy.
        
        Returns:
            Current strategy instance or None
        """
        return self._loaded_strategy


def load_strategy(config: Config, strategy_name: str) -> IStrategy:
    """
    Factory function to load a strategy.
    
    Args:
        config: Configuration dictionary
        strategy_name: Strategy name
        
    Returns:
        Strategy instance
    """
    loader = StrategyLoader(config)
    return loader.load_strategy(strategy_name)


def list_available_strategies(config: Config) -> list[str]:
    """
    List all available strategies.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of strategy names
    """
    loader = StrategyLoader(config)
    return loader.list_strategies()
