"""
Base resolver for Reco-Trading.
Dynamic loading of strategies, exchanges, and other components.
Based on FreqTrade's resolver pattern.
"""

import importlib.util
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from reco_trading.constants import Config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IResolver(ABC):
    """Base class for resolvers."""

    object_type: type
    object_type_str: str
    user_subdir: str | None = None
    initial_search_path: Path | None = None

    @classmethod
    def build_search_paths(
        cls,
        config: Config,
        user_subdir: str | None = None,
        extra_dirs: list[str] | None = None,
    ) -> list[Path]:
        """Build search paths for the resolver."""
        abs_paths: list[Path] = []
        
        if cls.initial_search_path:
            abs_paths.append(cls.initial_search_path)

        if user_subdir:
            user_data = config.get("user_data_dir", Path.cwd() / "user_data")
            abs_paths.insert(0, user_data / user_subdir)

        if extra_dirs:
            for directory in extra_dirs:
                abs_paths.insert(0, Path(directory).resolve())

        return abs_paths

    @classmethod
    def _getattr_from_module(cls, module: Any, object_name: str) -> Any:
        """Get attribute from module."""
        return getattr(module, object_name, None)

    @classmethod
    def _load_object(cls, module: Any, object_name: str) -> type | None:
        """Load object from module."""
        obj = cls._getattr_from_module(module, object_name)
        if obj is None:
            return None
        if not isinstance(obj, type):
            return None
        return obj


class StrategyResolver(IResolver):
    """Resolver for trading strategies."""

    object_type = type
    object_type_str = "strategy"
    user_subdir = "strategies"
    initial_search_path = Path(__file__).parent.parent / "strategy" / "strategies"

    @classmethod
    def resolve(
        cls,
        config: Config,
        strategy_name: str,
        extra_dirs: list[str] | None = None,
    ) -> type:
        """Resolve a strategy by name."""
        from reco_trading.strategy.interface import IStrategy

        search_paths = cls.build_search_paths(config, cls.user_subdir, extra_dirs)
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            for file_path in search_path.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                
                module = cls._load_module(file_path)
                if module is None:
                    continue
                
                obj = cls._load_object(module, strategy_name)
                if obj is not None and issubclass(obj, IStrategy):
                    logger.info(f"Loaded strategy {strategy_name} from {file_path}")
                    return obj

        raise ValueError(f"Strategy '{strategy_name}' not found in paths: {search_paths}")

    @classmethod
    def _load_module(cls, file_path: Path) -> Any:
        """Load module from file."""
        module_name = f"reco_strategy_{file_path.stem}"
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning(f"Error loading module {file_path}: {e}")
            return None
        
        return module

    @classmethod
    def list_available(cls, config: Config) -> list[str]:
        """List all available strategies."""
        search_paths = cls.build_search_paths(config, cls.user_subdir)
        
        strategies = []
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            for file_path in search_path.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                
                module = cls._load_module(file_path)
                if module is None:
                    continue
                
                for attr_name in dir(module):
                    if attr_name.startswith("_"):
                        continue
                    
                    obj = getattr(module, attr_name, None)
                    if obj and isinstance(obj, type) and hasattr(obj, "populate_indicators"):
                        strategies.append(attr_name)
        
        return sorted(set(strategies))


class ExchangeResolver(IResolver):
    """Resolver for exchange implementations."""

    object_type = type
    object_type_str = "exchange"
    user_subdir = "exchange"

    EXCHANGE_MAP = {
        "binance": "reco_trading.exchange.binance_client:BinanceClient",
        "bybit": "reco_trading.exchange.bybit_client:BybitClient",
    }

    @classmethod
    def resolve(
        cls,
        config: Config,
        exchange_name: str,
    ) -> type:
        """Resolve an exchange by name."""
        if exchange_name.lower() in cls.EXCHANGE_MAP:
            module_path = cls.EXCHANGE_MAP[exchange_name.lower()]
            return cls._load_from_module_path(module_path)
        
        raise ValueError(f"Exchange '{exchange_name}' not supported")

    @classmethod
    def _load_from_module_path(cls, module_path: str) -> type:
        """Load class from module path (e.g., 'module:ClassName')."""
        if ":" not in module_path:
            raise ValueError(f"Invalid module path format: {module_path}")
        
        module_name, class_name = module_path.split(":")
        
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load {module_path}: {e}")

    @classmethod
    def list_available(cls) -> list[str]:
        """List available exchanges."""
        return list(cls.EXCHANGE_MAP.keys())


class FreqAIModelResolver(IResolver):
    """Resolver for FreqAI models."""

    object_type = type
    object_type_str = "freqaimodel"
    user_subdir = "freqai"
    initial_search_path = Path(__file__).parent.parent / "freqai" / "models"

    @classmethod
    def resolve(
        cls,
        config: Config,
        model_name: str,
        extra_dirs: list[str] | None = None,
    ) -> type:
        """Resolve a FreqAI model by name."""
        search_paths = cls.build_search_paths(config, cls.user_subdir, extra_dirs)
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            for file_path in search_path.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                
                module = cls._load_module(file_path)
                if module is None:
                    continue
                
                obj = cls._load_object(module, model_name)
                if obj is not None:
                    logger.info(f"Loaded FreqAI model {model_name} from {file_path}")
                    return obj

        raise ValueError(f"FreqAI model '{model_name}' not found")

    @classmethod
    def _load_module(cls, file_path: Path) -> Any:
        """Load module from file."""
        module_name = f"reco_freqai_{file_path.stem}"
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception:
            return None
        
        return module

    @classmethod
    def list_available(cls, config: Config) -> list[str]:
        """List all available FreqAI models."""
        search_paths = cls.build_search_paths(config, cls.user_subdir)
        
        models = []
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            for file_path in search_path.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                
                module = cls._load_module(file_path)
                if module is None:
                    continue
                
                for attr_name in dir(module):
                    if attr_name.startswith("_"):
                        continue
                    
                    obj = getattr(module, attr_name, None)
                    if obj and isinstance(obj, type) and hasattr(obj, "train"):
                        models.append(attr_name)
        
        return sorted(set(models))


class PairListResolver(IResolver):
    """Resolver for pairlist handlers."""

    object_type = type
    object_type_str = "pairlist"
    user_subdir = "pairlists"

    PAIRLIST_MAP = {
        "StaticPairList": "reco_trading.plugins.pairlist:StaticPairList",
        "VolumePairList": "reco_trading.plugins.pairlist:VolumePairList",
        "PriceFilter": "reco_trading.plugins.pairlist:PriceFilter",
    }

    @classmethod
    def resolve(
        cls,
        config: Config,
        pairlist_name: str,
    ) -> type:
        """Resolve a pairlist by name."""
        if pairlist_name in cls.PAIRLIST_MAP:
            return cls._load_from_module_path(cls.PAIRLIST_MAP[pairlist_name])
        
        raise ValueError(f"PairList '{pairlist_name}' not supported")

    @classmethod
    def _load_from_module_path(cls, module_path: str) -> type:
        """Load class from module path."""
        if ":" not in module_path:
            raise ValueError(f"Invalid module path format: {module_path}")
        
        module_name, class_name = module_path.split(":")
        
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load {module_path}: {e}")

    @classmethod
    def list_available(cls) -> list[str]:
        """List available pairlists."""
        return list(cls.PAIRLIST_MAP.keys())


class ProtectionResolver(IResolver):
    """Resolver for protection handlers."""

    object_type = type
    object_type_str = "protection"
    user_subdir = "protections"

    PROTECTION_MAP = {
        "CooldownPeriod": "reco_trading.plugins.protections:CooldownPeriod",
        "StoplossGuard": "reco_trading.plugins.protections:StoplossGuard",
    }

    @classmethod
    def resolve(
        cls,
        config: Config,
        protection_name: str,
    ) -> type:
        """Resolve a protection by name."""
        if protection_name in cls.PROTECTION_MAP:
            return cls._load_from_module_path(cls.PROTECTION_MAP[protection_name])
        
        raise ValueError(f"Protection '{protection_name}' not supported")

    @classmethod
    def _load_from_module_path(cls, module_path: str) -> type:
        """Load class from module path."""
        if ":" not in module_path:
            raise ValueError(f"Invalid module path format: {module_path}")
        
        module_name, class_name = module_path.split(":")
        
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load {module_path}: {e}")

    @classmethod
    def list_available(cls) -> list[str]:
        """List available protections."""
        return list(cls.PROTECTION_MAP.keys())
