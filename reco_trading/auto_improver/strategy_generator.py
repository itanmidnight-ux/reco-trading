"""
Strategy Generator Module for Auto-Improver.
Generates new strategy variants with adjusted parameters.
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StrategyVariant:
    """A generated strategy variant."""
    id: str
    name: str
    parameters: dict[str, Any]
    base_strategy: str
    generation: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "parameters": self.parameters,
            "base_strategy": self.base_strategy,
            "generation": self.generation,
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent_id,
        }

    def get_hash(self) -> str:
        """Get unique hash of strategy."""
        content = json.dumps(self.parameters, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class IndicatorConfig:
    """Configuration for trading indicators."""

    DEFAULT_CONFIGS = {
        "rsi": {
            "period": 14,
            "overbought": 70,
            "oversold": 30,
        },
        "macd": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
        },
        "sma": {
            "period": 20,
        },
        "ema": {
            "period": 20,
        },
        "bb": {
            "period": 20,
            "std_dev": 2,
        },
        "atr": {
            "period": 14,
        },
        "stochastic": {
            "k_period": 14,
            "d_period": 3,
        },
    }

    PARAM_RANGES = {
        "rsi": {
            "period": (7, 28),
            "overbought": (60, 90),
            "oversold": (10, 40),
        },
        "macd": {
            "fast_period": (5, 20),
            "slow_period": (20, 50),
            "signal_period": (5, 15),
        },
        "sma": {
            "period": (5, 100),
        },
        "ema": {
            "period": (5, 100),
        },
        "bb": {
            "period": (10, 50),
            "std_dev": (1, 3),
        },
        "atr": {
            "period": (7, 28),
        },
        "stochastic": {
            "k_period": (5, 28),
            "d_period": (2, 10),
        },
    }


class StrategyGenerator:
    """Generates new strategy variants."""

    def __init__(self, strategies_dir: Path | None = None):
        self.strategies_dir = strategies_dir or Path("./user_data/strategies")
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        
        self._variants: dict[str, StrategyVariant] = {}
        self._generation_count = 0

    def generate_variant(
        self,
        base_strategy: str,
        base_params: dict[str, Any],
        mutation_rate: float = 0.2,
        parent_id: str | None = None,
    ) -> StrategyVariant:
        """Generate a new strategy variant through mutation."""
        self._generation_count += 1
        
        new_params = self._mutate_parameters(base_params, mutation_rate)
        
        variant = StrategyVariant(
            id=self._generate_id(),
            name=f"{base_strategy}_gen{self._generation_count}",
            parameters=new_params,
            base_strategy=base_strategy,
            generation=self._generation_count,
            parent_id=parent_id,
        )
        
        self._variants[variant.id] = variant
        self._save_variant(variant)
        
        logger.info(f"Generated variant {variant.id}: {variant.name}")
        return variant

    def _mutate_parameters(
        self,
        params: dict[str, Any],
        mutation_rate: float,
    ) -> dict[str, Any]:
        """Mutate parameters with given rate."""
        new_params = params.copy()
        
        for key, value in params.items():
            if random.random() < mutation_rate:
                if isinstance(value, int):
                    new_params[key] = value + random.randint(-3, 3)
                elif isinstance(value, float):
                    change = value * random.uniform(-0.2, 0.2)
                    new_params[key] = max(0.001, value + change)
                elif isinstance(value, bool):
                    new_params[key] = not value
        
        return new_params

    def generate_crossover(
        self,
        parent1: StrategyVariant,
        parent2: StrategyVariant,
    ) -> StrategyVariant:
        """Generate new variant through crossover of two parents."""
        self._generation_count += 1
        
        params1 = parent1.parameters
        params2 = parent2.parameters
        
        common_keys = set(params1.keys()) & set(params2.keys())
        
        new_params = {}
        for key in common_keys:
            new_params[key] = random.choice([params1[key], params2[key]])
        
        for key in params1:
            if key not in new_params:
                new_params[key] = params1[key]
        for key in params2:
            if key not in new_params:
                new_params[key] = params2[key]
        
        variant = StrategyVariant(
            id=self._generate_id(),
            name=f"crossover_gen{self._generation_count}",
            parameters=new_params,
            base_strategy=parent1.base_strategy,
            generation=self._generation_count,
            parent_id=f"{parent1.id}+{parent2.id}",
        )
        
        self._variants[variant.id] = variant
        self._save_variant(variant)
        
        logger.info(f"Generated crossover {variant.id} from {parent1.id} x {parent2.id}")
        return variant

    def generate_from_template(
        self,
        template_name: str,
        target_metrics: dict[str, float] | None = None,
    ) -> StrategyVariant:
        """Generate strategy optimized for target metrics."""
        self._generation_count += 1
        
        base_templates = {
            "conservative": {
                "stop_loss": 0.02,
                "take_profit": 0.03,
                "rsi_period": 21,
                "rsi_overbought": 75,
                "rsi_oversold": 25,
                "ma_period": 50,
                "position_size": 0.1,
            },
            "aggressive": {
                "stop_loss": 0.05,
                "take_profit": 0.10,
                "rsi_period": 7,
                "rsi_overbought": 65,
                "rsi_oversold": 35,
                "ma_period": 10,
                "position_size": 0.3,
            },
            "balanced": {
                "stop_loss": 0.03,
                "take_profit": 0.06,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "ma_period": 20,
                "position_size": 0.2,
            },
        }
        
        params = base_templates.get(template_name, base_templates["balanced"]).copy()
        
        variant = StrategyVariant(
            id=self._generate_id(),
            name=f"{template_name}_gen{self._generation_count}",
            parameters=params,
            base_strategy=template_name,
            generation=self._generation_count,
        )
        
        self._variants[variant.id] = variant
        self._save_variant(variant)
        
        logger.info(f"Generated template variant {variant.id}: {template_name}")
        return variant

    def _generate_id(self) -> str:
        """Generate unique variant ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"var_{timestamp}_{random.randint(1000, 9999)}"

    def _save_variant(self, variant: StrategyVariant) -> None:
        """Save variant to file."""
        file_path = self.strategies_dir / f"{variant.id}.json"
        
        with open(file_path, "w") as f:
            json.dump(variant.to_dict(), f, indent=2)

    def load_variant(self, variant_id: str) -> StrategyVariant | None:
        """Load variant from file."""
        if variant_id in self._variants:
            return self._variants[variant_id]
        
        file_path = self.strategies_dir / f"{variant_id}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        variant = StrategyVariant(
            id=data["id"],
            name=data["name"],
            parameters=data["parameters"],
            base_strategy=data["base_strategy"],
            generation=data["generation"],
            created_at=datetime.fromisoformat(data["created_at"]),
            parent_id=data.get("parent_id"),
        )
        
        self._variants[variant_id] = variant
        return variant

    def list_variants(self, base_strategy: str | None = None) -> list[StrategyVariant]:
        """List all variants, optionally filtered by base strategy."""
        variants = list(self._variants.values())
        
        if base_strategy:
            variants = [v for v in variants if v.base_strategy == base_strategy]
        
        return sorted(variants, key=lambda v: v.created_at, reverse=True)

    def delete_variant(self, variant_id: str) -> bool:
        """Delete a variant."""
        if variant_id not in self._variants:
            return False
        
        file_path = self.strategies_dir / f"{variant_id}.json"
        
        if file_path.exists():
            file_path.unlink()
        
        del self._variants[variant_id]
        
        logger.info(f"Deleted variant {variant_id}")
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get generator statistics."""
        variants = list(self._variants.values())
        
        return {
            "total_variants": len(variants),
            "max_generation": max((v.generation for v in variants), default=0),
            "by_base_strategy": {
                bs: len([v for v in variants if v.base_strategy == bs])
                for bs in set(v.base_strategy for v in variants)
            },
        }
