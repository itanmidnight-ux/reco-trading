"""
Auto-Improver Main Orchestrator.
Coordinates all auto-improvement components for continuous strategy optimization.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reco_trading.auto_improver.data_collector import DataCollector, DataSet
from reco_trading.auto_improver.deployment_manager import DeploymentManager
from reco_trading.auto_improver.evaluator_engine import EvaluationResult, EvaluatorEngine
from reco_trading.auto_improver.strategy_generator import StrategyGenerator, StrategyVariant
from reco_trading.auto_improver.strategy_selector import StrategySelector
from reco_trading.auto_improver.training_engine import TrainingEngine

logger = logging.getLogger(__name__)


@dataclass
class ImprovementCycle:
    """Result of an improvement cycle."""
    cycle_id: str
    timestamp: datetime
    phase: str
    status: str
    details: dict[str, Any]
    error: str | None = None


class AutoImproverConfig:
    """Configuration for Auto-Improver."""

    def __init__(self):
        self.data_dir = Path("./user_data/data")
        self.models_dir = Path("./user_data/models")
        self.strategies_dir = Path("./user_data/strategies")
        
        self.training_interval_hours = 24
        self.evaluation_interval_hours = 6
        self.deployment_min_improvement = 0.1
        
        self.min_win_rate = 40.0
        self.min_sharpe_ratio = 0.5
        self.max_drawdown = 30.0
        self.min_trades = 10
        
        self.population_size = 10
        self.generations = 5
        
        self.auto_deploy = False
        self.rollback_on_failure = True


class AutoImprover:
    """
    Main orchestrator for autonomous strategy improvement.
    
    Coordinates:
    - Data Collection
    - Strategy Generation
    - Model Training
    - Evaluation
    - Selection
    - Deployment
    """

    def __init__(self, config: AutoImproverConfig | None = None):
        self.config = config or AutoImproverConfig()
        
        self.data_collector = DataCollector(data_dir=self.config.data_dir)
        self.training_engine = TrainingEngine(models_dir=self.config.models_dir)
        self.strategy_generator = StrategyGenerator(strategies_dir=self.config.strategies_dir)
        self.evaluator = EvaluatorEngine()
        self.selector = StrategySelector(
            min_win_rate=self.config.min_win_rate,
            min_sharpe_ratio=self.config.min_sharpe_ratio,
            max_drawdown=self.config.max_drawdown,
            min_trades=self.config.min_trades,
        )
        self.deployment_manager = DeploymentManager(
            active_strategy_path=self.config.strategies_dir / "active.json",
            backup_dir=self.config.strategies_dir / "backups",
        )
        
        self._running = False
        self._cycles: list[ImprovementCycle] = []
        self._current_dataset: DataSet | None = None
        
        self._training_task: asyncio.Task | None = None
        self._evaluation_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the auto-improvement process."""
        if self._running:
            logger.warning("Auto-Improver already running")
            return
        
        self._running = True
        logger.info("Auto-Improver started")
        
        await self._run_initial_cycle()
        
        self._training_task = asyncio.create_task(self._training_loop())
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())

    async def stop(self) -> None:
        """Stop the auto-improvement process."""
        self._running = False
        
        if self._training_task:
            self._training_task.cancel()
        
        if self._evaluation_task:
            self._evaluation_task.cancel()
        
        logger.info("Auto-Improver stopped")

    async def _run_initial_cycle(self) -> None:
        """Run initial data collection and baseline."""
        logger.info("Running initial improvement cycle")
        
        cycle = ImprovementCycle(
            cycle_id=self._generate_cycle_id(),
            timestamp=datetime.now(timezone.utc),
            phase="initial",
            status="running",
            details={},
        )
        
        try:
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
            
            dataset = await self.data_collector.collect_historical(
                symbols=symbols,
                timeframe="5m",
                days_back=30,
            )
            
            self._current_dataset = dataset
            
            cycle.phase = "data_collection"
            cycle.details = {"symbols": symbols, "data_points": len(dataset.data_points)}
            self._cycles.append(cycle)
            
            logger.info(f"Collected {len(dataset.data_points)} data points")
            
        except Exception as e:
            logger.exception(f"Initial cycle failed: {e}")
            cycle.status = "failed"
            cycle.error = str(e)

    async def _training_loop(self) -> None:
        """Periodic training loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.training_interval_hours * 3600)
                
                if not self._running:
                    break
                
                await self._run_training_cycle()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Training loop error: {e}")

    async def _evaluation_loop(self) -> None:
        """Periodic evaluation loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.evaluation_interval_hours * 3600)
                
                if not self._running:
                    break
                
                await self._run_evaluation_cycle()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Evaluation loop error: {e}")

    async def _run_training_cycle(self) -> None:
        """Run a training cycle."""
        cycle = ImprovementCycle(
            cycle_id=self._generate_cycle_id(),
            timestamp=datetime.now(timezone.utc),
            phase="training",
            status="running",
            details={},
        )
        
        try:
            logger.info("Running training cycle")
            
            if not self._current_dataset:
                cycle.status = "skipped"
                cycle.details = {"reason": "No dataset available"}
                self._cycles.append(cycle)
                return
            
            result = await self.training_engine.train_signal_predictor(
                dataset=self._current_dataset,
                model_name=f"signal_model_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            )
            
            cycle.status = "completed"
            cycle.details = {
                "model_name": result.model_name,
                "status": result.status,
                "metrics": result.metrics,
            }
            
            logger.info(f"Training cycle completed: {result.status}")
            
        except Exception as e:
            logger.exception(f"Training cycle failed: {e}")
            cycle.status = "failed"
            cycle.error = str(e)
        
        self._cycles.append(cycle)

    async def _run_evaluation_cycle(self) -> None:
        """Run a complete evaluation and selection cycle."""
        cycle = ImprovementCycle(
            cycle_id=self._generate_cycle_id(),
            timestamp=datetime.now(timezone.utc),
            phase="evaluation",
            status="running",
            details={},
        )
        
        try:
            logger.info("Running evaluation cycle")
            
            if not self._current_dataset:
                cycle.status = "skipped"
                cycle.details = {"reason": "No dataset available"}
                self._cycles.append(cycle)
                return
            
            current_variant = self.deployment_manager.get_current_strategy()
            
            variants = []
            
            for i in range(self.config.population_size):
                variant = self.strategy_generator.generate_variant(
                    base_strategy="DefaultStrategy",
                    base_params=self._get_default_params(),
                    mutation_rate=0.2,
                )
                variants.append(variant)
            
            results = await self.evaluator.evaluate_multiple(variants, self._current_dataset)
            
            variants_dict = {v.id: v for v in variants}
            
            current_evaluation = None
            if current_variant:
                current_results = self.evaluator.get_evaluation_history(current_variant.id)
                if current_results:
                    current_evaluation = current_results[-1]
            
            selection = self.selector.select_best(results, variants_dict)
            
            if selection.selected_variant and selection.evaluation:
                should_deploy, reason = await self.deployment_manager.deploy(
                    variant=selection.selected_variant,
                    evaluation=selection.evaluation,
                    current_variant=current_variant,
                    current_evaluation=current_evaluation,
                )
                
                cycle.details = {
                    "evaluated": len(results),
                    "selected": selection.selected_variant.name,
                    "deployed": should_deploy,
                    "reason": reason,
                }
                
                if should_deploy and self.config.auto_deploy:
                    cycle.status = "deployed"
                else:
                    cycle.status = "completed"
            else:
                cycle.status = "no_improvement"
                cycle.details = {"reason": "No suitable strategy found"}
            
            logger.info(f"Evaluation cycle completed: {cycle.status}")
            
        except Exception as e:
            logger.exception(f"Evaluation cycle failed: {e}")
            cycle.status = "failed"
            cycle.error = str(e)
        
        self._cycles.append(cycle)

    def _get_default_params(self) -> dict[str, Any]:
        """Get default strategy parameters."""
        return {
            "stop_loss": 0.03,
            "take_profit": 0.06,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "ma_period": 20,
            "position_size": 0.2,
        }

    def _generate_cycle_id(self) -> str:
        """Generate unique cycle ID."""
        return f"cycle_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    async def run_manual_cycle(self) -> ImprovementCycle:
        """Manually trigger an improvement cycle."""
        cycle = ImprovementCycle(
            cycle_id=self._generate_cycle_id(),
            timestamp=datetime.now(timezone.utc),
            phase="manual",
            status="running",
            details={},
        )
        
        try:
            await self._run_initial_cycle()
            await self._run_training_cycle()
            await self._run_evaluation_cycle()
            
            cycle.status = "completed"
            
        except Exception as e:
            logger.exception(f"Manual cycle failed: {e}")
            cycle.status = "failed"
            cycle.error = str(e)
        
        self._cycles.append(cycle)
        return cycle

    def get_status(self) -> dict[str, Any]:
        """Get current status of auto-improver."""
        return {
            "running": self._running,
            "current_dataset": self._current_dataset.name if self._current_dataset else None,
            "cycles_count": len(self._cycles),
            "deployment": self.deployment_manager.get_status(),
            "generator_stats": self.strategy_generator.get_statistics(),
        }

    def get_cycles(self, limit: int = 10) -> list[ImprovementCycle]:
        """Get recent improvement cycles."""
        return self._cycles[-limit:]

    def export_state(self, output_dir: Path) -> None:
        """Export current state to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_path = output_dir / f"auto_improver_state_{timestamp}.json"
        
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": self.get_status(),
            "cycles": [
                {
                    "cycle_id": c.cycle_id,
                    "timestamp": c.timestamp.isoformat(),
                    "phase": c.phase,
                    "status": c.status,
                    "details": c.details,
                    "error": c.error,
                }
                for c in self._cycles
            ],
        }
        
        with open(file_path, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State exported to {file_path}")
