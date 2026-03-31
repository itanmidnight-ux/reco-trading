from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MarketRegime:
    name: str
    volatility_level: str
    trend_strength: float
    volume_activity: str
    recommended_confidence: float
    recommended_position_size: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    atr_multiplier: float = 2.0
    ema_short: int = 9
    ema_long: int = 21


class AdaptiveConfigurationEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._regime_configs: dict[str, MarketRegime] = {}
        self._initialize_regime_configs()
        
        self._current_regime: MarketRegime | None = None
        self._config_history: list[dict] = []
        
        self.logger.info("AdaptiveConfigurationEngine initialized")

    def _initialize_regime_configs(self) -> None:
        self._regime_configs = {
            "BULL_HIGH_VOL": MarketRegime(
                name="BULL_HIGH_VOL",
                volatility_level="high",
                trend_strength=0.8,
                volume_activity="high",
                recommended_confidence=0.70,
                recommended_position_size=0.08,
                stop_loss_multiplier=2.5,
                take_profit_multiplier=1.5,
                rsi_oversold=25,
                rsi_overbought=75,
                rsi_period=10,
                macd_fast=8,
                macd_slow=21,
                macd_signal=9,
                atr_multiplier=2.5,
                ema_short=9,
                ema_long=21
            ),
            "BULL_LOW_VOL": MarketRegime(
                name="BULL_LOW_VOL",
                volatility_level="low",
                trend_strength=0.6,
                volume_activity="medium",
                recommended_confidence=0.50,
                recommended_position_size=0.12,
                stop_loss_multiplier=1.5,
                take_profit_multiplier=2.0,
                rsi_oversold=35,
                rsi_overbought=65,
                rsi_period=21,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
                atr_multiplier=1.5,
                ema_short=12,
                ema_long=26
            ),
            "BEAR_HIGH_VOL": MarketRegime(
                name="BEAR_HIGH_VOL",
                volatility_level="high",
                trend_strength=-0.8,
                volume_activity="high",
                recommended_confidence=0.75,
                recommended_position_size=0.05,
                stop_loss_multiplier=3.0,
                take_profit_multiplier=1.0,
                rsi_oversold=20,
                rsi_overbought=80,
                rsi_period=7,
                macd_fast=5,
                macd_slow=15,
                macd_signal=5,
                atr_multiplier=3.0,
                ema_short=5,
                ema_long=15
            ),
            "BEAR_LOW_VOL": MarketRegime(
                name="BEAR_LOW_VOL",
                volatility_level="low",
                trend_strength=-0.4,
                volume_activity="low",
                recommended_confidence=0.65,
                recommended_position_size=0.07,
                stop_loss_multiplier=2.0,
                take_profit_multiplier=1.2,
                rsi_oversold=30,
                rsi_overbought=70,
                rsi_period=14,
                macd_fast=10,
                macd_slow=20,
                macd_signal=7,
                atr_multiplier=2.0,
                ema_short=8,
                ema_long=18
            ),
            "SIDEWAYS_HIGH_VOL": MarketRegime(
                name="SIDEWAYS_HIGH_VOL",
                volatility_level="high",
                trend_strength=0.0,
                volume_activity="high",
                recommended_confidence=0.80,
                recommended_position_size=0.05,
                stop_loss_multiplier=2.0,
                take_profit_multiplier=1.0,
                rsi_oversold=25,
                rsi_overbought=75,
                rsi_period=10,
                macd_fast=8,
                macd_slow=17,
                macd_signal=7,
                atr_multiplier=2.0,
                ema_short=7,
                ema_long=14
            ),
            "SIDEWAYS_LOW_VOL": MarketRegime(
                name="SIDEWAYS_LOW_VOL",
                volatility_level="low",
                trend_strength=0.0,
                volume_activity="low",
                recommended_confidence=0.55,
                recommended_position_size=0.10,
                stop_loss_multiplier=1.5,
                take_profit_multiplier=1.5,
                rsi_oversold=35,
                rsi_overbought=65,
                rsi_period=21,
                macd_fast=15,
                macd_slow=30,
                macd_signal=10,
                atr_multiplier=1.5,
                ema_short=10,
                ema_long=25
            ),
            "TRANSITION_BULL_TO_BEAR": MarketRegime(
                name="TRANSITION_BULL_TO_BEAR",
                volatility_level="high",
                trend_strength=-0.3,
                volume_activity="high",
                recommended_confidence=0.75,
                recommended_position_size=0.05,
                stop_loss_multiplier=2.5,
                take_profit_multiplier=0.8,
                rsi_oversold=25,
                rsi_overbought=70,
                rsi_period=10,
                macd_fast=6,
                macd_slow=15,
                macd_signal=5,
                atr_multiplier=2.5,
                ema_short=6,
                ema_long=15
            ),
            "TRANSITION_BEAR_TO_BULL": MarketRegime(
                name="TRANSITION_BEAR_TO_BULL",
                volatility_level="high",
                trend_strength=0.3,
                volume_activity="high",
                recommended_confidence=0.65,
                recommended_position_size=0.08,
                stop_loss_multiplier=2.0,
                take_profit_multiplier=1.8,
                rsi_oversold=30,
                rsi_overbought=75,
                rsi_period=12,
                macd_fast=10,
                macd_slow=20,
                macd_signal=7,
                atr_multiplier=2.0,
                ema_short=8,
                ema_long=18
            ),
        }

    def get_regime_config(self, regime_name: str) -> MarketRegime | None:
        return self._regime_configs.get(regime_name)

    def determine_regime(
        self,
        hmm_regime: str,
        volatility: float,
        trend_strength: float,
        volume_level: str,
    ) -> MarketRegime:
        vol_level = "high" if volatility > 0.05 else "low"
        
        if hmm_regime == "BULL" and vol_level == "high":
            regime_key = "BULL_HIGH_VOL"
        elif hmm_regime == "BULL" and vol_level == "low":
            regime_key = "BULL_LOW_VOL"
        elif hmm_regime == "BEAR" and vol_level == "high":
            regime_key = "BEAR_HIGH_VOL"
        elif hmm_regime == "BEAR" and vol_level == "low":
            regime_key = "BEAR_LOW_VOL"
        elif hmm_regime == "SIDEWAYS" and vol_level == "high":
            regime_key = "SIDEWAYS_HIGH_VOL"
        elif hmm_regime == "SIDEWAYS" and vol_level == "low":
            regime_key = "SIDEWAYS_LOW_VOL"
        elif abs(trend_strength) < 0.2 and vol_level == "high":
            regime_key = "TRANSITION_BULL_TO_BEAR" if trend_strength < 0 else "TRANSITION_BEAR_TO_BULL"
        else:
            regime_key = "SIDEWAYS_LOW_VOL"
        
        self._current_regime = self._regime_configs.get(regime_key, self._regime_configs["SIDEWAYS_LOW_VOL"])
        
        return self._current_regime

    def get_recommended_config(self) -> dict[str, Any]:
        if not self._current_regime:
            return self._get_default_config()
        
        return {
            "confidence_threshold": self._current_regime.recommended_confidence,
            "position_size_pct": self._current_regime.recommended_position_size * 100,
            "stop_loss_atr_multiplier": self._current_regime.stop_loss_multiplier,
            "take_profit_atr_multiplier": self._take_profit_from_entry(
                self._current_regime.stop_loss_multiplier,
                self._current_regime.take_profit_multiplier
            ),
            "rsi": {
                "period": self._current_regime.rsi_period,
                "oversold": self._current_regime.rsi_oversold,
                "overbought": self._current_regime.rsi_overbought,
            },
            "macd": {
                "fast": self._current_regime.macd_fast,
                "slow": self._current_regime.macd_slow,
                "signal": self._current_regime.macd_signal,
            },
            "atr": {
                "period": self._current_regime.atr_period,
                "multiplier": self._current_regime.atr_multiplier,
            },
            "ema": {
                "short": self._current_regime.ema_short,
                "long": self._current_regime.ema_long,
            },
            "regime_name": self._current_regime.name,
        }

    def _take_profit_from_entry(self, sl_mult: float, tp_mult: float) -> float:
        return sl_mult * tp_mult

    def _get_default_config(self) -> dict[str, Any]:
        return {
            "confidence_threshold": 0.55,
            "position_size_pct": 10.0,
            "stop_loss_atr_multiplier": 2.0,
            "take_profit_atr_multiplier": 3.0,
            "rsi": {"period": 14, "oversold": 30, "overbought": 70},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "atr": {"period": 14, "multiplier": 2.0},
            "ema": {"short": 9, "long": 21},
            "regime_name": "DEFAULT",
        }

    def adapt_to_performance(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        trade_count: int,
    ) -> dict[str, Any]:
        config = self.get_recommended_config()
        
        if trade_count < 10:
            return config
        
        if win_rate < 0.35:
            config["confidence_threshold"] = min(0.85, config["confidence_threshold"] + 0.15)
            config["position_size_pct"] = max(3, config["position_size_pct"] * 0.5)
            self.logger.warning(f"Poor performance - raising confidence to {config['confidence_threshold']}")
        
        elif win_rate < 0.45:
            config["confidence_threshold"] = min(0.75, config["confidence_threshold"] + 0.08)
            config["position_size_pct"] = config["position_size_pct"] * 0.75
        
        elif win_rate > 0.70:
            config["confidence_threshold"] = max(0.40, config["confidence_threshold"] - 0.08)
            config["position_size_pct"] = min(20, config["position_size_pct"] * 1.2)
            self.logger.info(f"Excellent performance - lowering confidence to {config['confidence_threshold']}")
        
        elif win_rate > 0.60:
            config["confidence_threshold"] = max(0.45, config["confidence_threshold"] - 0.03)
        
        profit_factor = avg_win / abs(avg_loss) if avg_loss != 0 else 1.0
        if profit_factor > 2.0:
            config["take_profit_atr_multiplier"] = config["take_profit_atr_multiplier"] * 1.3
        
        self._config_history.append({
            "timestamp": datetime.now(timezone.utc),
            "win_rate": win_rate,
            "trade_count": trade_count,
            "new_config": config,
        })
        
        return config

    def get_config_history(self, limit: int = 20) -> list[dict]:
        return self._config_history[-limit:]


class ConfidenceAdaptiveEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._base_confidence = 0.55
        self._current_confidence = 0.55
        self._min_confidence = 0.30
        self._max_confidence = 0.90
        
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._performance_window: list[dict] = []
        
        self._confidence_history: list[dict] = []
        
        self.logger.info("ConfidenceAdaptiveEngine initialized")

    def calculate_confidence(
        self,
        regime_config: MarketRegime | None,
        market_data: dict,
        ensemble_confidence: float,
        llm_agent_confidence: float | None,
    ) -> float:
        components = []
        weights = []
        
        base_conf = regime_config.recommended_confidence if regime_config else self._base_confidence
        components.append(base_conf)
        weights.append(0.25)
        
        if ensemble_confidence > 0:
            components.append(ensemble_confidence)
            weights.append(0.35)
        
        if llm_agent_confidence is not None and llm_agent_confidence > 0:
            components.append(llm_agent_confidence)
            weights.append(0.20)
        
        trend = market_data.get("trend_strength", 0)
        if abs(trend) > 0.6:
            components.append(0.65 if trend > 0 else 0.60)
            weights.append(0.10)
        
        volatility = market_data.get("volatility", 0.03)
        if volatility > 0.10:
            components.append(0.70)
            weights.append(0.05)
        elif volatility < 0.02:
            components.append(0.50)
            weights.append(0.05)
        
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            confidence = sum(c * w for c, w in zip(components, normalized_weights))
        else:
            confidence = sum(components) / len(components) if components else self._base_confidence
        
        confidence = self._apply_performance_adjustment(confidence)
        
        confidence = self._apply_consecutive_adjustment(confidence)
        
        self._current_confidence = max(self._min_confidence, min(self._max_confidence, confidence))
        
        return self._current_confidence

    def _apply_performance_adjustment(self, confidence: float) -> float:
        if len(self._performance_window) < 5:
            return confidence
        
        recent = self._performance_window[-10:]
        
        win_rate = sum(1 for p in recent if p.get("pnl", 0) > 0) / len(recent)
        
        if win_rate > 0.7:
            return confidence - 0.05
        elif win_rate > 0.6:
            return confidence - 0.02
        elif win_rate < 0.4:
            return confidence + 0.08
        elif win_rate < 0.3:
            return confidence + 0.15
        
        return confidence

    def _apply_consecutive_adjustment(self, confidence: float) -> float:
        if self._consecutive_wins >= 5:
            return confidence - 0.08
        
        if self._consecutive_losses >= 3:
            return confidence + 0.10
        elif self._consecutive_losses >= 2:
            return confidence + 0.05
        
        return confidence

    def record_trade_result(self, pnl: float) -> None:
        self._performance_window.append({"pnl": pnl, "timestamp": datetime.now(timezone.utc)})
        
        if len(self._performance_window) > 100:
            self._performance_window = self._performance_window[-50:]
        
        if pnl > 0:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        elif pnl < 0:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
        
        self._confidence_history.append({
            "timestamp": datetime.now(timezone.utc),
            "confidence": self._current_confidence,
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
            "pnl": pnl,
        })

    def get_confidence_stats(self) -> dict:
        return {
            "current_confidence": self._current_confidence,
            "base_confidence": self._base_confidence,
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
            "recent_win_rate": (
                sum(1 for p in self._performance_window[-10:] if p.get("pnl", 0) > 0)
                / max(min(len(self._performance_window), 10), 1)
            ),
        }


class AdaptiveFilterManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._config_engine = AdaptiveConfigurationEngine()
        self._confidence_engine = ConfidenceAdaptiveEngine()
        
        self._filters = {
            "min_volume": 1_000_000,
            "max_spread": 0.5,
            "min_volatility": 0.0,
            "max_volatility": 0.20,
        }
        
        self.logger.info("AdaptiveFilterManager initialized")

    def update(
        self,
        hmm_regime: str,
        volatility: float,
        trend_strength: float,
        volume_level: str,
        market_data: dict,
        ensemble_confidence: float,
        llm_confidence: float | None,
        performance_metrics: dict | None,
    ) -> dict:
        regime = self._config_engine.determine_regime(
            hmm_regime, volatility, trend_strength, volume_level
        )
        
        recommended_config = self._config_engine.get_recommended_config()
        
        if performance_metrics:
            win_rate = performance_metrics.get("win_rate", 0.5)
            avg_win = performance_metrics.get("avg_win", 0)
            avg_loss = performance_metrics.get("avg_loss", 0)
            trade_count = performance_metrics.get("trade_count", 0)
            
            recommended_config = self._config_engine.adapt_to_performance(
                win_rate, avg_win, avg_loss, trade_count
            )
        
        confidence = self._confidence_engine.calculate_confidence(
            regime, market_data, ensemble_confidence, llm_confidence
        )
        
        self._filters["min_volume"] = self._calculate_min_volume(volume_level)
        self._filters["max_volatility"] = regime.atr_multiplier * 0.08
        self._filters["min_volatility"] = max(0, volatility * 0.5)
        
        return {
            "regime": regime.name,
            "confidence_threshold": confidence,
            "position_size_pct": recommended_config["position_size_pct"],
            "stop_loss_multiplier": recommended_config["stop_loss_atr_multiplier"],
            "take_profit_multiplier": recommended_config["take_profit_atr_multiplier"],
            "filters": self._filters.copy(),
            "indicators": {
                "rsi": recommended_config["rsi"],
                "macd": recommended_config["macd"],
                "atr": recommended_config["atr"],
                "ema": recommended_config["ema"],
            },
        }

    def _calculate_min_volume(self, volume_level: str) -> float:
        if volume_level == "high":
            return 5_000_000
        elif volume_level == "medium":
            return 2_000_000
        else:
            return 500_000

    def record_trade(self, pnl: float) -> None:
        self._confidence_engine.record_trade_result(pnl)

    def get_current_config(self) -> dict:
        return {
            "confidence": self._confidence_engine.get_confidence_stats(),
            "filters": self._filters.copy(),
        }