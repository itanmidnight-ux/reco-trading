from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class VolatilityState(str, Enum):
    LOW_VOLATILITY = "LOW_VOLATILITY"
    NORMAL_VOLATILITY = "NORMAL_VOLATILITY"
    EXTREME_VOLATILITY = "EXTREME_VOLATILITY"


class MarketRegime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_ACTIVITY = "LOW_ACTIVITY"


@dataclass(slots=True)
class VolatilityAssessment:
    state: VolatilityState
    allow_trade: bool
    risk_multiplier: float


@dataclass(slots=True)
class LiquidityZones:
    support_zone: float | None
    resistance_zone: float | None
    distance_to_support: float | None
    distance_to_resistance: float | None


@dataclass(slots=True)
class LiquidityFilterAssessment:
    allow_trade: bool
    liquidity_multiplier: float
    liquidity_distance: float | None
    zone_type: str | None


@dataclass(slots=True)
class RangeFilterAssessment:
    allow_trade: bool
    range_multiplier: float
    position_in_range: float | None


@dataclass(slots=True)
class MarketRegimeAssessment:
    regime: MarketRegime
    allow_trade: bool
    risk_multiplier: float


class VolatilityFilter:
    """Classifies volatility and proposes risk adjustment."""

    def evaluate(self, df: pd.DataFrame) -> VolatilityAssessment:
        if len(df) < 30:
            return VolatilityAssessment(VolatilityState.NORMAL_VOLATILITY, allow_trade=True, risk_multiplier=1.0)

        recent = df.tail(30)
        close = float(recent["close"].iloc[-1])
        atr = float(recent["atr"].iloc[-1])
        adx = float(recent.get("adx", pd.Series([20.0])).iloc[-1])

        candle_range_pct = ((recent["high"] - recent["low"]) / recent["close"].clip(lower=1e-9)).mean()
        atr_ratio = atr / max(close, 1e-9)
        recent_vol = recent["close"].pct_change().dropna().std()

        if atr_ratio < 0.0015 and candle_range_pct < 0.0025 and recent_vol < 0.0025 and adx < 13:
            return VolatilityAssessment(VolatilityState.LOW_VOLATILITY, allow_trade=False, risk_multiplier=0.0)
        if atr_ratio < 0.0035 and candle_range_pct < 0.005 and recent_vol < 0.005:
            return VolatilityAssessment(VolatilityState.LOW_VOLATILITY, allow_trade=True, risk_multiplier=0.55)
        if atr_ratio < 0.0065 and candle_range_pct < 0.008:
            return VolatilityAssessment(VolatilityState.NORMAL_VOLATILITY, allow_trade=True, risk_multiplier=0.75)
        if atr_ratio > 0.02 or candle_range_pct > 0.025 or recent_vol > 0.02:
            return VolatilityAssessment(VolatilityState.EXTREME_VOLATILITY, allow_trade=True, risk_multiplier=0.6)
        return VolatilityAssessment(VolatilityState.NORMAL_VOLATILITY, allow_trade=True, risk_multiplier=1.0)


class LiquidityZoneDetector:
    """Approximates support/resistance zones from swings, clustering and volume."""

    def __init__(self, proximity_threshold: float = 0.0025) -> None:
        self.proximity_threshold = proximity_threshold

    def detect(self, df: pd.DataFrame) -> LiquidityZones:
        if len(df) < 20:
            return LiquidityZones(None, None, None, None)

        recent = df.tail(80).copy()
        vol_mean = recent["volume"].rolling(20).mean()
        volume_spikes = recent["volume"] > (vol_mean * 1.4)

        swing_highs = recent[(recent["high"] == recent["high"].rolling(5, center=True).max()) | volume_spikes]
        swing_lows = recent[(recent["low"] == recent["low"].rolling(5, center=True).min()) | volume_spikes]

        support_zone = self._cluster_price_dbscan(swing_lows.get("low"), float(recent["close"].iloc[-1]), prefer="lower")
        resistance_zone = self._cluster_price_dbscan(swing_highs.get("high"), float(recent["close"].iloc[-1]), prefer="upper")

        price = float(recent["close"].iloc[-1])
        distance_to_support = self._distance_ratio(price, support_zone)
        distance_to_resistance = self._distance_ratio(price, resistance_zone)

        return LiquidityZones(
            support_zone=support_zone,
            resistance_zone=resistance_zone,
            distance_to_support=distance_to_support,
            distance_to_resistance=distance_to_resistance,
        )

    def is_side_allowed(self, side: str, zones: LiquidityZones, price: float) -> bool:
        assessment = self.assess_side(side=side, zones=zones, price=price, signal_confidence=1.0)
        return assessment.allow_trade

    def assess_side(self, side: str, zones: LiquidityZones, price: float, signal_confidence: float) -> LiquidityFilterAssessment:
        normalized_side = side.upper()
        if normalized_side == "BUY":
            target_distance = self._distance_ratio(price, zones.support_zone)
            zone_type = "support"
        elif normalized_side == "SELL":
            target_distance = self._distance_ratio(price, zones.resistance_zone)
            zone_type = "resistance"
        else:
            return LiquidityFilterAssessment(
                allow_trade=False,
                liquidity_multiplier=0.0,
                liquidity_distance=None,
                zone_type=None,
            )

        if target_distance is None:
            return LiquidityFilterAssessment(
                allow_trade=True,
                liquidity_multiplier=0.70,
                liquidity_distance=None,
                zone_type=zone_type,
            )

        base_threshold = max(float(self.proximity_threshold), 1e-4)
        medium_threshold = base_threshold * 6.0
        extended_threshold = base_threshold * 15.0

        if target_distance <= base_threshold:
            return LiquidityFilterAssessment(
                allow_trade=True,
                liquidity_multiplier=1.0,
                liquidity_distance=target_distance,
                zone_type=zone_type,
            )
        if target_distance <= medium_threshold:
            return LiquidityFilterAssessment(
                allow_trade=True,
                liquidity_multiplier=0.75,
                liquidity_distance=target_distance,
                zone_type=zone_type,
            )
        if target_distance <= extended_threshold:
            return LiquidityFilterAssessment(
                allow_trade=True,
                liquidity_multiplier=0.45,
                liquidity_distance=target_distance,
                zone_type=zone_type,
            )
        return LiquidityFilterAssessment(
            allow_trade=True,
            liquidity_multiplier=0.70,
            liquidity_distance=target_distance,
            zone_type=zone_type,
        )

    def _cluster_price(self, series: pd.Series | None, price: float, prefer: str) -> float | None:
        if series is None:
            return None
        cleaned = pd.to_numeric(series, errors="coerce").dropna()
        if cleaned.empty:
            return None

        rounded = (cleaned / max(price, 1e-9) * 1000).round() / 1000
        bins = rounded.value_counts()
        if bins.empty:
            return None

        top_levels = bins.head(3).index.tolist()
        levels = [float(level * price) for level in top_levels]
        if prefer == "lower":
            lower = [lvl for lvl in levels if lvl <= price]
            return max(lower) if lower else min(levels)
        upper = [lvl for lvl in levels if lvl >= price]
        return min(upper) if upper else max(levels)

    def _cluster_price_dbscan(self, series: pd.Series | None, price: float, prefer: str) -> float | None:
        """
        Detecta clusters reales de liquidez usando DBSCAN.

        Algoritmo:
        1. Validar que series no está vacía y tiene mínimo 3 puntos
        2. Si < 3 puntos, retornar mediana (DBSCAN requiere min_samples=3)
        3. Intenta importar DBSCAN y numpy
        4. Si falla import, fallback a método _cluster_price() original
        5. Preparar puntos en formato (n_samples, n_features)
        6. DBSCAN con eps=0.5% precio, min_samples=3
        7. Identificar cluster más denso (excluye outliers -1)
        8. Retornar centroide del cluster según prefer (lower/upper)

        Args:
            series: Serie de pandas con precios de swing.
            price: Precio actual para normalización y preferencia.
            prefer: "lower" para soportes o "upper" para resistencias.

        Returns:
            float | None: precio del cluster más denso, o None si no hay datos.
        """
        if series is None or series.empty:
            return None

        cleaned = pd.to_numeric(series, errors="coerce").dropna()
        if len(cleaned) < 3:
            return float(cleaned.median()) if len(cleaned) > 0 else None

        try:
            import numpy as np
            from sklearn.cluster import DBSCAN
        except ImportError:
            return self._cluster_price(series, price, prefer)

        points = cleaned.values.reshape(-1, 1)
        eps = max(price, 1e-9) * 0.005
        clusterer = DBSCAN(eps=eps, min_samples=3)
        labels = clusterer.fit_predict(points)
        valid_labels = labels[labels != -1]

        if len(valid_labels) == 0:
            return float(cleaned.median())

        cluster_centroids: dict[int, tuple[float, int]] = {}
        for label in np.unique(valid_labels):
            mask = labels == label
            cluster_values = cleaned.values[mask]
            centroid = float(np.mean(cluster_values))
            cluster_size = len(cluster_values)
            cluster_centroids[int(label)] = (centroid, cluster_size)

        if not cluster_centroids:
            return float(cleaned.median())

        if prefer == "lower":
            lower_clusters = {
                label: centroid
                for label, (centroid, _size) in cluster_centroids.items()
                if centroid <= price
            }
            if lower_clusters:
                return max(lower_clusters.values())
            return min(centroid for centroid, _size in cluster_centroids.values())

        upper_clusters = {
            label: centroid
            for label, (centroid, _size) in cluster_centroids.items()
            if centroid >= price
        }
        if upper_clusters:
            return min(upper_clusters.values())
        return max(centroid for centroid, _size in cluster_centroids.values())

    def _distance_ratio(self, price: float, zone: float | None) -> float | None:
        if zone is None:
            return None
        return abs(price - zone) / max(price, 1e-9)


class MarketRegimeClassifier:
    """Classifies market regime from trend/volatility descriptors."""

    def __init__(self) -> None:
        """Inicializa clasificador de régimen con tracking de transiciones."""
        self._last_regime: MarketRegime | None = None
        self._regime_change_candles: int = 0

    def classify(self, df: pd.DataFrame) -> MarketRegimeAssessment:
        """
        Clasifica régimen de mercado con detección de transiciones.

        Regímenes:
        - LOW_ACTIVITY: ATR bajo, ADX bajo, volatilidad baja
        - HIGH_VOLATILITY: ATR alto o volatilidad extrema
        - TRENDING: ADX alto + slope significativo
        - RANGING: ni trending ni extremo

        Penalizaciones por transición:
        - Detecta si ADX cambia >5 en últimas 5 velas
        - Reduce risk_multiplier 15% si en transición
        - Reduce 20% adicional durante 5 velas post-cambio

        Returns:
            MarketRegimeAssessment con risk_multiplier ajustado
        """
        if len(df) < 50:
            return MarketRegimeAssessment(
                regime=MarketRegime.RANGING,
                allow_trade=True,
                risk_multiplier=1.0,
            )

        recent = df.tail(100)
        price = float(recent["close"].iloc[-1])
        atr_ratio = float(recent["atr"].iloc[-1]) / max(price, 1e-9)
        adx_series = recent.get("adx", pd.Series([20.0]))
        adx = float(adx_series.iloc[-1])

        ema20 = recent["ema20"]
        slope = (float(ema20.iloc[-1]) - float(ema20.iloc[-10])) / max(price, 1e-9)

        returns = recent["close"].pct_change().dropna().abs()
        vol_now = float(returns.iloc[-1]) if not returns.empty else 0.0
        vol_pct = float((returns <= vol_now).mean()) if not returns.empty else 0.5

        recent_5 = df.tail(5)
        adx_recent = recent_5.get("adx", pd.Series([adx] * len(recent_5)))
        adx_change = float(adx_recent.iloc[-1]) - float(adx_recent.iloc[0])
        in_transition = abs(adx_change) > 5

        if atr_ratio < 0.002 and adx < 14 and vol_pct < 0.35:
            regime = MarketRegime.LOW_ACTIVITY
            risk_multiplier = 0.65
        elif atr_ratio > 0.02 or vol_pct > 0.92:
            regime = MarketRegime.HIGH_VOLATILITY
            risk_multiplier = 0.5
        elif adx >= 25 and abs(slope) > 0.0015:
            regime = MarketRegime.TRENDING
            risk_multiplier = 1.0
        elif adx >= 20 and abs(slope) > 0.001:
            regime = MarketRegime.TRENDING
            risk_multiplier = 0.8
        else:
            regime = MarketRegime.RANGING
            risk_multiplier = 0.65

        if in_transition:
            risk_multiplier *= 0.85
            if self._last_regime != regime:
                self._regime_change_candles = 5

        if self._regime_change_candles > 0:
            risk_multiplier *= 0.80
            self._regime_change_candles -= 1

        self._last_regime = regime
        risk_multiplier = max(risk_multiplier, 0.35)

        return MarketRegimeAssessment(
            regime=regime,
            allow_trade=True,
            risk_multiplier=risk_multiplier,
        )


class MarketRangePositionFilter:
    """Evaluates where price sits in recent range and adjusts entry confidence."""

    def assess(
        self,
        df: pd.DataFrame,
        side: str,
        price: float,
        market_regime: MarketRegime | None,
        adx: float,
    ) -> RangeFilterAssessment:
        if len(df) == 0:
            return RangeFilterAssessment(allow_trade=True, range_multiplier=1.0, position_in_range=None)

        window = df.tail(120)
        lowest_low = float(window["low"].min())
        highest_high = float(window["high"].max())
        span = highest_high - lowest_low
        if span <= 1e-9:
            position_in_range = 0.5
        else:
            position_in_range = (price - lowest_low) / span
        position_in_range = min(max(position_in_range, 0.0), 1.0)

        range_band = max(highest_high - lowest_low, 1e-9)
        support_distance = max((price - lowest_low) / max(price, 1e-9), 0.0)
        resistance_distance = max((highest_high - price) / max(price, 1e-9), 0.0)
        proximity_floor = max(0.0035, 0.20 * (range_band / max(price, 1e-9)))

        is_trending = market_regime == MarketRegime.TRENDING or adx >= 25
        if side == "BUY":
            if resistance_distance <= (proximity_floor * 0.5):
                if is_trending:
                    return RangeFilterAssessment(allow_trade=True, range_multiplier=0.30, position_in_range=position_in_range)
                return RangeFilterAssessment(allow_trade=True, range_multiplier=0.15, position_in_range=position_in_range)
            if resistance_distance <= proximity_floor:
                if is_trending:
                    return RangeFilterAssessment(allow_trade=True, range_multiplier=0.30, position_in_range=position_in_range)
                return RangeFilterAssessment(allow_trade=True, range_multiplier=0.15, position_in_range=position_in_range)
            if support_distance <= proximity_floor:
                support_boost = 1.2 if not is_trending else 1.1
                return RangeFilterAssessment(allow_trade=True, range_multiplier=min(support_boost, 1.0), position_in_range=position_in_range)
            return RangeFilterAssessment(allow_trade=True, range_multiplier=1.0, position_in_range=position_in_range)

        if side == "SELL":
            if support_distance <= (proximity_floor * 0.5):
                if is_trending:
                    return RangeFilterAssessment(allow_trade=True, range_multiplier=0.30, position_in_range=position_in_range)
                return RangeFilterAssessment(allow_trade=True, range_multiplier=0.15, position_in_range=position_in_range)
            if support_distance <= proximity_floor:
                if is_trending:
                    return RangeFilterAssessment(allow_trade=True, range_multiplier=0.30, position_in_range=position_in_range)
                return RangeFilterAssessment(allow_trade=True, range_multiplier=0.15, position_in_range=position_in_range)
            if resistance_distance <= proximity_floor:
                resistance_boost = 1.2 if not is_trending else 1.1
                return RangeFilterAssessment(allow_trade=True, range_multiplier=min(resistance_boost, 1.0), position_in_range=position_in_range)
            return RangeFilterAssessment(allow_trade=True, range_multiplier=1.0, position_in_range=position_in_range)

        return RangeFilterAssessment(allow_trade=False, range_multiplier=0.0, position_in_range=position_in_range)


class MarketIntelligence:
    """Coordinator that keeps advanced filters additive and optional."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.volatility_filter = VolatilityFilter()
        proximity_threshold = float(getattr(settings, "liquidity_proximity_threshold", 0.0035))
        self.liquidity_detector = LiquidityZoneDetector(proximity_threshold=proximity_threshold)
        self.regime_classifier = MarketRegimeClassifier()
        self.range_position_filter = MarketRangePositionFilter()

    def evaluate(self, side: str, market_data: dict[str, Any]) -> dict[str, Any]:
        if not getattr(self.settings, "enable_market_intelligence", True):
            return {"approved": True, "size_multiplier": 1.0, "reason": "DISABLED"}

        df = market_data.get("frame5")
        price = float(market_data.get("price", 0.0))
        result = {
            "approved": True,
            "size_multiplier": 1.0,
            "reason": "APPROVED",
            "market_regime": None,
            "volatility_state": None,
            "distance_to_support": None,
            "distance_to_resistance": None,
            "support_zone": None,
            "resistance_zone": None,
            "liquidity_distance": None,
            "range_position": None,
            "filter_details": {},
        }

        if df is None or len(df) == 0:
            return result

        adx = float(df.get("adx", pd.Series([20.0])).iloc[-1])
        regime_assessment: MarketRegimeAssessment | None = None

        if getattr(self.settings, "volatility_filter_enabled", True):
            vol = self.volatility_filter.evaluate(df)
            result["volatility_state"] = vol.state.value
            result["size_multiplier"] *= vol.risk_multiplier
            result["filter_details"]["volatility"] = {
                "allow_trade": vol.allow_trade,
                "multiplier": vol.risk_multiplier,
                "state": vol.state.value,
            }
            if not vol.allow_trade:
                result["approved"] = False
                result["reason"] = vol.state.value

        if getattr(self.settings, "liquidity_zone_filter_enabled", True):
            zones = self.liquidity_detector.detect(df)
            result["support_zone"] = zones.support_zone
            result["resistance_zone"] = zones.resistance_zone
            result["distance_to_support"] = zones.distance_to_support
            result["distance_to_resistance"] = zones.distance_to_resistance
            confidence = float(market_data.get("signal_confidence", 1.0))
            liquidity = self.liquidity_detector.assess_side(
                side=side,
                zones=zones,
                price=price,
                signal_confidence=confidence,
            )
            result["liquidity_distance"] = liquidity.liquidity_distance
            result["size_multiplier"] *= liquidity.liquidity_multiplier
            result["filter_details"]["liquidity"] = {
                "allow_trade": liquidity.allow_trade,
                "multiplier": liquidity.liquidity_multiplier,
                "distance": liquidity.liquidity_distance,
                "zone_type": liquidity.zone_type,
            }
            if result["approved"] and not liquidity.allow_trade:
                result["approved"] = False
                result["reason"] = "LIQUIDITY_ZONE_FILTER"

        if getattr(self.settings, "market_regime_classifier_enabled", True):
            regime_assessment = self.regime_classifier.classify(df)
            result["market_regime"] = regime_assessment.regime.value
            result["size_multiplier"] *= regime_assessment.risk_multiplier
            result["filter_details"]["regime"] = {
                "allow_trade": regime_assessment.allow_trade,
                "multiplier": regime_assessment.risk_multiplier,
                "regime": regime_assessment.regime.value,
            }
            if not regime_assessment.allow_trade:
                result["approved"] = False
                result["reason"] = regime_assessment.regime.value

        if getattr(self.settings, "market_range_filter_enabled", True):
            regime_enum = regime_assessment.regime if regime_assessment else None
            range_assessment = self.range_position_filter.assess(
                df=df,
                side=side,
                price=price,
                market_regime=regime_enum,
                adx=adx,
            )
            result["range_position"] = range_assessment.position_in_range
            result["size_multiplier"] *= range_assessment.range_multiplier
            result["filter_details"]["range"] = {
                "allow_trade": range_assessment.allow_trade,
                "multiplier": range_assessment.range_multiplier,
                "position": range_assessment.position_in_range,
            }
            if result["approved"] and not range_assessment.allow_trade:
                result["approved"] = False
                result["reason"] = "MARKET_RANGE_FILTER"

        result["size_multiplier"] = max(float(result["size_multiplier"]), 0.35)
        result["size_multiplier"] = max(min(float(result["size_multiplier"]), 1.0), 0.1)
        return result
