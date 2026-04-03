from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TradeConfirmation:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    signal: str
    confidence: float
    confirmed: bool = False
    reason: str = ""
    analysis_time_ms: float = 0.0
    timestamp: str = ""


class LLMTradeConfirmator:
    """Ultra-fast trade confirmation using rule-based analysis (simulates LLM in <5ms)."""

    def __init__(
        self,
        llm_mode: str = "base",
        local_model: str = "qwen2.5:0.5b",
        ollama_base_url: str = "http://localhost:11434",
        local_timeout_seconds: float = 3.0,
        remote_timeout_seconds: float = 5.0,
        keep_alive: str = "10m",
        remote_endpoint: str = "https://api.openai.com/v1/chat/completions",
        remote_model: str = "gpt-4o-mini",
        remote_api_key: str = "",
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self._confirmation_count = 0
        self._rejection_count = 0
        self._avg_time_ms = 0.0
        self.llm_mode = (llm_mode or "base").lower()
        self.local_model = local_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.local_timeout_seconds = max(float(local_timeout_seconds), 0.5)
        self.remote_timeout_seconds = max(float(remote_timeout_seconds), 0.5)
        self.keep_alive = str(keep_alive or "10m")
        self.remote_endpoint = remote_endpoint
        self.remote_model = remote_model
        self.remote_api_key = remote_api_key

    def confirm_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        signal: str,
        confidence: float,
        trend: str = "NEUTRAL",
        adx: float = 0.0,
        volatility_regime: str = "NORMAL",
        order_flow: str = "NEUTRAL",
        spread: float = 0.0,
        atr: float = 0.0,
        volume: float = 0.0,
        risk_per_trade: float = 0.01,
        daily_pnl: float = 0.0,
        trades_today: int = 0,
        max_trades_per_day: int = 20,
    ) -> TradeConfirmation:
        start = time.perf_counter()

        score = 0.0
        reasons: list[str] = []

        if confidence >= 0.75:
            score += 25
        elif confidence >= 0.60:
            score += 15
        elif confidence >= 0.50:
            score += 5
        else:
            reasons.append(f"Low confidence: {confidence:.2f}")

        if trend in ("BULLISH", "BEARISH"):
            score += 15
            if (side == "BUY" and trend == "BULLISH") or (side == "SELL" and trend == "BEARISH"):
                score += 10
            else:
                score -= 10
                reasons.append("Trend/side mismatch")
        else:
            reasons.append("Neutral trend")

        if adx >= 25:
            score += 10
        elif adx >= 20:
            score += 5
        else:
            reasons.append("Weak trend strength (ADX low)")

        if volatility_regime in ("NORMAL", "TRENDING"):
            score += 10
        elif volatility_regime == "HIGH_VOLATILITY":
            score += 5
            reasons.append("High volatility - reduced size recommended")
        else:
            score -= 10
            reasons.append(f"Unfavorable regime: {volatility_regime}")

        if order_flow in ("BULLISH", "BEARISH"):
            score += 10
            if (side == "BUY" and order_flow == "BULLISH") or (side == "SELL" and order_flow == "BEARISH"):
                score += 5
        else:
            reasons.append("Neutral order flow")

        if spread < 0.001:
            score += 5
        elif spread < 0.005:
            score += 3
        else:
            score -= 5
            reasons.append(f"Wide spread: {spread:.4f}")

        if atr > 0:
            atr_pct = (atr / entry_price) * 100 if entry_price > 0 else 0
            if 0.5 <= atr_pct <= 5.0:
                score += 5
            elif atr_pct > 5.0:
                reasons.append(f"Extreme ATR: {atr_pct:.1f}%")

        if volume > 1000:
            score += 5

        if trades_today >= max_trades_per_day:
            score -= 30
            reasons.append(f"Daily trade limit reached: {trades_today}/{max_trades_per_day}")

        risk_amount = entry_price * quantity * risk_per_trade
        if risk_amount > 100:
            score -= 5
            reasons.append(f"High risk amount: ${risk_amount:.2f}")

        if daily_pnl < -50:
            score -= 15
            reasons.append(f"Significant daily loss: ${daily_pnl:.2f}")

        if self.llm_mode == "base":
            confirmed = score >= 50
            reasons.append("LLM_MODE=base (decisión por reglas)")
        elif self.llm_mode == "llm_remote":
            remote_confirmed, remote_reason = self._remote_confirm(
                symbol=symbol,
                side=side,
                signal=signal,
                confidence=confidence,
                score=score,
                default_confirmed=(score >= 50),
            )
            confirmed = remote_confirmed
            reasons.append(remote_reason)
        elif self.llm_mode == "llm_local":
            local_confirmed, local_reason = self._local_confirm(
                symbol=symbol,
                side=side,
                signal=signal,
                confidence=confidence,
                score=score,
                default_confirmed=(score >= 50),
            )
            confirmed = local_confirmed
            reasons.append(local_reason)
        else:
            confirmed = score >= 50
            reasons.append(f"LLM_MODE desconocido ({self.llm_mode}), fallback a reglas")
        analysis_time = (time.perf_counter() - start) * 1000

        total_events = self._confirmation_count + self._rejection_count + 1
        self._avg_time_ms = ((self._avg_time_ms * (total_events - 1)) + analysis_time) / total_events
        if confirmed:
            self._confirmation_count += 1
        else:
            self._rejection_count += 1

        result = TradeConfirmation(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            signal=signal,
            confidence=confidence,
            confirmed=confirmed,
            reason="; ".join(reasons) if reasons else "All checks passed",
            analysis_time_ms=analysis_time,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if confirmed:
            self.logger.info(
                f"trade_confirmed symbol={symbol} side={side} score={score} "
                f"time={analysis_time:.1f}ms"
            )
        else:
            self.logger.warning(
                f"trade_rejected symbol={symbol} side={side} score={score} "
                f"reasons={result.reason} time={analysis_time:.1f}ms"
            )

        return result

    def _local_confirm(
        self,
        symbol: str,
        side: str,
        signal: str,
        confidence: float,
        score: float,
        default_confirmed: bool,
    ) -> tuple[bool, str]:
        try:
            import requests

            prompt = (
                f"Symbol={symbol} Side={side} Signal={signal} Confidence={confidence:.2f} "
                f"RuleScore={score:.2f}. Return JSON only with key decision: "
                '{"decision":"APPROVE"} or {"decision":"REJECT"}.'
            )
            payload = {
                "model": self.local_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": self.keep_alive,
                "options": {"temperature": 0},
            }
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=self.local_timeout_seconds,
            )
            response.raise_for_status()
            raw_response = str(response.json().get("response", "")).strip()
            decision = self._extract_decision(raw_response)
            if decision == "REJECT":
                return False, "LLM local rechazó la operación"
            if decision == "APPROVE":
                return True, "LLM local aprobó la operación"
            return default_confirmed, "LLM local ambiguo, fallback a reglas"
        except Exception as exc:
            self.logger.warning(f"llm_local_fallback reason={exc}")
            return default_confirmed, "LLM local no disponible, fallback a reglas"

    def _remote_confirm(
        self,
        symbol: str,
        side: str,
        signal: str,
        confidence: float,
        score: float,
        default_confirmed: bool,
    ) -> tuple[bool, str]:
        try:
            import requests

            prompt = (
                f"Symbol={symbol} Side={side} Signal={signal} Confidence={confidence:.2f} "
                f"RuleScore={score:.2f}. Return only APPROVE or REJECT."
            )
            headers = {"Content-Type": "application/json"}
            api_key = self.remote_api_key or os.getenv("LLM_REMOTE_API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": self.remote_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            response = requests.post(
                self.remote_endpoint,
                headers=headers,
                json=payload,
                timeout=self.remote_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            content = ""
            if isinstance(body, dict):
                choices = body.get("choices", [])
                if choices:
                    content = str(choices[0].get("message", {}).get("content", ""))
            llm_text = content.upper()
            decision = self._extract_decision(llm_text)
            if decision == "REJECT":
                return False, "LLM remoto rechazó la operación"
            if decision == "APPROVE":
                return True, "LLM remoto aprobó la operación"
            return default_confirmed, "LLM remoto ambiguo, fallback a reglas"
        except Exception as exc:
            self.logger.warning(f"llm_remote_fallback reason={exc}")
            return default_confirmed, "LLM remoto no disponible, fallback a reglas"

    @staticmethod
    def _extract_decision(text: str) -> str | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            import json

            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                candidate = str(parsed.get("decision", "")).strip().upper()
                if candidate in {"APPROVE", "REJECT"}:
                    return candidate
        except Exception:
            pass
        normalized = raw.upper()
        if not normalized:
            return None
        match = re.search(r"\b(APPROVE|REJECT)\b", normalized)
        if not match:
            return None
        return match.group(1)

    @property
    def stats(self) -> dict[str, Any]:
        total = self._confirmation_count + self._rejection_count
        return {
            "total_analyzed": total,
            "confirmed": self._confirmation_count,
            "rejected": self._rejection_count,
            "confirmation_rate": (self._confirmation_count / total * 100) if total > 0 else 0,
            "avg_analysis_time_ms": self._avg_time_ms,
        }
