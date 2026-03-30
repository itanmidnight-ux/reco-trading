"""
Data Collector Module for Auto-Improver.
Collects historical and real-time market data for training and evaluation.
"""

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from reco_trading.database.repository import Repository

try:
    from reco_trading.exchange.exchange import Exchange
except ImportError:
    Exchange = None

logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """Single market data point."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str


@dataclass
class DataSet:
    """Collected dataset for training."""
    name: str
    symbols: list[str]
    start_date: datetime
    end_date: datetime
    timeframe: str
    data_points: list[MarketDataPoint] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "symbols": self.symbols,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "timeframe": self.timeframe,
            "data_points_count": len(self.data_points),
            "created_at": self.created_at.isoformat(),
        }


class DataCollector:
    """Collects market data from exchange and database."""

    def __init__(
        self,
        exchange: Any = None,
        repository: Any = None,
        data_dir: Any = None,
    ):
        self.exchange = exchange
        self.repository = repository
        self.data_dir = data_dir or Path("./user_data/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._realtime_buffers: dict[str, deque] = {}
        self._collecting = False

    async def collect_historical(
        self,
        symbols: list[str],
        timeframe: str = "5m",
        days_back: int = 30,
    ) -> DataSet:
        """Collect historical data for specified symbols."""
        logger.info(f"Collecting historical data for {len(symbols)} symbols")
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        all_data: list[MarketDataPoint] = []
        
        for symbol in symbols:
            try:
                if self.exchange:
                    candles = await self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=int(start_date.timestamp() * 1000),
                    )
                    
                    for c in candles:
                        all_data.append(MarketDataPoint(
                            timestamp=datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                            symbol=symbol,
                            open=float(c[1]),
                            high=float(c[2]),
                            low=float(c[3]),
                            close=float(c[4]),
                            volume=float(c[5]),
                            timeframe=timeframe,
                        ))
                        
                if self.repository:
                    await self._save_to_database(all_data)
                    
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
        
        dataset = DataSet(
            name=f"historical_{timeframe}_{start_date.date()}_{end_date.date()}",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            data_points=all_data,
        )
        
        await self._save_dataset(dataset)
        
        logger.info(f"Collected {len(all_data)} data points")
        return dataset

    async def _save_to_database(self, data_points: list[MarketDataPoint]) -> None:
        """Save collected data to database."""
        if not self.repository:
            return
        
        for point in data_points:
            await self.repository.record_market_candle(
                symbol=point.symbol,
                timeframe=point.timeframe,
                candle={
                    "timestamp": point.timestamp,
                    "open": point.open,
                    "high": point.high,
                    "low": point.low,
                    "close": point.close,
                    "volume": point.volume,
                },
            )

    async def _save_dataset(self, dataset: DataSet) -> None:
        """Save dataset to JSON file."""
        file_path = self.data_dir / f"{dataset.name}.json"
        
        data = {
            "name": dataset.name,
            "symbols": dataset.symbols,
            "start_date": dataset.start_date.isoformat(),
            "end_date": dataset.end_date.isoformat(),
            "timeframe": dataset.timeframe,
            "data": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "symbol": p.symbol,
                    "open": p.open,
                    "high": p.high,
                    "low": p.low,
                    "close": p.close,
                    "volume": p.volume,
                }
                for p in dataset.data_points
            ],
            "created_at": dataset.created_at.isoformat(),
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Dataset saved to {file_path}")

    async def load_dataset(self, name: str) -> DataSet | None:
        """Load dataset from file."""
        file_path = self.data_dir / f"{name}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        data_points = [
            MarketDataPoint(
                timestamp=datetime.fromisoformat(p["timestamp"]),
                symbol=p["symbol"],
                open=p["open"],
                high=p["high"],
                low=p["low"],
                close=p["close"],
                volume=p["volume"],
                timeframe=data["timeframe"],
            )
            for p in data["data"]
        ]
        
        return DataSet(
            name=data["name"],
            symbols=data["symbols"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]),
            timeframe=data["timeframe"],
            data_points=data_points,
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def list_datasets(self) -> list[dict[str, Any]]:
        """List all available datasets."""
        datasets = []
        
        for file_path in self.data_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    datasets.append({
                        "name": data["name"],
                        "symbols": data["symbols"],
                        "start_date": data["start_date"],
                        "end_date": data["end_date"],
                        "timeframe": data["timeframe"],
                        "data_points": len(data["data"]),
                    })
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        return sorted(datasets, key=lambda x: x["start_date"], reverse=True)

    async def start_realtime_collection(self, symbols: list[str], timeframe: str = "5m") -> None:
        """Start real-time data collection."""
        if self._collecting:
            logger.warning("Real-time collection already running")
            return
        
        self._collecting = True
        
        for symbol in symbols:
            self._realtime_buffers[symbol] = deque(maxlen=1000)
        
        asyncio.create_task(self._realtime_collection_loop(symbols, timeframe))
        logger.info(f"Started real-time collection for {len(symbols)} symbols")

    async def _realtime_collection_loop(self, symbols: list[str], timeframe: str) -> None:
        """Real-time collection loop."""
        while self._collecting:
            for symbol in symbols:
                try:
                    if self.exchange:
                        candles = await self.exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=1,
                        )
                        
                        if candles:
                            c = candles[-1]
                            point = MarketDataPoint(
                                timestamp=datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                                symbol=symbol,
                                open=float(c[1]),
                                high=float(c[2]),
                                low=float(c[3]),
                                close=float(c[4]),
                                volume=float(c[5]),
                                timeframe=timeframe,
                            )
                            
                            if symbol in self._realtime_buffers:
                                self._realtime_buffers[symbol].append(point)
                                
                except Exception as e:
                    logger.error(f"Error in realtime collection for {symbol}: {e}")
            
            await asyncio.sleep(60)

    def stop_realtime_collection(self) -> None:
        """Stop real-time collection."""
        self._collecting = False
        logger.info("Stopped real-time collection")

    def get_realtime_buffer(self, symbol: str) -> list[MarketDataPoint]:
        """Get current buffer for symbol."""
        if symbol in self._realtime_buffers:
            return list(self._realtime_buffers[symbol])
        return []
