from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BlacklistEntry:
    """Blacklist entry with reason and expiration."""
    pair: str
    reason: str
    added_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    is_permanent: bool = False


class BlacklistManager:
    """
    Manages pair blacklisting with temporary and permanent entries.
    Similar to FreqTrade's blacklist functionality.
    """

    def __init__(self, blacklist_file: str = "./user_data/blacklist.json"):
        self.logger = logging.getLogger(__name__)
        self.blacklist_file = Path(blacklist_file)
        self.blacklist: dict[str, BlacklistEntry] = {}
        self._load_blacklist()

    def _load_blacklist(self) -> None:
        """Load blacklist from file."""
        try:
            if self.blacklist_file.exists():
                with open(self.blacklist_file) as f:
                    data = json.load(f)
                    
                for pair, entry_data in data.items():
                    expires_at = None
                    if entry_data.get("expires_at"):
                        expires_at = datetime.fromisoformat(entry_data["expires_at"])
                    
                    self.blacklist[pair] = BlacklistEntry(
                        pair=pair,
                        reason=entry_data.get("reason", "unknown"),
                        added_at=datetime.fromisoformat(entry_data["added_at"]),
                        expires_at=expires_at,
                        is_permanent=entry_data.get("is_permanent", False),
                    )
                
                self.logger.info(f"Loaded {len(self.blacklist)} blacklist entries")
        except Exception as exc:
            self.logger.error(f"Failed to load blacklist: {exc}")

    def _save_blacklist(self) -> None:
        """Save blacklist to file."""
        try:
            self.blacklist_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for pair, entry in self.blacklist.items():
                data[pair] = {
                    "reason": entry.reason,
                    "added_at": entry.added_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                    "is_permanent": entry.is_permanent,
                }
            
            with open(self.blacklist_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as exc:
            self.logger.error(f"Failed to save blacklist: {exc}")

    def add(
        self,
        pair: str,
        reason: str = "manual",
        duration_minutes: int | None = None,
    ) -> bool:
        """Add a pair to blacklist."""
        
        expires_at = None
        is_permanent = False
        
        if duration_minutes is None or duration_minutes <= 0:
            is_permanent = True
        else:
            expires_at = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        
        self.blacklist[pair.upper()] = BlacklistEntry(
            pair=pair.upper(),
            reason=reason,
            is_permanent=is_permanent,
            expires_at=expires_at,
        )
        
        self._save_blacklist()
        self.logger.info(f"Added {pair} to blacklist: {reason} (permanent: {is_permanent})")
        
        return True

    def remove(self, pair: str) -> bool:
        """Remove a pair from blacklist."""
        
        pair = pair.upper()
        
        if pair in self.blacklist:
            del self.blacklist[pair]
            self._save_blacklist()
            self.logger.info(f"Removed {pair} from blacklist")
            return True
        
        return False

    def is_blacklisted(self, pair: str) -> bool:
        """Check if a pair is blacklisted."""
        
        pair = pair.upper()
        
        if pair not in self.blacklist:
            return False
        
        entry = self.blacklist[pair]
        
        if entry.is_permanent:
            return True
        
        if entry.expires_at and datetime.now(timezone.utc) > entry.expires_at:
            del self.blacklist[pair]
            self._save_blacklist()
            self.logger.info(f"Expired blacklist entry removed: {pair}")
            return False
        
        return True

    def get_all(self) -> list[str]:
        """Get all blacklisted pairs."""
        
        self._cleanup_expired()
        
        return list(self.blacklist.keys())

    def get_blacklist_info(self) -> list[dict]:
        """Get detailed blacklist information."""
        
        self._cleanup_expired()
        
        info = []
        
        for pair, entry in self.blacklist.items():
            info.append({
                "pair": entry.pair,
                "reason": entry.reason,
                "added_at": entry.added_at.isoformat(),
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                "is_permanent": entry.is_permanent,
            })
        
        return info

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        
        expired = []
        
        for pair, entry in self.blacklist.items():
            if not entry.is_permanent and entry.expires_at:
                if datetime.now(timezone.utc) > entry.expires_at:
                    expired.append(pair)
        
        for pair in expired:
            del self.blacklist[pair]
        
        if expired:
            self._save_blacklist()

    def clear(self) -> None:
        """Clear all blacklist entries."""
        
        self.blacklist.clear()
        self._save_blacklist()
        self.logger.info("Blacklist cleared")

    def add_pairs_from_list(self, pairs: list[str], reason: str = "batch") -> int:
        """Add multiple pairs to blacklist."""
        
        count = 0
        
        for pair in pairs:
            if self.add(pair, reason):
                count += 1
        
        return count

    def get_stats(self) -> dict:
        """Get blacklist statistics."""
        
        self._cleanup_expired()
        
        permanent = sum(1 for e in self.blacklist.values() if e.is_permanent)
        temporary = len(self.blacklist) - permanent
        
        return {
            "total": len(self.blacklist),
            "permanent": permanent,
            "temporary": temporary,
            "file": str(self.blacklist_file),
        }


def create_blacklist_from_volume(
    tickers: dict,
    min_volume: float = 100000,
    max_volume: float | None = None,
) -> list[str]:
    """Create blacklist based on volume (useful for low volume pairs)."""
    
    blacklist = []
    
    for pair, ticker in tickers.items():
        volume = float(ticker.get("quoteVolume", 0))
        
        if volume < min_volume:
            blacklist.append(pair)
        
        if max_volume and volume > max_volume:
            blacklist.append(pair)
    
    return blacklist


def create_blacklist_from_price(
    tickers: dict,
    min_price: float = 0.0001,
    max_price: float | None = None,
) -> list[str]:
    """Create blacklist based on price."""
    
    blacklist = []
    
    for pair, ticker in tickers.items():
        price = float(ticker.get("last", 0))
        
        if price < min_price:
            blacklist.append(pair)
        
        if max_price and price > max_price:
            blacklist.append(pair)
    
    return blacklist
