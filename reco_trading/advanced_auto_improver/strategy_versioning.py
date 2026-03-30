"""
Strategy Versioning System Module.
Manages strategy versions with rollback capability.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyVersion:
    """A versioned strategy."""
    version_id: str
    strategy_id: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = False
    is_stable: bool = False
    parent_version: Optional[str] = None
    notes: str = ""


class StrategyVersioning:
    """Manages strategy versions."""

    def __init__(self, versions_dir: Optional[Path] = None):
        self.versions_dir = versions_dir or Path("./user_data/strategies/versions")
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        self._versions: dict[str, StrategyVersion] = {}
        self._load_existing_versions()

    def _load_existing_versions(self) -> None:
        """Load existing versions from disk."""
        for file_path in self.versions_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                version = StrategyVersion(
                    version_id=data["version_id"],
                    strategy_id=data["strategy_id"],
                    parameters=data["parameters"],
                    metrics=data.get("metrics", {}),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    is_active=data.get("is_active", False),
                    is_stable=data.get("is_stable", False),
                    parent_version=data.get("parent_version"),
                    notes=data.get("notes", ""),
                )
                
                self._versions[version.version_id] = version
                
            except Exception as e:
                logger.warning(f"Error loading version {file_path}: {e}")

    def create_version(
        self,
        strategy_id: str,
        parameters: dict[str, Any],
        metrics: dict[str, float],
        parent_version: Optional[str] = None,
        notes: str = "",
    ) -> StrategyVersion:
        """Create a new strategy version."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_id = f"{strategy_id}_v{timestamp}"
        
        version = StrategyVersion(
            version_id=version_id,
            strategy_id=strategy_id,
            parameters=parameters,
            metrics=metrics,
            parent_version=parent_version,
            notes=notes,
        )
        
        self._versions[version_id] = version
        self._save_version(version)
        
        logger.info(f"Created strategy version: {version_id}")
        
        return version

    def _save_version(self, version: StrategyVersion) -> None:
        """Save version to disk."""
        file_path = self.versions_dir / f"{version.version_id}.json"
        
        data = {
            "version_id": version.version_id,
            "strategy_id": version.strategy_id,
            "parameters": version.parameters,
            "metrics": version.metrics,
            "created_at": version.created_at.isoformat(),
            "is_active": version.is_active,
            "is_stable": version.is_stable,
            "parent_version": version.parent_version,
            "notes": version.notes,
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def activate_version(self, version_id: str) -> bool:
        """Activate a strategy version."""
        if version_id not in self._versions:
            return False
        
        for v in self._versions.values():
            v.is_active = False
        
        self._versions[version_id].is_active = True
        self._save_version(self._versions[version_id])
        
        logger.info(f"Activated strategy version: {version_id}")
        
        return True

    def mark_stable(self, version_id: str) -> bool:
        """Mark a version as stable."""
        if version_id not in self._versions:
            return False
        
        self._versions[version_id].is_stable = True
        self._save_version(self._versions[version_id])
        
        logger.info(f"Marked version as stable: {version_id}")
        
        return True

    def get_active_version(self, strategy_id: str) -> Optional[StrategyVersion]:
        """Get active version for strategy."""
        for v in self._versions.values():
            if v.strategy_id == strategy_id and v.is_active:
                return v
        return None

    def get_latest_version(self, strategy_id: str) -> Optional[StrategyVersion]:
        """Get latest version for strategy."""
        strategy_versions = [
            v for v in self._versions.values()
            if v.strategy_id == strategy_id
        ]
        
        if not strategy_versions:
            return None
        
        return max(strategy_versions, key=lambda v: v.created_at)

    def get_stable_version(self, strategy_id: str) -> Optional[StrategyVersion]:
        """Get latest stable version for strategy."""
        strategy_versions = [
            v for v in self._versions.values()
            if v.strategy_id == strategy_id and v.is_stable
        ]
        
        if not strategy_versions:
            return None
        
        return max(strategy_versions, key=lambda v: v.created_at)

    def get_version_history(self, strategy_id: str) -> list[StrategyVersion]:
        """Get version history for strategy."""
        versions = [
            v for v in self._versions.values()
            if v.strategy_id == strategy_id
        ]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback to a specific version."""
        if version_id not in self._versions:
            return False
        
        version = self._versions[version_id]
        
        self.activate_version(version_id)
        
        logger.info(f"Rolled back to version: {version_id}")
        
        return True

    def compare_versions(
        self,
        version_id1: str,
        version_id2: str,
    ) -> dict[str, Any]:
        """Compare two strategy versions."""
        if version_id1 not in self._versions or version_id2 not in self._versions:
            return {"error": "Version not found"}
        
        v1 = self._versions[version_id1]
        v2 = self._versions[version_id2]
        
        metrics_diff = {}
        for key in set(v1.metrics.keys()) | set(v2.metrics.keys()):
            m1 = v1.metrics.get(key, 0)
            m2 = v2.metrics.get(key, 0)
            metrics_diff[key] = {
                "v1": m1,
                "v2": m2,
                "diff": m2 - m1,
                "diff_pct": ((m2 - m1) / abs(m1) * 100) if m1 != 0 else 0,
            }
        
        return {
            "version1": {
                "id": v1.version_id,
                "created_at": v1.created_at.isoformat(),
                "is_active": v1.is_active,
                "is_stable": v1.is_stable,
            },
            "version2": {
                "id": v2.version_id,
                "created_at": v2.created_at.isoformat(),
                "is_active": v2.is_active,
                "is_stable": v2.is_stable,
            },
            "metrics_comparison": metrics_diff,
        }

    def delete_version(self, version_id: str) -> bool:
        """Delete a version (cannot delete active)."""
        if version_id not in self._versions:
            return False
        
        if self._versions[version_id].is_active:
            logger.warning("Cannot delete active version")
            return False
        
        del self._versions[version_id]
        
        file_path = self.versions_dir / f"{version_id}.json"
        if file_path.exists():
            file_path.unlink()
        
        logger.info(f"Deleted version: {version_id}")
        
        return True

    def to_dict(self, strategy_id: str) -> dict[str, Any]:
        """Export strategy versioning info."""
        history = self.get_version_history(strategy_id)
        
        return {
            "total_versions": len(history),
            "active_version": self.get_active_version(strategy_id).version_id if self.get_active_version(strategy_id) else None,
            "stable_version": self.get_stable_version(strategy_id).version_id if self.get_stable_version(strategy_id) else None,
            "latest_version": self.get_latest_version(strategy_id).version_id if self.get_latest_version(strategy_id) else None,
            "versions": [
                {
                    "version_id": v.version_id,
                    "created_at": v.created_at.isoformat(),
                    "is_active": v.is_active,
                    "is_stable": v.is_stable,
                    "metrics": v.metrics,
                }
                for v in history
            ],
        }
