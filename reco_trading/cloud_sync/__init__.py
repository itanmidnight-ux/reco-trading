"""Cloud Sync System for Reco-Trading
Provides backup, synchronization and multi-device access
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import aiofiles

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Sync operation status"""
    IDLE = "idle"
    SYNCING = "syncing"
    ERROR = "error"
    COMPLETED = "completed"


class SyncItemType(Enum):
    """Type of item to sync"""
    CONFIG = "config"
    STRATEGY = "strategy"
    BOT_STATE = "bot_state"
    TRADING_HISTORY = "trading_history"
    MODELS = "models"
    SETTINGS = "settings"
    DATABASE = "database"


@dataclass
class SyncItem:
    """Item to be synced"""
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    item_type: SyncItemType = SyncItemType.CONFIG
    name: str = ""
    local_path: str = ""
    remote_path: str = ""
    checksum: str = ""
    size: int = 0
    
    # Version control
    version: int = 1
    previous_version: Optional[int] = None
    
    # Sync status
    last_synced: Optional[datetime] = None
    sync_status: SyncStatus = SyncStatus.IDLE
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class CloudConfig:
    """Configuration for cloud sync"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Provider settings
    provider: str = "local"  # local, s3, gdrive, dropbox
    access_token: str = ""
    refresh_token: str = ""
    bucket_name: str = "reco-trading-backup"
    
    # Sync settings
    auto_sync: bool = True
    sync_interval_minutes: int = 15
    max_retries: int = 3
    
    # What to sync
    sync_configs: bool = True
    sync_strategies: bool = True
    sync_models: bool = True
    sync_history: bool = True
    sync_database: bool = False  # Can be large
    
    # Exclusions
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "*.log", "*.tmp", "__pycache__", ".git", "node_modules"
    ])
    
    # Status
    is_active: bool = True
    last_sync: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SyncResult:
    """Result of sync operation"""
    success: bool = False
    items_synced: int = 0
    items_failed: int = 0
    bytes_transferred: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class CloudSyncManager:
    """Manages cloud synchronization"""
    
    def __init__(self, config: Optional[CloudConfig] = None):
        self.config = config or CloudConfig()
        self.sync_items: dict[str, SyncItem] = {}
        self.sync_queue: list[SyncItem] = []
        self.is_syncing = False
        
        self.local_backup_dir = Path("./backup")
        self.local_backup_dir.mkdir(exist_ok=True)
        
    async def initialize(self) -> bool:
        """Initialize cloud sync"""
        logger.info(f"Initializing cloud sync with provider: {self.config.provider}")
        
        # Initialize storage based on provider
        if self.config.provider == "local":
            return await self._init_local_storage()
        elif self.config.provider in ["s3", "gdrive", "dropbox"]:
            # TODO: Implement cloud provider integration
            logger.warning(f"Provider {self.config.provider} not yet implemented, using local")
            return await self._init_local_storage()
            
        return False
    
    async def _init_local_storage(self) -> bool:
        """Initialize local backup storage"""
        try:
            self.local_backup_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Local backup directory: {self.local_backup_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize local storage: {e}")
            return False
    
    async def add_sync_item(self, item: SyncItem) -> str:
        """Add an item to sync"""
        self.sync_items[item.item_id] = item
        self.sync_queue.append(item)
        logger.info(f"Added to sync queue: {item.name}")
        return item.item_id
    
    async def sync_now(self) -> SyncResult:
        """Perform sync operation"""
        if self.is_syncing:
            logger.warning("Sync already in progress")
            return SyncResult(success=False, errors=["Sync already in progress"])
            
        self.is_syncing = True
        result = SyncResult()
        
        start_time = datetime.now()
        
        try:
            while self.sync_queue:
                item = self.sync_queue.pop(0)
                sync_result = await self._sync_item(item)
                
                if sync_result:
                    result.items_synced += 1
                    result.bytes_transferred += item.size
                    item.last_synced = datetime.now()
                    item.sync_status = SyncStatus.COMPLETED
                else:
                    result.items_failed += 1
                    
            result.success = result.items_failed == 0
            
        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Sync error: {e}")
            
        finally:
            self.is_syncing = False
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            self.config.last_sync = datetime.now()
            
        logger.info(f"Sync completed: {result.items_synced} items, {result.bytes_transferred} bytes")
        return result
    
    async def _sync_item(self, item: SyncItem) -> bool:
        """Sync a single item"""
        try:
            # Calculate checksum
            if os.path.exists(item.local_path):
                item.checksum = await self._calculate_checksum(item.local_path)
                item.size = os.path.getsize(item.local_path)
            else:
                logger.warning(f"Local path not found: {item.local_path}")
                return False
                
            # Upload to cloud/local backup
            if self.config.provider == "local":
                return await self._sync_to_local(item)
            else:
                # TODO: Cloud provider sync
                return await self._sync_to_local(item)
                
        except Exception as e:
            logger.error(f"Failed to sync item {item.name}: {e}")
            item.sync_status = SyncStatus.ERROR
            return False
    
    async def _sync_to_local(self, item: SyncItem) -> bool:
        """Sync to local backup"""
        try:
            # Create remote path
            remote_path = self.local_backup_dir / item.item_type.value / item.name
            
            # Create parent directories
            remote_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(item.local_path, remote_path)
            
            # Update item
            item.remote_path = str(remote_path)
            item.version += 1
            item.updated_at = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Local sync failed: {e}")
            return False
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def restore_item(self, item_id: str) -> bool:
        """Restore an item from cloud"""
        if item_id not in self.sync_items:
            return False
            
        item = self.sync_items[item_id]
        
        try:
            if os.path.exists(item.remote_path):
                shutil.copy2(item.remote_path, item.local_path)
                logger.info(f"Restored: {item.name}")
                return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            
        return False
    
    async def restore_all(self) -> SyncResult:
        """Restore all items from backup"""
        result = SyncResult()
        
        for item in self.sync_items.values():
            if await self.restore_item(item.item_id):
                result.items_synced += 1
            else:
                result.items_failed += 1
                
        result.success = result.items_failed == 0
        return result
    
    async def backup_database(self, db_path: str) -> str:
        """Create database backup"""
        backup_name = f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_path = self.local_backup_dir / "database" / backup_name
        
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(db_path, backup_path)
            logger.info(f"Database backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return ""
    
    async def get_backup_list(self) -> list[dict]:
        """Get list of available backups"""
        backups = []
        
        for item in self.sync_items.values():
            backups.append({
                "name": item.name,
                "type": item.item_type.value,
                "version": item.version,
                "size": item.size,
                "last_synced": item.last_synced.isoformat() if item.last_synced else None,
                "status": item.sync_status.value,
            })
            
        return backups
    
    async def cleanup_old_backups(self, days: int = 30) -> int:
        """Clean up backups older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        cleaned = 0
        
        for item in list(self.sync_items.values()):
            if item.last_synced and item.last_synced < cutoff:
                if os.path.exists(item.remote_path):
                    os.remove(item.remote_path)
                    cleaned += 1
                    
        logger.info(f"Cleaned up {cleaned} old backups")
        return cleaned
    
    def get_status(self) -> dict:
        """Get sync status"""
        return {
            "provider": self.config.provider,
            "is_syncing": self.is_syncing,
            "last_sync": self.config.last_sync.isoformat() if self.config.last_sync else None,
            "items_queued": len(self.sync_queue),
            "items_tracked": len(self.sync_items),
            "backup_dir": str(self.local_backup_dir),
        }


# Global instance
_cloud_sync_manager: Optional[CloudSyncManager] = None


def get_cloud_sync_manager(config: Optional[CloudConfig] = None) -> CloudSyncManager:
    """Get or create cloud sync manager"""
    global _cloud_sync_manager
    if _cloud_sync_manager is None:
        _cloud_sync_manager = CloudSyncManager(config)
    return _cloud_sync_manager


__all__ = [
    "SyncStatus",
    "SyncItemType",
    "SyncItem",
    "CloudConfig",
    "SyncResult",
    "CloudSyncManager",
    "get_cloud_sync_manager",
]