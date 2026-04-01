"""Database Models for new trading features
Extends the existing database with models for Copy Trading, Grid Bots, Marketplace, etc.
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, JSON, Enum as SQLEnum
from sqlalchemy.sql import func

from reco_trading.database.models import Base


class TraderProfileModel(Base):
    """Trader profile for copy trading"""
    __tablename__ = "trader_profiles"
    
    trader_id = Column(String(100), primary_key=True)
    username = Column(String(100))
    display_name = Column(String(200))
    avatar_url = Column(String(500))
    bio = Column(Text)
    tier = Column(String(50), default="newcomer")
    
    # Statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    total_pnl_percent = Column(Float, default=0.0)
    avg_trade_duration = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    
    # Follower metrics
    follower_count = Column(Integer, default=0)
    total_aum = Column(Float, default=0.0)
    success_rate = Column(Float, default=0.0)
    
    # Status
    is_public = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)
    allow_copy = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class FollowerConfigModel(Base):
    """Follower configuration for copy trading"""
    __tablename__ = "follower_configs"
    
    config_id = Column(String(100), primary_key=True)
    follower_id = Column(String(100))
    trader_id = Column(String(100))
    
    copy_ratio = Column(Float, default=1.0)
    max_investment = Column(Float, default=10000.0)
    stop_loss_percent = Column(Float, default=5.0)
    take_profit_percent = Column(Float, default=10.0)
    
    auto_close_on_sl = Column(Boolean, default=True)
    auto_close_on_tp = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=func.now())


class CopiedTradeModel(Base):
    """Record of copied trades"""
    __tablename__ = "copied_trades"
    
    trade_id = Column(String(100), primary_key=True)
    original_trader_id = Column(String(100))
    follower_id = Column(String(100))
    original_signal_id = Column(String(100))
    
    symbol = Column(String(50))
    action = Column(String(20))
    amount = Column(Float)
    entry_price = Column(Float)
    current_price = Column(Float)
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float, default=0.0)
    
    status = Column(String(20), default="open")
    opened_at = Column(DateTime, default=func.now())
    closed_at = Column(DateTime, nullable=True)


class GridBotConfigModel(Base):
    """Grid bot configuration"""
    __tablename__ = "grid_bot_configs"
    
    config_id = Column(String(100), primary_key=True)
    name = Column(String(200))
    symbol = Column(String(50))
    mode = Column(String(50), default="classic")
    
    lower_price = Column(Float)
    upper_price = Column(Float)
    grid_count = Column(Integer, default=10)
    total_investment = Column(Float, default=1000.0)
    order_quantity = Column(Float)
    
    take_profit_percent = Column(Float, default=2.0)
    stop_loss_percent = Column(Float, default=10.0)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class StrategyTemplateModel(Base):
    """Strategy template for marketplace"""
    __tablename__ = "strategy_templates"
    
    template_id = Column(String(100), primary_key=True)
    name = Column(String(200))
    description = Column(Text)
    category = Column(String(50))
    
    author_id = Column(String(100))
    author_name = Column(String(200))
    
    config = Column(JSON)
    indicators = Column(JSON)
    entry_conditions = Column(JSON)
    exit_conditions = Column(JSON)
    risk_rules = Column(JSON)
    
    pricing = Column(String(20), default="free")
    price = Column(Float, default=0.0)
    
    downloads = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    ratings_count = Column(Integer, default=0)
    
    status = Column(String(20), default="draft")
    is_featured = Column(Boolean, default=False)
    
    tags = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class TemplateReviewModel(Base):
    """Review of strategy template"""
    __tablename__ = "template_reviews"
    
    review_id = Column(String(100), primary_key=True)
    template_id = Column(String(100))
    user_id = Column(String(100))
    user_name = Column(String(200))
    
    rating = Column(Integer, default=5)
    title = Column(String(200))
    content = Column(Text)
    pros = Column(JSON)
    cons = Column(JSON)
    
    is_helpful = Column(Boolean, default=False)
    helpful_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=func.now())


class SyncItemModel(Base):
    """Cloud sync items"""
    __tablename__ = "sync_items"
    
    item_id = Column(String(100), primary_key=True)
    item_type = Column(String(50))
    name = Column(String(200))
    local_path = Column(String(500))
    remote_path = Column(String(500))
    checksum = Column(String(100))
    size = Column(Integer, default=0)
    
    version = Column(Integer, default=1)
    last_synced = Column(DateTime, nullable=True)
    sync_status = Column(String(20), default="idle")
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class MobileDeviceModel(Base):
    """Registered mobile devices"""
    __tablename__ = "mobile_devices"
    
    device_id = Column(String(100), primary_key=True)
    user_id = Column(String(100))
    device_type = Column(String(20))
    device_name = Column(String(200))
    push_token = Column(String(500))
    app_version = Column(String(20))
    
    notifications_enabled = Column(Boolean, default=True)
    trade_notifications = Column(Boolean, default=True)
    signal_notifications = Column(Boolean, default=True)
    
    is_active = Column(Boolean, default=True)
    last_seen = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())


class WebhookSignalModel(Base):
    """TradingView webhook signals"""
    __tablename__ = "webhook_signals"
    
    signal_id = Column(String(100), primary_key=True)
    source = Column(String(50))
    signal_type = Column(String(20))
    symbol = Column(String(50))
    price = Column(Float)
    
    quantity = Column(Float, nullable=True)
    order_type = Column(String(20), default="market")
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    
    strategy_name = Column(String(200))
    timeframe = Column(String(20))
    
    is_valid = Column(Boolean, default=False)
    validation_error = Column(Text, nullable=True)
    processed = Column(Boolean, default=False)
    
    raw_data = Column(JSON)
    
    timestamp = Column(DateTime, default=func.now())


# Function to create all tables
def create_all_tables():
    """Create all new tables"""
    from sqlalchemy import create_engine
    from reco_trading.config.settings import Settings
    
    settings = Settings()
    dsn = settings.database_url or settings.postgres_dsn
    
    if dsn:
        engine = create_engine(dsn.replace("+asyncpg", "").replace("postgresql", "postgresql+psycopg2"))
        Base.metadata.create_all(engine)
        print("✅ All new tables created successfully")


__all__ = [
    "TraderProfileModel",
    "FollowerConfigModel", 
    "CopiedTradeModel",
    "GridBotConfigModel",
    "StrategyTemplateModel",
    "TemplateReviewModel",
    "SyncItemModel",
    "MobileDeviceModel",
    "WebhookSignalModel",
    "create_all_tables",
]