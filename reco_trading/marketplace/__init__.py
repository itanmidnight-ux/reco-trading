"""Strategy Templates Marketplace for Reco-Trading
Provides community sharing of trading strategies and bot configurations
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TemplateCategory(Enum):
    """Category of strategy template"""
    DCA = "dca"
    GRID = "grid"
    SCALPING = "scalping"
    SWING = "swing"
    ARBITRAGE = "arbitrage"
    MARTINGALE = "martingale"
    CUSTOM = "custom"
    AI_ML = "ai_ml"


class TemplateStatus(Enum):
    """Status of template"""
    DRAFT = "draft"
    PUBLISHED = "published"
    FEATURED = "featured"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class TemplatePricing(Enum):
    """Pricing model for template"""
    FREE = "free"
    PAID = "paid"
    SUBSCRIPTION = "subscription"
    DONATION = "donation"


@dataclass
class StrategyTemplate:
    """Trading strategy template"""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Basic info
    name: str = ""
    description: str = ""
    category: TemplateCategory = TemplateCategory.CUSTOM
    
    # Author
    author_id: str = ""
    author_name: str = ""
    
    # Content
    config: dict = field(default_factory=dict)
    indicators: dict = field(default_factory=dict)
    entry_conditions: list[str] = field(default_factory=list)
    exit_conditions: list[str] = field(default_factory=list)
    risk_rules: dict = field(default_factory=dict)
    
    # Performance (if backtested)
    backtest_result: Optional[dict] = None
    
    # Metadata
    version: str = "1.0.0"
    compatibility: list[str] = field(default_factory=list)  # e.g., ["binance", "bybit"]
    
    # Pricing
    pricing: TemplatePricing = TemplatePricing.FREE
    price: float = 0.0
    currency: str = "USD"
    
    # Stats
    downloads: int = 0
    rating: float = 0.0
    ratings_count: int = 0
    reviews_count: int = 0
    
    # Status
    status: TemplateStatus = TemplateStatus.DRAFT
    is_featured: bool = False
    is_verified: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    published_at: Optional[datetime] = None
    
    # Tags
    tags: list[str] = field(default_factory=list)
    
    # Requirements
    min_capital: float = 0.0
    recommended_capital: float = 0.0
    risk_level: str = "medium"  # low, medium, high
    
    # Support
    support_url: str = ""
    documentation_url: str = ""


@dataclass
class TemplateReview:
    """Review of a template"""
    review_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    template_id: str = ""
    user_id: str = ""
    user_name: str = ""
    
    rating: int = 5  # 1-5
    title: str = ""
    content: str = ""
    
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    
    is_verified_purchase: bool = False
    is_helpful: bool = False
    helpful_count: int = 0
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BotTemplate:
    """Pre-configured bot template (ready to use)"""
    bot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Template reference
    template_id: str = ""
    
    # Bot config
    name: str = ""
    strategy: str = ""
    pairs: list[str] = field(default_factory=list)
    timeframe: str = "5m"
    
    # Parameters
    params: dict = field(default_factory=dict)
    
    # Risk
    max_position_size: float = 10.0
    stop_loss_percent: float = 5.0
    take_profit_percent: float = 10.0
    
    # Status
    is_active: bool = False
    
    # Stats
    total_trades: int = 0
    winning_trades: int = 0
    pnl: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class TemplateMarketplace:
    """Marketplace for trading strategy templates"""
    
    def __init__(self):
        self.templates: dict[str, StrategyTemplate] = {}
        self.reviews: dict[str, list[TemplateReview]] = {}
        self.bot_templates: dict[str, BotTemplate] = {}
        
    async def create_template(self, template: StrategyTemplate) -> str:
        """Create a new template"""
        self.templates[template.template_id] = template
        logger.info(f"Template created: {template.name}")
        return template.template_id
    
    async def update_template(self, template_id: str, 
                              updates: dict) -> Optional[StrategyTemplate]:
        """Update a template"""
        if template_id not in self.templates:
            return None
            
        template = self.templates[template_id]
        
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
                
        template.updated_at = datetime.now()
        return template
    
    async def publish_template(self, template_id: str) -> bool:
        """Publish a template"""
        if template_id not in self.templates:
            return False
            
        template = self.templates[template_id]
        template.status = TemplateStatus.PUBLISHED
        template.published_at = datetime.now()
        template.updated_at = datetime.now()
        
        logger.info(f"Template published: {template.name}")
        return True
    
    async def delete_template(self, template_id: str) -> bool:
        """Delete a template"""
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False
    
    async def get_template(self, template_id: str) -> Optional[StrategyTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    async def search_templates(
        self,
        query: str = "",
        category: Optional[TemplateCategory] = None,
        pricing: Optional[TemplatePricing] = None,
        min_rating: float = 0.0,
        tags: list[str] = None,
        limit: int = 20
    ) -> list[StrategyTemplate]:
        """Search templates"""
        results = []
        
        for template in self.templates.values():
            if template.status != TemplateStatus.PUBLISHED:
                continue
                
            # Filter by query
            if query:
                query_lower = query.lower()
                if (query_lower not in template.name.lower() and 
                    query_lower not in template.description.lower()):
                    continue
                    
            # Filter by category
            if category and template.category != category:
                continue
                
            # Filter by pricing
            if pricing and template.pricing != pricing:
                continue
                
            # Filter by rating
            if template.rating < min_rating:
                continue
                
            # Filter by tags
            if tags and not any(t in template.tags for t in tags):
                continue
                
            results.append(template)
            
        # Sort by rating and downloads
        results.sort(key=lambda x: (x.rating, x.downloads), reverse=True)
        return results[:limit]
    
    async def get_featured_templates(self, limit: int = 10) -> list[StrategyTemplate]:
        """Get featured templates"""
        featured = [
            t for t in self.templates.values()
            if t.status == TemplateStatus.PUBLISHED and t.is_featured
        ]
        featured.sort(key=lambda x: x.rating, reverse=True)
        return featured[:limit]
    
    async def get_popular_templates(self, limit: int = 10) -> list[StrategyTemplate]:
        """Get most popular templates"""
        popular = [
            t for t in self.templates.values()
            if t.status == TemplateStatus.PUBLISHED
        ]
        popular.sort(key=lambda x: x.downloads, reverse=True)
        return popular[:limit]
    
    async def get_templates_by_category(
        self, 
        category: TemplateCategory,
        limit: int = 20
    ) -> list[StrategyTemplate]:
        """Get templates by category"""
        category_templates = [
            t for t in self.templates.values()
            if t.status == TemplateStatus.PUBLISHED and t.category == category
        ]
        category_templates.sort(key=lambda x: x.rating, reverse=True)
        return category_templates[:limit]
    
    async def add_review(self, review: TemplateReview) -> str:
        """Add a review to a template"""
        if review.template_id not in self.reviews:
            self.reviews[review.template_id] = []
            
        self.reviews[review.template_id].append(review)
        
        # Update template rating
        template = self.templates.get(review.template_id)
        if template:
            reviews = self.reviews[review.template_id]
            template.rating = sum(r.rating for r in reviews) / len(reviews)
            template.ratings_count = len(reviews)
            
        return review.review_id
    
    async def get_reviews(self, template_id: str) -> list[TemplateReview]:
        """Get all reviews for a template"""
        return self.reviews.get(template_id, [])
    
    async def download_template(self, template_id: str) -> Optional[dict]:
        """Download a template (increment download count)"""
        template = self.templates.get(template_id)
        if template:
            template.downloads += 1
            return {
                "config": template.config,
                "indicators": template.indicators,
                "entry_conditions": template.entry_conditions,
                "exit_conditions": template.exit_conditions,
                "risk_rules": template.risk_rules,
            }
        return None
    
    # Bot Templates
    async def create_bot_template(self, bot: BotTemplate) -> str:
        """Create a bot template from strategy template"""
        self.bot_templates[bot.bot_id] = bot
        return bot.bot_id
    
    async def get_bot_template(self, bot_id: str) -> Optional[BotTemplate]:
        """Get bot template by ID"""
        return self.bot_templates.get(bot_id)
    
    async def get_user_bots(self, user_id: str) -> list[BotTemplate]:
        """Get all bot templates for a user"""
        return [b for b in self.bot_templates.values() if b.is_active]
    
    def get_marketplace_stats(self) -> dict:
        """Get marketplace statistics"""
        templates = [t for t in self.templates.values() 
                     if t.status == TemplateStatus.PUBLISHED]
        
        return {
            "total_templates": len(templates),
            "total_downloads": sum(t.downloads for t in templates),
            "avg_rating": sum(t.rating for t in templates) / len(templates) if templates else 0,
            "by_category": {
                c.value: len([t for t in templates if t.category == c])
                for c in TemplateCategory
            },
            "by_pricing": {
                p.value: len([t for t in templates if t.pricing == p])
                for p in TemplatePricing
            },
        }


# Global instance
_template_marketplace: Optional[TemplateMarketplace] = None


def get_template_marketplace() -> TemplateMarketplace:
    """Get or create template marketplace"""
    global _template_marketplace
    if _template_marketplace is None:
        _template_marketplace = TemplateMarketplace()
    return _template_marketplace


__all__ = [
    "TemplateCategory",
    "TemplateStatus",
    "TemplatePricing",
    "StrategyTemplate",
    "TemplateReview",
    "BotTemplate",
    "TemplateMarketplace",
    "get_template_marketplace",
]