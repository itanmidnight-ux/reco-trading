from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AgentCapability(Enum):
    MARKET_ANALYSIS = "market_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_EXECUTION = "trade_execution"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    REGIME_DETECTION = "regime_detection"
    PORTFOLIO_MANAGEMENT = "portfolio_management"


class AgentRole(Enum):
    ANALYST = "analyst"
    RISK_MANAGER = "risk_manager"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"
    OBSERVER = "observer"


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: dict
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: str = "info"
    priority: int = 0

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
            "priority": self.priority
        }


@dataclass
class MarketContext:
    symbol: str
    price: float
    volume_24h: float
    volatility: float
    trend: str
    regime: str
    timestamp: datetime = field(default_factory=datetime.now)
    additional_data: dict = field(default_factory=dict)


@dataclass
class TradingDecision:
    action: str
    confidence: float
    reasoning: str
    risk_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentConfig:
    name: str
    role: AgentRole
    capabilities: list[AgentCapability]
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 30
    enable_caching: bool = True
    max_retries: int = 3


class BaseLLMAgent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._is_running = False
        self._last_response: dict | None = None
        self._response_cache: dict = {}
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0
        }

    @abstractmethod
    async def process(self, input_data: dict) -> dict:
        pass

    async def _call_llm(self, prompt: str, system_prompt: str | None = None) -> dict:
        start_time = time.time()
        
        cache_key = f"{prompt[:100]}:{system_prompt or ''}"
        
        if self.config.enable_caching and cache_key in self._response_cache:
            self.logger.debug(f"Cache hit for {cache_key[:50]}...")
            return self._response_cache[cache_key]
        
        try:
            result = await self._ollama_request(prompt, system_prompt)
            
            self._stats["total_requests"] += 1
            self._stats["successful_requests"] += 1
            
            response_time = time.time() - start_time
            self._stats["avg_response_time"] = (
                (self._stats["avg_response_time"] * (self._stats["total_requests"] - 1) + response_time)
                / self._stats["total_requests"]
            )
            
            if self.config.enable_caching:
                self._response_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self._stats["total_requests"] += 1
            self._stats["failed_requests"] += 1
            self.logger.error(f"LLM request failed: {e}")
            return self._generate_fallback_response(prompt)

    async def _ollama_request(self, prompt: str, system_prompt: str | None = None) -> dict:
        import httpx
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return {
                    "response": data.get("response", ""),
                    "done": data.get("done", True),
                    "context": data.get("context", {})
                }
            except httpx.ConnectError:
                return self._generate_fallback_response(prompt)

    def _generate_fallback_response(self, prompt: str) -> dict:
        prompt_lower = prompt.lower()
        
        if "analyze" in prompt_lower or "market" in prompt_lower:
            return {
                "response": "NEUTRAL: Insufficient data for analysis. Maintain current position.",
                "done": True
            }
        elif "risk" in prompt_lower:
            return {
                "response": "MEDIUM: Risk level acceptable. Continue with current parameters.",
                "done": True
            }
        elif "trade" in prompt_lower or "buy" in prompt_lower or "sell" in prompt_lower:
            return {
                "response": "HOLD: No clear signal. Wait for better opportunity.",
                "done": True
            }
        
        return {
            "response": "HOLD: Unable to process request. Using default response.",
            "done": True
        }

    async def send_message(self, receiver: str, content: dict, 
                          message_type: str = "info", priority: int = 0) -> None:
        message = AgentMessage(
            sender=self.config.name,
            receiver=receiver,
            content=content,
            message_type=message_type,
            priority=priority
        )
        await self._message_queue.put(message)

    async def receive_message(self) -> AgentMessage | None:
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "cache_size": len(self._response_cache),
            "queue_size": self._message_queue.qsize()
        }

    async def start(self) -> None:
        self._is_running = True
        self.logger.info(f"Agent {self.config.name} started")

    async def stop(self) -> None:
        self._is_running = False
        self.logger.info(f"Agent {self.config.name} stopped")

    def clear_cache(self) -> None:
        self._response_cache.clear()
        self.logger.info(f"Cache cleared for agent {self.config.name}")


class AgentMessageBus:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._subscribers: dict[str, list[str]] = {}
        self._message_log: list[dict] = []
        self._max_log_size = 1000

    def subscribe(self, agent_name: str, topics: list[str]) -> None:
        for topic in topics:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            if agent_name not in self._subscribers[topic]:
                self._subscribers[topic].append(agent_name)

    async def publish(self, topic: str, message: AgentMessage) -> None:
        self._message_log.append(message.to_dict())
        
        if len(self._message_log) > self._max_log_size:
            self._message_log = self._message_log[-self._max_log_size:]
        
        self.logger.debug(f"Published to {topic}: {message.sender} -> {message.receiver}")

    def get_subscribers(self, topic: str) -> list[str]:
        return self._subscribers.get(topic, [])

    def get_recent_messages(self, limit: int = 50) -> list[dict]:
        return self._message_log[-limit:]


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, BaseLLMAgent] = {}
        self._roles: dict[AgentRole, list[str]] = {}

    def register(self, agent: BaseLLMAgent) -> None:
        self._agents[agent.config.name] = agent
        
        role = agent.config.role
        if role not in self._roles:
            self._roles[role] = []
        self._roles[role].append(agent.config.name)

    def unregister(self, agent_name: str) -> None:
        if agent_name in self._agents:
            agent = self._agents[agent_name]
            role = agent.config.role
            self._roles[role].remove(agent_name)
            del self._agents[agent_name]

    def get_agent(self, name: str) -> BaseLLMAgent | None:
        return self._agents.get(name)

    def get_agents_by_role(self, role: AgentRole) -> list[BaseLLMAgent]:
        names = self._roles.get(role, [])
        return [self._agents[name] for name in names if name in self._agents]

    def get_all_agents(self) -> list[BaseLLMAgent]:
        return list(self._agents.values())

    def get_stats(self) -> dict:
        return {
            "total_agents": len(self._agents),
            "agents_by_role": {role.value: len(agents) for role, agents in self._roles.items()},
            "agent_names": list(self._agents.keys())
        }