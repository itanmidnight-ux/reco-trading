from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import email.utils
import json
import math
import os
import re
from typing import Any, Iterable
import urllib.request
import xml.etree.ElementTree as ET


@dataclass
class SentimentSnapshot:
    score: float
    attention_event: bool


@dataclass
class SourceDocument:
    body: str
    source_url: str
    fetched_at: datetime


@dataclass
class ParsedHeadline:
    text: str
    published_at: datetime
    source_url: str


class SentimentService:
    POSITIVE = {'surge', 'bullish', 'approval', 'growth', 'partnership', 'adoption', 'rally', 'record'}
    NEGATIVE = {'hack', 'lawsuit', 'ban', 'bearish', 'crash', 'liquidation', 'fraud', 'outage'}
    DATE_FIELDS = ('published', 'published_at', 'pubDate', 'updated', 'date', 'created_at', 'time')
    TEXT_FIELDS = ('title', 'headline', 'summary', 'description', 'content', 'text', 'name')

    def __init__(self) -> None:
        feeds = os.getenv('NEWS_FEED_URLS', '')
        self.feed_urls = [f.strip() for f in feeds.split(',') if f.strip()]
        self.decay_half_life_hours = float(os.getenv('SENTIMENT_DECAY_HOURS', '24'))
        self.zscore_window = int(os.getenv('SENTIMENT_ZSCORE_WINDOW', '100'))
        self.zscore_threshold = float(os.getenv('SENTIMENT_ZSCORE_THRESHOLD', '2.0'))
        self.minimum_zscore_samples = int(os.getenv('SENTIMENT_MIN_ZSCORE_SAMPLES', '5'))
        self._history: deque[float] = deque(maxlen=max(self.zscore_window, self.minimum_zscore_samples + 1))
        self._nlp_engine = self._build_nlp_engine()

    def _build_nlp_engine(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            return lambda text: analyzer.polarity_scores(text).get('compound', 0.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            from textblob import TextBlob

            return lambda text: float(TextBlob(text).sentiment.polarity)
        except Exception:  # noqa: BLE001
            return None

    async def latest(self) -> SentimentSnapshot:
        documents = await self.collect()
        headlines = self.parse(documents)
        scored = self.score(headlines)
        return self.aggregate(scored)

    async def collect(self) -> list[SourceDocument]:
        if not self.feed_urls:
            return []
        documents: list[SourceDocument] = []
        for url in self.feed_urls[:5]:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                    payload = resp.read().decode('utf-8', errors='ignore')
                documents.append(
                    SourceDocument(
                        body=payload,
                        source_url=url,
                        fetched_at=datetime.now(tz=timezone.utc),
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        return documents

    def parse(self, documents: Iterable[SourceDocument]) -> list[ParsedHeadline]:
        parsed: list[ParsedHeadline] = []
        for document in documents:
            body = document.body.strip()
            if not body:
                continue
            parsed.extend(self._parse_json(body, document))
            parsed.extend(self._parse_xml(body, document))
            if not parsed or parsed[-1].source_url != document.source_url:
                parsed.append(ParsedHeadline(text=body[:280], published_at=document.fetched_at, source_url=document.source_url))
        return parsed

    def score(self, headlines: Iterable[ParsedHeadline]) -> list[tuple[float, float]]:
        scored: list[tuple[float, float]] = []
        now = datetime.now(tz=timezone.utc)
        for headline in headlines:
            sentiment = self._score_text(headline.text)
            decay_weight = self._weight_for_recency(now=now, published_at=headline.published_at)
            scored.append((sentiment, decay_weight))
        return scored

    def aggregate(self, scored: Iterable[tuple[float, float]]) -> SentimentSnapshot:
        weighted_sum = 0.0
        total_weight = 0.0
        for score, weight in scored:
            weighted_sum += score * weight
            total_weight += weight
        raw = 0.0 if total_weight == 0 else weighted_sum / total_weight
        bounded = max(-1.0, min(1.0, raw))
        attention_event = self._is_attention_event(bounded)
        self._history.append(bounded)
        return SentimentSnapshot(score=bounded, attention_event=attention_event)

    def score_texts(self, texts: Iterable[str]) -> SentimentSnapshot:
        now = datetime.now(tz=timezone.utc)
        headlines = [ParsedHeadline(text=text, published_at=now, source_url='inline') for text in texts]
        return self.aggregate(self.score(headlines))

    def _score_text(self, text: str) -> float:
        if self._nlp_engine is not None:
            try:
                return max(-1.0, min(1.0, float(self._nlp_engine(text))))
            except Exception:  # noqa: BLE001
                pass

        words = {w.lower() for w in re.findall(r'[a-zA-Z]+', text)}
        positives = len(words & self.POSITIVE)
        negatives = len(words & self.NEGATIVE)
        total = positives + negatives
        return 0.0 if total == 0 else (positives - negatives) / total

    def _weight_for_recency(self, now: datetime, published_at: datetime) -> float:
        age_hours = max(0.0, (now - published_at).total_seconds() / 3600)
        if self.decay_half_life_hours <= 0:
            return 1.0
        return math.exp(-age_hours / self.decay_half_life_hours)

    def _is_attention_event(self, current_score: float) -> bool:
        if len(self._history) < self.minimum_zscore_samples:
            return abs(current_score) >= 0.85
        mean = sum(self._history) / len(self._history)
        variance = sum((item - mean) ** 2 for item in self._history) / len(self._history)
        std_dev = math.sqrt(variance)
        if std_dev == 0:
            return abs(current_score - mean) >= 0.5
        z_score = (current_score - mean) / std_dev
        return abs(z_score) >= self.zscore_threshold

    def _parse_json(self, body: str, document: SourceDocument) -> list[ParsedHeadline]:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return []
        items = payload if isinstance(payload, list) else [payload]
        parsed: list[ParsedHeadline] = []
        for item in items:
            parsed.extend(self._parse_json_item(item, document))
        return parsed

    def _parse_json_item(self, item: Any, document: SourceDocument) -> list[ParsedHeadline]:
        if isinstance(item, list):
            parsed: list[ParsedHeadline] = []
            for entry in item:
                parsed.extend(self._parse_json_item(entry, document))
            return parsed
        if not isinstance(item, dict):
            return []

        text_candidates = [str(item[field]).strip() for field in self.TEXT_FIELDS if field in item and str(item[field]).strip()]
        published_at = self._parse_datetime(next((item[field] for field in self.DATE_FIELDS if field in item), None), document.fetched_at)

        parsed = [ParsedHeadline(text=text, published_at=published_at, source_url=document.source_url) for text in text_candidates]
        for value in item.values():
            parsed.extend(self._parse_json_item(value, document))
        return parsed

    def _parse_xml(self, body: str, document: SourceDocument) -> list[ParsedHeadline]:
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return []

        parsed: list[ParsedHeadline] = []
        for entry in root.findall('.//item') + root.findall('.//entry'):
            text = self._find_xml_text(entry, self.TEXT_FIELDS)
            if not text:
                continue
            published = self._find_xml_text(entry, self.DATE_FIELDS)
            parsed.append(
                ParsedHeadline(
                    text=text,
                    published_at=self._parse_datetime(published, document.fetched_at),
                    source_url=document.source_url,
                )
            )

        if parsed:
            return parsed

        title = self._find_xml_text(root, ('title',))
        if title:
            return [ParsedHeadline(text=title, published_at=document.fetched_at, source_url=document.source_url)]
        return []

    def _find_xml_text(self, node: ET.Element, names: Iterable[str]) -> str | None:
        targets = set(names)
        for child in node.iter():
            tag = child.tag.split('}')[-1]
            if tag in targets and child.text and child.text.strip():
                return child.text.strip()
        return None

    def _parse_datetime(self, value: Any, default: datetime) -> datetime:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except Exception:  # noqa: BLE001
                return default
        raw = str(value).strip()
        if not raw:
            return default

        try:
            parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        try:
            parsed_rfc = email.utils.parsedate_to_datetime(raw)
            return parsed_rfc if parsed_rfc.tzinfo else parsed_rfc.replace(tzinfo=timezone.utc)
        except Exception:  # noqa: BLE001
            return default
