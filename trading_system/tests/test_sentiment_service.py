from datetime import datetime, timedelta, timezone

from trading_system.app.services.sentiment.service import ParsedHeadline, SentimentService, SourceDocument


def test_score_texts_positive_negative_and_ambiguous() -> None:
    service = SentimentService()
    service._nlp_engine = None

    positive = service.score_texts(["Bullish rally and record growth with major adoption"])
    negative = service.score_texts(["Market crash after fraud lawsuit and outage"])
    ambiguous = service.score_texts(["Asset trades sideways with little change"])

    assert positive.score > 0
    assert negative.score < 0
    assert ambiguous.score == 0


def test_parse_json_and_rss_sources() -> None:
    service = SentimentService()
    fetched = datetime(2024, 1, 1, tzinfo=timezone.utc)

    json_doc = SourceDocument(
        body='{"articles": [{"title": "Partnership boosts adoption", "published_at": "2024-01-01T10:00:00Z"}] }',
        source_url='https://example.com/api/news',
        fetched_at=fetched,
    )
    rss_doc = SourceDocument(
        body='''<?xml version="1.0"?><rss><channel><item><title>Exchange hack reported</title><pubDate>Tue, 02 Jan 2024 10:00:00 GMT</pubDate></item></channel></rss>''',
        source_url='https://example.com/rss',
        fetched_at=fetched,
    )

    parsed = service.parse([json_doc, rss_doc])

    assert any('Partnership boosts adoption' in headline.text for headline in parsed)
    assert any('Exchange hack reported' in headline.text for headline in parsed)


def test_parse_malformed_source_fallback_is_resilient() -> None:
    service = SentimentService()
    fetched = datetime(2024, 1, 1, tzinfo=timezone.utc)
    malformed = SourceDocument(
        body='not-json and <broken><xml',
        source_url='https://example.com/bad',
        fetched_at=fetched,
    )

    parsed = service.parse([malformed])

    assert len(parsed) == 1
    assert parsed[0].text.startswith('not-json and <broken><xml')


def test_recency_weighting_prefers_fresh_news() -> None:
    service = SentimentService()
    service._nlp_engine = None

    now = datetime.now(tz=timezone.utc)
    fresh = ParsedHeadline(text='bullish rally', published_at=now, source_url='fresh')
    stale = ParsedHeadline(text='crash fraud outage', published_at=now - timedelta(hours=72), source_url='stale')

    snapshot = service.aggregate(service.score([fresh, stale]))

    assert snapshot.score > 0


def test_attention_event_uses_historical_zscore() -> None:
    service = SentimentService()
    service._nlp_engine = None
    service.minimum_zscore_samples = 5

    for _ in range(10):
        service.aggregate([(0.05, 1.0)])

    snapshot = service.aggregate([(1.0, 1.0)])

    assert snapshot.attention_event is True
