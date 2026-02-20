from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Literal

from reco_trading.config.settings import Settings

try:
    from redis.asyncio import Redis
except Exception:  # pragma: no cover - redis is expected but keep import defensive.
    Redis = None  # type: ignore[assignment]

try:  # pragma: no cover - optional runtime dependency.
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.structs import OffsetAndMetadata
except Exception:  # pragma: no cover
    AIOKafkaConsumer = None  # type: ignore[assignment]
    AIOKafkaProducer = None  # type: ignore[assignment]
    OffsetAndMetadata = None  # type: ignore[assignment]

BrokerTaskType = Literal['features', 'inference', 'backtest', 'optimization', 'execution']


@dataclass(slots=True)
class BrokerMessage:
    task_type: BrokerTaskType
    payload: dict[str, Any]
    idempotency_key: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata.setdefault('attempt', 0)
        metadata.setdefault('created_at', datetime.now(UTC).isoformat())
        return {
            'task_type': self.task_type,
            'payload': self.payload,
            'idempotency_key': self.idempotency_key,
            'metadata': metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'BrokerMessage':
        return cls(
            task_type=data['task_type'],
            payload=data.get('payload', {}),
            idempotency_key=data['idempotency_key'],
            metadata=data.get('metadata', {}),
        )


@dataclass(slots=True)
class ReceivedMessage:
    broker_message: BrokerMessage
    receipt_handle: Any


class AbstractBroker(ABC):
    topic_types: tuple[BrokerTaskType, ...] = (
        'features',
        'inference',
        'backtest',
        'optimization',
        'execution',
    )

    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    async def publish(self, message: BrokerMessage) -> str | None:
        raise NotImplementedError

    @abstractmethod
    async def consume(
        self,
        task_type: BrokerTaskType,
        *,
        group: str,
        consumer: str,
        count: int = 1,
    ) -> AsyncIterator[ReceivedMessage]:
        raise NotImplementedError

    @abstractmethod
    async def ack(self, received: ReceivedMessage) -> None:
        raise NotImplementedError

    @abstractmethod
    async def nack(self, received: ReceivedMessage, *, reason: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class RedisStreamsBroker(AbstractBroker):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        if Redis is None:  # pragma: no cover - dependency guard.
            raise RuntimeError('redis no está instalado para RedisStreamsBroker.')

        self._redis = Redis.from_url(
            settings.redis_url,
            socket_timeout=settings.broker_operation_timeout_seconds,
            socket_connect_timeout=settings.broker_operation_timeout_seconds,
            decode_responses=True,
        )
        self._stream_names = {
            topic: f'{settings.broker_stream_prefix}:{topic}' for topic in self.topic_types
        }
        self._dlq_names = {topic: f'{stream}:dlq' for topic, stream in self._stream_names.items()}

    async def publish(self, message: BrokerMessage) -> str | None:
        idempotency_key = self._idempotency_key(message.idempotency_key)
        accepted = await self._redis.set(
            idempotency_key,
            '1',
            nx=True,
            ex=self.settings.broker_idempotency_ttl_seconds,
        )
        if not accepted:
            return None

        stream = self._stream_names[message.task_type]
        payload = {'data': json.dumps(message.to_dict())}
        message_id = await self._redis.xadd(
            stream,
            fields=payload,
            maxlen=self.settings.broker_stream_maxlen,
            approximate=True,
        )
        await self._trim_stream_retention(stream)
        return message_id

    async def consume(
        self,
        task_type: BrokerTaskType,
        *,
        group: str,
        consumer: str,
        count: int = 1,
    ) -> AsyncIterator[ReceivedMessage]:
        stream = self._stream_names[task_type]
        await self._ensure_group(stream=stream, group=group)

        messages = await self._redis.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: '>'},
            count=count,
            block=self.settings.broker_consume_block_ms,
        )

        for _, entries in messages:
            for entry_id, fields in entries:
                payload = json.loads(fields['data'])
                yield ReceivedMessage(
                    broker_message=BrokerMessage.from_dict(payload),
                    receipt_handle={'stream': stream, 'group': group, 'entry_id': entry_id},
                )

    async def ack(self, received: ReceivedMessage) -> None:
        handle = received.receipt_handle
        await self._redis.xack(handle['stream'], handle['group'], handle['entry_id'])

    async def nack(self, received: ReceivedMessage, *, reason: str) -> None:
        handle = received.receipt_handle
        message = received.broker_message
        metadata = dict(message.metadata)
        next_attempt = int(metadata.get('attempt', 0)) + 1
        metadata['attempt'] = next_attempt
        metadata['last_error'] = reason
        metadata['last_error_at'] = datetime.now(UTC).isoformat()

        if next_attempt > self.settings.broker_max_retries:
            dlq_payload = {
                'data': json.dumps(
                    BrokerMessage(
                        task_type=message.task_type,
                        payload=message.payload,
                        idempotency_key=message.idempotency_key,
                        metadata=metadata,
                    ).to_dict()
                )
            }
            dlq_stream = self._dlq_names[message.task_type]
            await self._redis.xadd(
                dlq_stream,
                fields=dlq_payload,
                maxlen=self.settings.broker_dlq_maxlen,
                approximate=True,
            )
            await self._trim_stream_retention(dlq_stream)
            await self.ack(received)
            return

        delay = min(
            self.settings.broker_retry_backoff_seconds * (2 ** (next_attempt - 1)),
            self.settings.broker_retry_backoff_max_seconds,
        )
        await asyncio.sleep(delay)

        retry_payload = {
            'data': json.dumps(
                BrokerMessage(
                    task_type=message.task_type,
                    payload=message.payload,
                    idempotency_key=message.idempotency_key,
                    metadata=metadata,
                ).to_dict()
            )
        }
        stream = self._stream_names[message.task_type]
        await self._redis.xadd(
            stream,
            fields=retry_payload,
            maxlen=self.settings.broker_stream_maxlen,
            approximate=True,
        )
        await self._trim_stream_retention(stream)
        await self.ack(received)

    async def close(self) -> None:
        await self._redis.aclose()

    async def _ensure_group(self, *, stream: str, group: str) -> None:
        try:
            await self._redis.xgroup_create(
                name=stream,
                groupname=group,
                id='0',
                mkstream=True,
            )
        except Exception as exc:  # pragma: no cover - response depends on redis server string.
            if 'BUSYGROUP' not in str(exc):
                raise

    def _idempotency_key(self, key: str) -> str:
        return f'{self.settings.broker_stream_prefix}:idempotency:{key}'

    async def _trim_stream_retention(self, stream: str) -> None:
        min_id = f"{int(datetime.now(UTC).timestamp() * 1000) - self.settings.broker_retention_ms}-0"
        await self._redis.xtrim(stream, minid=min_id, approximate=True)


class KafkaBroker(AbstractBroker):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        if AIOKafkaProducer is None or AIOKafkaConsumer is None:
            raise RuntimeError('KafkaBroker requiere aiokafka instalado.')

        self._topic_names = {
            topic: f'{settings.broker_topic_prefix}.{topic}' for topic in self.topic_types
        }
        self._dlq_names = {topic: f'{topic_name}.dlq' for topic, topic_name in self._topic_names.items()}
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            request_timeout_ms=int(settings.broker_operation_timeout_seconds * 1000),
        )
        self._consumers: dict[tuple[str, str], AIOKafkaConsumer] = {}
        self._started = False

    async def publish(self, message: BrokerMessage) -> str | None:
        await self._ensure_started()
        payload = json.dumps(message.to_dict()).encode('utf-8')
        metadata = await self._producer.send_and_wait(self._topic_names[message.task_type], payload)
        return f'{metadata.topic}:{metadata.partition}:{metadata.offset}'

    async def consume(
        self,
        task_type: BrokerTaskType,
        *,
        group: str,
        consumer: str,
        count: int = 1,
    ) -> AsyncIterator[ReceivedMessage]:
        await self._ensure_started()
        kafka_consumer = await self._get_consumer(task_type=task_type, group=group, consumer=consumer)
        records_map = await kafka_consumer.getmany(
            timeout_ms=self.settings.broker_consume_block_ms,
            max_records=count,
        )
        for topic_partition, records in records_map.items():
            for record in records:
                payload = json.loads(record.value.decode('utf-8'))
                yield ReceivedMessage(
                    broker_message=BrokerMessage.from_dict(payload),
                    receipt_handle={
                        'topic_partition': topic_partition,
                        'offset': record.offset,
                        'group': group,
                        'consumer': consumer,
                    },
                )

    async def ack(self, received: ReceivedMessage) -> None:
        handle = received.receipt_handle
        consumer = self._consumers[(handle['group'], handle['consumer'])]
        tp = handle['topic_partition']
        await consumer.commit({tp: OffsetAndMetadata(handle['offset'] + 1, '')})

    async def nack(self, received: ReceivedMessage, *, reason: str) -> None:
        message = received.broker_message
        metadata = dict(message.metadata)
        next_attempt = int(metadata.get('attempt', 0)) + 1
        metadata['attempt'] = next_attempt
        metadata['last_error'] = reason
        metadata['last_error_at'] = datetime.now(UTC).isoformat()

        if next_attempt > self.settings.broker_max_retries:
            await self._producer.send_and_wait(
                self._dlq_names[message.task_type],
                json.dumps(
                    BrokerMessage(
                        task_type=message.task_type,
                        payload=message.payload,
                        idempotency_key=message.idempotency_key,
                        metadata=metadata,
                    ).to_dict()
                ).encode('utf-8'),
            )
            await self.ack(received)
            return

        delay = min(
            self.settings.broker_retry_backoff_seconds * (2 ** (next_attempt - 1)),
            self.settings.broker_retry_backoff_max_seconds,
        )
        await asyncio.sleep(delay)

        await self.publish(
            BrokerMessage(
                task_type=message.task_type,
                payload=message.payload,
                idempotency_key=message.idempotency_key,
                metadata=metadata,
            )
        )
        await self.ack(received)

    async def close(self) -> None:
        for consumer in self._consumers.values():
            await consumer.stop()
        self._consumers.clear()
        if self._started:
            await self._producer.stop()
        self._started = False

    async def _ensure_started(self) -> None:
        if self._started:
            return
        await self._producer.start()
        self._started = True

    async def _get_consumer(self, *, task_type: BrokerTaskType, group: str, consumer: str) -> AIOKafkaConsumer:
        key = (group, consumer)
        if key in self._consumers:
            return self._consumers[key]

        consumer_client = AIOKafkaConsumer(
            self._topic_names[task_type],
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            group_id=group,
            client_id=consumer,
            enable_auto_commit=False,
            request_timeout_ms=int(self.settings.broker_operation_timeout_seconds * 1000),
        )
        await consumer_client.start()
        self._consumers[key] = consumer_client
        return consumer_client


def create_broker(settings: Settings) -> AbstractBroker:
    if settings.broker_backend == 'redis':
        return RedisStreamsBroker(settings)
    if settings.broker_backend == 'kafka':
        return KafkaBroker(settings)
    raise ValueError(f'Broker backend no soportado: {settings.broker_backend}')
