import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from llmq.core.config import Config
from llmq.core.models import Job, Result, QueueStats


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Config()
    config.rabbitmq_url = "amqp://test:test@localhost:5672/"
    config.vllm_queue_prefetch = 10
    config.vllm_gpu_memory_utilization = 0.9
    config.job_ttl_minutes = 5
    config.chunk_size = 100
    config.log_level = "DEBUG"
    return config


@pytest.fixture
def sample_job():
    """Sample job for testing."""
    return Job(id="test-job-001", prompt="Echo '{text}' back", text="Hello World")


@pytest.fixture
def sample_result():
    """Sample result for testing."""
    return Result(
        id="test-job-001",
        prompt="Echo 'Hello World' back",
        result="echo Hello World",
        worker_id="test-worker-001",
        duration_ms=150.5,
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def sample_queue_stats():
    """Sample queue statistics for testing."""
    return QueueStats(
        queue_name="test-queue",
        message_count=42,
        message_count_ready=40,
        message_count_unacknowledged=2,
        consumer_count=3,
        message_bytes=1024,
        message_bytes_ready=1000,
        message_bytes_unacknowledged=24,
        stats_source="management_api",
    )


@pytest.fixture
def mock_rabbitmq_message():
    """Mock RabbitMQ message for testing."""
    message = MagicMock()
    message.body = b'{"id": "test-job-001", "prompt": "Echo \\"{text}\\" back", "text": "Hello World"}'
    message.ack = AsyncMock()
    message.reject = AsyncMock()
    return message


@pytest.fixture
def mock_rabbitmq_channel():
    """Mock RabbitMQ channel for testing."""
    channel = AsyncMock()
    channel.set_qos = AsyncMock()
    channel.declare_queue = AsyncMock()
    channel.declare_exchange = AsyncMock()
    channel.default_exchange = AsyncMock()
    return channel


@pytest.fixture
def mock_rabbitmq_connection():
    """Mock RabbitMQ connection for testing."""
    connection = AsyncMock()
    connection.channel.return_value = mock_rabbitmq_channel()
    connection.is_closed = False
    connection.close = AsyncMock()
    return connection


@pytest.fixture
def mock_rabbitmq_queue():
    """Mock RabbitMQ queue for testing."""
    queue = AsyncMock()
    queue.consume = AsyncMock()
    queue.bind = AsyncMock()
    queue.message_count = 42
    queue.consumer_count = 3
    return queue


@pytest.fixture
def mock_rabbitmq_exchange():
    """Mock RabbitMQ exchange for testing."""
    exchange = AsyncMock()
    exchange.publish = AsyncMock()
    return exchange


@pytest.fixture
def mock_broker_manager(
    mock_config,
    mock_rabbitmq_connection,
    mock_rabbitmq_channel,
    mock_rabbitmq_queue,
    mock_rabbitmq_exchange,
):
    """Mock BrokerManager for testing."""
    from unittest.mock import patch

    with patch("llmq.core.broker.BrokerManager") as mock_broker:
        broker_instance = AsyncMock()
        broker_instance.config = mock_config
        broker_instance.connection = mock_rabbitmq_connection
        broker_instance.channel = mock_rabbitmq_channel

        # Mock methods
        broker_instance.connect = AsyncMock()
        broker_instance.disconnect = AsyncMock()
        broker_instance.setup_queue_infrastructure = AsyncMock(
            return_value=(
                mock_rabbitmq_queue,
                mock_rabbitmq_exchange,
            )
        )
        broker_instance.publish_job = AsyncMock()
        broker_instance.publish_result = AsyncMock()
        broker_instance.consume_jobs = AsyncMock()
        broker_instance.consume_results = AsyncMock()
        broker_instance.get_queue_stats = AsyncMock()
        broker_instance.get_failed_messages = AsyncMock(return_value=[])

        mock_broker.return_value = broker_instance
        yield broker_instance


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class MockAsyncContextManager:
    """Helper for mocking async context managers."""

    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for testing management API calls."""
    from unittest.mock import patch

    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": 42,
            "messages_ready": 40,
            "messages_unacknowledged": 2,
            "consumers": 3,
            "message_bytes": 1024,
            "message_bytes_ready": 1000,
            "message_bytes_unacknowledged": 24,
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client.return_value = MockAsyncContextManager(mock_client_instance)

        yield mock_client_instance
