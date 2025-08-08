import pytest
from unittest.mock import AsyncMock, patch

from llmq.core.broker import BrokerManager

# Apply asyncio marker to all async test methods in this module
pytestmark = pytest.mark.asyncio


class TestBrokerManager:
    """Test BrokerManager functionality with mocked RabbitMQ."""

    @pytest.mark.unit
    def test_broker_manager_init(self, mock_config):
        """Test broker manager initialization."""
        broker = BrokerManager(mock_config)

        assert broker.config == mock_config
        assert broker.connection is None
        assert broker.channel is None

    @pytest.mark.unit
    async def test_connect_success(self, mock_config):
        """Test successful RabbitMQ connection."""
        broker = BrokerManager(mock_config)

        with patch("aio_pika.connect_robust") as mock_connect:
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_connection.channel.return_value = mock_channel
            mock_connect.return_value = mock_connection

            await broker.connect()

            assert broker.connection == mock_connection
            assert broker.channel == mock_channel
            mock_connect.assert_called_once_with(
                mock_config.rabbitmq_url, client_properties={"application": "llmq"}
            )
            mock_channel.set_qos.assert_called_once_with(
                prefetch_count=mock_config.vllm_queue_prefetch
            )

    @pytest.mark.unit
    async def test_connect_with_retry(self, mock_config):
        """Test connection with retry logic."""
        broker = BrokerManager(mock_config)

        with patch("aio_pika.connect_robust") as mock_connect:
            # First call fails, second succeeds
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_connection.channel.return_value = mock_channel

            mock_connect.side_effect = [
                ConnectionError("First attempt fails"),
                mock_connection,
            ]

            with patch("asyncio.sleep"):  # Speed up retry delay
                await broker.connect()

            assert broker.connection == mock_connection
            assert broker.channel == mock_channel
            assert mock_connect.call_count == 2

    @pytest.mark.unit
    async def test_disconnect(self, mock_config):
        """Test RabbitMQ disconnection."""
        broker = BrokerManager(mock_config)

        # Set up a mock connection
        mock_connection = AsyncMock()
        mock_connection.is_closed = False
        broker.connection = mock_connection

        await broker.disconnect()

        mock_connection.close.assert_called_once()

    @pytest.mark.unit
    async def test_setup_queue_infrastructure(self, mock_config):
        """Test queue infrastructure setup."""
        broker = BrokerManager(mock_config)

        # Mock channel and its methods
        mock_channel = AsyncMock()
        broker.channel = mock_channel

        mock_job_queue = AsyncMock()
        mock_results_exchange = AsyncMock()

        mock_channel.declare_queue.side_effect = [
            mock_job_queue,  # Job queue
        ]
        mock_channel.declare_exchange.return_value = mock_results_exchange

        # Call the method
        (
            job_queue,
            results_exchange,
        ) = await broker.setup_queue_infrastructure("test-queue")

        # Verify results
        assert job_queue == mock_job_queue
        assert results_exchange == mock_results_exchange

        # Verify calls
        assert mock_channel.declare_queue.call_count == 1
        assert mock_channel.declare_exchange.call_count == 1

    @pytest.mark.unit
    async def test_publish_job(self, mock_config, sample_job):
        """Test job publishing."""
        broker = BrokerManager(mock_config)

        # Mock channel
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        mock_channel.default_exchange = mock_exchange
        broker.channel = mock_channel

        await broker.publish_job("test-queue", sample_job)

        mock_exchange.publish.assert_called_once()
        args, kwargs = mock_exchange.publish.call_args
        message = args[0]

        # Verify message properties
        assert kwargs["routing_key"] == "test-queue"
        assert message.message_id == sample_job.id

    @pytest.mark.unit
    async def test_publish_result(self, mock_config, sample_result):
        """Test result publishing."""
        broker = BrokerManager(mock_config)

        mock_results_exchange = AsyncMock()

        await broker.publish_result(mock_results_exchange, sample_result)

        mock_results_exchange.publish.assert_called_once()
        args, kwargs = mock_results_exchange.publish.call_args
        message = args[0]

        # Verify message properties
        assert kwargs["routing_key"] == ""
        assert message.message_id == sample_result.id

    @pytest.mark.unit
    async def test_get_queue_stats_via_api_success(
        self, mock_config, mock_httpx_client
    ):
        """Test getting queue stats via management API."""
        broker = BrokerManager(mock_config)

        stats = await broker._get_queue_stats_via_api("test-queue")

        assert stats is not None
        assert stats.queue_name == "test-queue"
        assert stats.message_count == 42
        assert stats.message_count_ready == 40
        assert stats.message_count_unacknowledged == 2
        assert stats.consumer_count == 3
        assert stats.message_bytes == 1024
        assert stats.stats_source == "management_api"

    @pytest.mark.unit
    async def test_get_queue_stats_via_api_failure(self, mock_config):
        """Test getting queue stats when management API fails."""
        broker = BrokerManager(mock_config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 404

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response

            async def mock_context():
                return mock_client_instance

            mock_client.return_value.__aenter__ = mock_context
            mock_client.return_value.__aexit__ = AsyncMock()

            stats = await broker._get_queue_stats_via_api("test-queue")

            assert stats is None

    @pytest.mark.unit
    async def test_get_queue_stats_fallback_to_amqp(self, mock_config):
        """Test queue stats fallback to AMQP when management API fails."""
        broker = BrokerManager(mock_config)

        # Mock channel for AMQP fallback
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()
        mock_channel.declare_queue.return_value = mock_queue
        broker.channel = mock_channel

        # Mock management API to fail
        with patch.object(broker, "_get_queue_stats_via_api", return_value=None):
            stats = await broker.get_queue_stats("test-queue")

        assert stats.queue_name == "test-queue"
        assert stats.stats_source == "amqp_fallback"
        assert stats.message_count is None  # Unknown via AMQP
        assert stats.consumer_count is None

    @pytest.mark.unit
    async def test_get_queue_stats_queue_not_found(self, mock_config):
        """Test queue stats when queue doesn't exist."""
        broker = BrokerManager(mock_config)

        # Mock channel to raise exception (queue not found)
        mock_channel = AsyncMock()
        mock_channel.declare_queue.side_effect = Exception("Queue not found")
        broker.channel = mock_channel

        # Mock management API to fail
        with patch.object(broker, "_get_queue_stats_via_api", return_value=None):
            stats = await broker.get_queue_stats("nonexistent-queue")

        assert stats.queue_name == "nonexistent-queue"
        assert stats.stats_source == "unavailable"
        assert stats.message_count is None
        assert stats.consumer_count is None

    @pytest.mark.unit
    async def test_consume_jobs(self, mock_config):
        """Test job consumption setup."""
        broker = BrokerManager(mock_config)

        # Mock channel and queue
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()
        broker.channel = mock_channel

        with patch.object(broker, "setup_queue_infrastructure") as mock_setup:
            mock_setup.return_value = (mock_queue, AsyncMock())

            callback = AsyncMock()
            result_queue = await broker.consume_jobs("test-queue", callback)

            assert result_queue == mock_queue
            mock_queue.consume.assert_called_once_with(callback)

    @pytest.mark.unit
    async def test_consume_results(self, mock_config):
        """Test result consumption setup."""
        broker = BrokerManager(mock_config)

        # Mock channel and components
        mock_channel = AsyncMock()
        mock_results_queue = AsyncMock()
        mock_results_exchange = AsyncMock()

        mock_channel.declare_queue.return_value = mock_results_queue
        mock_channel.declare_exchange.return_value = mock_results_exchange
        broker.channel = mock_channel

        callback = AsyncMock()
        result_queue = await broker.consume_results("test-queue", callback)

        assert result_queue == mock_results_queue
        mock_results_queue.bind.assert_called_once_with(mock_results_exchange)
        mock_results_queue.consume.assert_called_once_with(callback)
