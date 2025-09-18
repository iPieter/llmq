import pytest
from unittest.mock import patch

from llmq.workers.dummy_worker import DummyWorker
from llmq.core.models import Job

# Apply asyncio marker to all async test methods in this module
pytestmark = pytest.mark.asyncio


class TestDummyWorker:
    """Test DummyWorker functionality."""

    @pytest.mark.unit
    def test_dummy_worker_init(self):
        """Test dummy worker initialization."""
        worker = DummyWorker("test-queue")

        assert worker.queue_name == "test-queue"
        assert worker.worker_id.startswith("dummy-worker-")
        assert worker.concurrency is None

    @pytest.mark.unit
    def test_dummy_worker_init_with_concurrency(self):
        """Test dummy worker initialization with concurrency."""
        worker = DummyWorker("test-queue", concurrency=5)

        assert worker.concurrency == 5

    @pytest.mark.unit
    def test_dummy_worker_init_with_custom_id(self):
        """Test dummy worker initialization with custom worker ID."""
        worker = DummyWorker("test-queue", worker_id="custom-worker-123")

        assert worker.worker_id == "custom-worker-123"

    @pytest.mark.unit
    def test_generate_worker_id(self):
        """Test worker ID generation."""
        worker = DummyWorker("test-queue")
        worker_id = worker._generate_worker_id()

        assert worker_id.startswith("dummy-worker-")
        assert len(worker_id) == len("dummy-worker-") + 4  # 4-digit number

    @pytest.mark.unit
    async def test_initialize_processor(self):
        """Test processor initialization (should be no-op)."""
        worker = DummyWorker("test-queue")

        # Should not raise any exceptions
        await worker._initialize_processor()

    @pytest.mark.unit
    async def test_cleanup_processor(self):
        """Test processor cleanup (should be no-op)."""
        worker = DummyWorker("test-queue")

        # Should not raise any exceptions
        await worker._cleanup_processor()

    @pytest.mark.unit
    async def test_process_job_with_text(self):
        """Test job processing with text field."""
        worker = DummyWorker("test-queue")
        job = Job(id="test-001", prompt="Echo test", text="Hello World")

        # Mock the sleep to make test faster
        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            result = await worker._process_job(job)

        assert result == "echo Hello World"

    @pytest.mark.unit
    async def test_process_job_with_different_text_values(self):
        """Test job processing with various text values."""
        worker = DummyWorker("test-queue")

        test_cases = [
            ("simple text", "echo simple text"),
            ("Text with spaces", "echo Text with spaces"),
            ("123 numbers", "echo 123 numbers"),
            ("Special chars!@#$%", "echo Special chars!@#$%"),
            ("", "echo "),
        ]

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None

            for text_input, expected_output in test_cases:
                job = Job(
                    id=f"test-{hash(text_input)}", prompt="Echo test", text=text_input
                )

                result = await worker._process_job(job)
                assert result == expected_output
