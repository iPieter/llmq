import pytest
from unittest.mock import patch

from llmq.workers.dummy_worker import DummyWorker, FilterWorker
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


class TestFilterWorker:
    """Test FilterWorker functionality."""

    @pytest.mark.unit
    def test_filter_worker_init(self):
        """Test filter worker initialization."""
        worker = FilterWorker("test-queue", "category", "important")

        assert worker.queue_name == "test-queue"
        assert worker.filter_field == "category"
        assert worker.filter_value == "important"
        assert worker.worker_id.startswith("filter-worker-")

    @pytest.mark.unit
    def test_filter_worker_init_with_custom_id(self):
        """Test filter worker initialization with custom worker ID."""
        worker = FilterWorker(
            "test-queue", "priority", "high", worker_id="custom-filter-123"
        )

        assert worker.worker_id == "custom-filter-123"

    @pytest.mark.unit
    def test_generate_worker_id(self):
        """Test filter worker ID generation."""
        worker = FilterWorker("test-queue", "field", "value")
        worker_id = worker._generate_worker_id()

        assert worker_id.startswith("filter-worker-")
        assert len(worker_id) == len("filter-worker-") + 4  # 4-digit number

    @pytest.mark.unit
    async def test_initialize_processor(self):
        """Test filter processor initialization."""
        worker = FilterWorker("test-queue", "category", "important")

        # Should not raise any exceptions
        await worker._initialize_processor()

    @pytest.mark.unit
    async def test_cleanup_processor(self):
        """Test filter processor cleanup."""
        worker = FilterWorker("test-queue", "category", "important")

        # Should not raise any exceptions
        await worker._cleanup_processor()

    @pytest.mark.unit
    async def test_process_job_matching_filter(self):
        """Test job processing when filter matches."""
        worker = FilterWorker("test-queue", "category", "important")
        job = Job(
            id="test-001",
            prompt="Process important task",
            category="important",
            task="do something",
        )

        result = await worker._process_job(job)

        assert result.startswith("ACCEPTED:")
        assert "Process important task" in result
        assert "matched filter: category=important" in result

    @pytest.mark.unit
    async def test_process_job_not_matching_filter(self):
        """Test job processing when filter doesn't match."""
        worker = FilterWorker("test-queue", "category", "important")
        job = Job(
            id="test-001",
            prompt="Process routine task",
            category="routine",
            task="do something",
        )

        result = await worker._process_job(job)

        assert result.startswith("REJECTED:")
        assert "Process routine task" in result
        assert "did not match filter: category=important" in result

    @pytest.mark.unit
    async def test_process_job_missing_filter_field(self):
        """Test job processing when filter field is missing."""
        worker = FilterWorker("test-queue", "category", "important")
        job = Job(
            id="test-001", prompt="Process task without category", task="do something"
        )

        result = await worker._process_job(job)

        assert result.startswith("REJECTED:")
        assert "did not match filter: category=important" in result

    @pytest.mark.unit
    async def test_process_job_case_insensitive_matching(self):
        """Test that filter matching is case insensitive."""
        worker = FilterWorker("test-queue", "priority", "high")

        test_cases = [
            ("HIGH", True),  # Exact match uppercase
            ("high", True),  # Exact match lowercase
            ("High", True),  # Mixed case
            ("very high", True),  # Contains filter value
            ("medium", False),  # Does not contain
            ("higher", True),  # Contains as substring
        ]

        for priority_value, should_match in test_cases:
            job = Job(
                id=f"test-{hash(priority_value)}",
                prompt="Test task",
                priority=priority_value,
            )

            result = await worker._process_job(job)

            if should_match:
                assert result.startswith(
                    "ACCEPTED:"
                ), f"Failed for priority='{priority_value}'"
            else:
                assert result.startswith(
                    "REJECTED:"
                ), f"Failed for priority='{priority_value}'"


class TestWorkerIntegration:
    """Test worker integration scenarios."""

    @pytest.mark.unit
    async def test_multiple_workers_different_types(self):
        """Test that different worker types can coexist."""
        dummy_worker = DummyWorker("test-queue", worker_id="dummy-1")
        filter_worker = FilterWorker("test-queue", "type", "test", worker_id="filter-1")

        # Both workers should have different IDs and behavior
        assert dummy_worker.worker_id == "dummy-1"
        assert filter_worker.worker_id == "filter-1"

        # Test they process jobs differently
        job = Job(id="test-001", prompt="Test job", text="Hello", type="test")

        with patch("asyncio.sleep"):
            dummy_result = await dummy_worker._process_job(job)
            filter_result = await filter_worker._process_job(job)

        assert dummy_result == "echo Hello"
        assert filter_result.startswith("ACCEPTED:")
        assert dummy_result != filter_result
