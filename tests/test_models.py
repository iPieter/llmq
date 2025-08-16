import pytest
from datetime import datetime
from pydantic import ValidationError

from llmq.core.models import Job, Result, QueueStats, WorkerHealth, ErrorInfo


class TestJob:
    """Test Job model."""

    def test_job_creation(self):
        """Test basic job creation."""
        job = Job(
            id="test-001",
            prompt="Say {greeting} to {name}",
            greeting="hello",
            name="world",
        )

        assert job.id == "test-001"
        assert job.prompt == "Say {greeting} to {name}"
        assert job.greeting == "hello"
        assert job.name == "world"

    def test_job_formatted_prompt(self):
        """Test prompt formatting with job data."""
        job = Job(
            id="test-001",
            prompt="Translate '{text}' to {language}",
            text="Hello",
            language="Spanish",
        )

        formatted = job.get_formatted_prompt()
        assert formatted == "Translate 'Hello' to Spanish"

    def test_job_formatted_prompt_with_brackets_in_data(self):
        """Test prompt formatting when data contains curly braces."""
        job = Job(
            id="test-001",
            prompt="Translate: {text}",
            text="This text has {brackets} and {more_brackets} in it",
        )

        # Should not raise KeyError for 'brackets' or 'more_brackets'
        formatted = job.get_formatted_prompt()
        assert (
            formatted == "Translate: This text has {brackets} and {more_brackets} in it"
        )

    def test_job_with_messages_only(self):
        """Test job creation with messages field only (no extra dataset fields)."""
        job = Job(
            id="test-001",
            messages=[{"role": "user", "content": "Translate this text"}],
            chat_mode=True,
        )

        assert job.id == "test-001"
        assert job.messages == [{"role": "user", "content": "Translate this text"}]
        assert job.chat_mode is True
        assert job.prompt is None
        # Should not have extra fields like 'text'
        job_dict = job.dict()
        assert "text" not in job_dict

    def test_job_extra_fields_allowed(self):
        """Test that extra fields are allowed in job."""
        job = Job(
            id="test-001", prompt="Test", custom_field="custom_value", another_field=42
        )

        assert job.custom_field == "custom_value"
        assert job.another_field == 42

    def test_job_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Job(prompt="Test")  # Missing id

        assert "id" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Job(id="test-001")  # Missing prompt

        assert "prompt" in str(exc_info.value)

    def test_job_stop_sequences(self):
        """Test job creation with custom stop sequences."""
        job = Job(
            id="test-001",
            prompt="Generate text",
            stop=["</end>", "\n\n", "STOP"],
        )

        assert job.stop == ["</end>", "\n\n", "STOP"]

    def test_job_stop_sequences_defaults(self):
        """Test job creation with default stop sequence behavior."""
        job = Job(
            id="test-001",
            prompt="Generate text",
        )

        assert job.stop is None  # Will use EOS token by default

    def test_job_formatted_prompt_excludes_stop_fields(self):
        """Test that stop sequence fields are excluded from formatting."""
        job = Job(
            id="test-001",
            prompt="Say {greeting}",
            greeting="hello",
            stop=["</end>"],
        )

        formatted = job.get_formatted_prompt()
        assert formatted == "Say hello"


class TestResult:
    """Test Result model."""

    def test_result_creation(self):
        """Test basic result creation."""
        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        result = Result(
            id="test-001",
            prompt="Say hello to world",
            result="Hello world!",
            worker_id="worker-001",
            duration_ms=150.5,
            timestamp=timestamp,
        )

        assert result.id == "test-001"
        assert result.prompt == "Say hello to world"
        assert result.result == "Hello world!"
        assert result.worker_id == "worker-001"
        assert result.duration_ms == 150.5
        assert result.timestamp == timestamp

    def test_result_default_timestamp(self):
        """Test that timestamp defaults to current time."""
        result = Result(
            id="test-001",
            prompt="Test",
            result="Test result",
            worker_id="worker-001",
            duration_ms=100.0,
        )

        # Should have a timestamp close to now
        now = datetime.utcnow()
        time_diff = abs((now - result.timestamp).total_seconds())
        assert time_diff < 1  # Within 1 second

    def test_result_json_serialization(self):
        """Test that result can be serialized to JSON."""
        result = Result(
            id="test-001",
            prompt="Test",
            result="Test result",
            worker_id="worker-001",
            duration_ms=100.0,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
        )

        json_str = result.json()
        assert "test-001" in json_str
        assert "Test result" in json_str
        assert "worker-001" in json_str
        assert "100.0" in json_str


class TestQueueStats:
    """Test QueueStats model."""

    def test_queue_stats_creation(self):
        """Test basic queue stats creation."""
        stats = QueueStats(
            queue_name="test-queue",
            message_count=100,
            message_count_ready=90,
            message_count_unacknowledged=10,
            consumer_count=5,
            message_bytes=2048,
            message_bytes_ready=1800,
            message_bytes_unacknowledged=248,
            processing_rate=25.5,
            stats_source="management_api",
        )

        assert stats.queue_name == "test-queue"
        assert stats.message_count == 100
        assert stats.message_count_ready == 90
        assert stats.message_count_unacknowledged == 10
        assert stats.consumer_count == 5
        assert stats.message_bytes == 2048
        assert stats.message_bytes_ready == 1800
        assert stats.message_bytes_unacknowledged == 248
        assert stats.processing_rate == 25.5
        assert stats.stats_source == "management_api"

    def test_queue_stats_optional_fields(self):
        """Test that optional fields can be None."""
        stats = QueueStats(queue_name="test-queue", stats_source="unavailable")

        assert stats.queue_name == "test-queue"
        assert stats.message_count is None
        assert stats.consumer_count is None
        assert stats.message_bytes is None
        assert stats.processing_rate is None
        assert stats.stats_source == "unavailable"

    def test_queue_stats_default_source(self):
        """Test default stats source."""
        stats = QueueStats(queue_name="test-queue")
        assert stats.stats_source == "unknown"


class TestWorkerHealth:
    """Test WorkerHealth model."""

    def test_worker_health_creation(self):
        """Test worker health creation."""
        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        health = WorkerHealth(
            worker_id="worker-001",
            status="active",
            last_seen=timestamp,
            jobs_processed=150,
            avg_duration_ms=75.5,
        )

        assert health.worker_id == "worker-001"
        assert health.status == "active"
        assert health.last_seen == timestamp
        assert health.jobs_processed == 150
        assert health.avg_duration_ms == 75.5

    def test_worker_health_optional_avg_duration(self):
        """Test that avg_duration_ms is optional."""
        health = WorkerHealth(
            worker_id="worker-001",
            status="active",
            last_seen=datetime.utcnow(),
            jobs_processed=0,
        )

        assert health.avg_duration_ms is None


class TestErrorInfo:
    """Test ErrorInfo model."""

    def test_error_info_creation(self):
        """Test error info creation."""
        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        error = ErrorInfo(
            job_id="job-001",
            error_message="Connection timeout",
            timestamp=timestamp,
            worker_id="worker-001",
        )

        assert error.job_id == "job-001"
        assert error.error_message == "Connection timeout"
        assert error.timestamp == timestamp
        assert error.worker_id == "worker-001"

    def test_error_info_optional_worker_id(self):
        """Test that worker_id is optional."""
        error = ErrorInfo(
            job_id="job-001",
            error_message="Validation error",
            timestamp=datetime.utcnow(),
        )

        assert error.worker_id is None
