import asyncio
import pytest
import os
from typing import AsyncGenerator

from llmq.workers.dummy_worker import DummyWorker
from llmq.core.broker import BrokerManager
from llmq.core.models import Job, Result
from llmq.core.config import Config

# Apply asyncio marker to all async test methods in this module
pytestmark = pytest.mark.asyncio


@pytest.fixture
async def rabbitmq_url() -> str:
    """Get RabbitMQ URL for integration tests."""
    # Check environment variable first, otherwise skip tests
    url = os.environ.get("RABBITMQ_URL")
    if not url:
        pytest.skip("RABBITMQ_URL environment variable not set for integration tests")
    return url


@pytest.fixture
async def broker(rabbitmq_url: str) -> AsyncGenerator[BrokerManager, None]:
    """Create a real BrokerManager for integration tests."""
    config = Config(rabbitmq_url=rabbitmq_url)
    broker_manager = BrokerManager(config)

    try:
        await broker_manager.connect()
        yield broker_manager
    except Exception as e:
        pytest.skip(f"RabbitMQ not available for integration tests: {e}")
    finally:
        try:
            await broker_manager.disconnect()
        except Exception:
            pass  # Ignore disconnect errors


@pytest.fixture
async def test_queue_name() -> str:
    """Generate unique test queue name."""
    import uuid

    return f"integration-test-{uuid.uuid4().hex[:8]}"


@pytest.mark.rabbitmq
class TestDummyWorkerIntegration:
    """Integration tests for DummyWorker with real RabbitMQ."""

    @pytest.mark.integration
    async def test_dummy_worker_end_to_end(
        self, broker: BrokerManager, test_queue_name: str
    ):
        """Test complete job processing flow with DummyWorker."""
        # Create and start worker
        worker = DummyWorker(test_queue_name)

        # Start worker in background
        worker_task = asyncio.create_task(worker.run())

        try:
            # Wait a moment for worker to initialize
            await asyncio.sleep(1.0)

            # Set up result collection first
            results = []

            async def collect_result(message):
                try:
                    result = Result.parse_raw(message.body)
                    results.append(result)
                    await message.ack()
                except Exception as e:
                    print(f"Error parsing result: {e}")
                    await message.reject(requeue=False)

            # Start consuming results
            result_queue = await broker.consume_results(test_queue_name, collect_result)

            # Submit a job
            test_job = Job(  # type: ignore
                id="integration-test-job-001",
                prompt="{text}",
                text="Test prompt with integration test",
            )

            await broker.publish_job(test_queue_name, test_job)

            # Wait for results with timeout
            for _ in range(50):  # 5 seconds timeout
                if len(results) >= 1:
                    break
                await asyncio.sleep(0.1)

            # Clean up result consumer
            try:
                await result_queue.cancel("")
            except Exception:
                pass

            # Verify result
            assert len(results) == 1
            result = results[0]
            assert result.id == test_job.id
            assert "Test prompt with integration test" in result.result
            assert result.result.startswith("echo ")

        finally:
            # Clean up worker
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.integration
    async def test_dummy_worker_multiple_jobs(
        self, broker: BrokerManager, test_queue_name: str
    ):
        """Test DummyWorker processing multiple jobs."""
        # Create and start worker
        worker = DummyWorker(test_queue_name)

        # Start worker in background
        worker_task = asyncio.create_task(worker.run())

        try:
            # Wait a moment for worker to initialize
            await asyncio.sleep(1.0)

            # Set up result collection first
            results = []

            async def collect_result(message):
                try:
                    result = Result.parse_raw(message.body)
                    results.append(result)
                    await message.ack()
                except Exception as e:
                    print(f"Error parsing result: {e}")
                    await message.reject(requeue=False)

            # Start consuming results
            result_queue = await broker.consume_results(test_queue_name, collect_result)

            # Submit multiple jobs
            jobs = []
            for i in range(3):
                job = Job(  # type: ignore
                    id=f"integration-test-job-{i:03d}",
                    prompt=f"Job {i}: Hello from job {i}",
                )
                jobs.append(job)
                await broker.publish_job(test_queue_name, job)

            # Wait for results with timeout
            for _ in range(100):  # 10 seconds timeout
                if len(results) >= len(jobs):
                    break
                await asyncio.sleep(0.1)

            # Clean up result consumer
            try:
                await result_queue.cancel("")
            except Exception:
                pass

            # Verify all results received
            assert len(results) == len(jobs)

            # Verify each result
            result_ids = {r.id for r in results}
            job_ids = {j.id for j in jobs}
            assert result_ids == job_ids

            # Check content format
            for result in results:
                assert result.result.startswith("echo ")

        finally:
            # Clean up worker
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.integration
    async def test_worker_queue_setup(
        self, broker: BrokerManager, test_queue_name: str
    ):
        """Test that worker properly sets up queues."""
        # Create worker (but don't start it yet)
        worker = DummyWorker(test_queue_name)

        # Start worker briefly to set up queues
        worker_task = asyncio.create_task(worker.run())

        try:
            # Wait for worker to initialize queues
            await asyncio.sleep(2.0)

            # Verify queue exists and is ready
            stats = await broker.get_queue_stats(test_queue_name)
            assert stats.queue_name == test_queue_name  # Queue exists and is queryable

        finally:
            # Clean up
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.integration
    async def test_queue_status_monitoring(
        self, broker: BrokerManager, test_queue_name: str
    ):
        """Test queue status monitoring functionality."""
        # Submit a few jobs without a worker
        jobs = []
        for i in range(3):
            job = Job(  # type: ignore
                id=f"status-test-job-{i:03d}", prompt=f"Status test job {i}"
            )
            jobs.append(job)
            await broker.publish_job(test_queue_name, job)

        # Check queue stats
        stats = await broker.get_queue_stats(test_queue_name)
        assert stats.queue_name == test_queue_name
        # Should show pending jobs (exact count depends on RabbitMQ state)
        assert stats.stats_source in ["management_api", "amqp_fallback", "unavailable"]
