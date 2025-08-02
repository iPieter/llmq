import asyncio
import signal
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from aio_pika.abc import AbstractIncomingMessage

from llmq.core.config import get_config
from llmq.core.broker import BrokerManager
from llmq.core.models import Job, Result


class BaseWorker(ABC):
    """Abstract base class for all worker implementations."""

    def __init__(
        self,
        queue_name: str,
        worker_id: Optional[str] = None,
        concurrency: Optional[int] = None,
    ):
        self.queue_name = queue_name
        self.worker_id = worker_id or self._generate_worker_id()
        self.config = get_config()
        self.concurrency = concurrency  # Override prefetch if specified

        # Set up structured logging for workers
        from llmq.utils.logging import setup_logging

        self.logger = setup_logging(f"llmq.worker.{self.worker_id}", structured=True)

        self.broker: Optional[BrokerManager] = None
        self.results_exchange = None
        self.running = False
        self.jobs_processed = 0
        self.total_duration_ms = 0.0

        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    @abstractmethod
    def _generate_worker_id(self) -> str:
        """Generate a unique worker ID for this worker type."""
        pass

    @abstractmethod
    async def _initialize_processor(self) -> None:
        """Initialize the processing engine (vLLM, dummy, etc.)."""
        pass

    @abstractmethod
    async def _process_job(self, job: Job) -> str:
        """Process a single job and return the result text."""
        pass

    @abstractmethod
    async def _cleanup_processor(self) -> None:
        """Clean up processing resources."""
        pass

    async def initialize(self) -> None:
        """Initialize worker components."""
        self.logger.info(f"Initializing worker {self.worker_id}")

        # Initialize processing engine
        await self._initialize_processor()

        # Initialize RabbitMQ connection
        self.broker = BrokerManager(self.config)
        await self.broker.connect()

        # Override prefetch if concurrency is specified
        if self.concurrency is not None:
            await self.broker.channel.set_qos(prefetch_count=self.concurrency)
            self.logger.info(f"Set concurrency to {self.concurrency} jobs")

        # Set up queue infrastructure
        _, self.results_exchange, _ = await self.broker.setup_queue_infrastructure(
            self.queue_name
        )

        self.logger.info("Worker initialization complete")

    async def run(self) -> None:
        """Main worker loop."""
        try:
            await self.initialize()
            self.running = True

            self.logger.info(
                f"Worker {self.worker_id} starting to consume from queue {self.queue_name}"
            )

            # Start consuming jobs
            await self.broker.consume_jobs(self.queue_name, self._process_message)

            # Keep the worker running
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"Worker error: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def _process_message(self, message: AbstractIncomingMessage) -> None:
        """Process a single job message."""
        start_time = time.time()
        job_id = None

        try:
            # Parse job
            job = Job.parse_raw(message.body)
            job_id = job.id

            self.logger.info(f"Processing job {job_id}")

            # Process job using implementation-specific method
            result_text = await self._process_job(job)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Create result
            result = Result(
                id=job_id,
                prompt=job.get_formatted_prompt(),
                result=result_text,
                worker_id=self.worker_id,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
            )

            # Publish result
            await self.broker.publish_result(self.results_exchange, result)

            # Acknowledge message
            await message.ack()

            # Update stats
            self.jobs_processed += 1
            self.total_duration_ms += duration_ms

            self.logger.info(
                f"Completed job {job_id}",
                extra={
                    "job_id": job_id,
                    "duration_ms": duration_ms,
                    "jobs_processed": self.jobs_processed,
                    "avg_duration_ms": self.total_duration_ms / self.jobs_processed,
                },
            )

        except Exception as e:
            self.logger.error(
                f"Error processing job {job_id}: {e}",
                extra={"job_id": job_id, "error": str(e)},
                exc_info=True,
            )

            # Reject message (will go to dead letter queue after retries)
            await message.reject(requeue=True)

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info(f"Worker {self.worker_id} shutting down...")

        # Clean up processing resources
        await self._cleanup_processor()

        # Clean up broker connection
        if self.broker:
            await self.broker.disconnect()

        self.logger.info(
            f"Worker {self.worker_id} shutdown complete. Processed {self.jobs_processed} jobs."
        )
