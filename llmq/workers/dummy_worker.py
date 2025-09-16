import asyncio
import random
from typing import Optional

from llmq.core.models import Job
from llmq.workers.base import BaseWorker


class DummyWorker(BaseWorker):
    """Dummy worker that simulates LLM processing without vLLM dependency."""

    def __init__(
        self,
        queue_name: str,
        worker_id: Optional[str] = None,
        concurrency: Optional[int] = None,
        pipeline_name: Optional[str] = None,
        stage_name: Optional[str] = None,
        pipeline_stages: Optional[list[str]] = None,
    ):
        super().__init__(
            queue_name,
            worker_id,
            concurrency,
            pipeline_name,
            stage_name,
            pipeline_stages,
        )

    def _generate_worker_id(self) -> str:
        """Generate unique dummy worker ID."""
        return f"dummy-worker-{random.randint(1000, 9999)}"

    async def _initialize_processor(self) -> None:
        """Initialize dummy processor (no-op)."""
        self.logger.info("Initializing dummy processor")
        # No actual initialization needed for dummy worker
        pass

    async def _process_job(self, job: Job) -> str:
        """Process job using simple echo logic."""
        # Consistent 1 second delay
        await asyncio.sleep(1.0)

        # Simple echo response with any 'text' field from the job
        text = job.model_dump().get("text", "no text found")
        return f"echo {text}"

    async def _cleanup_processor(self) -> None:
        """Clean up dummy processor (no-op)."""
        pass
