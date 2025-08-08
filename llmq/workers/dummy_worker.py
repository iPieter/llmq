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
    ):
        super().__init__(queue_name, worker_id, concurrency)

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


class FilterWorker(BaseWorker):
    """Simple filtering worker that can accept/reject jobs based on criteria."""

    def __init__(
        self,
        queue_name: str,
        filter_field: str = "content",
        filter_value: str = "",
        worker_id: Optional[str] = None,
    ):
        super().__init__(queue_name, worker_id)
        self.filter_field = filter_field
        self.filter_value = filter_value

    def _generate_worker_id(self) -> str:
        """Generate unique filter worker ID."""
        return f"filter-worker-{random.randint(1000, 9999)}"

    async def _initialize_processor(self) -> None:
        """Initialize filter processor."""
        self.logger.info(
            f"Initializing filter processor (field={self.filter_field}, value={self.filter_value})"
        )

    async def _process_job(self, job: Job) -> str:
        """Process job using filtering logic."""
        # Simple filtering example
        formatted_prompt = job.get_formatted_prompt()
        job_data = job.model_dump()

        # Check if job should be accepted
        field_value = job_data.get(self.filter_field, "")

        if self.filter_value.lower() in str(field_value).lower():
            return f"ACCEPTED: {formatted_prompt} (matched filter: {self.filter_field}={self.filter_value})"
        else:
            return f"REJECTED: {formatted_prompt} (did not match filter: {self.filter_field}={self.filter_value})"

    async def _cleanup_processor(self) -> None:
        """Clean up filter processor."""
        pass
