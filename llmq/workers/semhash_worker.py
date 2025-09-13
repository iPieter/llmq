import random
from typing import Optional, List

from semhash import SemHash  # type: ignore

from llmq.core.models import Job
from llmq.workers.base import BaseWorker


class SemHashWorker(BaseWorker):
    """SemHash worker for semantic deduplication and outlier filtering using MinishLab/semhash."""

    def __init__(
        self,
        queue_name: str,
        worker_id: Optional[str] = None,
        concurrency: Optional[int] = None,
        pipeline_name: Optional[str] = None,
        stage_name: Optional[str] = None,
        pipeline_stages: Optional[list[str]] = None,
        batch_size: int = 1000,
        mode: str = "deduplicate",  # "deduplicate", "filter_outliers", "find_representative"
    ):
        super().__init__(
            queue_name,
            worker_id,
            concurrency,
            pipeline_name,
            stage_name,
            pipeline_stages,
        )
        self.batch_size = batch_size
        self.mode = mode
        self.text_batch: List[str] = []
        self.job_batch: List[Job] = []
        self.semhash = None

    def _generate_worker_id(self) -> str:
        """Generate unique SemHash worker ID."""
        return f"semhash-worker-{random.randint(1000, 9999)}"

    async def _initialize_processor(self) -> None:
        """Initialize SemHash processor."""
        self.logger.info(
            f"Initializing SemHash processor with mode '{self.mode}', "
            f"batch size {self.batch_size}"
        )

    async def _process_job(self, job: Job) -> str:
        """Process job using SemHash batch processing."""
        text_content = self._extract_text_from_job(job)

        if not text_content:
            return "REJECTED: No text content found in job"

        # Add to batch
        self.text_batch.append(text_content)
        self.job_batch.append(job)

        # Process batch when it reaches the configured size
        if len(self.text_batch) >= self.batch_size:
            return await self._process_batch()
        else:
            # For now, accept individual jobs - in production you'd want to
            # implement proper batching with timeouts
            return await self._process_single_item(text_content, job)

    async def _process_single_item(self, text_content: str, job: Job) -> str:
        """Process a single item for immediate response."""
        try:

            SemHash.from_records([text_content])

            if self.mode == "deduplicate":
                return f"ACCEPTED: Single item processed (mode: {self.mode})"

            elif self.mode == "filter_outliers":
                return f"ACCEPTED: Single item processed (mode: {self.mode})"

            elif self.mode == "find_representative":
                return f"ACCEPTED: Single item is representative (mode: {self.mode})"

            return f"ACCEPTED: Single item processed (mode: {self.mode})"

        except Exception as e:
            self.logger.error(f"SemHash processing error: {e}")
            return f"ACCEPTED: Processing fallback (error: {e})"

    async def _process_batch(self) -> str:
        """Process accumulated batch using SemHash."""
        if not self.text_batch:
            return "No items in batch"

        try:

            self.logger.info(
                f"Processing batch of {len(self.text_batch)} items with mode '{self.mode}'"
            )

            # Create SemHash instance from batch
            semhash = SemHash.from_records(self.text_batch)

            if self.mode == "deduplicate":
                # Perform deduplication
                deduplicated_result = semhash.self_deduplicate()
                selected_texts = deduplicated_result.selected

                self.logger.info(
                    f"Deduplication: {len(self.text_batch)} -> {len(selected_texts)} items"
                )

                # Clear batch
                self.text_batch.clear()
                self.job_batch.clear()

                return f"BATCH_PROCESSED: Deduplicated {len(self.text_batch)} -> {len(selected_texts)} items"

            elif self.mode == "filter_outliers":
                # Filter outliers
                filtered_result = semhash.self_filter_outliers()
                selected_texts = filtered_result.selected

                self.logger.info(
                    f"Outlier filtering: {len(self.text_batch)} -> {len(selected_texts)} items"
                )

                # Clear batch
                self.text_batch.clear()
                self.job_batch.clear()

                return f"BATCH_PROCESSED: Filtered outliers {len(self.text_batch)} -> {len(selected_texts)} items"

            elif self.mode == "find_representative":
                # Find representative texts
                representative_result = semhash.self_find_representative()
                selected_texts = representative_result.selected

                self.logger.info(
                    f"Representative finding: {len(self.text_batch)} -> {len(selected_texts)} items"
                )

                # Clear batch
                self.text_batch.clear()
                self.job_batch.clear()

                return f"BATCH_PROCESSED: Found {len(selected_texts)} representative items from {len(self.text_batch)}"

            else:
                self.logger.warning(f"Unknown mode: {self.mode}")
                return f"ERROR: Unknown mode {self.mode}"

        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            # Clear batch on error
            self.text_batch.clear()
            self.job_batch.clear()
            return f"ERROR: Batch processing failed: {e}"

    def _extract_text_from_job(self, job: Job) -> str:
        """Extract text content from job for deduplication."""
        job_data = job.model_dump()

        # Try different common text fields
        text_fields = ["text", "content", "source_text", "document", "body"]

        for field in text_fields:
            if field in job_data and job_data[field]:
                return str(job_data[field])

        # Try extracting from messages
        if job.messages:
            contents = []
            for msg in job.messages:
                if isinstance(msg, dict) and "content" in msg:
                    contents.append(str(msg["content"]))
            if contents:
                return "\n".join(contents)

        # Try formatted prompt as fallback
        try:
            return job.get_formatted_prompt()
        except Exception:
            return ""

    async def _cleanup_processor(self) -> None:
        """Clean up SemHash processor."""
        # Process any remaining items in batch
        if self.text_batch:
            await self._process_batch()

        self.logger.info("SemHash worker cleanup complete")
