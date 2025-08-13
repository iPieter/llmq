import asyncio
import json
import sys
import signal
import time
from typing import Dict, Optional, Iterator, Any
from pathlib import Path
import uuid

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)
from aio_pika.abc import AbstractIncomingMessage

from llmq.core.config import get_config
from llmq.core.broker import BrokerManager
from llmq.core.models import Job, Result
from llmq.utils.logging import setup_logging


class JobSubmitter:
    """Handles job submission and result streaming."""

    def __init__(
        self,
        queue_name: str,
        jobs_source: str,
        timeout: int = 300,
        column_mapping: Optional[Dict[str, str]] = None,
        max_samples: Optional[int] = None,
        split: str = "train",
        subset: Optional[str] = None,
    ):
        self.queue_name = queue_name
        self.jobs_source = jobs_source
        self.timeout = timeout
        self.column_mapping = column_mapping or {}
        self.max_samples = max_samples
        self.split = split
        self.subset = subset
        self.config = get_config()
        self.logger = setup_logging("llmq.submit")

        self.broker: Optional[BrokerManager] = None
        self.console = Console(file=sys.stderr)
        self.running = True
        self.shutting_down = False
        self.submitted_count = 0
        self.completed_count = 0
        self.pending_jobs_count = 0
        self.start_time: Optional[float] = None  # Set when first job is submitted
        self.last_result_time: Optional[float] = (
            None  # Track when we last received a result
        )

        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Detect if source is a dataset or file
        self.is_dataset = self._is_huggingface_dataset()
        if self.is_dataset:
            self.console.print(
                f"[blue]Detected Hugging Face dataset: {self.jobs_source}[/blue]"
            )
        else:
            self.jobs_file = Path(self.jobs_source)

    def _is_huggingface_dataset(self) -> bool:
        """Check if the source appears to be a Hugging Face dataset."""
        # Special case: "-" means stdin, not a dataset
        if self.jobs_source == "-":
            return False

        # If it's a file path that exists, it's not a dataset
        if Path(self.jobs_source).exists():
            return False

        # Check if it looks like a dataset identifier (contains / or doesn't end with common file extensions)
        if "/" in self.jobs_source or not any(
            self.jobs_source.endswith(ext) for ext in [".jsonl", ".json", ".txt"]
        ):
            return True

        return False

    def _load_dataset_iterator(self) -> Iterator[Dict[str, Any]]:
        """Load and iterate through a Hugging Face dataset."""
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError(
                "datasets package is required for Hugging Face dataset support. Install with: pip install datasets"
            )

        self.console.print(f"[blue]Loading dataset: {self.jobs_source}[/blue]")

        # Load dataset in streaming mode for memory efficiency
        try:
            if self.subset:
                dataset = load_dataset(
                    self.jobs_source, self.subset, streaming=True, split=self.split
                )
            else:
                dataset = load_dataset(
                    self.jobs_source, streaming=True, split=self.split
                )
        except Exception:
            # Try without specifying split
            self.console.print(
                f"[yellow]Failed to load '{self.split}' split, trying default...[/yellow]"
            )
            if self.subset:
                dataset = load_dataset(self.jobs_source, self.subset, streaming=True)
            else:
                dataset = load_dataset(self.jobs_source, streaming=True)
            # Take the first split available
            dataset = next(iter(dataset.values()))

        self.console.print("[green]Dataset loaded successfully[/green]")

        count = 0
        for item in dataset:
            if self.max_samples and count >= self.max_samples:
                break
            count += 1
            yield item

    def _create_job_from_dataset_item(self, item: Dict[str, Any], index: int) -> Job:
        """Create a Job from a dataset item using column mapping."""
        job_data: Dict[str, Any] = {"id": f"dataset-{index:08d}-{uuid.uuid4().hex[:8]}"}

        # Debug: log the first few items to understand the data structure
        if index < 3:
            self.logger.info(f"Dataset item {index} keys: {list(item.keys())}")
            if "text" in item:
                text_preview = (
                    str(item["text"])[:100] + "..."
                    if len(str(item["text"])) > 100
                    else str(item["text"])
                )
                self.logger.info(f"Dataset item {index} text preview: {text_preview}")

        # Apply column mapping
        for job_field, mapping_value in self.column_mapping.items():
            self.logger.debug(f"Processing mapping: {job_field} = {mapping_value}")
            if (mapping_value.startswith("{") and mapping_value.endswith("}")) or (
                mapping_value.startswith("[") and mapping_value.endswith("]")
            ):
                # Handle JSON mapping for complex fields like messages
                try:
                    # Parse as JSON and format any template strings
                    import json as json_module

                    json_template = json_module.loads(mapping_value)
                    job_data[job_field] = self._format_json_template(
                        json_template, item
                    )
                except json_module.JSONDecodeError as e:
                    self.logger.error(
                        f"Invalid JSON in mapping for field '{job_field}': {mapping_value}. Error: {e}"
                    )
                    # Don't set the field at all if JSON parsing fails
                    continue
            elif "{" in mapping_value and "}" in mapping_value:
                # Handle template string mapping
                try:
                    job_data[job_field] = mapping_value.format(**item)
                except KeyError as e:
                    self.logger.warning(
                        f"Template variable {e} not found in dataset item for field '{job_field}'"
                    )
            elif mapping_value in item:
                # Simple column mapping
                job_data[job_field] = item[mapping_value]
            else:
                self.logger.warning(
                    f"Column '{mapping_value}' not found in dataset item. Available columns: {list(item.keys())}"
                )

        # If no mapping provided and 'text' column exists, use it as prompt
        if not self.column_mapping and "text" in item:
            job_data["prompt"] = str(item["text"])

        # Don't add any additional fields - only keep what was explicitly mapped
        # The column_mapping processing above already handled all the mapped fields

        # Set chat_mode=True if we have messages
        if "messages" in job_data and job_data["messages"] is not None:
            job_data["chat_mode"] = True

        # Ensure we have either prompt or messages
        if "messages" not in job_data and "prompt" not in job_data:
            # Fallback: use text column as prompt if available
            if "text" in item:
                job_data["prompt"] = str(item["text"])
            else:
                raise ValueError(
                    f"No messages or prompt could be created from item. Available keys: {list(item.keys())}"
                )

        return Job(**job_data)

    def _format_json_template(self, json_obj: Any, item: Dict[str, Any]) -> Any:
        """Recursively format JSON template with dataset item values."""
        if isinstance(json_obj, str):
            # Format string templates
            try:
                return json_obj.format(**item)
            except KeyError as e:
                self.logger.warning(f"Template variable {e} not found in dataset item")
                return json_obj
        elif isinstance(json_obj, dict):
            # Recursively format dictionary values
            return {
                key: self._format_json_template(value, item)
                for key, value in json_obj.items()
            }
        elif isinstance(json_obj, list):
            # Recursively format list items
            return [self._format_json_template(value, item) for value in json_obj]
        else:
            # Return as-is for other types
            return json_obj

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle Ctrl+C gracefully - stop submitting, wait for pending results."""
        if not self.shutting_down:
            self.console.print(
                "\n[yellow]Received interrupt signal. Stopping submission, waiting for pending results...[/yellow]"
            )
            self.console.print("[dim]Press Ctrl+C again to force quit[/dim]")
            self.running = False
            self.shutting_down = True
        else:
            self.console.print("\n[red]Force quitting...[/red]")
            sys.exit(1)

    async def run(self):
        """Main submission process."""
        try:
            # Initialize broker connection
            self.broker = BrokerManager(self.config)
            await self.broker.connect()
            await self.broker.setup_queue_infrastructure(self.queue_name)

            # Start job submission
            submit_task = asyncio.create_task(self._submit_jobs())

            # Wait for submission to complete
            await submit_task

            # Start result consumer
            result_task = asyncio.create_task(self._consume_results())

            # Initialize timeout tracking now that submission is complete
            self.last_result_time = time.time()

            # Wait for all pending results if we have any
            if self.pending_jobs_count > 0 and not self.shutting_down:
                initial_pending = self.pending_jobs_count
                self.console.print(
                    f"[blue]Waiting for {initial_pending} pending results...[/blue]"
                )
                self.console.print(
                    f"[dim]Idle timeout: {self.timeout}s (resets when results arrive)[/dim]"
                )

                # Wait for all results with idle timeout (resets when results come in)
                while self.pending_jobs_count > 0 and not self.shutting_down:
                    time_since_last_result = time.time() - self.last_result_time

                    if time_since_last_result >= self.timeout:
                        self.console.print(
                            f"[yellow]Idle timeout: No results received for {self.timeout}s. Exiting.[/yellow]"
                        )
                        break

                    await asyncio.sleep(0.5)

                if self.shutting_down:
                    self.console.print(
                        f"[yellow]Force quit requested. Abandoning {self.pending_jobs_count} pending results.[/yellow]"
                    )

            # Cancel result consumer
            result_task.cancel()
            try:
                await result_task
            except asyncio.CancelledError:
                pass

        except Exception as e:
            self.logger.error(f"Submit error: {e}", exc_info=True)
            self.console.print(f"[red]Error: {e}[/red]")
        finally:
            # Show final completion stats
            if self.start_time is not None:
                total_time = time.time() - self.start_time
                if total_time > 0 and self.completed_count > 0:
                    completion_rate = self.completed_count / total_time
                    self.console.print(
                        f"[green]Completed {self.completed_count} jobs in {total_time:.1f}s "
                        f"({completion_rate:.1f} jobs/sec)[/green]"
                    )
                elif self.completed_count > 0:
                    self.console.print(
                        f"[green]Completed {self.completed_count} jobs[/green]"
                    )
            elif self.completed_count > 0:
                self.console.print(
                    f"[green]Completed {self.completed_count} jobs[/green]"
                )

            if self.broker:
                await self.broker.disconnect()

    async def _submit_jobs(self):
        """Submit jobs from dataset or JSONL file."""
        if self.is_dataset:
            await self._submit_jobs_from_dataset()
        else:
            await self._submit_jobs_from_file()

    async def _submit_jobs_from_dataset(self):
        """Submit jobs from Hugging Face dataset."""

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("Processed: {task.completed}"),
            TextColumn("Submit: {task.fields[submit_rate]:.1f}/sec"),
            TextColumn("Complete: {task.fields[complete_rate]:.1f}/sec"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            submit_task = progress.add_task(
                "Submitting from dataset",
                total=None,
                submit_rate=0.0,
                complete_rate=0.0,
            )

            chunk = []
            index = 0

            try:
                for item in self._load_dataset_iterator():
                    if not self.running:
                        break

                    try:
                        job = self._create_job_from_dataset_item(item, index)
                        chunk.append(job)
                        index += 1

                        # Process in chunks
                        if len(chunk) >= self.config.chunk_size:
                            await self._submit_chunk(chunk)
                            chunk = []

                            # Update progress
                            elapsed = (
                                time.time() - self.start_time
                                if self.start_time
                                else 0 if self.start_time else 0
                            )
                            submit_rate = (
                                self.submitted_count / elapsed if elapsed > 0 else 0
                            )
                            complete_rate = (
                                self.completed_count / elapsed if elapsed > 0 else 0
                            )
                            progress.update(
                                submit_task,
                                completed=index,
                                submit_rate=submit_rate,
                                complete_rate=complete_rate,
                            )

                            # Small delay to prevent overwhelming RabbitMQ
                            await asyncio.sleep(0.01)

                    except Exception as e:
                        self.logger.error(f"Error processing dataset item {index}: {e}")
                        continue

                # Submit remaining jobs
                if chunk and self.running:
                    await self._submit_chunk(chunk)

                # Final progress update
                elapsed = time.time() - self.start_time if self.start_time else 0
                submit_rate = self.submitted_count / elapsed if elapsed > 0 else 0
                complete_rate = self.completed_count / elapsed if elapsed > 0 else 0
                progress.update(
                    submit_task,
                    completed=index,
                    submit_rate=submit_rate,
                    complete_rate=complete_rate,
                )

            except Exception as e:
                self.console.print(f"[red]Error loading dataset: {e}[/red]")
                raise

        self.console.print(
            f"[green]Submitted {self.submitted_count} jobs from dataset '{self.jobs_source}' to queue '{self.queue_name}'[/green]"
        )

    async def _submit_jobs_from_file(self):
        """Submit jobs from JSONL file."""
        total_lines = self._count_lines()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("Submit: {task.fields[submit_rate]:.1f}/sec"),
            TextColumn("Complete: {task.fields[complete_rate]:.1f}/sec"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            submit_task = progress.add_task(
                "Submitting jobs", total=total_lines, submit_rate=0.0, complete_rate=0.0
            )

            # Handle stdin or regular file
            f = sys.stdin if self.jobs_source == "-" else open(self.jobs_file, "r")

            try:
                chunk = []

                for line_num, line in enumerate(f, 1):
                    if not self.running:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        job_data = json.loads(line)
                        job = Job(**job_data)
                        chunk.append(job)

                        # Process in chunks
                        if len(chunk) >= self.config.chunk_size:
                            await self._submit_chunk(chunk)
                            chunk = []

                            # Update progress
                            elapsed = (
                                time.time() - self.start_time
                                if self.start_time
                                else 0 if self.start_time else 0
                            )
                            submit_rate = (
                                self.submitted_count / elapsed if elapsed > 0 else 0
                            )
                            complete_rate = (
                                self.completed_count / elapsed if elapsed > 0 else 0
                            )
                            progress.update(
                                submit_task,
                                completed=line_num,
                                submit_rate=submit_rate,
                                complete_rate=complete_rate,
                            )

                            # Small delay to prevent overwhelming RabbitMQ
                            await asyncio.sleep(0.01)

                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON on line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(
                            f"Error processing job on line {line_num}: {e}"
                        )
                        continue

                # Submit remaining jobs
                if chunk and self.running:
                    await self._submit_chunk(chunk)

                # Final progress update
                elapsed = time.time() - self.start_time if self.start_time else 0
                submit_rate = self.submitted_count / elapsed if elapsed > 0 else 0
                complete_rate = self.completed_count / elapsed if elapsed > 0 else 0
                progress.update(
                    submit_task,
                    completed=self.submitted_count,
                    submit_rate=submit_rate,
                    complete_rate=complete_rate,
                )
            finally:
                # Close file if it's not stdin
                if self.jobs_source != "-":
                    f.close()

        self.console.print(
            f"[green]Submitted {self.submitted_count} jobs to queue '{self.queue_name}'[/green]"
        )

    async def _submit_chunk(self, jobs: list[Job]):
        """Submit a chunk of jobs concurrently."""
        submit_tasks = []
        for job in jobs:
            submit_tasks.append(self._submit_single_job(job))

        await asyncio.gather(*submit_tasks, return_exceptions=True)

    async def _submit_single_job(self, job: Job):
        """Submit a single job and track it."""
        try:
            if self.broker is not None:
                # Set start_time when first job is submitted (exclude setup/loading time)
                if self.start_time is None:
                    self.start_time = time.time()

                await self.broker.publish_job(self.queue_name, job)
                self.submitted_count += 1
                self.pending_jobs_count += 1
        except Exception as e:
            self.logger.error(f"Failed to submit job {job.id}: {e}")

    async def _consume_results(self):
        """Consume results and output to stdout."""

        async def result_handler(message: AbstractIncomingMessage):
            try:
                result = Result.parse_raw(message.body)

                # Output result to stdout (ensure it's not buffered)
                result_json = result.model_dump_json() + "\n"
                sys.stdout.write(result_json)
                sys.stdout.flush()

                # Track completion
                self.completed_count += 1
                self.pending_jobs_count -= 1
                self.last_result_time = time.time()  # Reset idle timeout

                await message.ack()

            except Exception as e:
                self.logger.error(f"Error processing result: {e}")
                await message.reject(requeue=False)

        try:
            await self.broker.consume_results(self.queue_name, result_handler)

            # Keep consuming until cancelled
            while True:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Result consumer error: {e}")

    def _count_lines(self) -> int:
        """Count total lines in the jobs file."""
        if self.is_dataset:
            return self.max_samples or 0  # Can't easily count dataset items

        # Can't count lines from stdin, return 0 to indicate unknown total
        if self.jobs_source == "-":
            return 0

        try:
            with open(self.jobs_file, "r") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0


def run_submit(
    queue_name: str,
    jobs_source: str,
    timeout: int = 300,
    column_mapping: Optional[Dict[str, str]] = None,
    max_samples: Optional[int] = None,
    split: str = "train",
    subset: Optional[str] = None,
):
    """Run the job submission process."""
    submitter = JobSubmitter(
        queue_name, jobs_source, timeout, column_mapping, max_samples, split, subset
    )

    try:
        asyncio.run(submitter.run())
    except KeyboardInterrupt:
        pass  # Handled gracefully by signal handler
