import asyncio
import json
import sys
import signal
import time
from typing import Dict, Any, TextIO
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from aio_pika.abc import AbstractIncomingMessage

from llmq.core.config import get_config
from llmq.core.broker import BrokerManager
from llmq.core.models import Job, Result
from llmq.utils.logging import setup_logging


class JobSubmitter:
    """Handles job submission and result streaming."""
    
    def __init__(self, queue_name: str, jobs_file: str, timeout: int = 300):
        self.queue_name = queue_name
        self.jobs_file = Path(jobs_file)
        self.timeout = timeout
        self.config = get_config()
        self.logger = setup_logging("llmq.submit")
        
        self.broker: BrokerManager = None
        self.console = Console(file=sys.stderr)
        self.running = True
        self.shutting_down = False
        self.submitted_count = 0
        self.completed_count = 0
        self.pending_jobs: Dict[str, float] = {}  # job_id -> submit_time
        self.start_time = time.time()
        self.last_result_time = time.time()  # Track when we last received a result
        
        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully - stop submitting, wait for pending results."""
        if not self.shutting_down:
            self.console.print("\n[yellow]Received interrupt signal. Stopping submission, waiting for pending results...[/yellow]")
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
            
            # Start result consumer
            result_task = asyncio.create_task(self._consume_results())
            
            # Start job submission
            submit_task = asyncio.create_task(self._submit_jobs())
            
            # Wait for submission to complete
            await submit_task
            
            # Wait for all pending results if we have any
            if self.pending_jobs and not self.shutting_down:
                self.console.print(f"[blue]Waiting for {len(self.pending_jobs)} pending results...[/blue]")
                self.console.print(f"[dim]Idle timeout: {self.timeout}s (resets when results arrive)[/dim]")
                
                # Wait for all results with idle timeout (resets when results come in)
                while self.pending_jobs and not self.shutting_down:
                    time_since_last_result = time.time() - self.last_result_time
                    
                    if time_since_last_result >= self.timeout:
                        self.console.print(f"[yellow]Idle timeout: No results received for {self.timeout}s. Exiting.[/yellow]")
                        break
                        
                    await asyncio.sleep(0.5)
                
                if self.shutting_down:
                    self.console.print(f"[yellow]Force quit requested. Abandoning {len(self.pending_jobs)} pending results.[/yellow]")
            
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
            if self.broker:
                await self.broker.disconnect()
    
    async def _submit_jobs(self):
        """Submit jobs from JSONL file."""
        total_lines = self._count_lines()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("Rate: {task.fields[rate]:.1f} jobs/sec"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            submit_task = progress.add_task(
                "Submitting jobs",
                total=total_lines,
                rate=0.0
            )
            
            with open(self.jobs_file, 'r') as f:
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
                            rate = self.submitted_count / (time.time() - self.start_time) if time.time() > self.start_time else 0
                            progress.update(submit_task, completed=line_num, rate=rate)
                            
                            # Small delay to prevent overwhelming RabbitMQ
                            await asyncio.sleep(0.01)
                            
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON on line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing job on line {line_num}: {e}")
                        continue
                
                # Submit remaining jobs
                if chunk and self.running:
                    await self._submit_chunk(chunk)
                
                # Final progress update
                rate = self.submitted_count / (time.time() - self.start_time) if time.time() > self.start_time else 0
                progress.update(submit_task, completed=self.submitted_count, rate=rate)
        
        self.console.print(f"[green]Submitted {self.submitted_count} jobs to queue '{self.queue_name}'[/green]")
    
    async def _submit_chunk(self, jobs: list[Job]):
        """Submit a chunk of jobs concurrently."""
        submit_tasks = []
        for job in jobs:
            submit_tasks.append(self._submit_single_job(job))
        
        await asyncio.gather(*submit_tasks, return_exceptions=True)
    
    async def _submit_single_job(self, job: Job):
        """Submit a single job and track it."""
        try:
            await self.broker.publish_job(self.queue_name, job)
            self.submitted_count += 1
            self.pending_jobs[job.id] = time.time()
        except Exception as e:
            self.logger.error(f"Failed to submit job {job.id}: {e}")
    
    async def _consume_results(self):
        """Consume results and output to stdout."""
        async def result_handler(message: AbstractIncomingMessage):
            try:
                result = Result.parse_raw(message.body)
                
                # Output result to stdout
                print(result.json(), file=sys.stdout, flush=True)
                
                # Track completion
                if result.id in self.pending_jobs:
                    del self.pending_jobs[result.id]
                    self.completed_count += 1
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
        try:
            with open(self.jobs_file, 'r') as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0


def run_submit(queue_name: str, jobs_file: str, timeout: int = 300):
    """Run the job submission process."""
    submitter = JobSubmitter(queue_name, jobs_file, timeout)
    
    try:
        asyncio.run(submitter.run())
    except KeyboardInterrupt:
        pass  # Handled gracefully by signal handler