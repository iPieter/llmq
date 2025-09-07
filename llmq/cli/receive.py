import asyncio
import sys
import signal
import time
from typing import Optional, Any

from rich.console import Console
from aio_pika.abc import AbstractIncomingMessage

from llmq.core.config import get_config
from llmq.core.broker import BrokerManager
from llmq.core.models import Result
from llmq.core.pipeline import PipelineConfig
from llmq.utils.logging import setup_logging


class ResultReceiver:
    """Handles receiving results from queues."""

    def __init__(self, queue_name: str, timeout: int = 300):
        self.queue_name = queue_name
        self.timeout = timeout
        self.config = get_config()
        self.logger = setup_logging("llmq.receive")

        self.broker: Optional[BrokerManager] = None
        self.console = Console(file=sys.stderr)
        self.running = True
        self.shutting_down = False
        self.received_count = 0
        self.start_time: Optional[float] = None
        self.last_result_time: Optional[float] = None

        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle Ctrl+C gracefully."""
        if not self.shutting_down:
            self.console.print(
                "\n[yellow]Received interrupt signal. Stopping...[/yellow]"
            )
            self.running = False
            self.shutting_down = True
        else:
            self.console.print("\n[red]Force quitting...[/red]")
            sys.exit(1)

    async def run(self):
        """Main receive process."""
        try:
            # Initialize broker connection
            self.broker = BrokerManager(self.config)
            await self.broker.connect()

            self.console.print(
                f"[blue]Receiving results from queue: {self.queue_name}[/blue]"
            )
            self.console.print(
                f"[dim]Timeout: {self.timeout}s (use Ctrl+C to stop)[/dim]"
            )

            # Start result consumer
            result_task = asyncio.create_task(self._consume_results())
            self.start_time = time.time()
            self.last_result_time = time.time()

            # Wait for results with timeout
            while self.running:
                time_since_last_result = time.time() - self.last_result_time

                if time_since_last_result >= self.timeout:
                    self.console.print(
                        f"[yellow]Timeout: No results received for {self.timeout}s. Exiting.[/yellow]"
                    )
                    break

                await asyncio.sleep(0.5)

            # Cancel result consumer
            result_task.cancel()
            try:
                await result_task
            except asyncio.CancelledError:
                pass

        except Exception as e:
            self.logger.error(f"Receive error: {e}", exc_info=True)
            self.console.print(f"[red]Error: {e}[/red]")
        finally:
            # Show final stats
            if self.start_time is not None and self.received_count > 0:
                total_time = time.time() - self.start_time
                if total_time > 0:
                    receive_rate = self.received_count / total_time
                    self.console.print(
                        f"[green]Received {self.received_count} results in {total_time:.1f}s "
                        f"({receive_rate:.1f} results/sec)[/green]"
                    )
                else:
                    self.console.print(
                        f"[green]Received {self.received_count} results[/green]"
                    )

            if self.broker:
                await self.broker.disconnect()

    async def _consume_results(self):
        """Consume results and output to stdout."""

        async def result_handler(message: AbstractIncomingMessage):
            try:
                result = Result.parse_raw(message.body)

                # Output result to stdout
                result_json = result.model_dump_json() + "\n"
                sys.stdout.write(result_json)
                sys.stdout.flush()

                # Track completion
                self.received_count += 1
                self.last_result_time = time.time()

                await message.ack()

            except Exception as e:
                self.logger.error(f"Error processing result: {e}")
                await message.reject(requeue=False)

        try:
            if self.broker is None:
                raise RuntimeError("Broker not initialized")

            await self.broker.consume_results(self.queue_name, result_handler)

            # Keep consuming until cancelled
            while True:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Result consumer error: {e}")


class PipelineResultReceiver:
    """Handles receiving results from pipeline queues."""

    def __init__(self, pipeline_config_path: str, timeout: int = 300):
        self.pipeline_config_path = pipeline_config_path
        self.timeout = timeout
        self.config = get_config()
        self.logger = setup_logging("llmq.pipeline.receive")

        self.broker: Optional[BrokerManager] = None
        self.console = Console(file=sys.stderr)
        self.running = True
        self.shutting_down = False
        self.received_count = 0
        self.start_time: Optional[float] = None
        self.last_result_time: Optional[float] = None

        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle Ctrl+C gracefully."""
        if not self.shutting_down:
            self.console.print(
                "\n[yellow]Received interrupt signal. Stopping...[/yellow]"
            )
            self.running = False
            self.shutting_down = True
        else:
            self.console.print("\n[red]Force quitting...[/red]")
            sys.exit(1)

    async def run(self):
        """Main pipeline receive process."""
        try:
            # Load pipeline configuration
            from pathlib import Path

            pipeline_config = PipelineConfig.from_yaml_file(
                Path(self.pipeline_config_path)
            )
            results_queue_name = pipeline_config.get_pipeline_results_queue_name()

            # Initialize broker connection
            self.broker = BrokerManager(self.config)
            await self.broker.connect()

            self.console.print(
                f"[blue]Receiving pipeline results from: {pipeline_config.name}[/blue]"
            )
            self.console.print(f"[dim]Queue: {results_queue_name}[/dim]")
            self.console.print(
                f"[dim]Timeout: {self.timeout}s (use Ctrl+C to stop)[/dim]"
            )

            # Start result consumer
            result_task = asyncio.create_task(self._consume_results(results_queue_name))
            self.start_time = time.time()
            self.last_result_time = time.time()

            # Wait for results with timeout
            while self.running:
                time_since_last_result = time.time() - self.last_result_time

                if time_since_last_result >= self.timeout:
                    self.console.print(
                        f"[yellow]Timeout: No results received for {self.timeout}s. Exiting.[/yellow]"
                    )
                    break

                await asyncio.sleep(0.5)

            # Cancel result consumer
            result_task.cancel()
            try:
                await result_task
            except asyncio.CancelledError:
                pass

        except Exception as e:
            self.logger.error(f"Pipeline receive error: {e}", exc_info=True)
            self.console.print(f"[red]Error: {e}[/red]")
        finally:
            # Show final stats
            if self.start_time is not None and self.received_count > 0:
                total_time = time.time() - self.start_time
                if total_time > 0:
                    receive_rate = self.received_count / total_time
                    self.console.print(
                        f"[green]Received {self.received_count} pipeline results in {total_time:.1f}s "
                        f"({receive_rate:.1f} results/sec)[/green]"
                    )
                else:
                    self.console.print(
                        f"[green]Received {self.received_count} pipeline results[/green]"
                    )

            if self.broker:
                await self.broker.disconnect()

    async def _consume_results(self, results_queue_name: str):
        """Consume pipeline results and output to stdout."""

        async def result_handler(message: AbstractIncomingMessage):
            try:
                result = Result.parse_raw(message.body)

                # Output result to stdout
                result_json = result.model_dump_json() + "\n"
                sys.stdout.write(result_json)
                sys.stdout.flush()

                # Track completion
                self.received_count += 1
                self.last_result_time = time.time()

                await message.ack()

            except Exception as e:
                self.logger.error(f"Error processing pipeline result: {e}")
                await message.reject(requeue=False)

        try:
            if self.broker is None:
                raise RuntimeError("Broker not initialized")

            await self.broker.consume_results(results_queue_name, result_handler)

            # Keep consuming until cancelled
            while True:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Pipeline result consumer error: {e}")


def run_receive(queue_name: str, timeout: int = 300):
    """Run the result receiving process."""
    receiver = ResultReceiver(queue_name, timeout)

    try:
        asyncio.run(receiver.run())
    except KeyboardInterrupt:
        pass  # Handled gracefully by signal handler


def run_pipeline_receive(pipeline_config_path: str, timeout: int = 300):
    """Run the pipeline result receiving process."""
    receiver = PipelineResultReceiver(pipeline_config_path, timeout)

    try:
        asyncio.run(receiver.run())
    except KeyboardInterrupt:
        pass  # Handled gracefully by signal handler
