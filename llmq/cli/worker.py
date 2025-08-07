import asyncio
import sys
from typing import Optional

from rich.console import Console
from llmq.utils.logging import setup_logging


def run_vllm_worker(
    model_name: str,
    queue_name: str,
    tensor_parallel_size: Optional[int] = None,
    data_parallel_size: Optional[int] = None,
):
    """Run vLLM worker with configurable parallelism."""
    console = Console()

    try:
        # Lazy import to avoid dependency issues
        from llmq.workers.vllm_worker import VLLMWorker

        console.print(
            f"[blue]Starting vLLM worker for model '{model_name}' on queue '{queue_name}'[/blue]"
        )

        if tensor_parallel_size:
            console.print(
                f"[dim]Tensor parallel size: {tensor_parallel_size} GPUs per replica[/dim]"
            )

        if data_parallel_size:
            console.print(
                f"[dim]Data parallel size: {data_parallel_size} replicas[/dim]"
            )

        if not tensor_parallel_size and not data_parallel_size:
            console.print("[dim]Worker will use all visible GPUs automatically[/dim]")

        worker = VLLMWorker(
            model_name,
            queue_name,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
        )
        asyncio.run(worker.run())

    except ImportError as e:
        console.print("[red]vLLM not installed. Install with: pip install vllm[/red]")
        console.print(f"[dim]Error: {e}[/dim]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]vLLM worker stopped by user[/yellow]")
    except Exception as e:
        logger = setup_logging("llmq.cli.worker")
        logger.error(f"vLLM worker error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def run_dummy_worker(queue_name: str, concurrency: Optional[int] = None):
    """Run dummy worker for testing (no vLLM required)."""
    console = Console()

    try:
        # Lazy import
        from llmq.workers.dummy_worker import DummyWorker

        console.print(f"[blue]Starting dummy worker for queue '{queue_name}'[/blue]")

        if concurrency:
            console.print(f"[dim]Concurrency set to {concurrency} jobs at a time[/dim]")
        else:
            console.print("[dim]Using default concurrency (VLLM_QUEUE_PREFETCH)[/dim]")

        worker = DummyWorker(queue_name, concurrency=concurrency)
        asyncio.run(worker.run())

    except KeyboardInterrupt:
        console.print("\n[yellow]Dummy worker stopped by user[/yellow]")
    except Exception as e:
        logger = setup_logging("llmq.cli.worker")
        logger.error(f"Dummy worker error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def run_filter_worker(queue_name: str, filter_field: str, filter_value: str):
    """Run filter worker for simple job filtering."""
    console = Console()

    try:
        # Lazy import
        from llmq.workers.dummy_worker import FilterWorker

        console.print(f"[blue]Starting filter worker for queue '{queue_name}'[/blue]")
        console.print(f"[dim]Filter: {filter_field} contains '{filter_value}'[/dim]")

        worker = FilterWorker(queue_name, filter_field, filter_value)
        asyncio.run(worker.run())

    except KeyboardInterrupt:
        console.print("\n[yellow]Filter worker stopped by user[/yellow]")
    except Exception as e:
        logger = setup_logging("llmq.cli.worker")
        logger.error(f"Filter worker error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
