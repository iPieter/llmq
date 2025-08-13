import asyncio
import sys
from typing import Optional, Union

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


def run_pipeline_worker(
    pipeline_config_path: str, stage_name: str, concurrency: Optional[int] = None
):
    """Run worker for a specific pipeline stage."""
    console = Console()

    try:
        from pathlib import Path
        from llmq.core.pipeline import PipelineConfig

        # Load pipeline configuration
        pipeline_config = PipelineConfig.from_yaml_file(Path(pipeline_config_path))

        # Find the stage
        stage = pipeline_config.get_stage_by_name(stage_name)
        if not stage:
            console.print(
                f"[red]Stage '{stage_name}' not found in pipeline '{pipeline_config.name}'[/red]"
            )
            console.print(
                f"[yellow]Available stages: {', '.join(s.name for s in pipeline_config.stages)}[/yellow]"
            )
            sys.exit(1)

        # Get queue name for this stage
        queue_name = pipeline_config.get_stage_queue_name(stage_name)

        console.print(
            f"[blue]Starting {stage.worker} worker for pipeline stage '{stage_name}'[/blue]"
        )
        console.print(f"[dim]Pipeline: {pipeline_config.name}[/dim]")
        console.print(f"[dim]Queue: {queue_name}[/dim]")

        # Launch appropriate worker type
        worker: Optional[Union["VLLMWorker", "DummyWorker", "FilterWorker"]] = None
        if stage.worker == "vllm":
            # Need model name from stage config
            if stage.config is None:
                console.print(
                    "[red]vLLM worker requires stage config with 'model' field[/red]"
                )
                sys.exit(1)

            model_name = stage.config.get("model")
            if not model_name:
                console.print("[red]vLLM worker requires 'model' in stage config[/red]")
                sys.exit(1)

            # Import and create vLLM worker
            from llmq.workers.vllm_worker import VLLMWorker

            worker = VLLMWorker(
                model_name,
                queue_name,
                concurrency=concurrency,
                pipeline_name=pipeline_config.name,
                stage_name=stage_name,
            )

        elif stage.worker == "dummy":
            # Import and create dummy worker
            from llmq.workers.dummy_worker import DummyWorker

            worker = DummyWorker(
                queue_name,
                concurrency=concurrency,
                pipeline_name=pipeline_config.name,
                stage_name=stage_name,
            )

        elif stage.worker == "filter":
            # Need filter config
            if stage.config is None:
                console.print(
                    "[red]Filter worker requires stage config with 'filter_field' and 'filter_value'[/red]"
                )
                sys.exit(1)

            filter_field = stage.config.get("filter_field")
            filter_value = stage.config.get("filter_value")
            if not filter_field or not filter_value:
                console.print(
                    "[red]Filter worker requires 'filter_field' and 'filter_value' in stage config[/red]"
                )
                sys.exit(1)

            from llmq.workers.dummy_worker import FilterWorker

            worker = FilterWorker(
                queue_name,
                filter_field,
                filter_value,
                pipeline_name=pipeline_config.name,
                stage_name=stage_name,
            )

        else:
            console.print(f"[red]Unknown worker type: {stage.worker}[/red]")
            console.print(
                "[yellow]Supported worker types: vllm, dummy, filter[/yellow]"
            )
            sys.exit(1)

        # Run the worker
        if worker is None:
            console.print("[red]Failed to create worker[/red]")
            sys.exit(1)

        asyncio.run(worker.run())

    except FileNotFoundError as e:
        console.print(f"[red]Pipeline configuration file not found: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        logger = setup_logging("llmq.cli.worker")
        logger.error(f"Pipeline worker error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline worker stopped by user[/yellow]")
