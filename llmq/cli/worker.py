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


def run_semhash_worker(
    queue_name: str,
    batch_size: int = 1000,
    mode: str = "deduplicate",
    concurrency: Optional[int] = None,
):
    """Run SemHash worker for semantic deduplication."""
    console = Console()

    try:
        # Lazy import
        from llmq.workers.semhash_worker import SemHashWorker

        console.print(f"[blue]Starting SemHash worker for queue '{queue_name}'[/blue]")
        console.print(f"[dim]Mode: {mode}[/dim]")
        console.print(f"[dim]Batch size: {batch_size}[/dim]")

        if concurrency:
            console.print(f"[dim]Concurrency set to {concurrency} jobs at a time[/dim]")

        worker = SemHashWorker(
            queue_name,
            concurrency=concurrency,
            batch_size=batch_size,
            mode=mode,
        )
        asyncio.run(worker.run())

    except ImportError as e:
        console.print(
            "[red]SemHash not installed. Install with: pip install semhash[/red]"
        )
        console.print(f"[dim]Error: {e}[/dim]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]SemHash worker stopped by user[/yellow]")
    except Exception as e:
        logger = setup_logging("llmq.cli.worker")
        logger.error(f"SemHash worker error: {e}", exc_info=True)
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

        # Get ordered list of stage names for routing
        pipeline_stages = [s.name for s in pipeline_config.stages]

        console.print(
            f"[blue]Starting {stage.worker} worker for pipeline stage '{stage_name}'[/blue]"
        )
        console.print(f"[dim]Pipeline: {pipeline_config.name}[/dim]")
        console.print(f"[dim]Queue: {queue_name}[/dim]")

        # Launch appropriate worker type
        worker: Optional[Union["VLLMWorker", "DummyWorker", "SemHashWorker"]] = None
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
                pipeline_stages=pipeline_stages,
            )

        elif stage.worker == "dummy":
            # Import and create dummy worker
            from llmq.workers.dummy_worker import DummyWorker

            worker = DummyWorker(
                queue_name,
                concurrency=concurrency,
                pipeline_name=pipeline_config.name,
                stage_name=stage_name,
                pipeline_stages=pipeline_stages,
            )

        elif stage.worker == "semhash":
            # Import and create SemHash worker
            from llmq.workers.semhash_worker import SemHashWorker

            # Get SemHash config or use defaults
            batch_size = 1000
            mode = "deduplicate"

            if stage.config:
                batch_size = stage.config.get("batch_size", batch_size)
                mode = stage.config.get("mode", mode)

            worker = SemHashWorker(
                queue_name,
                concurrency=concurrency,
                pipeline_name=pipeline_config.name,
                stage_name=stage_name,
                pipeline_stages=pipeline_stages,
                batch_size=batch_size,
                mode=mode,
            )

        else:
            console.print(f"[red]Unknown worker type: {stage.worker}[/red]")
            console.print(
                "[yellow]Supported worker types: vllm, dummy, semhash[/yellow]"
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
