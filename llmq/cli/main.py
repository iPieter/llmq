import click
from typing import Optional
from llmq import __version__


@click.group()
@click.version_option(version=__version__, prog_name="llmq")
@click.pass_context
def cli(ctx):
    """High-Performance vLLM Job Queue Package"""
    ctx.ensure_object(dict)


@cli.group()
def worker():
    """Worker management commands"""
    pass


@cli.command()
@click.argument("first_arg")
@click.argument("second_arg", required=False)  # Can be file path or dataset name
@click.option(
    "-p",
    "--pipeline",
    "pipeline_config",
    help="Pipeline configuration file (TOML/YAML format)",
)
@click.option(
    "--timeout",
    default=300,
    help="Timeout in seconds to wait for results (only with --stream)",
)
@click.option(
    "--map",
    "column_mapping",
    multiple=True,
    help="Column mapping: --map prompt=text --map target_lang=language",
)
@click.option(
    "--max-samples", type=int, help="Maximum number of samples to process from dataset"
)
@click.option("--split", default="train", help="Dataset split to use (default: train)")
@click.option("--subset", help="Dataset subset/config to use")
@click.option(
    "--stream",
    is_flag=True,
    help="Stream results back immediately (backwards compatibility mode)",
)
def submit(
    first_arg: str,
    second_arg: Optional[str],
    pipeline_config: Optional[str],
    timeout: int,
    column_mapping: tuple,
    max_samples: int,
    split: str,
    subset: str,
    stream: bool,
):
    """Submit jobs from JSONL file or Hugging Face dataset to queue or pipeline

    Usage patterns:
    \b
    # Submit to queue (traditional mode)
    llmq submit QUEUE_NAME JOBS_SOURCE

    # Submit to pipeline (new simplified mode)
    llmq submit -p pipeline.toml JOBS_SOURCE

    By default, jobs are submitted and you can receive results separately using:
    \b
    llmq receive QUEUE_NAME
    llmq receive -p pipeline.toml

    The --stream flag provides backwards compatibility - it submits jobs AND
    streams results back immediately (like the old behavior).

    The --map option supports three types of mappings:
    1. Simple column mapping: --map field=column
    2. Template strings: --map field="Template with {column}"
    3. JSON templates: --map field='{"key": "value with {column}"}'

    Examples:
    \b
    # Traditional queue submission
    llmq submit translation-queue example_jobs.jsonl
    llmq receive translation-queue > results.jsonl

    # New simplified pipeline submission
    llmq submit -p example-pipeline.toml test-jobs.jsonl
    llmq receive -p example-pipeline.toml > results.jsonl

    # Submit with streaming (backwards compatibility)
    llmq submit translation-queue example_jobs.jsonl --stream > results.jsonl
    llmq submit -p example-pipeline.toml test-jobs.jsonl --stream > results.jsonl

    # Dataset with mapping
    llmq submit translation-queue HuggingFaceFW/fineweb --map source_text=text --max-samples 1000
    llmq submit -p pipeline.toml HuggingFaceFW/fineweb --map prompt="Translate: {text}" --max-samples 1000
    """
    from llmq.cli.submit import run_submit, run_pipeline_submit

    # Parse column mapping from CLI format
    mapping_dict = {}
    for mapping in column_mapping:
        if "=" in mapping:
            key, value = mapping.split("=", 1)
            mapping_dict[key] = value
        else:
            click.echo(
                f"Warning: Invalid mapping format '{mapping}'. Use key=value format."
            )

    # Handle pipeline mode vs regular queue mode
    if pipeline_config:
        # Pipeline mode: llmq submit -p pipeline.toml jobs.jsonl
        # In this case, first_arg is the jobs source
        jobs_source = first_arg
        if jobs_source is None:
            click.echo("Error: jobs source is required when using -p/--pipeline flag")
            raise click.Abort()

        run_pipeline_submit(
            pipeline_config,
            jobs_source,
            timeout,
            mapping_dict if mapping_dict else None,
            max_samples,
            split,
            subset,
            stream,
        )
    else:
        # Regular queue mode: llmq submit queue-name jobs.jsonl
        if second_arg is None:
            click.echo("Error: jobs source is required when not using pipeline mode")
            raise click.Abort()

        run_submit(
            first_arg,  # This is the queue name in regular mode
            second_arg,  # This is the jobs source
            timeout,
            mapping_dict if mapping_dict else None,
            max_samples,
            split,
            subset,
            stream,
        )


@cli.command("pipeline", deprecated=True)
@click.argument("pipeline_config_path")
@click.argument("jobs_source")  # Can be file path or dataset name
@click.option(
    "--timeout",
    default=300,
    help="Timeout in seconds to wait for results (only with --stream)",
)
@click.option(
    "--map",
    "column_mapping",
    multiple=True,
    help="Column mapping: --map prompt=text --map target_lang=language",
)
@click.option(
    "--max-samples", type=int, help="Maximum number of samples to process from dataset"
)
@click.option("--split", default="train", help="Dataset split to use (default: train)")
@click.option("--subset", help="Dataset subset/config to use")
@click.option(
    "--stream",
    is_flag=True,
    help="Stream results back immediately (backwards compatibility mode)",
)
def pipeline_submit(
    pipeline_config_path: str,
    jobs_source: str,
    timeout: int,
    column_mapping: tuple,
    max_samples: int,
    split: str,
    subset: str,
    stream: bool,
):
    """[DEPRECATED] Submit jobs through a multi-stage pipeline

    This command is deprecated. Use the simplified syntax instead:
    \b
    # Instead of: llmq pipeline pipeline.yaml jobs.jsonl
    # Use:        llmq submit -p pipeline.yaml jobs.jsonl

    Pipeline configuration is defined in YAML format:
    \b
    name: document-processing
    stages:
      - name: deduplication
        worker: bloom-filter
      - name: translation
        worker: vllm
        config:
          model: microsoft/DialoGPT-medium
      - name: formatting
        worker: vllm

    Workers are launched separately using the pipeline command:
    \b
    llmq worker pipeline pipeline.yaml deduplication
    llmq worker pipeline pipeline.yaml translation
    llmq worker pipeline pipeline.yaml formatting

    Examples:
    \b
    # DEPRECATED - Run pipeline on JSONL file
    llmq pipeline pipeline.yaml jobs.jsonl

    # NEW SYNTAX - Run pipeline on JSONL file
    llmq submit -p pipeline.yaml jobs.jsonl

    # DEPRECATED - Run pipeline on dataset with column mapping
    llmq pipeline pipeline.yaml HuggingFaceFW/fineweb --map prompt=text --max-samples 100

    # NEW SYNTAX - Run pipeline on dataset with column mapping
    llmq submit -p pipeline.yaml HuggingFaceFW/fineweb --map prompt=text --max-samples 100
    """
    from llmq.cli.submit import run_pipeline_submit

    # Show deprecation warning
    click.echo(
        "⚠️  WARNING: This command is deprecated. Use 'llmq submit -p PIPELINE_CONFIG JOBS_SOURCE' instead.",
        err=True,
    )

    # Parse column mapping from CLI format
    mapping_dict = {}
    for mapping in column_mapping:
        if "=" in mapping:
            key, value = mapping.split("=", 1)
            mapping_dict[key] = value
        else:
            click.echo(
                f"Warning: Invalid mapping format '{mapping}'. Use key=value format."
            )

    run_pipeline_submit(
        pipeline_config_path,
        jobs_source,
        timeout,
        mapping_dict if mapping_dict else None,
        max_samples,
        split,
        subset,
        stream,
    )


@cli.command()
@click.argument("queue_name", required=False)
@click.option(
    "-p",
    "--pipeline",
    "pipeline_config",
    help="Pipeline configuration file (YAML format)",
)
def status(queue_name: Optional[str] = None, pipeline_config: Optional[str] = None):
    """Show connection status, queue statistics, or pipeline visualization

    Usage patterns:
    \b
    # Show connection status
    llmq status

    # Show queue statistics
    llmq status QUEUE_NAME

    # Show pipeline visualization
    llmq status -p pipeline.yaml

    Examples:
    \b
    # Check RabbitMQ connection
    llmq status

    # Monitor specific queue
    llmq status translation-queue

    # Visualize pipeline status
    llmq status -p example-pipeline.yaml
    """
    from llmq.cli.monitor import (
        show_status,
        show_connection_status,
        show_pipeline_status,
    )

    if pipeline_config:
        # Pipeline mode: llmq status -p pipeline.yaml
        show_pipeline_status(pipeline_config)
    elif queue_name:
        # Queue mode: llmq status queue-name
        show_status(queue_name)
    else:
        # Connection status mode: llmq status
        show_connection_status()


@cli.command()
@click.argument("queue_name")
def health(queue_name: str):
    """Basic health check for queue"""
    from llmq.cli.monitor import check_health

    check_health(queue_name)


@cli.command()
@click.argument("queue_name")
@click.option("--limit", default=100, help="Maximum number of errors to show")
def errors(queue_name: str, limit: int):
    """Show recent errors from dead letter queue"""
    from llmq.cli.monitor import show_errors

    show_errors(queue_name, limit)


@cli.command()
@click.argument("queue_name_or_pipeline", required=False)
@click.option(
    "-p",
    "--pipeline",
    "pipeline_config",
    help="Pipeline configuration file (TOML/YAML format)",
)
@click.option("--timeout", default=300, help="Timeout in seconds to wait for results")
def receive(
    queue_name_or_pipeline: Optional[str], pipeline_config: Optional[str], timeout: int
):
    """Receive results from a queue or pipeline

    Usage patterns:
    \b
    # Receive from queue (traditional mode)
    llmq receive QUEUE_NAME

    # Receive from pipeline (new simplified mode)
    llmq receive -p pipeline.toml

    Examples:
    \b
    # Traditional queue receiving
    llmq receive translation-queue > results.jsonl
    llmq receive translation-queue --timeout 600

    # New simplified pipeline receiving
    llmq receive -p example-pipeline.toml > results.jsonl
    llmq receive -p example-pipeline.toml --timeout 600
    """
    from llmq.cli.receive import run_receive, run_pipeline_receive

    # Handle pipeline mode vs regular queue mode
    if pipeline_config:
        # Pipeline mode: llmq receive -p pipeline.toml
        run_pipeline_receive(pipeline_config, timeout)
    else:
        # Regular queue mode: llmq receive queue-name
        if queue_name_or_pipeline is None:
            click.echo(
                "Error: queue name is required when not using -p/--pipeline flag"
            )
            raise click.Abort()

        run_receive(queue_name_or_pipeline, timeout)


@cli.command("receive-pipeline", deprecated=True)
@click.argument("pipeline_config_path")
@click.option("--timeout", default=300, help="Timeout in seconds to wait for results")
def receive_pipeline(pipeline_config_path: str, timeout: int):
    """[DEPRECATED] Receive results from a pipeline

    This command is deprecated. Use the simplified syntax instead:
    \b
    # Instead of: llmq receive-pipeline pipeline.yaml
    # Use:        llmq receive -p pipeline.yaml

    Examples:
    \b
    # DEPRECATED - Receive pipeline results
    llmq receive-pipeline pipeline.yaml > results.jsonl

    # NEW SYNTAX - Receive pipeline results
    llmq receive -p pipeline.yaml > results.jsonl

    # DEPRECATED - With custom timeout
    llmq receive-pipeline pipeline.yaml --timeout 600

    # NEW SYNTAX - With custom timeout
    llmq receive -p pipeline.yaml --timeout 600
    """
    from llmq.cli.receive import run_pipeline_receive

    # Show deprecation warning
    click.echo(
        "⚠️  WARNING: This command is deprecated. Use 'llmq receive -p PIPELINE_CONFIG' instead.",
        err=True,
    )

    run_pipeline_receive(pipeline_config_path, timeout)


@cli.command()
@click.argument("queue_name")
@click.confirmation_option(
    prompt="Are you sure you want to clear all messages from the queue?"
)
def clear(queue_name: str):
    """Clear all messages from a queue"""
    from llmq.cli.monitor import clear_queue

    clear_queue(queue_name)


@worker.command("run")
@click.argument("model_name")
@click.argument("queue_name")
@click.option(
    "--tensor-parallel-size",
    "-tp",
    default=None,
    type=int,
    help="Tensor parallel size (number of GPUs per model replica)",
)
@click.option(
    "--data-parallel-size",
    "-dp",
    default=None,
    type=int,
    help="Data parallel size (number of model replicas)",
)
def worker_run(
    model_name: str,
    queue_name: str,
    tensor_parallel_size: Optional[int],
    data_parallel_size: Optional[int],
):
    """Run vLLM worker using all visible GPUs

    Examples:
    \b
    # Use all GPUs with automatic configuration
    llmq worker run model-name queue-name

    # Tensor parallelism: split model across 4 GPUs
    llmq worker run model-name queue-name --tensor-parallel-size 4

    # Data parallelism: 2 model replicas, each using 2 GPUs
    llmq worker run model-name queue-name --data-parallel-size 2 --tensor-parallel-size 2
    """
    from llmq.cli.worker import run_vllm_worker

    run_vllm_worker(model_name, queue_name, tensor_parallel_size, data_parallel_size)


@worker.command("dummy")
@click.argument("queue_name")
@click.option(
    "--concurrency",
    "-c",
    default=None,
    type=int,
    help="Number of jobs to process concurrently",
)
def worker_dummy(queue_name: str, concurrency: int):
    """Run dummy worker for testing (no vLLM required)"""
    from llmq.cli.worker import run_dummy_worker

    run_dummy_worker(queue_name, concurrency)


@worker.command("semhash")
@click.argument("queue_name")
@click.option(
    "--batch-size",
    default=1000,
    type=int,
    help="Batch size for processing texts together",
)
@click.option(
    "--mode",
    default="deduplicate",
    type=click.Choice(["deduplicate", "filter_outliers", "find_representative"]),
    help="SemHash processing mode",
)
@click.option(
    "--concurrency",
    "-c",
    default=None,
    type=int,
    help="Number of jobs to process concurrently",
)
def worker_semhash(
    queue_name: str,
    batch_size: int,
    mode: str,
    concurrency: int,
):
    """Run SemHash worker for semantic deduplication and filtering

    Modes:
    - deduplicate: Remove semantically similar texts
    - filter_outliers: Remove outlier texts
    - find_representative: Keep only representative texts
    """
    from llmq.cli.worker import run_semhash_worker

    run_semhash_worker(queue_name, batch_size, mode, concurrency)


@worker.command("pipeline")
@click.argument("pipeline_config_path")
@click.argument("stage_name")
@click.option(
    "--concurrency",
    "-c",
    default=None,
    type=int,
    help="Number of jobs to process concurrently",
)
def worker_pipeline(pipeline_config_path: str, stage_name: str, concurrency: int):
    """Run worker for a specific pipeline stage

    Loads the pipeline configuration and runs the appropriate worker type
    for the specified stage with its configuration.

    Examples:
    \b
    # Run deduplication stage
    llmq worker pipeline document-pipeline.yaml deduplication

    # Run with custom concurrency
    llmq worker pipeline document-pipeline.yaml translation --concurrency 4
    """
    from llmq.cli.worker import run_pipeline_worker

    run_pipeline_worker(pipeline_config_path, stage_name, concurrency)


if __name__ == "__main__":
    cli()
