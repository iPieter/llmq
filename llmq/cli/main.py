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
@click.argument("queue_name")
@click.argument("jobs_source")  # Can be file path or dataset name
@click.option("--timeout", default=300, help="Timeout in seconds to wait for results")
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
def submit(
    queue_name: str,
    jobs_source: str,
    timeout: int,
    column_mapping: tuple,
    max_samples: int,
    split: str,
    subset: str,
):
    """Submit jobs from JSONL file or Hugging Face dataset to queue

    The --map option supports three types of mappings:
    1. Simple column mapping: --map field=column
    2. Template strings: --map field="Template with {column}"  
    3. JSON templates: --map field='{"key": "value with {column}"}'

    Examples:
    \b
    # From JSONL file
    llmq submit translation-queue example_jobs.jsonl

    # Simple mapping from dataset column to job field
    llmq submit translation-queue HuggingFaceFW/fineweb --map source_text=text --max-samples 1000

    # Template string mapping
    llmq submit translation-queue HuggingFaceFW/fineweb --map prompt="Translate to Dutch: {text}" --max-samples 1000

    # JSON mapping for chat messages
    llmq submit translation-queue HuggingFaceFW/fineweb \\
      --map 'messages=[{"role": "user", "content": "Translate to Dutch:\\nEnglish: {text}\\nDutch: "}]' \\
      --map source_text=text --max-samples 1000
    """
    from llmq.cli.submit import run_submit

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

    run_submit(
        queue_name,
        jobs_source,
        timeout,
        mapping_dict if mapping_dict else None,
        max_samples,
        split,
        subset,
    )


@cli.command()
@click.argument("queue_name", required=False)
def status(queue_name: Optional[str] = None):
    """Show connection status or queue statistics"""
    from llmq.cli.monitor import show_status, show_connection_status

    if queue_name:
        show_status(queue_name)
    else:
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


@worker.command("filter")
@click.argument("queue_name")
@click.argument("filter_field")
@click.argument("filter_value")
def worker_filter(queue_name: str, filter_field: str, filter_value: str):
    """Run filter worker for simple job filtering"""
    from llmq.cli.worker import run_filter_worker

    run_filter_worker(queue_name, filter_field, filter_value)


if __name__ == "__main__":
    cli()
