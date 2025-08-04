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
def submit(
    queue_name: str,
    jobs_source: str,
    timeout: int,
    column_mapping: tuple,
    max_samples: int,
):
    """Submit jobs from JSONL file or Hugging Face dataset to queue

    Examples:
    \b
    # From JSONL file
    llmq submit translation-queue example_jobs.jsonl

    # From Hugging Face dataset
    llmq submit translation-queue HuggingFaceFW/fineweb --map prompt=text --max-samples 1000

    # Column mapping for translation task
    llmq submit translation-queue wmt14 --map prompt="Translate to Dutch: {en}" --map source_lang=en --map target_lang=nl
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


@worker.command("run")
@click.argument("model_name")
@click.argument("queue_name")
def worker_run(model_name: str, queue_name: str):
    """Run vLLM worker using all visible GPUs"""
    from llmq.cli.worker import run_vllm_worker

    run_vllm_worker(model_name, queue_name)


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
