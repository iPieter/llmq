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
@click.argument("jobs_file", type=click.Path(exists=True))
@click.option("--timeout", default=300, help="Timeout in seconds to wait for results")
def submit(queue_name: str, jobs_file: str, timeout: int):
    """Submit jobs from JSONL file to queue"""
    from llmq.cli.submit import run_submit

    run_submit(queue_name, jobs_file, timeout)


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
