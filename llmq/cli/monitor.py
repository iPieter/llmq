import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED

from llmq.core.config import get_config
from llmq.core.broker import BrokerManager
from llmq.core.pipeline import PipelineConfig
from llmq.core.models import QueueStats
from llmq.utils.logging import setup_logging


async def check_connection_async():
    """Check basic RabbitMQ connection."""
    broker = BrokerManager(get_config())
    try:
        await broker.connect()
        return True, "Connected to RabbitMQ"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            await broker.disconnect()
        except Exception:
            # Ignore disconnection errors during cleanup
            pass


async def get_queue_stats_async(queue_name: str):
    """Get queue statistics asynchronously."""
    broker = BrokerManager(get_config())
    try:
        await broker.connect()
        stats = await broker.get_queue_stats(queue_name)
        return stats
    except Exception as e:
        return None, str(e)
    finally:
        await broker.disconnect()


async def check_queue_health_async(queue_name: str):
    """Check queue health asynchronously."""
    broker = BrokerManager(get_config())
    try:
        await broker.connect()

        # Try to get queue stats as a basic health check
        stats = await broker.get_queue_stats(queue_name)

        # Basic health criteria
        is_healthy = True
        issues = []

        if stats.consumer_count is not None and stats.consumer_count == 0:
            is_healthy = False
            issues.append("No active consumers")

        if (
            stats.message_count is not None and stats.message_count > 10000
        ):  # Configurable threshold
            issues.append(f"High message backlog: {stats.message_count}")

        return is_healthy, issues, stats

    except Exception as e:
        return False, [f"Connection error: {str(e)}"], None
    finally:
        await broker.disconnect()


async def get_failed_messages_async(queue_name: str, limit: int):
    """Get failed messages asynchronously."""
    broker = BrokerManager(get_config())
    try:
        await broker.connect()
        failed_messages = await broker.get_failed_messages(queue_name, limit)
        return failed_messages
    except Exception as e:
        return None, str(e)
    finally:
        await broker.disconnect()


async def clear_queue_async(queue_name: str):
    """Clear queue asynchronously."""
    broker = BrokerManager(get_config())
    try:
        await broker.connect()
        purged_count = await broker.clear_queue(queue_name)
        return purged_count
    except Exception as e:
        return None, str(e)
    finally:
        await broker.disconnect()


def show_status(queue_name: str):
    """Show queue status and statistics."""
    console = Console()

    try:
        with console.status(f"Getting status for queue '{queue_name}'..."):
            result = asyncio.run(get_queue_stats_async(queue_name))

        if isinstance(result, tuple):
            stats, error = result
            if stats is None:
                console.print(f"[red]Error getting queue stats: {error}[/red]")
                return
        else:
            stats = result

        # Create status table
        table = Table(title=f"Queue Status: {queue_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Queue Name", stats.queue_name)

        # Show detailed message counts if available
        if stats.stats_source == "management_api":
            # Detailed stats from management API
            if stats.message_count is not None:
                table.add_row("Total Messages", str(stats.message_count))
            if stats.message_count_ready is not None:
                table.add_row(
                    "├─ Ready (awaiting processing)", str(stats.message_count_ready)
                )
            if stats.message_count_unacknowledged is not None:
                table.add_row(
                    "└─ Unacknowledged (processing)",
                    str(stats.message_count_unacknowledged),
                )

            # Byte information
            if stats.message_bytes is not None:
                table.add_row(
                    "Total Bytes",
                    f"{stats.message_bytes:,} bytes ({stats.message_bytes / 1024 / 1024:.1f} MB)",
                )
            if stats.message_bytes_ready is not None:
                table.add_row("├─ Ready Bytes", f"{stats.message_bytes_ready:,} bytes")
            if stats.message_bytes_unacknowledged is not None:
                table.add_row(
                    "└─ Unacked Bytes", f"{stats.message_bytes_unacknowledged:,} bytes"
                )

            if stats.consumer_count is not None:
                table.add_row("Active Consumers", str(stats.consumer_count))
        else:
            # Limited stats
            if stats.stats_source == "amqp_fallback":
                table.add_row("Messages in Queue", "Unknown (queue exists)")
                table.add_row("Active Consumers", "Unknown (queue exists)")
                table.add_row(
                    "Stats Source",
                    "AMQP fallback - enable management plugin for details",
                )
            else:
                table.add_row("Messages in Queue", "Unknown (queue may not exist)")
                table.add_row("Active Consumers", "Unknown (queue may not exist)")
                table.add_row(
                    "Stats Source", "Unavailable - check queue name and connection"
                )

        if stats.processing_rate:
            table.add_row("Processing Rate", f"{stats.processing_rate:.1f} jobs/sec")

        table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        console.print(table)

        # Show warnings if any
        if stats.consumer_count is not None and stats.consumer_count == 0:
            console.print(
                Panel(
                    "[yellow]⚠️  No active consumers - jobs will not be processed[/yellow]",
                    title="Warning",
                )
            )

        if stats.message_count is not None and stats.message_count > 1000:
            console.print(
                Panel(
                    f"[yellow]⚠️  High message backlog: {stats.message_count} messages[/yellow]",
                    title="Warning",
                )
            )

        # Show info about stats source
        if stats.stats_source == "amqp_fallback":
            console.print(
                Panel(
                    "[blue]💡 Enable RabbitMQ management plugin for detailed statistics[/blue]",
                    title="Info",
                )
            )
        elif stats.stats_source == "unavailable":
            console.print(
                Panel(
                    "[red]❌ Could not retrieve queue statistics - check queue name and connection[/red]",
                    title="Error",
                )
            )

    except Exception as e:
        logger = setup_logging("llmq.cli.monitor")
        logger.error(f"Status error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")


def check_health(queue_name: str):
    """Basic health check for queue."""
    console = Console()

    try:
        with console.status(f"Checking health for queue '{queue_name}'..."):
            is_healthy, issues, stats = asyncio.run(
                check_queue_health_async(queue_name)
            )

        if is_healthy:
            console.print(f"[green]✅ Queue '{queue_name}' is healthy[/green]")
        else:
            console.print(f"[red]❌ Queue '{queue_name}' has issues:[/red]")
            for issue in issues:
                console.print(f"  [red]• {issue}[/red]")

        if stats:
            console.print(
                f"[dim]Messages: {stats.message_count}, Consumers: {stats.consumer_count}[/dim]"
            )

    except Exception as e:
        logger = setup_logging("llmq.cli.monitor")
        logger.error(f"Health check error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")


def show_errors(queue_name: str, limit: int):
    """Show recent errors from dead letter queue."""
    console = Console()

    try:
        with console.status(f"Getting errors for queue '{queue_name}'..."):
            result = asyncio.run(get_failed_messages_async(queue_name, limit))

        if isinstance(result, tuple):
            failed_messages, error = result
            if failed_messages is None:
                console.print(f"[red]Error getting failed messages: {error}[/red]")
                return
        else:
            failed_messages = result

        if not failed_messages:
            console.print(
                f"[green]No failed messages found in queue '{queue_name}'[/green]"
            )
            return

        # Create errors table
        table = Table(title=f"Failed Messages: {queue_name}")
        table.add_column("Job ID", style="cyan")
        table.add_column("Timestamp", style="yellow")
        table.add_column("Error Details", style="red")

        for msg in failed_messages[:limit]:
            job_id = msg.get("job_id", "Unknown")
            timestamp = msg.get("timestamp", "Unknown")

            # Try to extract error info from job data
            error_details = "Job failed during processing"

            if isinstance(timestamp, datetime):
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp_str = str(timestamp)

            table.add_row(job_id, timestamp_str, error_details)

        console.print(table)

        if len(failed_messages) >= limit:
            console.print(
                f"[dim]Showing first {limit} errors. Use --limit to see more.[/dim]"
            )

    except Exception as e:
        logger = setup_logging("llmq.cli.monitor")
        logger.error(f"Errors command error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")


def show_connection_status():
    """Show basic RabbitMQ connection status."""
    console = Console()

    try:
        with console.status("Checking RabbitMQ connection..."):
            is_connected, message = asyncio.run(check_connection_async())

        config = get_config()

        if is_connected:
            console.print(f"[green]✅ {message}[/green]")
            console.print(f"[dim]URL: {config.rabbitmq_url}[/dim]")
        else:
            console.print(f"[red]❌ Connection failed: {message}[/red]")
            console.print(f"[dim]URL: {config.rabbitmq_url}[/dim]")
            console.print(
                "[yellow]💡 Make sure RabbitMQ is running and accessible[/yellow]"
            )

    except Exception as e:
        logger = setup_logging("llmq.cli.monitor")
        logger.error(f"Connection status error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")


def clear_queue(queue_name: str):
    """Clear all messages from a queue."""
    console = Console()

    try:
        with console.status(f"Clearing queue '{queue_name}'..."):
            result = asyncio.run(clear_queue_async(queue_name))

        if isinstance(result, tuple):
            purged_count, error = result
            if purged_count is None:
                console.print(f"[red]Error clearing queue: {error}[/red]")
                return
        else:
            purged_count = result

        if purged_count == 0:
            console.print(f"[yellow]Queue '{queue_name}' was already empty[/yellow]")
        else:
            console.print(
                f"[green]✅ Cleared {purged_count} messages from queue '{queue_name}'[/green]"
            )

    except Exception as e:
        logger = setup_logging("llmq.cli.monitor")
        logger.error(f"Clear queue error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")


async def get_pipeline_stats_async(pipeline_config: PipelineConfig):
    """Get statistics for all queues in a pipeline."""
    broker = BrokerManager(get_config())
    try:
        await broker.connect()

        pipeline_stats: Dict[str, Optional[QueueStats]] = {}

        # Get stats for each stage queue
        for stage in pipeline_config.stages:
            queue_name = pipeline_config.get_stage_queue_name(stage.name)
            try:
                stats = await broker.get_queue_stats(queue_name)
                pipeline_stats[stage.name] = stats
            except Exception:
                pipeline_stats[stage.name] = None

        # Get stats for results queue
        results_queue = pipeline_config.get_pipeline_results_queue_name()
        try:
            results_stats = await broker.get_queue_stats(results_queue)
            pipeline_stats["results"] = results_stats
        except Exception:
            pipeline_stats["results"] = None

        return pipeline_stats

    except Exception as e:
        return None, str(e)
    finally:
        await broker.disconnect()


def show_pipeline_status(pipeline_config_path: str):
    """Show pipeline status and visualization."""
    console = Console()

    try:
        # Load pipeline configuration
        pipeline_path = Path(pipeline_config_path)
        if not pipeline_path.exists():
            console.print(
                f"[red]Error: Pipeline configuration file not found: {pipeline_config_path}[/red]"
            )
            return

        try:
            pipeline_config = PipelineConfig.from_yaml_file(pipeline_path)
        except Exception as e:
            console.print(f"[red]Error loading pipeline configuration: {e}[/red]")
            return

        # Get pipeline statistics
        with console.status(f"Getting pipeline status for '{pipeline_config.name}'..."):
            result = asyncio.run(get_pipeline_stats_async(pipeline_config))

        if isinstance(result, tuple):
            pipeline_stats, error = result
            if pipeline_stats is None:
                console.print(f"[red]Error getting pipeline stats: {error}[/red]")
                return
        else:
            pipeline_stats = result

        # Create pipeline visualization
        console.print()
        console.print(
            Panel(
                f"[bold cyan]{pipeline_config.name}[/bold cyan]",
                title="Pipeline Status",
                subtitle=f"Configuration: {pipeline_config_path}",
                box=ROUNDED,
            )
        )

        # Create stages table
        stages_table = Table(title="Pipeline Stages", box=ROUNDED)
        stages_table.add_column("Stage", style="cyan", width=15)
        stages_table.add_column("Worker Type", style="yellow", width=12)
        stages_table.add_column("Queue", style="dim", width=25)
        stages_table.add_column("Messages", style="green", justify="right", width=8)
        stages_table.add_column("Consumers", style="blue", justify="right", width=9)
        stages_table.add_column("Status", style="bold", width=10)

        # Add each stage to the table
        for i, stage in enumerate(pipeline_config.stages):
            stage_stats = pipeline_stats.get(stage.name)
            queue_name = pipeline_config.get_stage_queue_name(stage.name)

            if stage_stats is None:
                messages = "N/A"
                consumers = "N/A"
                status = "[red]ERROR[/red]"
            else:
                messages = (
                    str(stage_stats.message_count)
                    if stage_stats.message_count is not None
                    else "N/A"
                )
                consumers = (
                    str(stage_stats.consumer_count)
                    if stage_stats.consumer_count is not None
                    else "N/A"
                )

                # Determine status based on consumers and message count
                if stage_stats.consumer_count == 0:
                    status = "[yellow]NO WORKERS[/yellow]"
                elif stage_stats.message_count and stage_stats.message_count > 1000:
                    status = "[yellow]BACKLOG[/yellow]"
                else:
                    status = "[green]HEALTHY[/green]"

            # Add arrow for flow visualization
            stage_name = stage.name
            if i < len(pipeline_config.stages) - 1:
                stage_name += " →"

            stages_table.add_row(
                stage_name, stage.worker, queue_name, messages, consumers, status
            )

        console.print(stages_table)

        # Show results queue
        results_stats = pipeline_stats.get("results")
        results_queue = pipeline_config.get_pipeline_results_queue_name()

        results_table = Table(title="Pipeline Results", box=ROUNDED)
        results_table.add_column("Queue", style="cyan")
        results_table.add_column("Ready Results", style="green", justify="right")
        results_table.add_column("Status", style="bold")

        if results_stats is None:
            ready_results = "N/A"
            status = "[red]ERROR[/red]"
        else:
            ready_results = (
                str(results_stats.message_count)
                if results_stats.message_count is not None
                else "N/A"
            )
            if results_stats.message_count and results_stats.message_count > 0:
                status = "[green]RESULTS AVAILABLE[/green]"
            else:
                status = "[dim]NO RESULTS[/dim]"

        results_table.add_row(results_queue, ready_results, status)
        console.print(results_table)

        # Show pipeline flow diagram
        console.print()
        flow_text = Text("Pipeline Flow: ", style="bold")

        for i, stage in enumerate(pipeline_config.stages):
            stage_stats = pipeline_stats.get(stage.name)

            # Color code based on status
            if stage_stats is None:
                color = "red"
            elif stage_stats.consumer_count == 0:
                color = "yellow"
            elif stage_stats.message_count and stage_stats.message_count > 1000:
                color = "yellow"
            else:
                color = "green"

            flow_text.append(stage.name, style=f"bold {color}")

            if i < len(pipeline_config.stages) - 1:
                flow_text.append(" → ", style="dim")

        flow_text.append(" → ", style="dim")

        # Results queue status
        results_stats = pipeline_stats.get("results")
        if results_stats is None:
            color = "red"
        elif results_stats.message_count and results_stats.message_count > 0:
            color = "green"
        else:
            color = "dim"

        flow_text.append("RESULTS", style=f"bold {color}")

        console.print(Panel(flow_text, title="Data Flow", box=ROUNDED))

        # Show warnings and suggestions
        warnings = []

        # Check for stages without consumers
        stages_without_consumers = []
        for stage in pipeline_config.stages:
            stage_stats = pipeline_stats.get(stage.name)
            if stage_stats and stage_stats.consumer_count == 0:
                stages_without_consumers.append(stage.name)

        if stages_without_consumers:
            warnings.append(
                f"Stages without workers: {', '.join(stages_without_consumers)}"
            )

        # Check for high backlogs
        stages_with_backlog = []
        for stage in pipeline_config.stages:
            stage_stats = pipeline_stats.get(stage.name)
            if (
                stage_stats
                and stage_stats.message_count
                and stage_stats.message_count > 1000
            ):
                stages_with_backlog.append(
                    f"{stage.name} ({stage_stats.message_count} msgs)"
                )

        if stages_with_backlog:
            warnings.append(f"High backlog in: {', '.join(stages_with_backlog)}")

        if warnings:
            warning_text = "\n".join(f"• {warning}" for warning in warnings)
            console.print(
                Panel(
                    f"[yellow]{warning_text}[/yellow]", title="⚠️  Warnings", box=ROUNDED
                )
            )

        # Show timestamp
        console.print(
            f"[dim]Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]"
        )

    except Exception as e:
        logger = setup_logging("llmq.cli.monitor")
        logger.error(f"Pipeline status error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
