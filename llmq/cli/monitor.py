import asyncio
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from llmq.core.config import get_config
from llmq.core.broker import BrokerManager
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
