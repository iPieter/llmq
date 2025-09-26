import asyncio
import logging
from typing import Optional, Callable, Any
from urllib.parse import urlparse
import aio_pika
from aio_pika.abc import (
    AbstractConnection,
    AbstractChannel,
    AbstractQueue,
)
from aio_pika import DeliveryMode
import httpx

from llmq.core.config import get_config
from llmq.core.models import Job, Result, QueueStats


class BrokerManager:
    """Manages RabbitMQ connections and queue operations."""

    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self.connection: Optional[AbstractConnection] = None
        self.channel: Optional[AbstractChannel] = None
        self.logger = logging.getLogger("llmq.broker")

    async def connect(self) -> None:
        """Establish connection to RabbitMQ with retry logic."""
        max_retries = 5
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                self.connection = await aio_pika.connect_robust(
                    self.config.rabbitmq_url, client_properties={"application": "llmq"}
                )
                self.channel = await self.connection.channel()
                await self.channel.set_qos(
                    prefetch_count=self.config.vllm_queue_prefetch
                )
                self.logger.info("Connected to RabbitMQ")
                return
            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    async def disconnect(self) -> None:
        """Close RabbitMQ connection."""
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            self.logger.info("Disconnected from RabbitMQ")

    async def setup_queue_infrastructure(
        self, queue_name: str
    ) -> tuple[AbstractQueue, AbstractQueue]:
        """
        Set up queue infrastructure for a given queue name.

        Returns:
            Tuple of (job_queue, results_queue)
        """
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        # Set up main job queue
        job_queue = await self.channel.declare_queue(
            queue_name,
            durable=True,
        )

        # Set up results queue (durable for resumable downloads)
        results_queue = await self.channel.declare_queue(
            f"{queue_name}.results", durable=True
        )

        self.logger.info(f"Queue infrastructure set up for {queue_name}")
        return job_queue, results_queue

    async def setup_pipeline_infrastructure(
        self, pipeline_name: str, stages: list[str]
    ) -> tuple[dict[str, AbstractQueue], AbstractQueue]:
        """
        Set up pipeline infrastructure with direct queue routing.

        Returns:
            Tuple of (stage_queues_dict, final_results_queue) where stage_queues_dict
            maps stage names to job queues, and final_results_queue is for final output
        """
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        stage_queues = {}

        # Set up stage queues
        for stage in stages:
            stage_queue_name = f"pipeline.{pipeline_name}.{stage}"
            job_queue = await self.channel.declare_queue(
                stage_queue_name,
                durable=True,
            )
            stage_queues[stage] = job_queue

        # Set up final results queue (only one for the entire pipeline)
        final_results_queue = await self.channel.declare_queue(
            f"pipeline.{pipeline_name}.results", durable=True
        )

        self.logger.info(f"Pipeline infrastructure set up for {pipeline_name}")
        return stage_queues, final_results_queue

    async def publish_job(self, queue_name: str, job: Job) -> None:
        """Publish a job to the specified queue."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        message = aio_pika.Message(
            job.model_dump_json().encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=job.id,
        )

        await self.channel.default_exchange.publish(message, routing_key=queue_name)

    async def publish_result(self, queue_name: str, result: Result) -> None:
        """Publish a result to the results queue."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        message = aio_pika.Message(
            result.model_dump_json().encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=result.id,
        )

        # Publish directly to results queue
        results_queue_name = f"{queue_name}.results"
        await self.channel.default_exchange.publish(
            message, routing_key=results_queue_name
        )

    async def publish_pipeline_result(
        self,
        pipeline_name: str,
        stage_name: str,
        stages: list[str],
        result: Result,
    ) -> None:
        """Publish a pipeline result - either to next stage or final results."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        # Find current stage index
        current_stage_idx = stages.index(stage_name)

        if current_stage_idx == len(stages) - 1:
            # This is the final stage - publish to pipeline results queue
            results_queue_name = f"pipeline.{pipeline_name}.results"
            message = aio_pika.Message(
                result.model_dump_json().encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                message_id=result.id,
            )
            await self.channel.default_exchange.publish(
                message, routing_key=results_queue_name
            )
            self.logger.info(f"Pipeline final result: {stage_name} -> results queue")
        else:
            # Route to next stage as a job
            next_stage = stages[current_stage_idx + 1]
            next_stage_queue = f"pipeline.{pipeline_name}.{next_stage}"

            # Convert result to next stage job, preserving metadata
            extra_fields = result.model_extra if result.model_extra else {}
            next_job = Job(
                id=result.id,  # Keep same ID for tracking
                prompt=result.result,  # Previous result becomes next prompt
                **extra_fields,
            )

            message = aio_pika.Message(
                next_job.model_dump_json().encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                message_id=result.id,
            )

            await self.channel.default_exchange.publish(
                message, routing_key=next_stage_queue
            )
            self.logger.info(f"Pipeline result routed: {stage_name} -> {next_stage}")

    async def consume_jobs(self, queue_name: str, callback: Callable) -> AbstractQueue:
        """Set up job consumption with the provided callback."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        job_queue, _ = await self.setup_queue_infrastructure(queue_name)
        await job_queue.consume(callback)
        return job_queue

    async def consume_results(
        self, queue_name: str, callback: Callable
    ) -> AbstractQueue:
        """Set up result consumption with the provided callback."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        # Check if this is already a results queue (pipeline results or explicit results queue)
        if queue_name.endswith(".results"):
            # Direct results queue - just declare it
            results_queue = await self.channel.declare_queue(queue_name, durable=True)
        else:
            # Regular queue - set up infrastructure and get results queue
            _, results_queue = await self.setup_queue_infrastructure(queue_name)

        await results_queue.consume(callback)
        return results_queue

    async def get_queue_stats(self, queue_name: str) -> QueueStats:
        """Get statistics for a queue using RabbitMQ Management API."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        try:
            # Try RabbitMQ Management API first
            stats = await self._get_queue_stats_via_api(queue_name)
            if stats:
                return stats
        except Exception as e:
            self.logger.debug(f"Management API failed for {queue_name}: {e}")

        try:
            # Fallback: just verify queue exists
            await self.channel.declare_queue(queue_name, passive=True)

            return QueueStats(queue_name=queue_name, stats_source="amqp_fallback")
        except Exception as e:
            self.logger.debug(f"Queue {queue_name} not found: {e}")
            return QueueStats(queue_name=queue_name, stats_source="unavailable")

    async def _get_queue_stats_via_api(self, queue_name: str) -> Optional[QueueStats]:
        """Get queue stats via RabbitMQ Management API."""
        try:
            # Parse RabbitMQ URL to get management API URL
            parsed_url = urlparse(self.config.rabbitmq_url)

            # Default management port is 15672
            management_port = 15672
            management_url = f"http://{parsed_url.hostname}:{management_port}"

            # Extract credentials
            username = parsed_url.username or "guest"
            password = parsed_url.password or "guest"

            # Get vhost (default is /)
            vhost = parsed_url.path.strip("/") or "%2F"  # URL encode /

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{management_url}/api/queues/{vhost}/{queue_name}",
                    auth=(username, password),
                    timeout=5.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    return QueueStats(
                        queue_name=queue_name,
                        message_count=data.get("messages"),
                        message_count_ready=data.get("messages_ready"),
                        message_count_unacknowledged=data.get(
                            "messages_unacknowledged"
                        ),
                        consumer_count=data.get("consumers"),
                        message_bytes=data.get("message_bytes"),
                        message_bytes_ready=data.get("message_bytes_ready"),
                        message_bytes_unacknowledged=data.get(
                            "message_bytes_unacknowledged"
                        ),
                        stats_source="management_api",
                    )

        except Exception as e:
            self.logger.debug(f"Management API error: {e}")

        return None

    async def get_failed_messages(
        self, queue_name: str, limit: int = 100
    ) -> list[dict]:
        """Get messages from the dead letter queue."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        failed_queue_name = f"{queue_name}.failed"
        failed_messages = []

        try:
            failed_queue = await self.channel.declare_queue(
                failed_queue_name, passive=True
            )

            # Consume messages without acking them (peek)
            count = 0

            async def message_handler(message):
                nonlocal count
                if count >= limit:
                    return

                try:
                    job_data = Job.parse_raw(message.body)
                    failed_messages.append(
                        {
                            "job_id": job_data.id,
                            "job_data": job_data.model_dump(),
                            "timestamp": message.timestamp,
                        }
                    )
                    count += 1
                except Exception as e:
                    self.logger.error(f"Failed to parse failed message: {e}")

                # Don't ack the message, just reject it back to queue
                await message.reject(requeue=True)

            await failed_queue.consume(message_handler, no_ack=False)

            # Wait a bit to collect messages, then stop consuming
            await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error getting failed messages: {e}")

        return failed_messages

    async def clear_queue(self, queue_name: str) -> int:
        """Clear all messages from a queue. Returns the number of messages purged."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        try:
            job_queue = await self.channel.declare_queue(queue_name, passive=True)
            purge_result = await job_queue.purge()
            purged_count = purge_result.message_count or 0
            self.logger.info(f"Cleared {purged_count} messages from queue {queue_name}")
            return purged_count
        except Exception as e:
            self.logger.error(f"Failed to clear queue {queue_name}: {e}")
            raise
