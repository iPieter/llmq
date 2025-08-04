from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Job(BaseModel):
    id: str = Field(..., description="Unique job identifier")
    prompt: str = Field(..., description="Template prompt with placeholders")
    messages: Optional[List[Dict[str, Any]]] = Field(
        None, description="Chat messages for chat-based models"
    )
    chat_mode: bool = Field(
        default=False, description="Use chat API instead of generate API"
    )

    class Config:
        extra = "allow"

    def get_formatted_prompt(self) -> str:
        """Format the prompt template with job data, excluding id and prompt fields."""
        format_data = {
            k: v
            for k, v in self.dict().items()
            if k not in ["id", "prompt", "messages", "chat_mode"]
        }
        return self.prompt.format(**format_data)

    def get_formatted_messages(self) -> List[Dict[str, Any]]:
        """Format chat messages with job data, excluding special fields."""
        if not self.messages:
            return []

        format_data = {
            k: v
            for k, v in self.dict().items()
            if k not in ["id", "prompt", "messages", "chat_mode"]
        }

        formatted_messages = []
        for message in self.messages:
            formatted_message = {}
            for key, value in message.items():
                if isinstance(value, str):
                    formatted_message[key] = value.format(**format_data)
                else:
                    formatted_message[key] = value
            formatted_messages.append(formatted_message)

        return formatted_messages


class Result(BaseModel):
    id: str = Field(..., description="Job ID this result corresponds to")
    prompt: str = Field(
        ..., description="The actual formatted prompt that was processed"
    )
    result: str = Field(..., description="Generated text result from vLLM")
    worker_id: str = Field(..., description="ID of the worker that processed this job")
    duration_ms: float = Field(..., description="Processing duration in milliseconds")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the result was generated"
    )


class QueueStats(BaseModel):
    queue_name: str
    message_count: Optional[int] = None  # Total messages
    message_count_ready: Optional[int] = None  # Ready (unacked) messages
    message_count_unacknowledged: Optional[int] = None  # Unacked messages
    consumer_count: Optional[int] = None
    message_bytes: Optional[int] = None  # Total bytes in queue
    message_bytes_ready: Optional[int] = None  # Bytes in ready messages
    message_bytes_unacknowledged: Optional[int] = None  # Bytes in unacked messages
    processing_rate: Optional[float] = None
    stats_source: str = "unknown"  # "management_api", "amqp_fallback", or "unavailable"


class WorkerHealth(BaseModel):
    worker_id: str
    status: str
    last_seen: datetime
    jobs_processed: int
    avg_duration_ms: Optional[float] = None


class ErrorInfo(BaseModel):
    job_id: str
    error_message: str
    timestamp: datetime
    worker_id: Optional[str] = None
