from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class Job(BaseModel):
    id: str = Field(..., description="Unique job identifier")
    prompt: Optional[str] = Field(None, description="Template prompt with placeholders")
    messages: Optional[List[Dict[str, Any]]] = Field(
        None, description="Chat messages for chat-based models"
    )
    chat_mode: bool = Field(
        default=False, description="Use chat API instead of generate API"
    )
    stop: Optional[List[str]] = Field(
        None, description="Stop sequences for generation. If None, uses EOS token only"
    )

    class Config:
        extra = "allow"

    @validator("messages", always=True)
    def validate_prompt_or_messages(cls, v, values):
        """Ensure either prompt OR messages is provided, not both or neither."""
        prompt = values.get("prompt")

        if prompt is not None and v is not None:
            raise ValueError(
                "Cannot specify both 'prompt' and 'messages'. Use one or the other."
            )

        if prompt is None and v is None:
            raise ValueError("Must specify either 'prompt' or 'messages'.")

        return v

    def get_formatted_prompt(self) -> str:
        """Format the prompt template with job data, excluding id and prompt fields."""
        if self.prompt is None:
            raise ValueError("Cannot format prompt: prompt is None")
        format_data = {
            k: v
            for k, v in self.model_dump().items()
            if k not in ["id", "prompt", "messages", "chat_mode", "stop"]
        }
        return self.prompt.format(**format_data)


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

    class Config:
        extra = "allow"  # Allow extra fields to be passed through


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
