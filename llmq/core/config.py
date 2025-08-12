import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Config(BaseModel):
    rabbitmq_url: str = Field(
        default_factory=lambda: os.getenv(
            "RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"
        ),
        description="RabbitMQ connection URL",
    )

    vllm_queue_prefetch: int = Field(
        default_factory=lambda: int(os.getenv("VLLM_QUEUE_PREFETCH", "100")),
        description="Number of messages to prefetch per worker",
    )

    vllm_gpu_memory_utilization: float = Field(
        default_factory=lambda: float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
        description="GPU memory utilization ratio for vLLM",
    )

    vllm_max_num_seqs: Optional[int] = Field(
        default_factory=lambda: (
            int(value) if (value := os.getenv("VLLM_MAX_NUM_SEQS")) else None
        ),
        description="Maximum number of sequences in a batch",
    )

    vllm_max_model_len: Optional[int] = Field(
        default_factory=lambda: (
            int(value) if (value := os.getenv("VLLM_MAX_MODEL_LEN")) else None
        ),
        description="Maximum model length (context window) for vLLM",
    )

    vllm_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("VLLM_MAX_TOKENS", "8192")),
        description="Maximum tokens to generate per request",
    )

    job_ttl_minutes: int = Field(
        default_factory=lambda: int(os.getenv("LLMQ_JOB_TTL_MINUTES", "30")),
        description="Job TTL in minutes",
    )

    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("LLMQ_CHUNK_SIZE", "10000")),
        description="Number of jobs to read from JSONL at once",
    )

    log_level: str = Field(
        default_factory=lambda: os.getenv("LLMQ_LOG_LEVEL", "INFO"),
        description="Logging level",
    )

    @property
    def job_ttl_ms(self) -> int:
        """Job TTL in milliseconds."""
        return self.job_ttl_minutes * 60 * 1000


def get_config() -> Config:
    """Get configuration instance."""
    return Config()
