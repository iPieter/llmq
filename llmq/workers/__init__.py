"""Worker implementations for llmq."""

from .base import BaseWorker
from .dummy_worker import DummyWorker, FilterWorker
from .semhash_worker import SemHashWorker

__all__ = ["BaseWorker", "DummyWorker", "FilterWorker", "SemHashWorker"]

try:
    from .vllm_worker import VLLMWorker  # noqa: F401

    __all__.append("VLLMWorker")
except ImportError:
    pass
