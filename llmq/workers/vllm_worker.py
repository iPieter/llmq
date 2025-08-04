import os
from typing import Optional, List, Dict, Any

from vllm import AsyncLLMEngine, SamplingParams  # type: ignore
from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore

from llmq.core.models import Job
from llmq.workers.base import BaseWorker


class VLLMWorker(BaseWorker):
    """vLLM worker that processes jobs using all visible GPUs."""

    def __init__(
        self, model_name: str, queue_name: str, worker_id: Optional[str] = None
    ):
        self.model_name = model_name
        super().__init__(queue_name, worker_id)
        self.engine: Optional[AsyncLLMEngine] = None

    def _generate_worker_id(self) -> str:
        """Generate worker ID based on visible GPUs."""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        gpu_suffix = cuda_visible.replace(",", "-") if cuda_visible else "auto"
        return f"vllm-{gpu_suffix}"

    async def _initialize_processor(self) -> None:
        """Initialize vLLM engine with all visible GPUs."""
        self.logger.info(f"Initializing vLLM engine for model {self.model_name}")

        # Count visible GPUs
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            gpu_count = len(
                [x.strip() for x in cuda_visible.split(",") if x.strip().isdigit()]
            )
        else:
            # Try to detect available GPUs
            try:
                import torch

                gpu_count = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 1
                )
            except ImportError:
                gpu_count = 1

        self.logger.info(f"Using {gpu_count} GPU(s): {cuda_visible or 'auto-detected'}")

        # Configure vLLM engine args
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            tensor_parallel_size=gpu_count,  # Use all visible GPUs
            disable_log_stats=True,
        )

        if self.config.vllm_max_num_seqs:
            engine_args.max_num_seqs = self.config.vllm_max_num_seqs

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.logger.info("vLLM engine initialized successfully")

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts)

    async def _process_job(self, job: Job) -> str:
        """Process job using vLLM engine."""
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7, max_tokens=1024, stop=["\n\n"]
        )

        if self.engine is None:
            raise RuntimeError("vLLM engine not initialized")

        results = []

        # Format prompt based on chat mode
        if job.chat_mode or job.messages:
            if not job.messages:
                raise ValueError("Chat mode enabled but no messages provided")

            # For chat mode, convert messages to a string prompt
            messages = job.get_formatted_messages()
            formatted_prompt = self._messages_to_prompt(messages)
        else:
            # Use traditional prompt formatting
            formatted_prompt = job.get_formatted_prompt()

        # Use generate method (AsyncLLMEngine doesn't have chat method)
        async for output in self.engine.generate(
            formatted_prompt, sampling_params, request_id=job.id
        ):
            results.append(output)

        if not results:
            raise ValueError("No results generated")

        # Extract generated text
        generated_text = results[0].outputs[0].text if results[0].outputs else ""
        return generated_text

    async def _cleanup_processor(self) -> None:
        """Clean up vLLM engine resources."""
        # Note: vLLM AsyncLLMEngine doesn't have an explicit cleanup method
        # The engine will be garbage collected
        self.engine = None
