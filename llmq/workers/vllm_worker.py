import os
from typing import Optional

from vllm import AsyncLLMEngine, SamplingParams  # type: ignore
from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore

from llmq.core.models import Job
from llmq.workers.base import BaseWorker


class VLLMWorker(BaseWorker):
    """vLLM worker that processes jobs using all visible GPUs."""

    def __init__(
        self,
        model_name: str,
        queue_name: str,
        worker_id: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        data_parallel_size: Optional[int] = None,
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        super().__init__(queue_name, worker_id)
        self.engine: Optional[AsyncLLMEngine] = None

    def _generate_worker_id(self) -> str:
        """Generate worker ID based on parallel configuration."""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        gpu_suffix = cuda_visible.replace(",", "-") if cuda_visible else "auto"

        # Include parallel configuration in worker ID
        if self.tensor_parallel_size or self.data_parallel_size:
            tp = self.tensor_parallel_size or 1
            dp = self.data_parallel_size or 1
            return f"vllm-{gpu_suffix}-tp{tp}-dp{dp}"
        else:
            return f"vllm-{gpu_suffix}"

    async def _initialize_processor(self) -> None:
        """Initialize vLLM engine with all visible GPUs."""
        self.logger.info("_initialize_processor() called")
        try:
            self.logger.info(f"Initializing vLLM engine for model {self.model_name}")

            # Determine parallelism configuration
            tensor_parallel_size = self.tensor_parallel_size
            data_parallel_size = self.data_parallel_size

            if not tensor_parallel_size and not data_parallel_size:
                # Auto-detect: use all visible GPUs for tensor parallelism
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                if cuda_visible:
                    gpu_count = len(
                        [
                            x.strip()
                            for x in cuda_visible.split(",")
                            if x.strip().isdigit()
                        ]
                    )
                else:
                    # Try to detect available GPUs
                    try:
                        import torch

                        gpu_count = (
                            torch.cuda.device_count()
                            if torch.cuda.is_available()
                            else 1
                        )
                    except ImportError:
                        gpu_count = 1

                tensor_parallel_size = gpu_count
                self.logger.info(
                    f"Auto-detected configuration: tensor_parallel_size={tensor_parallel_size}"
                )
            else:
                # Use explicit configuration
                if not tensor_parallel_size:
                    tensor_parallel_size = 1
                if not data_parallel_size:
                    data_parallel_size = 1

                total_gpus = tensor_parallel_size * data_parallel_size
                self.logger.info(
                    f"Explicit configuration: tensor_parallel_size={tensor_parallel_size}, "
                    f"data_parallel_size={data_parallel_size}, total_gpus={total_gpus}"
                )

            # Configure vLLM engine args
            self.logger.info("Creating AsyncEngineArgs...")
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                disable_log_stats=True,
            )

            # Add data parallel size if specified
            if data_parallel_size and data_parallel_size > 1:
                engine_args.data_parallel_size = data_parallel_size

            if self.config.vllm_max_num_seqs is not None:
                engine_args.max_num_seqs = self.config.vllm_max_num_seqs

            if self.config.vllm_max_model_len is not None:
                engine_args.max_model_len = self.config.vllm_max_model_len

            self.logger.info("Creating AsyncLLMEngine...")
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.logger.info(
                f"vLLM engine initialized successfully: {self.engine is not None}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to initialize vLLM engine: {type(e).__name__}: {e}"
            )
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _process_job(self, job: Job) -> str:
        """Process job using vLLM engine."""
        if self.engine is None:
            self.logger.error("vLLM engine is None - initialization must have failed")
            self.logger.error(
                f"Worker initialized: {hasattr(self, 'broker') and self.broker is not None}"
            )
            raise RuntimeError("vLLM engine not initialized")

        # Get tokenizer once for both EOS token and chat template needs
        tokenizer = await self.engine.get_tokenizer()

        # Determine stop sequences
        stop_sequences = []

        if job.stop is not None:
            # User provided custom stop sequences - use exactly what they specified
            stop_sequences.extend(job.stop)
        else:
            # Default behavior: use EOS token only
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token:
                stop_sequences.append(eos_token)

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=self.config.vllm_max_tokens,
            stop=stop_sequences if stop_sequences else None,
        )

        results = []

        # Prepare prompt based on chat mode
        if job.chat_mode or job.messages:
            if not job.messages:
                raise ValueError("Chat mode enabled but no messages provided")

            # Use tokenizer to format chat template
            prompt = tokenizer.apply_chat_template(  # type: ignore
                conversation=job.messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Use traditional string prompt
            prompt = job.get_formatted_prompt()

        # Use generate method
        async for output in self.engine.generate(
            str(prompt), sampling_params, request_id=job.id
        ):
            results.append(output)

        if not results:
            raise ValueError("No results generated")

        # Extract generated text from final output
        final_output = results[-1]
        generated_text = final_output.outputs[0].text if final_output.outputs else ""

        return generated_text

    async def _cleanup_processor(self) -> None:
        """Clean up vLLM engine resources."""
        # Note: vLLM AsyncLLMEngine doesn't have an explicit cleanup method
        # The engine will be garbage collected
        self.engine = None
