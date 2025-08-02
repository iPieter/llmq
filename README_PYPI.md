# llmq

High-Performance Inference Queueing

## Quick Start

```bash
# Install
pip install llmq

# Start RabbitMQ
docker run -d --name rabbitmq -p 5672:5672 rabbitmq:3

# Submit jobs
echo '{"id": "1", "prompt": "Say hello", "name": "world"}' > jobs.jsonl
llmq submit my-queue jobs.jsonl > output.jsonl

# Start worker on a GPU node (in another terminal)
llmq worker dummy my-queue
```

## Features

- **High-performance**: GPU-accelerated inference with vLLM
- **Scalable**: RabbitMQ-based job distribution
- **Simple**: Unix-friendly CLI with piped output
- **Async**: Non-blocking job processing
- **Flexible**: Support for multiple worker types

## Worker Types

- `llmq worker run <model> <queue>` - vLLM worker for real inference
- `llmq worker dummy <queue>` - Testing worker

## Configuration

Set via environment variables:

- `RABBITMQ_URL` - RabbitMQ connection
- `VLLM_GPU_MEMORY_UTILIZATION` - GPU memory usage (0.0-1.0)
- `VLLM_QUEUE_PREFETCH` - Concurrent jobs per worker

## Documentation

See the [GitHub repository](https://github.com/ipieter/llmq) for full documentation.

## License

MIT