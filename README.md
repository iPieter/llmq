# llmq

[![PyPI version](https://badge.fury.io/py/llmq.svg)](https://pypi.org/project/llmq/)
[![CI](https://github.com/ipieter/llmq/workflows/CI/badge.svg)](https://github.com/ipieter/llmq/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Performance Inference Queueing** - Process millions of LLM inference jobs with GPU acceleration using vLLM workers and RabbitMQ. Ideal for synthetic data generation, translation pipelines, and batch inference workloads.

> **Note**: API may change until v1.0 as I'm actively developing new features.

## Features

- **High-Performance**: GPU-accelerated inference with vLLM batching
- **Scalable**: RabbitMQ-based distributed queuing, so never let your GPUs idle  
- **Simple**: Unix-friendly CLI with piped input/output
- **Flexible**: Supports many standard LLM operations for synthetic data generation. You can combine different models and process Huggingface datasets directly

**Not for real-time use**: llmq is designed for (laaarge) batch processing, not chat applications or real-time inference. It doesn't support token streaming or optimized time-to-first-token (TTFT).

## Quick Start

### Installation

```bash
pip install llmq
```

### Start RabbitMQ

```bash
docker run -d --name rabbitmq \
  -p 5672:5672 -p 15672:15672 \
  -e RABBITMQ_DEFAULT_USER=llmq \
  -e RABBITMQ_DEFAULT_PASS=llmq123 \
  rabbitmq:3-management
```

### Run Your First Job

```bash
# Start a worker
llmq worker run Unbabel/Tower-Plus-9B translation-queue

# Submit jobs (in another terminal)
echo '{"id": "hello", "prompt": "Translate to Spanish: Hello world"}' | llmq submit translation-queue -

# Results stream back immediately
{"id": "hello", "result": "Hola mundo", "worker_id": "worker-gpu0", "duration_ms": 45.2}
```

## Use Cases

### Translation Pipeline

Process translation jobs with specialized multilingual models:

```bash
# Start translation worker
llmq worker run Unbabel/Tower-Plus-9B translation-queue

# Example jobs file (jobs.jsonl)
{"id": "job1", "messages": [{"role": "user", "content": "Translate to Spanish: {text}"}], "text": "Hello world"}
{"id": "job2", "messages": [{"role": "user", "content": "Translate to French: {text}"}], "text": "Good morning"}

# Process jobs
llmq submit translation-queue jobs.jsonl > results.jsonl
```

### Data Cleaning at Scale

Clean and process large datasets with custom prompts:

```bash
# Start worker for data cleaning
llmq worker run meta-llama/Llama-3.2-3B-Instruct cleaning-queue

# Submit HuggingFace dataset directly
llmq submit cleaning-queue HuggingFaceFW/fineweb \
  --map 'messages=[{"role": "user", "content": "Clean this text: {text}"}]' \
  --max-samples 10000 > cleaned_data.jsonl
```

### RL Training Rollouts

Currently requires manual orchestration - you need to manually switch between queues and manage workers for different training phases. For example, you'd start policy workers, submit rollout jobs, tear down those workers, then start reward model workers to score the rollouts.

Future versions will add automatic model switching and queue coordination to streamline complex RL workflows with policy models, reward models, and value functions.

## Worker Types

**Production Workers:**
- `llmq worker run <model-name> <queue-name>` - GPU-accelerated inference with vLLM

**Development & Testing:**
- `llmq worker dummy <queue-name>` - Simple echo worker for testing (no GPU required)

All workers support the same configuration options and can be scaled horizontally by running multiple instances.

## Core Commands

### Job Management

```bash
# Submit jobs from file or stdin
llmq submit <queue-name> <jobs.jsonl>
llmq submit <queue-name> -  # from stdin

# Monitor progress
llmq status <queue-name>
```

### Worker Management

```bash
# Start GPU-accelerated worker
llmq worker run <model-name> <queue-name>

# Start test worker (no GPU required)
llmq worker dummy <queue-name>

# Start filter worker (job filtering)
llmq worker filter <queue-name> <field> <value>

# Multiple workers: run command multiple times
```

### Monitoring

```bash
# Check connection and queues
llmq status
✅ Connected to RabbitMQ
URL: amqp://llmq:llmq123@localhost:5672/

# View queue statistics
llmq status <queue-name>

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                         ┃ Value               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ Queue Name                     │ translation-queue   │
│ Total Messages                 │ 0                   │
│ ├─ Ready (awaiting processing) │ 0                   │
│ └─ Unacknowledged (processing) │ 0                   │
│ Total Bytes                    │ 0 bytes (0.0 MB)    │
│ ├─ Ready Bytes                 │ 0 bytes             │
│ └─ Unacked Bytes               │ 0 bytes             │
│ Active Consumers               │ 0                   │
│ Timestamp                      │ 2025-08-08 11:36:31 │
└────────────────────────────────┴─────────────────────┘
```

## Configuration

Configure via environment variables or `.env` file:

```bash
# Connection
RABBITMQ_URL=amqp://llmq:llmq123@localhost:5672/

# Performance tuning
VLLM_QUEUE_PREFETCH=100              # Messages per worker
VLLM_GPU_MEMORY_UTILIZATION=0.9     # GPU memory usage
VLLM_MAX_NUM_SEQS=256               # Batch size

# Job processing
LLMQ_CHUNK_SIZE=10000               # Bulk submission size
```

## Job Formats

### Modern Chat Format (Recommended)

```json
{
  "id": "job-1",
  "messages": [
    {"role": "user", "content": "Translate to {language}: {text}"}
  ],
  "text": "Hello world",
  "language": "Spanish"
}
```

### Traditional Prompt Format

```json
{
  "id": "job-1", 
  "prompt": "Translate to {language}: {text}",
  "text": "Hello world",
  "language": "Spanish"
}
```

Both formats support template substitution with `{variable}` syntax.

## Architecture

llmq creates two components per queue:
- **Job Queue**: `<queue-name>` - Where jobs are submitted
- **Results Exchange**: `<queue-name>.results` - Streams results back

Workers use vLLM for GPU acceleration and RabbitMQ for reliable job distribution. Results stream back in real-time as jobs complete.

## Performance Tips

- **GPU Memory**: Adjust `VLLM_GPU_MEMORY_UTILIZATION` (default: 0.9)
- **Concurrency**: Tune `VLLM_QUEUE_PREFETCH` based on model size
- **Batch Size**: Set `VLLM_MAX_NUM_SEQS` for optimal throughput
- **Multiple GPUs**: vLLM automatically uses all visible GPUs. You can also start multiple workers yourself for data parallel processing, which [is actually recommended for larger deployements](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html#external-load-balancing).

## Testing

```bash
# Install with test dependencies
pip install llmq[test]

# Run unit tests (no external dependencies)
pytest -m unit

# Run integration tests (requires RabbitMQ)
pytest -m integration
```

## Links

- **PyPI**: https://pypi.org/project/llmq/
- **Issues**: https://github.com/ipieter/llmq/issues
- **Docker Compose Setup**: [docker-compose.yml](#docker-compose-setup)
- **HPC/SLURM/Singularity Setup**: [Singularity Setup](#singularity-setup)

---

## Advanced Setup

### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  rabbitmq:
    image: rabbitmq:3-management
    container_name: llmq-rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: llmq
      RABBITMQ_DEFAULT_PASS: llmq123
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: unless-stopped

volumes:
  rabbitmq_data:
```

Run with: `docker-compose up -d`

### Singularity Setup

For HPC clusters:

```bash
# Use provided utility
./utils/start_singularity_broker.sh

# Set connection URL  
export RABBITMQ_URL=amqp://guest:guest@$(hostname):5672/

# Test connection
llmq status
```

### Performance Tuning

#### GPU Memory Management
```bash
# Reduce for large models
export VLLM_GPU_MEMORY_UTILIZATION=0.7

# Increase for small models
export VLLM_GPU_MEMORY_UTILIZATION=0.95
```

#### Concurrency Tuning
```bash
# Higher throughput, more memory usage
export VLLM_QUEUE_PREFETCH=200

# Lower memory usage, potentially lower throughput
export VLLM_QUEUE_PREFETCH=50
```

#### Batch Processing
```bash
# Larger batches for better GPU utilization
export VLLM_MAX_NUM_SEQS=512

# Smaller batches for lower latency
export VLLM_MAX_NUM_SEQS=64
```

### Multi-GPU Setup

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 llmq worker run model-name queue-name

# vLLM automatically distributes across all visible GPUs
```

### Troubleshooting

#### Connection Issues
```bash
# Check RabbitMQ status
docker ps
docker logs rabbitmq

# Test management API
curl -u llmq:llmq123 http://localhost:15672/api/overview
```

#### Worker Issues
```bash
# Check GPU memory
nvidia-smi

# Reduce GPU utilization if needed
export VLLM_GPU_MEMORY_UTILIZATION=0.7

# View structured logs
llmq worker run model queue 2>&1 | jq .
```

#### Queue Issues
```bash
# Check queue health
llmq health queue-name

# View failed jobs
llmq errors queue-name --limit 10

# Access RabbitMQ management UI
open http://localhost:15672
```