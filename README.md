# llmq

[![PyPI version](https://badge.fury.io/py/llmq.svg)](https://pypi.org/project/llmq/)
[![CI](https://github.com/ipieter/llmq/workflows/CI/badge.svg)](https://github.com/ipieter/llmq/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/llmq/month)](https://pepy.tech/projects/llmq)

<img src="https://github.com/iPieter/llmq/raw/main/assets/render1755117250879.gif" alt="LLMQ Demo" width="600">


**A Scheduler for Batched LLM Inference** - Like OpenAI's Batch API, but for self-hosted open-source models. Submit millions of inference jobs, let workers process them with vLLM-backed inference, and stream results back to a single file. Ideal for synthetic data generation, translation pipelines, and batch inference workloads.

> **Note**: API may change until v1.0 as I'm actively developing new features.

<details>
<summary><strong>📋 Table of Contents</strong></summary>

- [Features](#features)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Start RabbitMQ](#start-rabbitmq)
  - [Run Your First Job](#run-your-first-job)
- [How It Works](#how-it-works)
- [Use Cases](#use-cases)
  - [Multi-Stage Pipelines](#multi-stage-pipelines)
  - [Translation Pipeline](#translation-pipeline)
  - [Data Cleaning at Scale](#data-cleaning-at-scale)
  - [RL Training Rollouts](#rl-training-rollouts)
- [Real-World Usage](#real-world-usage)
- [Worker Types](#worker-types)
- [Core Commands](#core-commands)
  - [Job Management](#job-management)
  - [Worker Management](#worker-management)
  - [Monitoring](#monitoring)
- [Configuration](#configuration)
- [Job Formats](#job-formats)
- [Architecture](#architecture)
- [Performance Tips](#performance-tips)
- [Testing](#testing)
- [Links](#links)
- [Advanced Setup](#advanced-setup)
  - [Docker Compose Setup](#docker-compose-setup)
  - [Singularity Setup](#singularity-setup)
  - [Performance Tuning](#performance-tuning)
  - [Multi-GPU Setup](#multi-gpu-setup)
  - [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

</details>

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

**Option 1: Simple Queue (Traditional)**
```bash
# Start a worker
llmq worker run Unbabel/Tower-Plus-9B translation-queue

# Submit jobs (in another terminal)
echo '{"id": "hello", "messages": [{"role": "user", "content": "Translate the following German source text to English:\\nGerman: Ich  bin eine Giraffe.\\nEnglish: "}]}' \
    | llmq submit translation-queue -

# Receive results (separate command for resumable downloads)
llmq receive translation-queue > results.jsonl
```

**Option 2: Pipeline (Simplified - Recommended)**
```bash
# Use the included example-pipeline.yaml (translation → formatting)
# Start pipeline workers (in separate terminals)
llmq worker pipeline example-pipeline.yaml translation
llmq worker pipeline example-pipeline.yaml formatting

# Submit jobs with clean syntax - just provide the data!
echo '{"id": "hello", "source_text": "Ich bin eine Giraffe", "source_lang": "German"}' \
    | llmq submit -p example-pipeline.yaml -

# Receive results
llmq receive -p example-pipeline.yaml > results.jsonl
```

## How It Works

llmq now provides two modes for handling jobs and results:

### Separate Submit/Receive (Default)
1. **Submit jobs** - Upload thousands of inference requests to a queue
2. **Workers process** - GPU-accelerated workers pull jobs and generate responses  
3. **Receive results** - Download results separately with resumable queue-based retrieval

### Streaming Mode (Backwards Compatible)  
1. **Submit jobs** - Upload inference requests to a queue
2. **Workers process** - GPU-accelerated workers pull jobs and generate responses
3. **Stream results** - Get real-time results as jobs complete, with automatic timeout handling

**Benefits of separate submit/receive:**
- **Resumable downloads** - If your connection drops, you can resume receiving results
- **Multiple consumers** - Different processes can consume from the same results queue
- **Better for large batches** - Submit thousands of jobs, then receive results at your own pace

## Use Cases

### Multi-Stage Pipelines

**NEW**: llmq now supports multi-stage pipelines with a simplified API. Perfect for complex workflows like translation → post-processing → formatting.

```bash
# Your pipeline configuration (example-pipeline.yaml)
name: translation-pipeline
stages:
  - name: translation
    worker: vllm
    config:
      model: "Unbabel/Tower-Plus-9B"
      messages:
        - role: "user"
          content: "Translate the following {source_lang} source text to English:\n{source_lang}: {source_text}\nEnglish: "
  - name: formatting
    worker: vllm
    config:
      model: "google/gemma-2-9b-it"
      messages:
        - role: "user"
          content: "Clean up and format the following translated text with proper markdown formatting. Keep the meaning intact but improve readability:\n\n{translation_result}"

# Submit jobs with simplified syntax
llmq submit -p example-pipeline.yaml jobs.jsonl

# Receive final results
llmq receive -p example-pipeline.yaml > results.jsonl

# Start workers for each stage (in separate terminals)
llmq worker pipeline example-pipeline.yaml translation
llmq worker pipeline example-pipeline.yaml formatting
```

**Benefits of pipelines:**
- **Automatic job routing** between stages
- **Parallel processing** - multiple workers per stage
- **Fault tolerance** - failed jobs don't break the entire pipeline
- **Built-in templates** - define prompts once in the pipeline config

### Translation Pipeline

Process translation jobs with specialized multilingual models:

```bash
# Start translation worker
llmq worker run Unbabel/Tower-Plus-9B translation-queue

# Example jobs file (jobs.jsonl)
{"id": "job1", "messages": [{"role": "user", "content": "Translate to Spanish: {text}"}], "text": "Hello world"}
{"id": "job2", "messages": [{"role": "user", "content": "Translate to French: {text}"}], "text": "Good morning"}

# Submit jobs
llmq submit translation-queue jobs.jsonl

# Receive results separately (resumable) 
llmq receive translation-queue > results.jsonl

# Or combine both (backwards compatible)
llmq submit translation-queue jobs.jsonl --stream > results.jsonl
```

### Data Cleaning at Scale

Clean and process large datasets with custom prompts:

```bash
# Start worker for data cleaning
llmq worker run meta-llama/Llama-3.2-3B-Instruct cleaning-queue

# Submit HuggingFace dataset directly
llmq submit cleaning-queue HuggingFaceFW/fineweb \
  --map 'messages=[{"role": "user", "content": "Clean this text: {text}"}]' \
  --max-samples 10000

# Receive cleaned results
llmq receive cleaning-queue > cleaned_data.jsonl
```

### RL Training Rollouts

Currently requires manual orchestration - you need to manually switch between queues and manage workers for different training phases. For example, you'd start policy workers, submit rollout jobs, tear down those workers, then start reward model workers to score the rollouts.

Future versions will add automatic model switching and queue coordination to streamline complex RL workflows with policy models, reward models, and value functions.

## Real-World Usage

`llmq` has been used in production to process large-scale translation datasets on HPC clusters. The table below shows the datasets processed, their outputs, and the SLURM scripts used:

| Dataset Created | Model Used | SLURM Script |
|---|---|---|
| 🤗 [fineweb-edu-german-mt](https://huggingface.co/datasets/pdelobelle/fineweb-edu-german-mt) | Tower-Plus-72B | [`run_german_72b_translation.slurm`](utils/run_german_72b_translation.slurm) |
| 🤗 [fineweb-edu-dutch-mt](https://huggingface.co/datasets/pdelobelle/fineweb-edu-dutch-mt) | Tower-Plus-9B | [`run_dutch_9b_translation.slurm`](utils/run_dutch_9b_translation.slurm) |
| 🤗 [nemotron-dutch-mt](https://huggingface.co/datasets/pdelobelle/nemotron-dutch-mt) | Tower-Plus-9B | [`run_dutch_nemotron.slurm`](utils/run_dutch_nemotron.slurm) |



## Worker Types

**Production Workers:**
- `llmq worker run <model-name> <queue-name>` - GPU-accelerated inference with vLLM

**Development & Testing:**
- `llmq worker dummy <queue-name>` - Simple echo worker for testing (no GPU required)

All workers support the same configuration options and can be scaled horizontally by running multiple instances.

## Core Commands

### Job Management

**Pipeline Mode (Simplified):**
```bash
# Submit to pipeline
llmq submit -p pipeline.yaml jobs.jsonl

# Receive from pipeline
llmq receive -p pipeline.yaml > results.jsonl

# Stream pipeline results (backwards compatible)
llmq submit -p pipeline.yaml jobs.jsonl --stream > results.jsonl
```

**Single Queue Mode (Traditional):**
```bash
# Submit jobs from file or stdin
llmq submit <queue-name> <jobs.jsonl>
llmq submit <queue-name> -  # from stdin

# Receive results (resumable)
llmq receive <queue-name> > results.jsonl

# Stream results (backwards compatible)
llmq submit <queue-name> <jobs.jsonl> --stream > results.jsonl

# Monitor progress
llmq status <queue-name>
```

### Worker Management

**Pipeline Workers:**
```bash
# Start workers for pipeline stages
llmq worker pipeline pipeline.yaml <stage-name>
llmq worker pipeline example-pipeline.yaml translation    # First stage
llmq worker pipeline example-pipeline.yaml formatting     # Second stage

# Multiple workers per stage: run command multiple times
```

**Single Queue Workers:**
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

**Pipelines** automatically handle prompt templates based on worker configuration, so you only need to provide the data:

```json
{"id": "job-1", "text": "Hello world", "language": "Spanish"}
```

For **single queues**, you need to provide the full prompt structure:

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

All formats support template substitution with `{variable}` syntax. **Pipelines make this simpler** by handling the prompt structure automatically based on your pipeline configuration.

## Architecture

llmq creates two components per queue:
- **Job Queue**: `<queue-name>` - Where jobs are submitted  
- **Results Queue**: `<queue-name>.results` - Where results are stored (durable for resumability)

**Pipeline Architecture:**
- **Stage Queues**: `pipeline.<name>.<stage>` - Individual pipeline stages
- **Final Results Queue**: `pipeline.<name>.results` - Final pipeline output

Workers use vLLM for GPU acceleration and RabbitMQ for reliable job distribution. The queue-based results system enables resumable downloads and better fault tolerance.

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

## Acknowledgments

🇪🇺 Development and testing of this project was supported by computational resources provided by EuroHPC under grant EHPC-AIF-2025PG01-128.
