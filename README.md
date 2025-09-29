# llmq

[![PyPI version](https://badge.fury.io/py/llmq.svg)](https://pypi.org/project/llmq/)
[![CI](https://github.com/ipieter/llmq/workflows/CI/badge.svg)](https://github.com/ipieter/llmq/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/llmq/month)](https://pepy.tech/projects/llmq)

<img src="https://github.com/iPieter/llmq/raw/main/assets/render1755117250879.gif" alt="LLMQ Demo" width="600">


**A Scheduler for Distributed LLM Inference.** Submit millions of inference jobs to distributed workers, get results back reliably. Built for research teams and production workloads that need control over their inference stack.

<h4 align="center">
    <p>
        <a href="#why-llmq">Why llmq?</a> •
        <a href="#quick-start">Quick Start</a> •
        <a href="#pipelines">Pipelines</a> •
        <a href="#production">Production</a> •
        <a href="#examples">Examples</a>
    </p>
</h4>

> **Note**: API may change until v1.0 as new features are being developed.

## Why llmq?

llmq provides distributed batched inference for self-hosted models. When you need to process large volumes of inference jobs but want full control over your models, infrastructure, and costs.

**Best for:**
- Research teams processing datasets with custom models
- Production workflows requiring specific model versions
- Organizations wanting to avoid per-token pricing
- Multi-stage processing pipelines

**Not suitable for:**
- Real-time chat applications
- Streaming token responses
- Sub-minute latency requirements

llmq uses vLLM for GPU acceleration and RabbitMQ for reliable job distribution. Workers can be scaled horizontally across multiple machines and GPUs.

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
echo '{"id": "hello", "messages": [{"role": "user", "content": "Translate the following German source text to English:\\nGerman: Ich  bin eine Giraffe.\\nEnglish: "}]}' \
    | llmq submit translation-queue -

# Receive results
llmq receive translation-queue > results.jsonl
```

For multi-stage workflows, see the [Pipelines](#pipelines) section.

## How It Works

llmq separates job submission from result collection for better reliability:

1. **Submit jobs** - Upload thousands of inference requests to a RabbitMQ queue
2. **Workers process** - GPU workers pull jobs and generate responses using vLLM
3. **Receive results** - Download results separately with resumable retrieval

This architecture enables resumable downloads, multiple result consumers, and better handling of large batches compared to streaming approaches.

## Pipelines

llmq supports multi-stage pipelines for complex workflows like translation followed by post-processing:

```yaml
# example-pipeline.yaml
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
```

```bash
# Start workers for each stage
llmq worker pipeline example-pipeline.yaml translation
llmq worker pipeline example-pipeline.yaml formatting

# Submit data (pipeline handles prompt templates)
echo '{"id": "job1", "source_text": "Bonjour", "source_lang": "French"}' \
    | llmq submit -p example-pipeline.yaml -

# Receive final results
llmq receive -p example-pipeline.yaml > results.jsonl
```

## Production

### Performance Configuration
```bash
# Optimize for your hardware
export VLLM_GPU_MEMORY_UTILIZATION=0.9     # GPU memory usage
export VLLM_MAX_NUM_SEQS=256               # Batch size
export VLLM_QUEUE_PREFETCH=100             # Messages per worker
```

### Multi-GPU Setup
```bash
# vLLM automatically uses all visible GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 llmq worker run model-name queue-name
```

### Monitoring
```bash
# Check system status
llmq status

# Monitor specific queue
llmq status queue-name
```

## Examples


### SLURM scripts

`llmq` has been used in production to process large-scale translation datasets on HPC clusters. The table below shows the datasets processed, their outputs, and the SLURM scripts used:

| Dataset Created | Model Used | SLURM Script |
|---|---|---|
| 🤗 [fineweb-edu-german-mt](https://huggingface.co/datasets/pdelobelle/fineweb-edu-german-mt) | Tower-Plus-72B | [`run_german_72b_translation.slurm`](utils/run_german_72b_translation.slurm) |
| 🤗 [fineweb-edu-dutch-mt](https://huggingface.co/datasets/pdelobelle/fineweb-edu-dutch-mt) | Tower-Plus-9B | [`run_dutch_9b_translation.slurm`](utils/run_dutch_9b_translation.slurm) |
| 🤗 [nemotron-dutch-mt](https://huggingface.co/datasets/pdelobelle/nemotron-dutch-mt) | Tower-Plus-9B | [`run_dutch_nemotron.slurm`](utils/run_dutch_nemotron.slurm) |


### Huggingface datasets

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

Future versions will add automatic model switching and queue coordination to streamline complex RL workflows.


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
CUDA_VISIBLE_DEVICES="0" llmq worker pipeline example-pipeline.yaml translation    # First stage
CUDA_VISIBLE_DEVICES="1" llmq worker pipeline example-pipeline.yaml formatting     # Second stage

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
