# llmq: High-Performance vLLM Job Queue Package

A Python package that efficiently processes millions of LLM inference jobs using vLLM workers and RabbitMQ. Maximize GPU utilization through intelligent batching while providing simple CLI tools for job submission and monitoring.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd llmq

# Install in development mode
pip install -e .
```

### 2. Start RabbitMQ

**Using Docker (Recommended):**

```bash
# Start RabbitMQ with management UI
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  -e RABBITMQ_DEFAULT_USER=llmq \
  -e RABBITMQ_DEFAULT_PASS=llmq123 \
  rabbitmq:3-management

# Or using docker-compose (see docker-compose.yml below)
docker-compose up -d
```

**Docker Compose Setup (`docker-compose.yml`):**

```yaml
version: '3.8'
services:
  rabbitmq:
    image: rabbitmq:3-management
    container_name: llmq-rabbitmq
    ports:
      - "5672:5672"    # AMQP port
      - "15672:15672"  # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: llmq
      RABBITMQ_DEFAULT_PASS: llmq123
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: unless-stopped

volumes:
  rabbitmq_data:
```

**HPC Clusters with Singularity:**

```bash
# Build Singularity container
singularity build rabbitmq.sif docker://rabbitmq:3-management

# Run RabbitMQ in background
singularity instance start --bind /tmp:/tmp rabbitmq.sif rabbitmq-instance

# Set connection URL for cluster node
export RABBITMQ_URL=amqp://guest:guest@$(hostname):5672/
```

### 3. Configure Environment

Create a `.env` file in your project root:

```bash
# RabbitMQ Configuration
RABBITMQ_URL=amqp://llmq:llmq123@localhost:5672/

# Worker Configuration  
VLLM_QUEUE_PREFETCH=100
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_MAX_NUM_SEQS=256

# Job Configuration
LLMQ_JOB_TTL_MINUTES=30
LLMQ_CHUNK_SIZE=10000

# Logging
LLMQ_LOG_LEVEL=INFO
```

### 4. Test Connection

```bash
# Check if RabbitMQ is accessible
llmq status

# Should show: ✅ Connected to RabbitMQ
```

### 5. Run Your First Job

**Create a jobs file (`jobs.jsonl`):**

```json
{"id": "job-001", "prompt": "Translate '{text}' to {language}", "text": "Hello world", "language": "Spanish"}
{"id": "job-002", "prompt": "Translate '{text}' to {language}", "text": "Good morning", "language": "French"}
{"id": "job-003", "prompt": "Summarize: {text}", "text": "AI is transforming industries globally."}
```

**Start a worker:**

```bash
# Dummy worker for testing (no vLLM/GPU required)
llmq worker dummy translation-queue

# Real vLLM worker (automatically uses all visible GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 llmq worker run microsoft/DialoGPT-medium translation-queue

# Filter worker for simple job processing
llmq worker filter translation-queue language spanish

# Multiple workers: open multiple terminals and run the same command
```

**Submit jobs:**

```bash
# Submit jobs and stream results
llmq submit translation-queue jobs.jsonl > results.jsonl

# Monitor progress in another terminal
llmq status translation-queue
```

## 📋 CLI Commands Reference

### Job Submission

```bash
# Submit jobs from JSONL file (streams results to stdout, progress to stderr)
llmq submit <queue-name> <jobs.jsonl> > results.jsonl

# Example job format:
{"id": "job-123", "prompt": "Translate {text} to {language}", "text": "Hello", "language": "Spanish"}

# Example result format:
{"id": "job-123", "prompt": "Translate Hello to Spanish", "result": "Hola", "worker_id": "worker-gpu0", "duration_ms": 23.5, "timestamp": "2024-01-01T00:00:00Z"}
```

### Worker Management

```bash
# Dummy workers (for testing, no GPU/vLLM required)
llmq worker dummy <queue-name>

# Real vLLM workers (automatically uses all visible GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 llmq worker run <model-name> <queue-name>

# Filter workers (for simple job processing)
llmq worker filter <queue-name> <field> <value>

# Examples:
llmq worker dummy test-queue
llmq worker run microsoft/DialoGPT-medium chat-queue
llmq worker run meta-llama/Llama-2-7b-chat-hf translation-queue
llmq worker filter content-queue category important

# Multiple workers: just run the same command in multiple terminals
```

### Monitoring & Health

```bash
# Check RabbitMQ connection
llmq status

# Show queue statistics
llmq status <queue-name>

# Basic health check
llmq health <queue-name>

# Show recent errors
llmq errors <queue-name>
llmq errors <queue-name> --limit 50
```

## ⚙️ Configuration

Configuration is loaded in the following order (later values override earlier ones):

1. **Default values**
2. **`.env` file**
3. **Environment variables**
4. **CLI arguments**

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection URL |
| `VLLM_QUEUE_PREFETCH` | `100` | Messages per worker to prefetch |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory utilization ratio |
| `VLLM_MAX_NUM_SEQS` | `None` | Max sequences in vLLM batch |
| `LLMQ_JOB_TTL_MINUTES` | `30` | Job time-to-live in minutes |
| `LLMQ_CHUNK_SIZE` | `10000` | Jobs to read from JSONL at once |
| `LLMQ_LOG_LEVEL` | `INFO` | Logging level |

### Docker Environment Variables

When running RabbitMQ in Docker, use these variables:

```bash
# Docker run
docker run -d --name rabbitmq \
  -p 5672:5672 -p 15672:15672 \
  -e RABBITMQ_DEFAULT_USER=llmq \
  -e RABBITMQ_DEFAULT_PASS=llmq123 \
  rabbitmq:3-management

# Then set your connection URL
export RABBITMQ_URL=amqp://llmq:llmq123@localhost:5672/
```

## 🏗️ Architecture

### Queue Infrastructure

**For each queue name, llmq creates:**

- **Job Queue**: `<queue-name>` - Where jobs are submitted
- **Results Exchange**: `<queue-name>.results` - Fanout exchange for results
- **Dead Letter Queue**: `<queue-name>.failed` - Failed jobs after retries

### Message Flow

```
Jobs File → Submit CLI → RabbitMQ → vLLM Workers → Results Exchange → Submit CLI → stdout
                ↓             ↓                         ↓
           Progress (stderr)  Job Queue          Dead Letter Queue
```

### Worker Design

- **Async Processing**: Non-blocking job consumption
- **Dynamic Batching**: Leverages vLLM's internal batching
- **Error Handling**: Failed jobs go to dead letter queue
- **Graceful Shutdown**: Handle SIGINT/SIGTERM properly
- **GPU Isolation**: Each worker uses specific GPU via `CUDA_VISIBLE_DEVICES`

## 🔧 Performance Tuning

### GPU Memory

```bash
# Adjust GPU memory utilization (0.0-1.0)
export VLLM_GPU_MEMORY_UTILIZATION=0.85

# Set maximum batch size
export VLLM_MAX_NUM_SEQS=128
```

### RabbitMQ Prefetch

```bash
# More messages per worker = higher throughput but more memory
export VLLM_QUEUE_PREFETCH=200

# Fewer messages = lower memory but potentially lower throughput  
export VLLM_QUEUE_PREFETCH=50
```

### Job Processing

```bash
# Larger chunks = faster submission but more memory
export LLMQ_CHUNK_SIZE=50000

# Smaller chunks = lower memory but slower submission
export LLMQ_CHUNK_SIZE=1000
```

## 📊 Example Workflows

### Quick Testing with Dummy Workers

```bash
# Terminal 1: Start multiple dummy workers
llmq worker dummy test-queue &
llmq worker dummy test-queue &
llmq worker dummy test-queue &

# Terminal 2: Submit test jobs
llmq submit test-queue example_jobs.jsonl > results.jsonl

# Terminal 3: Monitor in real-time
llmq status test-queue
```

### Large-Scale Translation

```bash
# Terminal 1: Start vLLM worker with all GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 llmq worker run Helsinki-NLP/opus-mt-en-es translation

# Terminal 2: Submit 1M translation jobs
llmq submit translation million_translations.jsonl > results.jsonl 2> progress.log

# Terminal 3: Monitor in real-time
watch -n 1 'llmq status translation'
```

### Multi-Worker Multi-Queue Processing

```bash
# Terminal 1: Dummy workers for testing
llmq worker dummy test-queue

# Terminal 2: Real vLLM worker for production
CUDA_VISIBLE_DEVICES=0,1 llmq worker run gpt2 chat-queue

# Terminal 3: Filter worker for preprocessing
llmq worker filter content-queue priority high

# Terminal 4: Submit to different queues
llmq submit test-queue test_jobs.jsonl > test_results.jsonl &
llmq submit chat-queue chat_jobs.jsonl > chat_results.jsonl &
llmq submit content-queue content_jobs.jsonl > content_results.jsonl &
```

## 🚨 Troubleshooting

### Connection Issues

```bash
# Check connection
llmq status

# Common fixes:
docker ps  # Ensure RabbitMQ is running
docker logs rabbitmq  # Check RabbitMQ logs

# Test with curl
curl -u llmq:llmq123 http://localhost:15672/api/overview
```

### Worker Issues

```bash
# Check worker logs (structured JSON)
llmq worker run model-name queue-name 2>&1 | jq .

# GPU memory issues
nvidia-smi  # Check GPU memory usage
export VLLM_GPU_MEMORY_UTILIZATION=0.7  # Reduce utilization
```

### Queue Issues

```bash
# Check queue health
llmq health queue-name

# View failed jobs
llmq errors queue-name --limit 10

# Purge failed jobs (RabbitMQ management UI)
# http://localhost:15672 → Queues → queue-name.failed → Purge
```

### Performance Issues

```bash
# Monitor processing rate
llmq status queue-name

# Check GPU utilization
nvidia-smi -l 1

# Adjust prefetch for your workload
export VLLM_QUEUE_PREFETCH=50   # Lower for large models
export VLLM_QUEUE_PREFETCH=200  # Higher for small models
```

## 🔒 Security Notes

- **Never commit credentials** to version control
- **Use environment variables** for sensitive configuration
- **Secure RabbitMQ** in production with proper authentication
- **Monitor resource usage** to prevent DoS from large jobs

## 🧪 Testing

### Unit Testing

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage report
pytest --cov=llmq --cov-report=html

# Run only unit tests (fast, no external dependencies)
pytest -m unit

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v

# Run integration tests (requires RabbitMQ)
pytest -m integration
```

### Test Categories

- **Unit Tests** (`pytest -m unit`): Fast tests with mocked dependencies
- **Integration Tests** (`pytest -m integration`): Tests with real RabbitMQ
- **Slow Tests** (`pytest -m "not slow"`): Exclude long-running tests

### Test Mode Setup

```bash
# Start test RabbitMQ
docker run -d --name test-rabbitmq -p 5673:5672 rabbitmq:3

# Use test environment
export RABBITMQ_URL=amqp://guest:guest@localhost:5673/

# Run basic tests
llmq status  # Should connect to test instance
```

### Performance Benchmarking

```bash
# Create test jobs
python -c "
import json
jobs = [{'id': f'test-{i}', 'prompt': 'Say hello {i} times', 'i': i} for i in range(1000)]
with open('test_jobs.jsonl', 'w') as f:
    for job in jobs: f.write(json.dumps(job) + '\n')
"

# Benchmark throughput
time llmq submit test-queue test_jobs.jsonl > /dev/null
```

### Development Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests with file watching
pytest-watch

# Check code formatting
black llmq tests
ruff check llmq tests

# Type checking
mypy llmq
```

## 📚 Advanced Usage

### Custom Sampling Parameters

Modify `llmq/core/worker.py` to customize vLLM sampling:

```python
sampling_params = SamplingParams(
    temperature=0.8,      # Creativity
    max_tokens=512,       # Response length
    top_p=0.9,           # Nucleus sampling
    stop=["\n\n", "END"] # Stop sequences
)
```

### Custom Job Processing

Extend the `Job` model in `llmq/core/models.py`:

```python
class CustomJob(Job):
    temperature: float = 0.7
    max_tokens: int = 100
    
    def get_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
```

---

**🎯 Need Help?**

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check `llmq --help` and command-specific help
- **Performance**: See the Performance Tuning section above
- **RabbitMQ UI**: Access http://localhost:15672 (guest/guest or llmq/llmq123)

The `llmq` package prioritizes **performance** and **reliability** while maintaining a **Unix-philosophy-friendly** interface where components can be easily piped together. Happy processing! 🚀