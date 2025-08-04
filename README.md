# llmq: High-Performance vLLM Job Queue Package

A Python package that efficiently processes millions of LLM inference jobs using vLLM workers and RabbitMQ. Maximize GPU utilization through intelligent batching while providing simple CLI tools for job submission and monitoring.

## 🚀 Quick Start: Translation Example

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
# Use the provided utility script
./utils/start_singularity_broker.sh

# Set connection URL
export RABBITMQ_URL=amqp://guest:guest@$(hostname):5672/

# test the connection
llmq status
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

### 5. Translation Workflow Example

**Use the provided translation jobs (`example_jobs.jsonl`):**

The example jobs file contains multilingual translation tasks using the chat message format for the Unbabel/Tower-Plus-9B model:

```json
{"id": "translate-001", "messages": [{"role": "user", "content": "Translate the following English source text to Portuguese (Portugal):\nEnglish: {source_text}\nPortuguese (Portugal): "}], "source_text": "Hello world!"}
{"id": "translate-002", "messages": [{"role": "user", "content": "Translate the following Spanish source text to French:\nSpanish: {source_text}\nFrench: "}], "source_text": "¿Cómo estás hoy?"}
```

**Start the translation worker:**

```bash
# Start Unbabel Tower-Plus-9B worker (uses all visible GPUs)
llmq worker run Unbabel/Tower-Plus-9B translation-queue
```

**Submit translation jobs:**

```bash
# Submit jobs and stream results to file
llmq submit translation-queue example_jobs.jsonl > results.jsonl

# Monitor progress in another terminal
llmq status translation-queue
```

**View results:**

```bash
# Check the translated results
cat results.jsonl
```

This workflow demonstrates:
- **Chat-based model support** for modern translation models
- **Template substitution** with `{source_text}` variables
- **Real-time result streaming** to `results.jsonl`
- **GPU-accelerated inference** with vLLM batching

## 📋 CLI Commands Reference

### Job Submission

```bash
# Submit jobs from JSONL file (streams results to stdout, progress to stderr)
llmq submit <queue-name> <jobs.jsonl> > results.jsonl

# Example result format:
{"id": "job-123", "prompt": "Translate Hello to Spanish", "result": "Hola", "worker_id": "worker-gpu0", "duration_ms": 23.5, "timestamp": "2024-01-01T00:00:00Z"}
```

### Worker Management

```bash
# Real vLLM workers (automatically uses all visible GPUs)
llmq worker run <model-name> <queue-name>

# Dummy workers (for testing, no GPU/vLLM required)
llmq worker dummy <queue-name>

# Filter workers (for simple job processing)
llmq worker filter <queue-name> <field> <value>

# Multiple workers: run the same command in multiple terminals
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

### Translation with Unbabel Tower-Plus-9B

```bash
# Terminal 1: Start vLLM worker with all GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 llmq worker run Unbabel/Tower-Plus-9B translation-queue

# Terminal 2: Submit translation jobs
llmq submit translation-queue example_jobs.jsonl > results.jsonl 2> progress.log

# Terminal 3: Monitor in real-time
watch -n 1 'llmq status translation-queue'
```

### Dataset Translation with HuggingFace Datasets

```bash
# Submit translation jobs directly from HuggingFace dataset
llmq submit translation-queue HuggingFaceFW/fineweb --map 'messages=[{"role": "user", "content": "Translate the following English source text to Dutch:\nEnglish: {text}\nDutch: "}]' --max-samples 5000 > results.jsonl
```

### Testing with Dummy Workers

```bash
# Terminal 1: Start multiple dummy workers
llmq worker dummy translation-queue &
llmq worker dummy translation-queue &

# Terminal 2: Submit and monitor
llmq submit translation-queue example_jobs.jsonl > results.jsonl
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


## 💬 Job Formats

llmq supports both traditional prompt-based and modern chat message formats:

### Traditional Prompt Format

```json
{"id": "job-1", "prompt": "Translate {text} to {language}", "text": "Hello", "language": "Spanish"}
```

### Chat Message Format (for modern models)

```json
{"id": "translation-1", "messages": [{"role": "user", "content": "Translate the following English source text to Portuguese (Portugal):\nEnglish: {source_text}\nPortuguese (Portugal): "}], "source_text": "Hello world!"}
```

**Features:**
- Template support with `{variable}` substitution
- Multi-turn conversations with system/user/assistant roles
- Automatic format detection
- Backward compatibility


---

**🎯 Need Help?**

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check `llmq --help` and command-specific help
- **Performance**: See the Performance Tuning section above
- **RabbitMQ UI**: Access http://localhost:15672 (guest/guest or llmq/llmq123)

The `llmq` package prioritizes **performance** and **reliability** while maintaining a **Unix-philosophy-friendly** interface where components can be easily piped together. Happy processing! 🚀