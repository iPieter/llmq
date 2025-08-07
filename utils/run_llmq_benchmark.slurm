#!/bin/bash
#SBATCH --job-name=llmq-benchmark
#SBATCH --account=ehpc-aif-2025pg01-128
#SBATCH --qos=ehpc-aif-2025pg01-128
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=01:00:00
#SBATCH --output=llmq-benchmark-%j.out
#SBATCH --error=llmq-benchmark-%j.err
#SBATCH --nodelist=dgx1

# Get the allocated node
NODE=$(hostname)
echo "Job allocated to node: $NODE"
echo "JobID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Project directory
PROJECT_DIR="/valhalla/projects/ehpc-aif-2025pg01-128/llm-training-scripts"
cd "$PROJECT_DIR"

echo "Working directory: $(pwd)"

# Activate environment
source "$PROJECT_DIR/.env/bin/activate"
echo "Environment activated"

# Export LLMQ configuration
export VLLM_MAX_NUM_SEQS=750
export VLLM_QUEUE_PREFETCH=1250
export RABBITMQ_URL=amqp://guest:guest@$(hostname):5672/
echo "LLMQ config: MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS, QUEUE_PREFETCH=$VLLM_QUEUE_PREFETCH"

# Step 1: Start RabbitMQ broker
echo "=========================================="
echo "Step 1: Starting RabbitMQ broker..."
echo "=========================================="
bash llmq/utils/start_singularity_broker.sh
echo "Waiting 20 seconds for RabbitMQ to initialize..."
sleep 20
echo "RabbitMQ should be ready"

# Step 2: Start workers on each GPU
echo "=========================================="
echo "Step 2: Starting LLMQ workers on 8 GPUs..."
echo "=========================================="

# Start workers in background
for gpu_id in 0 1 2 3 4 5 6 7; do
    echo "Starting worker on GPU $gpu_id..."
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        export VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS
        export VLLM_QUEUE_PREFETCH=$VLLM_QUEUE_PREFETCH
        export RABBITMQ_URL=amqp://guest:guest@$(hostname):5672/
        # Isolate each worker process completely
        export RAY_DISABLE_IMPORT_WARNING=1
        export RAY_ADDRESS=  # Disable Ray clustering
        export RAY_START_REDIS=0
        export NCCL_P2P_DISABLE=1
        export NCCL_TIMEOUT=600
        cd "$PROJECT_DIR"
        source "$PROJECT_DIR/.env/bin/activate"
        llmq worker run Unbabel/Tower-Plus-9B translation-queue --tensor-parallel-size 1
    ) > worker-gpu${gpu_id}-${SLURM_JOB_ID}.out 2> worker-gpu${gpu_id}-${SLURM_JOB_ID}.err &
    
    echo "Worker $gpu_id started"
    sleep 2
done

echo "All workers started"

# Wait for workers to initialize
echo "Waiting 45 seconds for all workers to initialize..."
sleep 45

# Step 3: Submit benchmark job
echo "=========================================="
echo "Step 3: Starting benchmark submission..."
echo "=========================================="

# Record start time
BENCHMARK_START=$(date +%s)

# Submit the benchmark job

llmq submit translation-queue Aleph-Alpha/Aleph-Alpha-GermanWeb \
    --subset synthetic \
    --map 'messages=[{"role": "user", "content": "Translate the following German source text to Dutch:\nGerman: {text}\nDutch: "}]' \
    --max-samples 15000 > results.jsonl

# Record completion
BENCHMARK_END=$(date +%s)
BENCHMARK_DURATION=$((BENCHMARK_END - BENCHMARK_START))

echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
echo "Duration: ${BENCHMARK_DURATION} seconds"
echo "Results written to: results.jsonl"

# Check results file
if [ -f "results.jsonl" ]; then
    RESULT_COUNT=$(wc -l < results.jsonl)
    echo "Results file contains $RESULT_COUNT lines"
    
    # Calculate throughput
    if [ $BENCHMARK_DURATION -gt 0 ]; then
        THROUGHPUT=$(echo "scale=2; $RESULT_COUNT / $BENCHMARK_DURATION" | bc)
        echo "Throughput: $THROUGHPUT jobs/sec"
    fi
else
    echo "WARNING: results.jsonl not found!"
fi

# Step 4: Cleanup
echo "=========================================="
echo "Step 4: SLURM will handle worker cleanup automatically"
echo "=========================================="

# Step 5: Final status
echo "=========================================="
echo "Final Status"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $NODE"
echo "Benchmark duration: ${BENCHMARK_DURATION} seconds"
echo "End time: $(date)"

# Optional: Show some sample results
if [ -f "results.jsonl" ] && [ $RESULT_COUNT -gt 0 ]; then
    echo ""
    echo "Sample results (first 3 lines):"
    head -3 results.jsonl
    echo ""
    echo "Sample results (last 3 lines):"
    tail -3 results.jsonl
fi

echo "LLMQ benchmark job completed successfully!"