#!/bin/bash
#SBATCH --job-name=german-translation
#SBATCH --account=ehpc-aif-2025pg01-128
#SBATCH --qos=ehpc-aif-2025pg01-128
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=german-translation-%j.out
#SBATCH --error=german-translation-%j.err
#SBATCH --nodelist=dgx2

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
export LLMQ_LOG_LEVEL=WARNING
echo "LLMQ config: MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS, QUEUE_PREFETCH=$VLLM_QUEUE_PREFETCH"

# Step 1: Start RabbitMQ broker
echo "=========================================="
echo "Step 1: Starting RabbitMQ broker..."
echo "=========================================="
bash llmq/utils/start_singularity_broker.sh
echo "Waiting 20 seconds for RabbitMQ to initialize..."
sleep 20
echo "RabbitMQ should be ready"

# Step 2: Start workers with tensor parallelism
echo "=========================================="
echo "Step 2: Starting LLMQ workers with TP=2..."
echo "=========================================="

# Start 4 workers with tensor-parallel-size 2 (using GPU pairs)
for worker_id in 0 1 2 3; do
    gpu_start=$((worker_id * 2))
    gpu_end=$((gpu_start + 1))
    echo "Starting worker $worker_id on GPUs $gpu_start,$gpu_end..."
    (
        export CUDA_VISIBLE_DEVICES=$gpu_start,$gpu_end
        export VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS
        export VLLM_QUEUE_PREFETCH=$VLLM_QUEUE_PREFETCH
        export RABBITMQ_URL=amqp://guest:guest@$(hostname):5672/
        export LLMQ_LOG_LEVEL=WARNING
        # Isolate each worker process completely
        cd "$PROJECT_DIR"
        source "$PROJECT_DIR/.env/bin/activate"
        llmq worker run Unbabel/Tower-Plus-72B translationqueue --tensor-parallel-size 2
    ) > worker-tp2-${worker_id}-${SLURM_JOB_ID}.out 2> worker-tp2-${worker_id}-${SLURM_JOB_ID}.err &
    
    echo "Worker $worker_id started"
    sleep 5
done

echo "All workers started"

# Wait for workers to initialize
echo "Waiting 16 min for all workers to initialize..."
sleep 1000

# Step 3: Submit translation job
echo "=========================================="
echo "Step 3: Starting German translation job..."
echo "=========================================="

# Record start time
TRANSLATION_START=$(date +%s)

# Submit the translation job
llmq submit translationqueue HuggingFaceFW/fineweb-edu \
    --subset sample-10BT \
    --map 'messages=[{"role": "user", "content": "Translate the following English source text to German:\nEnglish: {text}\nGerman: "}]' \
    --map 'url="{url}"' \
    --max-samples 500000 > fineweb_german.jsonl

# Record completion
TRANSLATION_END=$(date +%s)
TRANSLATION_DURATION=$((TRANSLATION_END - TRANSLATION_START))

echo "=========================================="
echo "Translation completed!"
echo "=========================================="
echm "Duration: ${TRANSLATION_DURATION} seconds"
