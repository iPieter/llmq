#!/bin/bash
#SBATCH --job-name=dutch-translation
#SBATCH --account=ehpc-aif-2025pg01-128
#SBATCH --qos=ehpc-aif-2025pg01-128
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=dutch-translation-%j.out
#SBATCH --error=dutch-translation-%j.err
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
export LLMQ_LOG_LEVEL=WARNING
echo "LLMQ config: MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS, QUEUE_PREFETCH=$VLLM_QUEUE_PREFETCH"

# Step 1: Start RabbitMQ broker
echo "=========================================="
echo "Step 1: Starting RabbitMQ broker..."
echo "=========================================="
bash llmq/utils/start_singularity_broker.sh
echo "Waiting 20 seconds for RabbitMQ to initialize..."
sleep 80
echo "RabbitMQ should be ready"

echo "=========================================="
echo "Step 3: Starting LLMQ workers with TP=1..."
echo "=========================================="
# Start 8 workers with tensor-parallel-size 1 (one GPU per worker)
for worker_id in 0 1 2 3 4 5 6 7; do
    gpu_id=$worker_id
    echo "Starting worker $worker_id on GPU $gpu_id..."
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        export VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS
        export VLLM_QUEUE_PREFETCH=$VLLM_QUEUE_PREFETCH
        export RABBITMQ_URL=amqp://guest:guest@$(hostname):5672/
        export LLMQ_LOG_LEVEL=WARNING
        # Isolate each worker process completely
        cd "$PROJECT_DIR"
        source "$PROJECT_DIR/.env/bin/activate"
        llmq worker run Unbabel/Tower-Plus-9B translationqueue --tensor-parallel-size 1
    ) > worker-tp1-${worker_id}-${SLURM_JOB_ID}.out 2> worker-tp1-${worker_id}-${SLURM_JOB_ID}.err &

    echo "Worker $worker_id started"
    sleep 5
done

echo "All workers started"


echo "=========================================="
echo "Step 2: Starting Dutch translation job..."
echo "=========================================="

sleep 120

# Record start time
TRANSLATION_START=$(date +%s)


# Submit the translation job
llmq submit translationqueue pdelobelle/nemotron-post-training-reordered \
  --map 'messages=[{"role": "user", "content": "Translate the following English source text to Dutch:\nEnglish: {message_content}\nDutch: "}]' \
  --map 'uuid="{uuid}"' \
  --map 'version="{version}"' \
  --map 'message_order="{message_order}"' \
  --map 'message_role="{message_role}"' \
  --map 'category="{category}"' > nemotron-dutch-chat-output.jsonl


# Record completion
TRANSLATION_END=$(date +%s)
TRANSLATION_DURATION=$((TRANSLATION_END - TRANSLATION_START))

echo "=========================================="
echo "Translation completed!"
echo "=========================================="
echo "Duration: ${TRANSLATION_DURATION} seconds"
