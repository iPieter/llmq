#!/bin/bash
set -e

# Utility script to start RabbitMQ broker using Singularity on HPC clusters
# Usage: ./utils/start_singularity_broker.sh

RABBITMQ_SIF="${RABBITMQ_SIF:-./rabbitmq.sif}"
RABBITMQ_DATA="${RABBITMQ_DATA:-$HOME/rabbitmq-data}"
INSTANCE_NAME="${INSTANCE_NAME:-rabbitmq-instance}"

echo "Starting RabbitMQ broker with Singularity..."

# Build container if it doesn't exist
if [ ! -f "$RABBITMQ_SIF" ]; then
    echo "Building Singularity container..."
    singularity build "$RABBITMQ_SIF" docker://rabbitmq:3-management
fi

# Create data directory
echo "Creating data directory: $RABBITMQ_DATA"
mkdir -p "$RABBITMQ_DATA"

# Stop existing instance if running
if singularity instance list | grep -q "$INSTANCE_NAME"; then
    echo "Stopping existing instance..."
    singularity instance stop "$INSTANCE_NAME"
fi

# Kill any existing RabbitMQ processes
pkill -f rabbitmq-server || true

# Start instance with bind mount
echo "Starting Singularity instance..."
singularity instance start --bind "$RABBITMQ_DATA:/var/lib/rabbitmq" "$RABBITMQ_SIF" "$INSTANCE_NAME"

# Start RabbitMQ service
echo "Starting RabbitMQ service..."
nohup singularity exec instance://"$INSTANCE_NAME" rabbitmq-server > /dev/null 2>&1 &

# Start RabbitMQ (may take a moment to bind to port)
echo "✅ RabbitMQ startup initiated"
echo "Set connection URL with:"
echo "export RABBITMQ_URL=amqp://guest:guest@\$(hostname):5672/"
echo "Test connection with: llmq status"