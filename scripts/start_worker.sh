#!/bin/bash
# ═══════════════════════════════════════════
# start_worker.sh — Run on Worker VMs
# Usage: bash start_worker.sh <worker_id> <ps_private_ip>
# Example: bash start_worker.sh 0 10.0.0.4
# ═══════════════════════════════════════════
WORKER_ID=${1:-0}
PS_IP=${2:-"10.0.0.4"}

echo "⚙️  Starting Worker $WORKER_ID → PS at $PS_IP:50051"

export PS_HOST=$PS_IP
export PS_PORT=50051
mkdir -p checkpoints outputs data_shards

python3 core/worker.py \
    --id $WORKER_ID \
    --data data_shards/shard_${WORKER_ID}.csv \
    --epochs 20
