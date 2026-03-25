#!/bin/bash
# ═══════════════════════════════════════════
# start_ps.sh — Run on Parameter Server VM
# ═══════════════════════════════════════════
echo "🔄 Starting Parameter Server..."
export PS_PORT=50051
export NUM_WORKERS=2
export LR=0.001
export STRAGGLER_TIMEOUT=30
mkdir -p checkpoints outputs
python3 core/parameter_server.py --workers 2 --mode sync
