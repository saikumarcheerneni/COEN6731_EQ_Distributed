#!/bin/bash
# ═══════════════════════════════════════════
# start_web.sh — Run on Web VM
# Usage: bash start_web.sh <ps_private_ip>
# Example: bash start_web.sh 10.0.0.4
# ═══════════════════════════════════════════
PS_IP=${1:-"10.0.0.4"}

echo "🌐 Starting Flask Web Dashboard → PS at $PS_IP"

export PS_HOST=$PS_IP
export PS_PORT=50051
export FLASK_PORT=5000

python3 flask_app/app.py
