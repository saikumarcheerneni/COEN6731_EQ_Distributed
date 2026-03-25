#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Azure VM Setup Script — EQ Distributed Training System
# Run this ONCE on each VM after creation
# ═══════════════════════════════════════════════════════════

set -e

echo "🚀 Setting up EQ Distributed Training System..."

# ── 1. Update & install Docker ──────────────────────────────
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose git python3-pip

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

echo "✅ Docker installed"

# ── 2. Clone project ────────────────────────────────────────
cd ~
if [ ! -d "eq-distributed" ]; then
    git clone https://github.com/YOUR_USERNAME/eq-distributed.git
    # OR: copy files manually via scp
fi
cd eq-distributed

echo "✅ Project ready"

# ── 3. Install Python deps (fallback without Docker) ────────
pip3 install numpy pandas matplotlib flask --break-system-packages 2>/dev/null || true

echo "✅ Python packages installed"
echo ""
echo "🎯 Setup complete! Now run the appropriate start script:"
echo "   PS VM:      bash scripts/start_ps.sh"
echo "   Worker VM:  bash scripts/start_worker.sh"  
echo "   Web VM:     bash scripts/start_web.sh"
