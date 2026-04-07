# EQ Distributed Training System

A distributed machine learning training system built for COEN 6731 — Concordia University.
Trains a linear regression model on earthquake data across multiple worker nodes using a
centralized Parameter Server architecture, deployed on Microsoft Azure.

---

## Architecture
Internet
↓
eq-web VM (Flask Dashboard :80) ← Public IP
↓ TCP
eq-ps VM (Parameter Server :50051) ← Private IP
↓              ↓
eq-worker0      eq-worker1 VM
(PS VM)         (Worker 1 + 2 + 3)

**Data Flow:**
1. User uploads CSV → Flask splits into shards → Azure Blob Storage
2. Workers download their shard → compute gradients → push to PS
3. PS aggregates gradients → updates weights → returns to workers
4. Dashboard shows live training progress and analytics

---

## System Components

| Component | File | Role |
|-----------|------|------|
| Parameter Server | `core/parameter_server.py` | Aggregates gradients, manages training state |
| Worker | `core/worker.py` | Loads shard, computes MSE gradients, pushes to PS |
| Master | `core/master.py` | Orchestrates local workers as subprocesses |
| Web Dashboard | `flask_app/app.py` | Upload, monitor, analytics UI |
| Data Prep | `prepare_data.py` | Splits CSV into worker shards |

---

## Quick Start — Docker (Recommended)
```bash
git clone https://github.com/<your-username>/COEN6731_EQ_Distributed
cd COEN6731_EQ_Distributed/docker
docker-compose up --build
```

Open browser: `http://localhost:5000`

For 3 workers:
```bash
NUM_WORKERS=3 docker-compose --profile workers3 up --build
```

For 4 workers:
```bash
NUM_WORKERS=4 docker-compose --profile workers4 up --build
```

---

## Azure Cloud Deployment (CI/CD)

Every push to `main` automatically builds Docker images, pushes to Docker Hub,
and deploys to Azure VMs via GitHub Actions.

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password or access token |
| `PS_PUBLIC_IP` | Public IP of Parameter Server VM |
| `PS_PRIVATE_IP` | Private IP of Parameter Server VM |
| `WORKER1_PUBLIC_IP` | Public IP of Worker VM |
| `WEB_PUBLIC_IP` | Public IP of Web Dashboard VM |
| `VM_SSH_KEY` | SSH private key (same key for all VMs) |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection string |

### VM Layout

| VM Name | Role | Contains |
|---------|------|----------|
| `eq-ps` | Parameter Server + Worker 0 | PS process + worker0 container |
| `eq-worker1` | Workers 1, 2, 3 | worker1, worker2, worker3 containers |
| `eq-web` | Flask Dashboard | Web UI on port 80 |

### Azure VM Setup

Create 3 VMs (Ubuntu 22.04 LTS, B1s size) in the same Virtual Network:
Resource Group: eq-distributed
Region:         Canada East
VNet:           eq-vnet (10.0.0.0/16)
Subnet:         eq-subnet (10.0.0.0/24)

Network Security Group rules:
eq-ps VM:   port 50051 (TCP) — from VNet only
eq-web VM:  port 80   (TCP) — from Internet
port 22   (SSH) — from your IP

SSH into each VM and run:
```bash
bash scripts/vm_setup.sh
```

---

## Manual Local Setup (without Docker)

### Prerequisites
- Python 3.11+
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

### Prepare data
```bash
python prepare_data.py --input earthquake.csv --workers 2
```

### Start Parameter Server
```bash
python core/parameter_server.py --workers 2 --mode sync
```

### Start Workers (separate terminals)
```bash
python core/worker.py --id 0 --epochs 20
python core/worker.py --id 1 --epochs 20
```

### Start Web Dashboard
```bash
PS_HOST=localhost python flask_app/app.py
```

Open browser: `http://localhost:5000`

---

## Usage — Web Dashboard

1. Open the dashboard at your web VM's public IP
2. Go to **Upload Dataset** — upload a CSV file, select number of workers (2/3/4)
3. On **Dashboard** — select SYNC or ASYNC mode using the toggle
4. Click **Start Training** — training begins automatically
5. Monitor live progress on the dashboard (auto-refreshes every 5s)
6. Go to **Training Analytics** for detailed charts and per-worker stats
7. Use **Predict Magnitude** to run inference with the trained model

---

## Required CSV Columns

| Column | Description | Aliases accepted |
|--------|-------------|-----------------|
| `Magnitude` | Target — earthquake magnitude | `mag`, `magnitude` |
| `Depth` | Depth in km | `depth` |
| `Latitude` | Latitude coordinate | `lat`, `latitude` |
| `Longitude` | Longitude coordinate | `lon`, `long`, `longitude` |

Column names are auto-detected — exact case and aliases all work.

---

## Training Modes

| Mode | Description | Use case |
|------|-------------|----------|
| **Sync** | PS waits for all workers before updating | Consistent, slower |
| **Async** | PS updates immediately on any gradient push | Faster, stale weights |

Switch modes from the Dashboard without restarting — PS resets state automatically.

---

## Fault Tolerance

- **Straggler detection** — workers slower than 30s timeout are skipped in sync mode
- **Worker crash recovery** — Docker `restart: on-failure:5` automatically restarts crashed containers
- **Reconnect logic** — workers retry PS connection with exponential backoff (up to 60s)
- **Memory watchdog** — PS trims history and runs GC if memory exceeds 80%
- **Checkpoint saving** — PS saves weights every 50 rounds to `checkpoints/weights.npy`

---

## Project Structure
COEN6731_EQ_Distributed/
├── core/
│   ├── parameter_server.py   # PS — gradient aggregation, sync/async, fault tolerance
│   ├── worker.py             # Worker — shard loading, gradient computation, PS comms
│   └── master.py             # Master — local subprocess orchestration
├── flask_app/
│   └── app.py                # Web dashboard — upload, monitor, analytics
├── docker/
│   ├── Dockerfile.ps         # Parameter Server image
│   ├── Dockerfile.worker     # Worker image
│   ├── Dockerfile.web        # Flask dashboard image
│   └── docker-compose.yml    # Local multi-container setup
├── scripts/
│   ├── vm_setup.sh           # VM dependency setup script
│   ├── start_ps.sh           # Manual PS start script
│   ├── start_worker.sh       # Manual worker start script
│   └── start_web.sh          # Manual web start script
├── .github/
│   └── workflows/
│       └── deploy.yml        # CI/CD — build, push, deploy on every push to main
├── prepare_data.py           # CSV → shards splitter
├── requirements.txt          # Python dependencies
└── earthquake.csv            # USGS earthquake dataset (797K rows, 83MB)

---

## Dependencies
numpy>=1.24.0
pandas>=2.0.0
flask>=3.0.0
werkzeug>=3.0.0
azure-storage-blob>=12.19.0

All handled automatically via Docker. For manual install:
```bash
pip install -r requirements.txt
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PS_HOST` | `localhost` | Parameter Server hostname |
| `PS_PORT` | `50051` | Parameter Server port |
| `NUM_WORKERS` | `4` | Number of workers PS waits for |
| `LR` | `0.001` | Learning rate |
| `MODE` | `sync` | Training mode (`sync` or `async`) |
| `MAX_ROUNDS` | `1000` | Training stops after this many rounds |
| `STRAGGLER_TIMEOUT` | `30` | Seconds before a worker is marked straggler |
| `WORKER_ID` | `0` | Worker identity (0–3) |
| `AZURE_STORAGE_CONNECTION_STRING` | `` | Azure Blob connection string |

---

## Evaluation Results

Tested on USGS earthquake dataset (797,178 rows, 83MB) on Azure B1s VMs, Canada East.

### Worker Scaling (Sync Mode)

| Workers | Best RMSE | Total Epochs | PS Rounds |
|---------|-----------|--------------|-----------|
| 2 | 0.7360 | 4 | 1193 |
| 3 | 0.7285 | 5 | 1193 |
| 4 | 0.7682 | 23 | 1193 |

### Sync vs Async Comparison

| Metric | Sync | Async |
|--------|------|-------|
| Gradient updates | 1193 rounds | 2855 updates |
| Consistency | High | Lower (stale weights) |
| Speed | Baseline | ~2.4x faster |

### PS Communication Latency

| Worker | VM | Avg Latency |
|--------|----|-------------|
| Worker 0 | PS VM (co-located) | 0.8ms |
| Worker 1 | Worker VM (cross-VM) | 3.1ms |
| Worker 2 | Worker VM (cross-VM) | 3.1ms |
| Worker 3 | Worker VM (cross-VM) | 3.4ms |

---

## Course Info

- **Course:** COEN 6731 — Distributed Systems, Winter 2026
- **Institution:** Concordia University
- **Topic:** Topic 3 — Distributed Training System
- **Team:** Varshit Beesetti (40326466), Saikumar Cheerneni (40323643)
- **Language:** Python
- **Infrastructure:** Microsoft Azure (3 B1s VMs, Canada East)