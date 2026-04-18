# Distributed Earthquake Training System

A distributed machine learning training system built for **COEN 6731 — Distributed Software Systems**, Concordia University, Winter 2026.

Trains a linear regression model to predict earthquake magnitude across multiple worker nodes using a centralized **Parameter Server** architecture, deployed on **Microsoft Azure**.

> **Course:** COEN 6731 — Distributed Software Systems, Winter 2026  
> **Institution:** Concordia University, Montreal, Canada  
> **Team:** Saikumar Cheerneni (40323643) · Varshit Beesetti (40326466)  
> **Topic:** Topic 3 — Distributed Training System

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [System Components](#system-components)
4. [Project Structure](#project-structure)
5. [Quick Start — Docker (Local)](#quick-start--docker-local)
6. [Manual Local Setup](#manual-local-setup)
7. [Azure Cloud Deployment](#azure-cloud-deployment)
8. [Web Dashboard Usage](#web-dashboard-usage)
9. [Dataset Format](#dataset-format)
10. [Training Modes](#training-modes)
11. [Fault Tolerance](#fault-tolerance)
12. [Environment Variables](#environment-variables)
13. [Dependencies](#dependencies)
14. [Evaluation Results](#evaluation-results)

---

## System Overview

The system distributes the training of a linear regression model across multiple worker nodes. Each worker receives a shard of the USGS earthquake dataset, computes gradients locally, and pushes them to a central Parameter Server over TCP. The PS aggregates gradients, updates the global model weights, and broadcasts the updated weights back to all workers.

The system supports:
- **Synchronous training** — PS waits for all workers before updating weights
- **Asynchronous training** — PS updates immediately on each gradient push
- **Straggler detection** — slow or failed workers are removed from the active set automatically
- **Fault tolerance** — worker crashes trigger Docker auto-restart and reconnection
- **Live web dashboard** — real-time training progress, loss curves, per-worker analytics, and magnitude prediction

---

## Architecture

```
Internet
    │
    ▼
eq-web VM  ─── Flask Dashboard (port 80)  ◄── Public IP
    │
    │ (reads training history from Azure Blob Storage)
    │
    ▼
Azure Blob Storage
    │   ▲
    │   │  (shards, weights, history CSVs)
    ▼   │
eq-ps VM  ─── Parameter Server (port 50051) ◄── Private IP only
    │               │
    │               │ TCP (JSON messages)
    ├── Worker 0    │
    │   (co-located on eq-ps)
    │
eq-worker1 VM
    ├── Worker 1
    ├── Worker 2
    └── Worker 3
```

**Data Flow:**
1. User uploads CSV via dashboard → system splits into equal shards → uploads to Azure Blob Storage
2. Each worker downloads its assigned shard → normalises features → computes mini-batch gradients
3. Workers push `(gradients, loss)` to PS via TCP on port 50051
4. PS aggregates gradients → updates global weights → sends updated weights back to workers
5. After every epoch, workers upload training history CSVs to Blob → dashboard reads and displays them live

---

## System Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Parameter Server | `core/parameter_server.py` | Accepts TCP connections, buffers gradients, updates weights in sync/async mode, detects stragglers, checkpoints every 50 rounds |
| Worker | `core/worker.py` | Downloads shard from Azure Blob, Z-score normalises features, runs mini-batch SGD, pushes gradients via TCP with exponential backoff |
| Master | `core/master.py` | Local coordinator — launches workers as OS subprocesses, streams logs, aggregates final weights |
| Web Dashboard | `flask_app/app.py` | CSV upload, live loss curve, per-worker analytics, sync/async toggle, magnitude prediction |
| Data Prep | `prepare_data.py` | Auto-detects CSV columns, removes missing rows, shuffles, splits into equal shards, uploads to Azure Blob |

---

## Project Structure

```
COEN6731_EQ_Distributed/
├── core/
│   ├── parameter_server.py     # PS — gradient aggregation, sync/async, fault tolerance
│   ├── worker.py               # Worker — shard loading, SGD, PS communication
│   └── master.py               # Master — local subprocess orchestration
├── flask_app/
│   └── app.py                  # Web dashboard — upload, monitor, analytics, prediction
├── docker/
│   ├── Dockerfile.ps           # Parameter Server Docker image
│   ├── Dockerfile.worker       # Worker Docker image
│   ├── Dockerfile.web          # Flask dashboard Docker image
│   └── docker-compose.yml      # Local multi-container setup (2 workers default)
├── scripts/
│   ├── vm_setup.sh             # VM dependency installation script
│   ├── start_ps.sh             # Manual PS start script
│   ├── start_worker.sh         # Manual worker start script
│   └── start_web.sh            # Manual web start script
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD — build, push Docker Hub, deploy to Azure on push to main
├── prepare_data.py             # CSV → shards splitter + Azure Blob uploader
├── requirements.txt            # Python dependencies
└── earthquake.csv              # USGS earthquake dataset (797,178 rows, 83 MB)
```

---

## Quick Start — Docker (Local)

**Prerequisites:** Docker and Docker Compose installed.

### 1. Clone the repository

```bash
git clone https://github.com/saikumarcheerneni/COEN6731_EQ_Distributed
cd COEN6731_EQ_Distributed/docker
```

### 2. Start with 2 workers (default)

```bash
docker compose up --build
```

### 3. Open the dashboard

```
http://localhost:5000
```

### 4. Start with 3 or 4 workers

```bash
# 3 workers
NUM_WORKERS=3 docker compose --profile workers3 up --build

# 4 workers
NUM_WORKERS=4 docker compose --profile workers4 up --build
```

### 5. Stop the system

```bash
docker compose down
```

---

## Manual Local Setup

**Prerequisites:** Python 3.11+

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data shards

```bash
python prepare_data.py --input earthquake.csv --workers 2
```

This reads `earthquake.csv`, shuffles the rows, splits them into 2 equal shards, and saves them to `data_shards/shard_0.csv` and `data_shards/shard_1.csv`.

### 3. Start the Parameter Server (Terminal 1)

```bash
python core/parameter_server.py --workers 2 --mode sync
```

### 4. Start Worker 0 (Terminal 2)

```bash
python core/worker.py --id 0 --epochs 20
```

### 5. Start Worker 1 (Terminal 3)

```bash
python core/worker.py --id 1 --epochs 20
```

### 6. Start the Web Dashboard (Terminal 4)

```bash
PS_HOST=localhost python flask_app/app.py
```

Open `http://localhost:5000` in your browser.

### Alternatively — use master.py to launch all workers at once

```bash
# Start PS first (Terminal 1)
python core/parameter_server.py --workers 2 --mode sync

# Then run master (Terminal 2) — launches both workers as subprocesses
python core/master.py --workers 2 --mode sync
```

---

## Azure Cloud Deployment

Every push to `main` automatically builds Docker images, pushes them to Docker Hub, and deploys to all three Azure VMs via GitHub Actions.

### Azure VM Layout

| VM Name | Role | Public IP | Private IP |
|---------|------|-----------|------------|
| `eq-ps` | Parameter Server + Worker 0 | Yes | 10.0.0.4 |
| `eq-worker1` | Workers 1, 2, 3 | Yes (SSH only) | 10.0.0.5 |
| `eq-web` | Flask Dashboard | Yes (port 80) | 10.0.0.6 |

### Azure Setup Steps

**1. Create 3 Ubuntu 22.04 LTS B1s VMs** in the same Virtual Network:

```
Resource Group:  eq-distributed
Region:          Canada East
VNet:            eq-vnet (10.0.0.0/16)
Subnet:          eq-subnet (10.0.0.0/24)
```

**2. Configure Network Security Group rules:**

| VM | Port | Protocol | Source |
|----|------|----------|--------|
| eq-ps | 50051 | TCP | VNet only |
| eq-web | 80 | TCP | Internet |
| all VMs | 22 | TCP | Your IP |

**3. SSH into each VM and run the setup script:**

```bash
bash scripts/vm_setup.sh
```

This installs Docker, Docker Compose, and required system dependencies.

### Required GitHub Secrets

Go to your GitHub repo → Settings → Secrets and Variables → Actions, and add:

| Secret | Description |
|--------|-------------|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password or access token |
| `PS_PUBLIC_IP` | Public IP of eq-ps VM |
| `PS_PRIVATE_IP` | Private IP of eq-ps VM (e.g. 10.0.0.4) |
| `WORKER1_PUBLIC_IP` | Public IP of eq-worker1 VM |
| `WEB_PUBLIC_IP` | Public IP of eq-web VM |
| `VM_SSH_KEY` | SSH private key (same key used for all VMs) |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection string |

### Deploy manually without CI/CD

```bash
# SSH into eq-ps VM
ssh -i eq-ps-key.pem azureuser@<PS_PUBLIC_IP>
docker pull <dockerhub-user>/eq-ps:latest
docker run -d --name eq-ps -p 50051:50051 <dockerhub-user>/eq-ps:latest
```

---

## Web Dashboard Usage

1. Open the dashboard at `http://<WEB_PUBLIC_IP>` (Azure) or `http://localhost:5000` (local)
2. **Upload Dataset** — upload a CSV file and select number of workers (2, 3, or 4)
3. **Dashboard** — select SYNC or ASYNC mode using the toggle button
4. **Click Start Training** — training begins automatically across all workers
5. **Monitor** — the dashboard auto-refreshes every 5 seconds showing live loss curves and round counts
6. **Training Analytics** — detailed per-worker RMSE charts, PS latency comparison, improvement percentage
7. **Predict Magnitude** — enter depth (km), latitude, and longitude to get a real-time prediction from the trained model

---

## Dataset Format

The system accepts any CSV file with earthquake records. Column names are auto-detected using hint matching — exact case and common aliases all work.

| Column | Description | Accepted aliases |
|--------|-------------|-----------------|
| `Magnitude` | Target — earthquake magnitude | `mag`, `magnitude` |
| `Depth` | Depth in km | `depth` |
| `Latitude` | Latitude coordinate | `lat`, `latitude` |
| `Longitude` | Longitude coordinate | `lon`, `long`, `longitude` |

The system automatically drops any other columns and removes rows with missing values.

**Dataset used in evaluation:** USGS earthquake dataset — 797,178 records, 83 MB.

---

## Training Modes

| Mode | How it works | Best for |
|------|-------------|----------|
| **Sync** | PS waits for all active workers to push gradients before updating weights | Consistent model, slower overall |
| **Async** | PS updates weights immediately when any worker pushes gradients | Faster updates, workers may use slightly stale weights |

Both modes converge to the same final accuracy for linear regression because the loss surface is convex. Async mode produces approximately 1.5x more gradient updates in the same wall-clock time.

Switch modes from the dashboard without restarting — the PS resets its round counter and buffer state automatically.

---

## Fault Tolerance

The system handles three types of failures:

| Failure Type | Detection | Recovery |
|-------------|-----------|----------|
| **Worker crash** | Docker detects container exit | `restart: always` policy auto-restarts, worker re-downloads shard and reconnects |
| **Slow / straggler worker** | PS tracks `last_seen` timestamp per worker | Worker marked as straggler after 30 seconds of silence; removed from active set; training continues |
| **Dropped network connection** | TCP send/receive timeout | Worker retries with exponential backoff — doubles wait from 2s up to 60s, up to 20 retries per batch |

**Known limitation:** The Parameter Server is a single point of failure. If the PS VM crashes, all workers lose their connections and training stops. Workers retry automatically and resume once the PS recovers. A peer-to-peer architecture would eliminate this at the cost of more complex coordination.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PS_HOST` | `localhost` | Hostname or IP of the Parameter Server |
| `PS_PORT` | `50051` | TCP port the PS listens on |
| `NUM_WORKERS` | `4` | Number of workers the PS expects |
| `LR` | `0.001` | Gradient descent learning rate |
| `MODE` | `sync` | Training mode: `sync` or `async` |
| `MAX_ROUNDS` | `1000` | Maximum PS rounds before stopping |
| `STRAGGLER_TIMEOUT` | `30` | Seconds of silence before a worker is marked a straggler |
| `WORKER_ID` | `0` | Worker identity number (0 through N-1) |
| `EPOCHS` | `20` | Maximum epochs per worker |
| `BATCH_SIZE` | `256` | Rows per mini-batch |
| `AZURE_STORAGE_CONNECTION_STRING` | — | Azure Blob Storage connection string |

---

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
flask>=3.0.0
werkzeug>=3.0.0
azure-storage-blob>=12.19.0
```

All dependencies are handled automatically by Docker. For manual installation:

```bash
pip install -r requirements.txt
```

---

## Evaluation Results

All experiments run on Azure B1s VMs (1 vCPU, 1 GB RAM, Ubuntu 22.04), Canada East region.  
Dataset: USGS earthquake dataset, 797,178 records, batch size 256, learning rate 0.001.

### Worker Scaling — Synchronous Mode

| Workers | Best RMSE | Epochs | PS Rounds |
|---------|-----------|--------|-----------|
| 2 | 0.7360 | 4 | 1,193 |
| 3 | 0.7285 | 5 | 1,193 |
| **4** | 0.7682 | 23 | 1,193 |

3 workers produced the best RMSE of 0.7285.

### PS Communication Latency by Worker Placement

| Worker | VM Placement | Avg Latency | Epochs Completed |
|--------|-------------|-------------|-----------------|
| Worker 0 | Same VM as PS (co-located) | 1.3 ms | 17–20 |
| Worker 1 | Cross-VM (Azure private network) | 3.1 ms | 2–3 |
| Worker 2 | Cross-VM (Azure private network) | 3.1 ms | 2–3 |
| Worker 3 | Cross-VM (Azure private network) | 3.4 ms | 2–3 |

### Sync vs Async Comparison

| Metric | Sync | Async |
|--------|------|-------|
| Total gradient updates | 1,006 rounds | 1,558 updates |
| Consistency | High | Lower (stale weights) |
| Final RMSE | 0.7285 | 0.7285 |
| Bottleneck | Slowest worker | None |

### Fault Tolerance Experiment

Worker 1 killed mid-training via `docker kill eq-worker1`:

| Event | Observed Outcome |
|-------|-----------------|
| Worker 1 killed | Stopped after 2 epochs |
| After 30 seconds | PS marked Worker 1 as straggler |
| Training continued | PS used Workers 0 and 2 only |
| Training completed | Round 1,045, RMSE 0.8157 |

The slightly higher RMSE (0.8157 vs 0.7285) is expected — the PS received gradients from only two thirds of the dataset after Worker 1 was removed.

---

## License

This project was developed for academic purposes as part of COEN 6731 at Concordia University.
