# 🌍 EQ Distributed Training System — Azure Deployment

## Architecture

```
Internet
    ↓
eq-web VM (Flask :5000) ← Public IP
    ↓ TCP
eq-ps VM (Parameter Server :50051) ← Private IP
    ↙              ↘
eq-worker0 VM    eq-worker1 VM
(Worker 0)       (Worker 1)
```

---

## Step 1: Azure Portal — Create Resources

### 1.1 Resource Group
```
Name:   eq-distributed
Region: Canada East
```

### 1.2 Virtual Network
```
Name:           eq-vnet
Address space:  10.0.0.0/16
Subnet:         eq-subnet (10.0.0.0/24)
```

### 1.3 Create 4 VMs (B1s — cheapest)

| VM Name    | Role             | Public IP |
|------------|------------------|-----------|
| eq-ps      | Parameter Server | No        |
| eq-worker0 | Worker 0         | No        |
| eq-worker1 | Worker 1         | No        |
| eq-web     | Flask Website    | YES ✅    |

**Settings for each VM:**
- OS: Ubuntu 22.04 LTS
- Size: B1s (1 CPU, 1GB RAM) — cheapest!
- VNet: eq-vnet / eq-subnet
- Auth: SSH public key

### 1.4 Network Security Group (NSG)
Add inbound rules:
```
eq-ps VM:   port 50051  (TCP) — from eq-vnet only
eq-web VM:  port 5000   (TCP) — from Internet
            port 22     (SSH) — from your IP
```

---

## Step 2: Note Private IPs

After VMs are created, note private IPs:
```
eq-ps:      10.0.0.4  (example)
eq-worker0: 10.0.0.5
eq-worker1: 10.0.0.6
eq-web:     10.0.0.7  + public IP
```

---

## Step 3: Upload Code to VMs

From your PC (PowerShell):
```powershell
# Upload to PS VM
scp -r C:\Users\saiku\Downloads\COEN6731_Azure_Deploy\eq-azure\* azureuser@<eq-ps-public-ip>:~/eq-distributed/

# Upload to workers
scp -r ... azureuser@<eq-worker0-public-ip>:~/eq-distributed/
scp -r ... azureuser@<eq-worker1-public-ip>:~/eq-distributed/
scp -r ... azureuser@<eq-web-public-ip>:~/eq-distributed/
```

---

## Step 4: Setup Each VM

SSH into each VM and run:
```bash
bash scripts/vm_setup.sh
```

---

## Step 5: Prepare Data (on PS VM)

```bash
ssh azureuser@<eq-ps-ip>
cd ~/eq-distributed
python3 prepare_data.py --input earthquake.csv --workers 2

# Copy shards to workers
scp data_shards/shard_0.csv azureuser@10.0.0.5:~/eq-distributed/data_shards/
scp data_shards/shard_1.csv azureuser@10.0.0.6:~/eq-distributed/data_shards/
```

---

## Step 6: Start Everything (in order!)

### Terminal 1 — SSH to eq-ps VM:
```bash
bash scripts/start_ps.sh
```

### Terminal 2 — SSH to eq-worker0 VM:
```bash
bash scripts/start_worker.sh 0 10.0.0.4
```

### Terminal 3 — SSH to eq-worker1 VM:
```bash
bash scripts/start_worker.sh 1 10.0.0.4
```

### Terminal 4 — SSH to eq-web VM:
```bash
bash scripts/start_web.sh 10.0.0.4
```

---

## Step 7: Access Website

Open browser: `http://<eq-web-public-ip>:5000`

You will see:
- Live training rounds
- Loss curve updating
- Worker status
- Predict magnitude form

---

## Docker Alternative (easier!)

On any VM with Docker installed:
```bash
cd docker/
# Edit docker-compose.yml — set PS_HOST to eq-ps private IP
docker-compose up --build
```

---

## Cost Estimate

```
4x B1s VMs = 4 × $0.012/hr = $0.048/hr
Training:   ~30 minutes = $0.024
Website 24/7: 1 VM = $8.76/month

Total for demo: < $1 💰
```

**Stop VMs when not using to save credits!**
```bash
# Azure CLI to stop all VMs
az vm deallocate --resource-group eq-distributed --name eq-ps
az vm deallocate --resource-group eq-distributed --name eq-worker0
az vm deallocate --resource-group eq-distributed --name eq-worker1
# Keep eq-web running for demo
```

---

## Project Info
- Course: COEN 6731 — Concordia University
- Team: Varshit Beesetti (40326466), Saikumar Cheerneni (40323643)
