"""
core/worker.py — Azure Blob Storage aware worker
"""
import argparse, json, math, socket, time, logging, os
import numpy as np
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [W%(worker_id)s] %(message)s", datefmt="%H:%M:%S")

PS_HOST        = os.environ.get("PS_HOST", "localhost")
PS_PORT        = int(os.environ.get("PS_PORT", 50051))
WORKER_ID      = int(os.environ.get("WORKER_ID", 0))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
CHECKPOINT_DIR.mkdir(exist_ok=True)
AZURE_CONN_STR = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
CONTAINER_NAME = "eq-data"
MAX_RECONNECT  = 8
RECONNECT_BASE = 1.0

def download_shard_from_blob(worker_id):
    log = logging.getLogger("W")
    if not AZURE_CONN_STR:
        log.info("No Azure connection string — using local shard")
        return f"data_shards/shard_{worker_id}.csv"
    try:
        log.info(f"Downloading shard_{worker_id}.csv from Azure Blob...")
        client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        blob   = client.get_blob_client(container=CONTAINER_NAME, blob=f"shard_{worker_id}.csv")
        path   = f"data_shards/shard_{worker_id}.csv"
        Path("data_shards").mkdir(exist_ok=True)
        with open(path, "wb") as f:
            f.write(blob.download_blob().readall())
        log.info(f"Downloaded shard_{worker_id}.csv successfully!")
        return path
    except Exception as e:
        log.warning(f"Blob download failed: {e} — using local shard if available")
        return f"data_shards/shard_{worker_id}.csv"

def predict(X, w): return X @ w[:-1] + w[-1]

def mse_gradients(X, y, w):
    err = predict(X, w) - y
    return np.append((2/len(y))*(X.T@err), (2/len(y))*err.sum()).astype(np.float32)

def send_recv(host, port, msg):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(json.dumps(msg).encode())
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk: break
            data += chunk
            try: return json.loads(data.decode())
            except json.JSONDecodeError: continue

def train(worker_id, epochs=20, batch_size=32):
    log = logging.LoggerAdapter(logging.getLogger("W"), {"worker_id": worker_id})

    # Download shard from Azure Blob
    data_path = download_shard_from_blob(worker_id)

    # Wait for shard to be available
    retries = 0
    while not Path(data_path).exists() and retries < 10:
        log.info(f"Waiting for shard file {data_path}...")
        time.sleep(5)
        data_path = download_shard_from_blob(worker_id)
        retries += 1

    if not Path(data_path).exists():
        log.error(f"Shard file not found: {data_path}")
        return

    df = pd.read_csv(data_path).dropna()
    X  = df[["Depth", "Latitude", "Longitude"]].values.astype(np.float32)
    y  = df["Magnitude"].values.astype(np.float32)
    X  = (X - X.mean(0)) / (X.std(0) + 1e-8)

    ckpt = CHECKPOINT_DIR / f"worker_{worker_id}.npy"
    start_epoch = 0
    if ckpt.exists():
        state = np.load(ckpt, allow_pickle=True).item()
        start_epoch = state.get("epoch", 0)
        log.info(f"Resuming from epoch {start_epoch}")

    # Get initial weights from PS
    for attempt in range(MAX_RECONNECT):
        try:
            resp = send_recv(PS_HOST, PS_PORT, {"type": "get_weights"})
            w = np.array(resp["weights"], dtype=np.float32)
            break
        except Exception as e:
            wait = RECONNECT_BASE * (2 ** attempt)
            log.info(f"PS not ready, retry in {wait:.1f}s ({e})")
            time.sleep(wait)

    history = []
    for epoch in range(start_epoch + 1, epochs + 1):
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        epoch_loss = 0
        batches = math.ceil(len(X) / batch_size)
        for i in range(batches):
            xb = X[i*batch_size:(i+1)*batch_size]
            yb = y[i*batch_size:(i+1)*batch_size]
            grads = mse_gradients(xb, yb, w)
            pred  = predict(xb, w)
            loss  = float(np.sqrt(np.mean((pred - yb)**2)))
            epoch_loss += loss
            try:
                resp = send_recv(PS_HOST, PS_PORT, {
                    "type": "push_gradients",
                    "worker_id": worker_id,
                    "gradients": grads.tolist(),
                    "loss": loss
                })
                w = np.array(resp["weights"], dtype=np.float32)
            except Exception as e:
                log.warning(f"PS error: {e}, using local weights")

        avg_loss = epoch_loss / batches
        rmse = float(np.sqrt(np.mean((predict(X, w) - y)**2)))
        log.info(f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | RMSE={rmse:.4f}")
        history.append({"epoch": epoch, "loss": avg_loss, "rmse": rmse})
        np.save(ckpt, {"epoch": epoch, "weights": w, "history": history})

    log.info(f"Training complete | final RMSE={rmse:.4f}")
    out = Path("outputs"); out.mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(out / f"worker_{worker_id}_history.csv", index=False)
    return w, history

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--id",         type=int, default=int(os.environ.get("WORKER_ID", 0)))
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    train(args.id, args.epochs, args.batch_size)