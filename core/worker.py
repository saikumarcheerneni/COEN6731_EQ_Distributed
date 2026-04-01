"""
core/worker.py — Azure Blob Storage aware worker
"""
import argparse, json, math, socket, time, logging, os
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [WORKER] %(message)s", datefmt="%H:%M:%S")

PS_HOST        = os.environ.get("PS_HOST", "localhost")
PS_PORT        = int(os.environ.get("PS_PORT", 50051))
WORKER_ID      = int(os.environ.get("WORKER_ID", 0))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
CHECKPOINT_DIR.mkdir(exist_ok=True)
AZURE_CONN_STR = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
CONTAINER_NAME = "eq-data"
MAX_RECONNECT_WAIT = 60
RECONNECT_BASE     = 2.0


def download_shard_from_blob(worker_id):
    log        = logging.getLogger("W")
    local_path = f"data_shards/shard_{worker_id}.csv"
    if not AZURE_CONN_STR:
        log.info("No Azure connection string — using local shard")
        return local_path
    try:
        from azure.storage.blob import BlobServiceClient
        log.info(f"Downloading shard_{worker_id}.csv from Azure Blob...")
        client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        blob   = client.get_blob_client(container=CONTAINER_NAME, blob=f"shard_{worker_id}.csv")
        Path("data_shards").mkdir(exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(blob.download_blob().readall())
        log.info(f"Downloaded shard_{worker_id}.csv from Azure Blob — ready to train on real data!")
        return local_path
    except ImportError:
        log.warning("azure-storage-blob not installed — using local shard")
        return local_path
    except Exception as e:
        log.warning(f"Blob download failed: {e} — falling back to local shard")
        return local_path


def _upload_history_to_blob(worker_id, history_path):
    """Upload worker history CSV to Azure Blob so dashboard can read it live."""
    if not AZURE_CONN_STR:
        return
    try:
        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        blob   = client.get_blob_client(
            container=CONTAINER_NAME,
            blob=f"worker_{worker_id}_history.csv"
        )
        with open(history_path, "rb") as f:
            blob.upload_blob(f, overwrite=True)
        logging.getLogger("W").info(f"Uploaded worker_{worker_id}_history.csv to Azure Blob")
    except Exception as e:
        logging.getLogger("W").warning(f"Failed to upload history to Blob: {e}")


def predict(X, w): return X @ w[:-1] + w[-1]


def mse_gradients(X, y, w):
    err = predict(X, w) - y
    return np.append((2/len(y))*(X.T@err), (2/len(y))*err.sum()).astype(np.float32)


def send_recv(host, port, msg, timeout=30):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect((host, port))
        s.sendall(json.dumps(msg).encode())
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk: break
            data += chunk
            try: return json.loads(data.decode())
            except json.JSONDecodeError: continue
    return {}


def connect_to_ps(worker_id):
    """Retry PS connection forever with exponential backoff."""
    log     = logging.getLogger("W")
    wait    = RECONNECT_BASE
    attempt = 0
    while True:
        try:
            resp = send_recv(PS_HOST, PS_PORT, {"type": "get_weights"})
            w = np.array(resp["weights"], dtype=np.float32)
            log.info(f"Connected to PS at {PS_HOST}:{PS_PORT}")
            return w
        except Exception as e:
            attempt += 1
            log.info(f"PS not ready (attempt {attempt}), retrying in {wait:.1f}s — {e}")
            time.sleep(wait)
            wait = min(wait * 1.5, MAX_RECONNECT_WAIT)


def train(worker_id, epochs=20, batch_size=32, straggler_delay=0.0):
    log = logging.LoggerAdapter(logging.getLogger("W"), {"worker_id": worker_id})

    # Download shard from Azure Blob
    data_path = download_shard_from_blob(worker_id)

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
    log.info(f"Loaded {len(df):,} rows from {data_path}")

    # Resume from checkpoint if container restarted
    ckpt = CHECKPOINT_DIR / f"worker_{worker_id}.npy"
    start_epoch = 0
    if ckpt.exists():
        try:
            state = np.load(ckpt, allow_pickle=True).item()
            start_epoch = state.get("epoch", 0)
            log.info(f"Resuming from checkpoint epoch {start_epoch}")
        except Exception:
            log.warning("Could not load checkpoint — starting fresh")

    # Connect to PS — retries forever with backoff
    w = connect_to_ps(worker_id)

    rmse    = 0.0
    history = []

    for epoch in range(start_epoch + 1, epochs + 1):

        # Straggler simulation
        if straggler_delay > 0:
            log.info(f"[STRAGGLER SIM] Sleeping {straggler_delay}s")
            time.sleep(straggler_delay)

        idx = np.random.permutation(len(X))
        Xs, ys     = X[idx], y[idx]
        epoch_loss = 0.0
        batches    = math.ceil(len(Xs) / batch_size)

        for i in range(batches):
            xb = Xs[i*batch_size:(i+1)*batch_size]
            yb = ys[i*batch_size:(i+1)*batch_size]
            grads = mse_gradients(xb, yb, w)
            loss  = float(np.sqrt(np.mean((predict(xb, w) - yb)**2)))
            epoch_loss += loss

            # Push gradients — reconnect if PS drops mid-training
            reconnect_wait = RECONNECT_BASE
            for _ in range(20):
                try:
                    resp = send_recv(PS_HOST, PS_PORT, {
                        "type":      "push_gradients",
                        "worker_id": worker_id,
                        "gradients": grads.tolist(),
                        "loss":      loss
                    })
                    w = np.array(resp["weights"], dtype=np.float32)
                    # PS signals training complete — exit cleanly
                    if resp.get("done"):
                        log.info("PS signalled training done. Exiting.")
                        _save_outputs(worker_id, w, history)
                        return w, history
                    break
                except Exception as e:
                    log.warning(f"PS comm error: {e} — retrying in {reconnect_wait:.1f}s")
                    time.sleep(reconnect_wait)
                    reconnect_wait = min(reconnect_wait * 2, MAX_RECONNECT_WAIT)

        avg_loss = epoch_loss / batches
        rmse     = float(np.sqrt(np.mean((predict(Xs, w) - ys)**2)))
        log.info(f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | RMSE={rmse:.4f}")
        history.append({"epoch": epoch, "loss": avg_loss, "rmse": rmse})

        # Checkpoint every epoch — survives container restart
        np.save(ckpt, {"epoch": epoch, "weights": w.tolist(), "history": history})

        # Upload history to Azure Blob every epoch — live dashboard charts
        out = Path("outputs"); out.mkdir(exist_ok=True)
        history_path = out / f"worker_{worker_id}_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)
        _upload_history_to_blob(worker_id, history_path)

    log.info(f"Training complete | final RMSE={rmse:.4f}")
    _save_outputs(worker_id, w, history)
    return w, history


def _save_outputs(worker_id, w, history):
    out = Path("outputs"); out.mkdir(exist_ok=True)
    history_path = out / f"worker_{worker_id}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    np.save(out / f"worker_{worker_id}_weights.npy", w)
    # Final upload to Azure Blob
    _upload_history_to_blob(worker_id, history_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--id",              type=int,   default=int(os.environ.get("WORKER_ID", 0)))
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--straggler_delay", type=float, default=0.0,
                   help="Extra sleep per epoch to simulate a straggler worker")
    args = p.parse_args()
    train(args.id, args.epochs, args.batch_size, args.straggler_delay)