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
        log.info(f"Downloaded shard_{worker_id}.csv — ready to train!")
        return local_path
    except ImportError:
        log.warning("azure-storage-blob not installed — using local shard")
        return local_path
    except Exception as e:
        log.warning(f"Blob download failed: {e} — falling back to local shard")
        return local_path


def _upload_history_to_blob(worker_id, history_path):
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
        logging.getLogger("W").warning(f"Failed to upload history: {e}")


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


def send_recv_timed(host, port, msg, timeout=30):
    """Send message and return (response, comm_time_ms)."""
    t_start = time.time()
    resp    = send_recv(host, port, msg, timeout)
    comm_ms = (time.time() - t_start) * 1000
    return resp, comm_ms


def connect_to_ps(worker_id):
    """Retry PS connection forever. Skip if PS is still in done state."""
    log     = logging.getLogger("W")
    wait    = RECONNECT_BASE
    attempt = 0
    while True:
        try:
            resp = send_recv(PS_HOST, PS_PORT, {"type": "get_weights"})
            if resp.get("done"):
                log.info("PS still in done state — waiting for reset...")
                time.sleep(15)
                continue
            w = np.array(resp["weights"], dtype=np.float32)
            log.info(f"Connected to PS at {PS_HOST}:{PS_PORT}")
            return w
        except Exception as e:
            attempt += 1
            log.info(f"PS not ready (attempt {attempt}), retrying in {wait:.1f}s — {e}")
            time.sleep(wait)
            wait = min(wait * 1.5, MAX_RECONNECT_WAIT)


def train_once(worker_id, epochs, batch_size, straggler_delay):
    """Run one full training session. Always saves history."""
    log = logging.LoggerAdapter(logging.getLogger("W"), {"worker_id": worker_id})

    data_path = download_shard_from_blob(worker_id)

    retries = 0
    while not Path(data_path).exists() and retries < 10:
        log.info(f"Waiting for shard file {data_path}...")
        time.sleep(5)
        data_path = download_shard_from_blob(worker_id)
        retries += 1

    if not Path(data_path).exists():
        log.error(f"Shard file not found: {data_path}")
        return False

    df = pd.read_csv(data_path).dropna()
    X  = df[["Depth", "Latitude", "Longitude"]].values.astype(np.float32)
    y  = df["Magnitude"].values.astype(np.float32)
    X  = (X - X.mean(0)) / (X.std(0) + 1e-8)
    log.info(f"Loaded {len(df):,} rows from {data_path}")

    # Clear old checkpoint — fresh training each session
    ckpt = CHECKPOINT_DIR / f"worker_{worker_id}.npy"
    if ckpt.exists():
        ckpt.unlink()

    w = connect_to_ps(worker_id)

    rmse         = 0.0
    history      = []
    ps_done      = False
    comm_times   = []   # track PS communication latency

    for epoch in range(1, epochs + 1):

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

            reconnect_wait = RECONNECT_BASE
            for _ in range(20):
                try:
                    resp, comm_ms = send_recv_timed(PS_HOST, PS_PORT, {
                        "type":      "push_gradients",
                        "worker_id": worker_id,
                        "gradients": grads.tolist(),
                        "loss":      loss
                    })
                    comm_times.append(comm_ms)
                    w = np.array(resp["weights"], dtype=np.float32)
                    if resp.get("done"):
                        ps_done = True
                    break
                except Exception as e:
                    log.warning(f"PS comm error: {e} — retrying in {reconnect_wait:.1f}s")
                    time.sleep(reconnect_wait)
                    reconnect_wait = min(reconnect_wait * 2, MAX_RECONNECT_WAIT)

        avg_loss = epoch_loss / batches
        rmse     = float(np.sqrt(np.mean((predict(Xs, w) - ys)**2)))
        avg_comm = sum(comm_times[-batches:]) / max(len(comm_times[-batches:]), 1)
        log.info(f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | RMSE={rmse:.4f} | avg_comm={avg_comm:.1f}ms")
        history.append({"epoch": epoch, "loss": avg_loss, "rmse": rmse, "avg_comm_ms": round(avg_comm, 2)})

        np.save(ckpt, {"epoch": epoch, "weights": w.tolist(), "history": history})

        out = Path("outputs"); out.mkdir(exist_ok=True)
        history_path = out / f"worker_{worker_id}_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)
        _upload_history_to_blob(worker_id, history_path)

        if ps_done:
            log.info("PS training complete — saved history and exiting epoch loop.")
            break

    # Log overall communication stats
    if comm_times:
        log.info(f"PS comm stats — avg: {sum(comm_times)/len(comm_times):.1f}ms | "
                 f"min: {min(comm_times):.1f}ms | max: {max(comm_times):.1f}ms")

    log.info(f"Session complete | final RMSE={rmse:.4f}")
    _save_outputs(worker_id, w, history)
    return True


def _save_outputs(worker_id, w, history):
    out = Path("outputs"); out.mkdir(exist_ok=True)
    history_path = out / f"worker_{worker_id}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    np.save(out / f"worker_{worker_id}_weights.npy", w)
    _upload_history_to_blob(worker_id, history_path)


def run_forever(worker_id, epochs, batch_size, straggler_delay):
    """
    Run training sessions forever.
    After each session — wait for PS to restart then retrain.
    VM restart = automatic retraining. No manual steps needed.
    """
    log     = logging.getLogger("W")
    session = 0

    while True:
        session += 1
        log.info(f"=== Training session {session} starting ===")

        try:
            train_once(worker_id, epochs, batch_size, straggler_delay)
        except Exception as e:
            log.error(f"Session {session} crashed: {e} — restarting in 30s")
            time.sleep(30)
            continue

        log.info("Session complete. Waiting for PS to restart for next session...")

        while True:
            try:
                resp = send_recv(PS_HOST, PS_PORT, {"type": "get_status"}, timeout=10)
                data = resp.get("data", resp)
                ps_done  = data.get("done", True)
                ps_round = data.get("round", 0)
                if not ps_done and ps_round == 0:
                    log.info("PS has reset — starting new training session!")
                    break
                log.info(f"PS still done (round={ps_round}) — waiting 30s...")
                time.sleep(30)
            except Exception:
                log.info("PS not reachable — waiting 15s...")
                time.sleep(15)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--id",              type=int,   default=int(os.environ.get("WORKER_ID", 0)))
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=256)
    p.add_argument("--straggler_delay", type=float, default=0.0)
    args = p.parse_args()
    run_forever(args.id, args.epochs, args.batch_size, args.straggler_delay)