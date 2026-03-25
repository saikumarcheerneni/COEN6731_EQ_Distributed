"""
gossip/gossip_worker.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gossip Protocol Worker — Decentralized Distributed Training

ఇది ఎలా పని చేస్తుందో:
  - PS లేదు. ప్రతి Worker directly తన peers తో మాట్లాడుతుంది.
  - ప్రతి round: నా weights → random peer కి పంపు
                 peer weights → నా weights తో average చేయి
  - Eventually అందరి weights converge అవుతాయి!

Real world: Bitcoin nodes, Cassandra DB ఇలాగే పని చేస్తాయి.

Usage:
    # Terminal 1:
    python gossip/gossip_worker.py --id 0 --peers 1,2 --epochs 20

    # Terminal 2:
    python gossip/gossip_worker.py --id 1 --peers 0,2 --epochs 20

    # Terminal 3 (optional):
    python gossip/gossip_worker.py --id 2 --peers 0,1 --epochs 20
"""

import argparse
import json
import logging
import math
import os
import random
import socket
import threading
import time
from pathlib import Path

import numpy as np

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [W%(message)s",
    datefmt="%H:%M:%S",
)

# ── Constants ──────────────────────────────────────────────────────
BASE_PORT      = 60000          # Worker i listens on 60000+i
LEARNING_RATE  = 0.001
BATCH_SIZE     = 32
NUM_FEATURES   = 3              # depth, lat, lon
GOSSIP_ALPHA   = 0.5            # averaging weight: 0.5 = equal mix
GOSSIP_TIMEOUT = 5.0            # seconds to wait for peer response
CHECKPOINT_DIR = Path("checkpoints/gossip")

# ── Send + Receive helpers ─────────────────────────────────────────

def send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode()
    # 4-byte length prefix so receiver knows message boundary
    sock.sendall(len(data).to_bytes(4, "big") + data)


def recv_msg(sock: socket.socket) -> dict | None:
    try:
        raw_len = _recv_exact(sock, 4)
        if not raw_len:
            return None
        length = int.from_bytes(raw_len, "big")
        raw    = _recv_exact(sock, length)
        return json.loads(raw.decode())
    except Exception:
        return None


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


# ── Data helpers ───────────────────────────────────────────────────

def load_shard(worker_id: int):
    path = Path(f"data/shard_{worker_id}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_data.py first."
        )
    import pandas as pd
    df = pd.read_csv(path).dropna()

    # normalise column names
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "mag"  in cl: rename[c] = "Magnitude"
        if "dep"  in cl: rename[c] = "Depth"
        if "lat"  in cl: rename[c] = "Latitude"
        if "lon"  in cl: rename[c] = "Longitude"
    df = df.rename(columns=rename)

    needed = ["Depth", "Latitude", "Longitude", "Magnitude"]
    df = df[needed].dropna()

    X = df[["Depth", "Latitude", "Longitude"]].values.astype(float)
    y = df["Magnitude"].values.astype(float)

    # standardise
    x_mean = X.mean(axis=0)
    x_std  = X.std(axis=0) + 1e-8
    X = (X - x_mean) / x_std

    # add bias column
    X = np.hstack([X, np.ones((len(X), 1))])
    return X, y, x_mean, x_std


def compute_gradient(X_batch, y_batch, weights):
    preds = X_batch @ weights
    error = preds - y_batch
    grad  = X_batch.T @ error / len(y_batch)
    rmse  = math.sqrt(np.mean(error ** 2))
    return grad, rmse


def get_batches(X, y, batch_size):
    idx = np.random.permutation(len(y))
    for i in range(0, len(y) - batch_size + 1, batch_size):
        b = idx[i : i + batch_size]
        yield X[b], y[b]


# ── Checkpoint ─────────────────────────────────────────────────────

def save_checkpoint(worker_id, weights, epoch):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CHECKPOINT_DIR / f"gossip_worker_{worker_id}.npz",
        weights=weights,
        epoch=[epoch],
    )


def load_checkpoint(worker_id):
    path = CHECKPOINT_DIR / f"gossip_worker_{worker_id}.npz"
    if path.exists():
        d = np.load(path)
        return d["weights"], int(d["epoch"][0])
    return None, 0


# ══════════════════════════════════════════════════════════════════
# GossipWorker
# ══════════════════════════════════════════════════════════════════

class GossipWorker:
    """
    Each worker:
      1. Trains on its local shard (mini-batch SGD)
      2. Every epoch: picks a random peer → sends weights → receives
         peer's weights → averages them (gossip exchange)
      3. Saves checkpoint after every epoch
    """

    def __init__(self, worker_id: int, peer_ids: list[int],
                 epochs: int, gossip_rounds: int = 1):
        self.id            = worker_id
        self.peers         = peer_ids          # list of peer worker ids
        self.epochs        = epochs
        self.gossip_rounds = gossip_rounds     # how many peers to gossip per epoch
        self.port          = BASE_PORT + worker_id
        self.weights       = np.zeros(NUM_FEATURES + 1)  # +1 for bias

        # Server socket — listens for incoming gossip from peers
        self.server        = None
        self._lock         = threading.Lock()

        # Metrics
        self.history = []   # list of (epoch, rmse, n_gossips)
        self.n_gossips  = 0
        self.n_timeouts = 0

        # Load existing checkpoint
        saved_w, self.start_epoch = load_checkpoint(worker_id)
        if saved_w is not None:
            self.weights = saved_w
            logging.info(f"{worker_id}] Resumed from checkpoint epoch {self.start_epoch}")

    # ── Server thread: handles incoming gossip requests ────────────

    def _start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("localhost", self.port))
        self.server.listen(10)
        self.server.settimeout(1.0)
        logging.info(f"{self.id}] Listening on port {self.port}")

        t = threading.Thread(target=self._accept_loop, daemon=True)
        t.start()

    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self.server.accept()
                threading.Thread(
                    target=self._handle_peer,
                    args=(conn,),
                    daemon=True,
                ).start()
            except socket.timeout:
                continue
            except Exception:
                break

    def _handle_peer(self, conn: socket.socket):
        """
        Peer ని handle చేయడం:
        1. Peer నా weights అడిగింది → నా weights పంపు
        2. Peer నా weights తో gossip చేయాలి →
           peer weights receive చేసి average చేయి
        """
        try:
            msg = recv_msg(conn)
            if not msg:
                return

            if msg["type"] == "gossip_exchange":
                peer_weights = np.array(msg["weights"])

                # Send my current weights back
                with self._lock:
                    my_w = self.weights.copy()

                send_msg(conn, {
                    "type":    "gossip_response",
                    "weights": my_w.tolist(),
                    "worker":  self.id,
                })

                # Average: w_new = alpha * peer + (1-alpha) * mine
                with self._lock:
                    self.weights = (
                        GOSSIP_ALPHA * peer_weights
                        + (1 - GOSSIP_ALPHA) * self.weights
                    )
                self.n_gossips += 1

        except Exception as e:
            logging.debug(f"W{self.id}] handle_peer error: {e}")
        finally:
            conn.close()

    # ── Client: initiate gossip with a random peer ─────────────────

    def _gossip_with(self, peer_id: int) -> bool:
        """
        Send my weights to peer_id, receive their weights, average.
        Returns True on success, False on timeout/failure.
        """
        peer_port = BASE_PORT + peer_id
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(GOSSIP_TIMEOUT)
            sock.connect(("localhost", peer_port))

            with self._lock:
                my_w = self.weights.copy()

            # Initiate exchange
            send_msg(sock, {
                "type":    "gossip_exchange",
                "weights": my_w.tolist(),
                "from":    self.id,
            })

            # Receive peer weights
            resp = recv_msg(sock)
            sock.close()

            if resp and resp["type"] == "gossip_response":
                peer_w = np.array(resp["weights"])
                with self._lock:
                    self.weights = (
                        GOSSIP_ALPHA * peer_w
                        + (1 - GOSSIP_ALPHA) * self.weights
                    )
                self.n_gossips += 1
                return True

        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            self.n_timeouts += 1
            logging.warning(
                f"W{self.id}] Gossip timeout with peer {peer_id}: {e}"
            )
        return False

    # ── Training loop ──────────────────────────────────────────────

    def train(self, X, y):
        self._running = True
        self._start_server()

        # Small wait so all workers start their servers
        time.sleep(1.5)

        logging.info(
            f"{self.id}] Starting training | "
            f"epochs={self.epochs} | peers={self.peers} | "
            f"data={len(y)} rows"
        )

        for epoch in range(self.start_epoch, self.epochs):
            epoch_rmse = []

            # ── Local SGD on my shard ──────────────────────────────
            for X_b, y_b in get_batches(X, y, BATCH_SIZE):
                with self._lock:
                    w = self.weights.copy()
                grad, rmse = compute_gradient(X_b, y_b, w)
                with self._lock:
                    self.weights -= LEARNING_RATE * grad
                epoch_rmse.append(rmse)

            avg_rmse = float(np.mean(epoch_rmse))

            # ── Gossip exchange with random peers ─────────────────
            # Pick `gossip_rounds` random peers (without replacement if possible)
            sample_size = min(self.gossip_rounds, len(self.peers))
            chosen = random.sample(self.peers, sample_size)

            for peer_id in chosen:
                success = self._gossip_with(peer_id)
                status  = "✓" if success else "✗ (timeout)"
                logging.info(
                    f"{self.id}] Epoch {epoch+1:2d} | "
                    f"RMSE={avg_rmse:.4f} | "
                    f"Gossip→W{peer_id} {status} | "
                    f"total_gossips={self.n_gossips}"
                )

            # ── Save checkpoint ────────────────────────────────────
            save_checkpoint(self.id, self.weights, epoch + 1)
            self.history.append((epoch + 1, avg_rmse, self.n_gossips))

        self._running = False
        self.server.close()
        logging.info(
            f"{self.id}] Training complete | "
            f"final_RMSE={self.history[-1][1]:.4f} | "
            f"gossips={self.n_gossips} | "
            f"timeouts={self.n_timeouts}"
        )
        self._save_history()

    def _save_history(self):
        out = Path("outputs")
        out.mkdir(exist_ok=True)
        path = out / f"gossip_worker_{self.id}_history.csv"
        with open(path, "w") as f:
            f.write("epoch,rmse,total_gossips\n")
            for ep, rmse, ng in self.history:
                f.write(f"{ep},{rmse:.6f},{ng}\n")
        logging.info(f"W{self.id}] History → {path}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Gossip Protocol Worker")
    p.add_argument("--id",     type=int, required=True,
                   help="Worker ID (0, 1, 2, ...)")
    p.add_argument("--peers",  type=str, required=True,
                   help="Comma-separated peer IDs, e.g. 1,2")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--gossip_rounds", type=int, default=1,
                   help="How many peers to gossip with per epoch")
    args = p.parse_args()

    peer_ids = [int(x) for x in args.peers.split(",")]

    print(f"\n{'='*55}")
    print(f"  GOSSIP WORKER {args.id}")
    print(f"  Peers : {peer_ids}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Port  : {BASE_PORT + args.id}")
    print(f"  Mode  : Decentralized (no Parameter Server!)")
    print(f"{'='*55}\n")

    X, y, x_mean, x_std = load_shard(args.id)
    worker = GossipWorker(args.id, peer_ids, args.epochs,
                          args.gossip_rounds)
    worker.train(X, y)


if __name__ == "__main__":
    main()
