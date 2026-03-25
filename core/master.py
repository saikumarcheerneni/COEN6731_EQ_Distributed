"""
core/master.py
─────────────────────────────────────────────────────────────────
Master coordinator — launches workers as subprocesses, monitors
them, and aggregates final weights.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MASTER] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("master")

PS_HOST = "localhost"
PS_PORT = 50051


def ps_stats() -> dict:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            s.connect((PS_HOST, PS_PORT))
            s.sendall((json.dumps({"type": "stats"}) + "\n").encode())
            data = b""
            while True:
                c = s.recv(4096)
                if not c: break
                data += c
                if data.endswith(b"\n"): break
        return json.loads(data.decode())
    except Exception:
        return {}


def stream(proc, label):
    for line in iter(proc.stdout.readline, b""):
        print(f"  {label} | {line.decode().rstrip()}", flush=True)


def aggregate_weights(num_workers: int) -> np.ndarray | None:
    ws = []
    for i in range(num_workers):
        p = Path("outputs") / f"worker_{i}_weights.npy"
        if p.exists():
            ws.append(np.load(str(p)))
            log.info(f"Loaded weights from Worker {i}")
        else:
            log.warning(f"Missing weights for Worker {i}")
    if not ws:
        return None
    final = np.mean(ws, axis=0)
    Path("outputs").mkdir(exist_ok=True)
    np.save("outputs/final_model_weights.npy", final)
    log.info(f"Final aggregated weights: {final.round(4)}")
    return final


def run(num_workers: int, epochs: int, batch_size: int,
        mode: str, straggler_worker: int = -1,
        straggler_delay: float = 5.0):

    # Verify shards
    for i in range(num_workers):
        if not Path(f"data_shards/shard_{i}.csv").exists():
            log.error(f"Missing shard_{i}.csv — run prepare_data.py first")
            sys.exit(1)

    log.info(f"Starting {num_workers} workers  "
             f"(epochs={epochs}, mode={mode})")
    t0     = time.time()
    procs  = []
    threads= []

    for i in range(num_workers):
        delay = straggler_delay if i == straggler_worker else 0.0
        cmd   = [
            sys.executable, "core/worker.py",
            "--id",         str(i),
            "--epochs",     str(epochs),
            "--batch_size", str(batch_size),
            "--straggler_delay", str(delay),
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append(proc)
        t = threading.Thread(
            target=stream, args=(proc, f"W{i}"), daemon=True)
        t.start(); threads.append(t)
        time.sleep(0.3)

    # Wait
    log.info("Waiting for workers to finish...")
    for i, proc in enumerate(procs):
        proc.wait()
        log.info(f"Worker {i} exited (code={proc.returncode})")
    for t in threads:
        t.join(timeout=3)

    elapsed = time.time() - t0
    log.info(f"All workers done in {elapsed:.1f}s")

    # Aggregate
    final = aggregate_weights(num_workers)

    # PS stats
    stats = ps_stats()
    if stats:
        log.info(f"PS stats: rounds={stats.get('rounds')}  "
                 f"elapsed={stats.get('elapsed_sec')}s")

    print("\n" + "="*55)
    print("  TRAINING COMPLETE")
    print("="*55)
    print(f"  Workers      : {num_workers}")
    print(f"  Epochs       : {epochs}")
    print(f"  Mode         : {mode}")
    print(f"  Total time   : {elapsed:.1f}s")
    if final is not None:
        print(f"  Final weights: {final.round(4)}")
    print(f"  Model saved  : outputs/final_model_weights.npy")
    print("="*55)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers",          type=int,   default=2)
    p.add_argument("--epochs",           type=int,   default=20)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--mode",             default="sync",
                   choices=["sync","async"])
    p.add_argument("--straggler_worker", type=int,   default=-1,
                   help="Worker ID to simulate as straggler (-1=none)")
    p.add_argument("--straggler_delay",  type=float, default=5.0,
                   help="Seconds of extra delay per epoch for straggler")
    args = p.parse_args()
    run(args.workers, args.epochs, args.batch_size,
        args.mode, args.straggler_worker, args.straggler_delay)
