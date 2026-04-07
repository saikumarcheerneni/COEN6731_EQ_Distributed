"""
core/parameter_server.py — Azure-ready (reads ENV vars)
"""
import json, socket, threading, time, logging, os, gc
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PS] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("PS")

PS_HOST           = "0.0.0.0"
PS_PORT           = int(os.environ.get("PS_PORT", 50051))
NUM_WORKERS       = int(os.environ.get("NUM_WORKERS", 2))
NUM_FEATURES      = 3
LEARNING_RATE     = float(os.environ.get("LR", 0.001))
STRAGGLER_TIMEOUT = int(os.environ.get("STRAGGLER_TIMEOUT", 30))
CHECKPOINT_DIR    = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
MAX_ROUNDS        = int(os.environ.get("MAX_ROUNDS", 1000))
MEMORY_LOG_INTERVAL = 60
CHECKPOINT_DIR.mkdir(exist_ok=True)

class ParameterServer:
    def __init__(self, num_workers, mode="sync"):
        self.num_workers   = num_workers
        self.mode          = mode
        self.weights       = np.zeros(NUM_FEATURES + 1, dtype=np.float32)
        self.lock          = threading.Lock()
        self.round         = 0
        self.grad_buffer   = {}
        self.last_seen     = {}
        self.stragglers    = set()
        self.total_updates = 0
        self.loss_history  = []
        self.start_time    = time.time()
        self.done          = False
        threading.Thread(target=self._memory_watchdog, daemon=True).start()

    def set_num_workers(self, n):
        with self.lock:
            old = self.num_workers
            self.num_workers = n
            self.grad_buffer.clear()
            self.stragglers.clear()
            self.last_seen.clear()
            self.round         = 0
            self.total_updates = 0
            self.loss_history  = []
            self.weights       = np.zeros(NUM_FEATURES + 1, dtype=np.float32)
            self.done          = False
            self.start_time    = time.time()
            log.info(f"Worker count changed {old} → {n}. PS state reset.")

    def push_gradients(self, worker_id, gradients, loss):
        with self.lock:
            self.grad_buffer[worker_id] = gradients
            self.last_seen[worker_id]   = time.time()
            self.stragglers.discard(worker_id)
            if self.mode == "async":
                self._update([gradients])
                self.total_updates += 1
                if self.total_updates >= MAX_ROUNDS:
                    self.done = True
                return self.weights.copy()
            now = time.time()
            for wid, last in self.last_seen.items():
                if now - last > STRAGGLER_TIMEOUT and wid not in self.grad_buffer:
                    self.stragglers.add(wid)
                    log.warning(f"Worker {wid} STRAGGLER")
            live = set(range(self.num_workers)) - self.stragglers
            if live and live.issubset(set(self.grad_buffer.keys())):
                self._update([self.grad_buffer[w] for w in live])
                self.grad_buffer.clear()
                self.round += 1
                self.total_updates += 1
                entry = {"round": self.round, "loss": round(float(loss), 6),
                         "time": round(time.time() - self.start_time, 1)}
                self.loss_history.append(entry)
                if len(self.loss_history) > 100:
                    self.loss_history = self.loss_history[-100:]
                if self.round % 50 == 0:
                    np.save(CHECKPOINT_DIR/"weights.npy", self.weights)
                log.info(f"Round {self.round}/{MAX_ROUNDS} | loss={loss:.4f} | workers={list(live)}")
                if self.round >= MAX_ROUNDS:
                    self.done = True
                    log.info(f"Reached MAX_ROUNDS={MAX_ROUNDS}. Training complete.")
                gc.collect()
            return self.weights.copy()

    def _update(self, grads):
        self.weights -= LEARNING_RATE * np.mean(grads, axis=0)

    def _memory_watchdog(self):
        while True:
            time.sleep(MEMORY_LOG_INTERVAL)
            try:
                with open("/proc/meminfo") as f:
                    lines = f.readlines()
                mem = {}
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        mem[parts[0].rstrip(":")] = int(parts[1])
                total = mem.get("MemTotal", 1)
                avail = mem.get("MemAvailable", total)
                used_pct = (1 - avail / total) * 100
                log.info(f"Memory: {used_pct:.1f}% used | round={self.round}")
                if used_pct > 80:
                    log.warning(f"HIGH MEMORY {used_pct:.1f}% — trimming history + GC")
                    with self.lock:
                        self.loss_history = self.loss_history[-20:]
                        self.grad_buffer.clear()
                    gc.collect()
            except Exception:
                pass

    def get_weights(self):
        with self.lock: return self.weights.copy()

    def get_status(self):
        with self.lock:
            return {"round": self.round, "max_rounds": MAX_ROUNDS,
                    "total_updates": self.total_updates,
                    "num_workers": self.num_workers, "stragglers": list(self.stragglers),
                    "loss_history": self.loss_history[-20:],
                    "uptime": round(time.time() - self.start_time, 1),
                    "mode": self.mode, "done": self.done}

def handle_client(conn, addr, ps):
    data = b""
    try:
        while True:
            chunk = conn.recv(65536)
            if not chunk: break
            data += chunk
            try:
                msg = json.loads(data.decode())
                msg_type = msg.get("type", "")
                if msg_type == "push_gradients":
                    w = ps.push_gradients(msg["worker_id"], np.array(msg["gradients"], dtype=np.float32), msg.get("loss", 0))
                    conn.sendall(json.dumps({"type": "weights", "weights": w.tolist(), "done": ps.done}).encode())
                elif msg_type == "get_weights":
                    w = ps.get_weights()
                    conn.sendall(json.dumps({"type": "weights", "weights": w.tolist(), "done": ps.done}).encode())
                elif msg_type in ("get_status", "stats"):
                    conn.sendall(json.dumps({"type": "status", "data": ps.get_status()}).encode())
                elif msg_type == "set_num_workers":
                    n = int(msg.get("num_workers", 2))
                    ps.set_num_workers(n)
                    conn.sendall(json.dumps({"type": "ok", "num_workers": n}).encode())
                data = b""
            except json.JSONDecodeError: continue
    except Exception as e: log.error(f"{addr}: {e}")
    finally: conn.close()

def run_server(num_workers=NUM_WORKERS, mode=os.environ.get("MODE", "sync")):
    ps = ParameterServer(num_workers, mode)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((PS_HOST, PS_PORT))
    srv.listen(50)
    log.info(f"PS online :{PS_PORT} | workers={num_workers} | mode={mode} | max_rounds={MAX_ROUNDS}")
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle_client, args=(conn, addr, ps), daemon=True).start()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--mode", default="sync")
    args = p.parse_args()
    run_server(args.workers, args.mode)