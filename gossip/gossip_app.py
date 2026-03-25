"""
gossip/gossip_app.py
Live Gossip Protocol Training Dashboard
Run with: python gossip/gossip_app.py
"""
from flask import Flask, render_template_string, jsonify, request
import json, os, threading, time, math, random
import numpy as np

app = Flask(__name__)

# ── In-memory state ────────────────────────────────────────────────
state = {
    "running": False,
    "round": 0,
    "workers": {
        0: {"status": "idle", "rmse": None, "gossips": 0, "port": 60000},
        1: {"status": "idle", "rmse": None, "gossips": 0, "port": 60001},
    },
    "loss_history": [],
    "gossip_events": [],   # list of {round, from, to, delta}
    "start_time": None,
    "weights": [np.zeros(4).tolist(), np.zeros(4).tolist()],
    "x_mean": [30.0, 0.0, 0.0],
    "x_std":  [50.0, 45.0, 90.0],
}
state_lock = threading.Lock()

# ── Load real shard data ───────────────────────────────────────────
def load_shard(shard_id):
    """Try multiple paths to find the shard CSV."""
    import pandas as pd
    candidates = [
        f"data_shards/shard_{shard_id}.csv",
        f"data/shard_{shard_id}.csv",
        f"../data_shards/shard_{shard_id}.csv",
        f"../data/shard_{shard_id}.csv",
        os.path.join(os.path.dirname(__file__), f"../data_shards/shard_{shard_id}.csv"),
        os.path.join(os.path.dirname(__file__), f"../data/shard_{shard_id}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path).dropna()
            # normalise column names
            rename = {}
            for c in df.columns:
                cl = c.lower()
                if "mag" in cl:  rename[c] = "Magnitude"
                if "dep" in cl:  rename[c] = "Depth"
                if "lat" in cl:  rename[c] = "Latitude"
                if "lon" in cl:  rename[c] = "Longitude"
            df = df.rename(columns=rename)
            df = df[["Depth","Latitude","Longitude","Magnitude"]].dropna()
            X = df[["Depth","Latitude","Longitude"]].values.astype(np.float32)
            y = df["Magnitude"].values.astype(np.float32)
            # standardise — save mean/std for prediction
            x_mean = X.mean(axis=0)
            x_std  = X.std(axis=0) + 1e-8
            X_norm = (X - x_mean) / x_std
            X_bias = np.hstack([X_norm, np.ones((len(X_norm),1), dtype=np.float32)])
            print(f"[Gossip] Loaded shard_{shard_id}: {len(y)} rows from {path}")
            return X_bias, y, x_mean, x_std
    print(f"[Gossip] WARNING: shard_{shard_id}.csv not found, using synthetic data")
    return None, None, None, None

# ── Simulation (runs in background thread) ─────────────────────────
def simulation_loop():
    rng = np.random.default_rng(42)

    # Try loading real shards
    X0, y0, xm0, xs0 = load_shard(0)
    X1, y1, xm1, xs1 = load_shard(1)

    # Fall back to synthetic if shards not found
    if X0 is None or X1 is None:
        N = 5000
        X_syn = rng.normal(0, 1, (N, 4)).astype(np.float32); X_syn[:, 3] = 1.0
        true_w = rng.normal(0, 0.3, 4).astype(np.float32)
        y_syn = (X_syn @ true_w + rng.normal(0, 0.1, N)).astype(np.float32)
        halves = np.array_split(np.arange(N), 2)
        X0, y0 = X_syn[halves[0]], y_syn[halves[0]]
        X1, y1 = X_syn[halves[1]], y_syn[halves[1]]
        xm0 = xm1 = np.array([30.0, 0.0, 0.0])
        xs0 = xs1 = np.array([50.0, 45.0, 90.0])

    # Store normalization stats for prediction
    with state_lock:
        state["x_mean"] = xm0.tolist()
        state["x_std"]  = xs0.tolist()

    shards_X = [X0, X1]
    shards_y = [y0, y1]
    weights   = [np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]
    lr = 0.001
    ep = 0

    while True:
        with state_lock:
            if not state["running"]:
                time.sleep(0.3)
                continue

        ep += 1
        ep_rmse = []
        for i in range(2):
            idx = rng.integers(0, len(shards_y[i]), 32)
            Xb, yb = shards_X[i][idx], shards_y[i][idx]
            err = Xb @ weights[i] - yb
            g = Xb.T @ err / 32
            weights[i] -= lr * g
            rmse = math.sqrt(float(np.mean(err**2)))
            ep_rmse.append(rmse)

        # Gossip exchange
        old_0 = weights[0].copy()
        weights[0] = 0.5 * weights[0] + 0.5 * weights[1]
        weights[1] = 0.5 * weights[1] + 0.5 * old_0
        delta = float(np.linalg.norm(weights[0] - old_0))

        avg_rmse = float(np.mean(ep_rmse))

        with state_lock:
            state["round"] = ep
            state["loss_history"] = (state["loss_history"] + [{"round": ep, "loss": avg_rmse}])[-20:]
            state["gossip_events"] = (state["gossip_events"] + [{"round": ep, "from": 0, "to": 1, "delta": round(delta, 5)}])[-10:]
            state["weights"] = [weights[0].tolist(), weights[1].tolist()]
            for i in range(2):
                state["workers"][i]["status"] = "training"
                state["workers"][i]["rmse"] = round(ep_rmse[i], 4)
                state["workers"][i]["gossips"] = ep

        time.sleep(0.05)

threading.Thread(target=simulation_loop, daemon=True).start()

# ── HTML ────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Gossip Protocol Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:     #03060d;
  --bg2:    #080f1e;
  --bg3:    #0d1828;
  --border: #112240;
  --cyan:   #00e5ff;
  --teal:   #00e5c8;
  --orange: #ff6b35;
  --purple: #a78bfa;
  --green:  #4ade80;
  --muted:  #3d5470;
  --text:   #c8d8ee;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:'Syne',sans-serif; min-height:100vh; overflow-x:hidden; }

/* animated background grid */
body::before {
  content:'';
  position:fixed; inset:0; z-index:0;
  background-image:
    linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events:none;
}

header {
  position:relative; z-index:10;
  background:rgba(8,15,30,0.95);
  border-bottom:1px solid var(--border);
  padding:18px 36px;
  display:flex; align-items:center; justify-content:space-between;
  backdrop-filter:blur(10px);
}
.logo { font-size:1.2rem; font-weight:800; letter-spacing:0.05em; }
.logo span { color:var(--teal); }
.badge {
  font-family:'Space Mono',monospace; font-size:0.7rem;
  padding:5px 14px; border-radius:3px;
  border:1px solid var(--teal); color:var(--teal);
  background:rgba(0,229,200,0.05);
  animation: pulse 2s infinite;
}
.badge.stopped { border-color:var(--muted); color:var(--muted); animation:none; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

main { position:relative; z-index:1; max-width:1200px; margin:0 auto; padding:32px 24px; }

/* Controls */
.controls { display:flex; gap:12px; margin-bottom:28px; }
.btn {
  font-family:'Space Mono',monospace; font-size:0.75rem; font-weight:700;
  padding:11px 24px; border-radius:3px; border:none; cursor:pointer;
  text-decoration:none; transition:all 0.2s;
}
.btn-start { background:var(--teal); color:#000; }
.btn-start:hover { background:#00ffdd; box-shadow:0 0 20px rgba(0,229,200,0.4); }
.btn-stop { background:transparent; border:1px solid var(--orange); color:var(--orange); }
.btn-stop:hover { background:rgba(255,107,53,0.1); }
.btn-reset { background:transparent; border:1px solid var(--border); color:var(--muted); }
.btn-reset:hover { border-color:var(--text); color:var(--text); }

/* Stats row */
.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:24px; }
.stat {
  background:var(--bg2); border:1px solid var(--border); border-radius:6px;
  padding:20px; position:relative; overflow:hidden;
}
.stat::after {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
}
.stat.s-cyan::after  { background:var(--cyan); }
.stat.s-teal::after  { background:var(--teal); }
.stat.s-orange::after{ background:var(--orange); }
.stat.s-purple::after{ background:var(--purple); }
.stat-val { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; margin-bottom:4px; }
.stat-lbl { font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.12em; }
.s-cyan .stat-val   { color:var(--cyan); }
.s-teal .stat-val   { color:var(--teal); }
.s-orange .stat-val { color:var(--orange); }
.s-purple .stat-val { color:var(--purple); }

/* Two column layout */
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px; }

/* Panel */
.panel {
  background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:24px;
}
.panel-title {
  font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--muted);
  letter-spacing:0.12em; margin-bottom:18px; display:flex; align-items:center; gap:8px;
}
.panel-title::before { content:''; display:block; width:16px; height:1px; background:var(--teal); }

/* Loss chart */
.loss-row { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.loss-round { font-family:'Space Mono',monospace; font-size:0.62rem; color:var(--muted); width:64px; }
.bar-wrap { flex:1; height:5px; background:rgba(255,255,255,0.04); border-radius:2px; }
.bar-fill { height:100%; border-radius:2px; background:linear-gradient(90deg,var(--teal),var(--cyan)); transition:width 0.4s; }
.loss-num { font-family:'Space Mono',monospace; font-size:0.62rem; color:var(--orange); width:54px; text-align:right; }

/* Gossip network viz */
.network { position:relative; height:200px; }
svg.net { width:100%; height:100%; }

/* Worker cards */
.workers { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.worker {
  background:var(--bg3); border:1px solid var(--border); border-radius:6px; padding:16px;
  position:relative; overflow:hidden; transition:border-color 0.3s;
}
.worker.active { border-color:rgba(0,229,200,0.3); }
.worker-id { font-weight:800; font-size:1rem; margin-bottom:4px; }
.worker-rmse { font-family:'Space Mono',monospace; font-size:0.75rem; color:var(--orange); margin-bottom:8px; }
.worker-gossips { font-family:'Space Mono',monospace; font-size:0.62rem; color:var(--muted); }
.worker-dot {
  position:absolute; top:16px; right:16px; width:8px; height:8px;
  border-radius:50%; background:var(--muted);
}
.worker-dot.active { background:var(--green); box-shadow:0 0 8px var(--green); animation:pulse 1.5s infinite; }

/* Gossip event log */
.event-log { font-family:'Space Mono',monospace; font-size:0.65rem; }
.event-row {
  display:flex; align-items:center; gap:8px; padding:7px 0;
  border-bottom:1px solid rgba(255,255,255,0.04); color:var(--muted);
}
.event-row:last-child { border-bottom:none; }
.event-round { color:var(--teal); width:60px; }
.event-arrow { color:var(--orange); }
.event-delta { color:var(--purple); margin-left:auto; }

/* Predict */
.predict-form { display:flex; flex-direction:column; gap:14px; }
.inp-group label { display:block; font-size:0.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:5px; }
.inp-row3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
input[type=number] {
  width:100%; background:var(--bg); border:1px solid var(--border); color:var(--text);
  padding:9px 12px; border-radius:3px; font-family:'Space Mono',monospace; font-size:0.8rem;
  outline:none; transition:border-color 0.2s;
}
input[type=number]:focus { border-color:var(--teal); }
.predict-result {
  font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700;
  color:var(--teal); text-align:center; padding:16px;
  background:var(--bg); border-radius:4px; border:1px solid var(--border);
}
.predict-sub { text-align:center; font-size:0.65rem; color:var(--muted); margin-top:4px; }

/* Comparison table */
.compare-table { width:100%; border-collapse:collapse; font-size:0.75rem; }
.compare-table th {
  font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.1em;
  color:var(--muted); text-transform:uppercase; padding:8px 12px; text-align:left;
  border-bottom:1px solid var(--border);
}
.compare-table td { padding:9px 12px; border-bottom:1px solid rgba(255,255,255,0.04); }
.compare-table tr:last-child td { border-bottom:none; }
.tag-good { color:var(--green); }
.tag-warn { color:var(--orange); }
.tag-neutral { color:var(--purple); }

</style>
</head>
<body>

<header>
  <div class="logo">🔗 <span>GOSSIP</span> PROTOCOL DASHBOARD</div>
  <div id="badge" class="badge stopped">● STOPPED</div>
</header>

<main>

  <!-- Controls -->
  <div class="controls">
    <button class="btn btn-start" onclick="startTraining()">▶ START GOSSIP</button>
    <button class="btn btn-stop"  onclick="stopTraining()">■ STOP</button>
    <button class="btn btn-reset" onclick="location.reload()">↺ RESET</button>
  </div>

  <!-- Stats -->
  <div class="stats">
    <div class="stat s-cyan">
      <div class="stat-val" id="stat-round">0</div>
      <div class="stat-lbl">Gossip Rounds</div>
    </div>
    <div class="stat s-teal">
      <div class="stat-val" id="stat-gossips">0</div>
      <div class="stat-lbl">Weight Exchanges</div>
    </div>
    <div class="stat s-orange">
      <div class="stat-val" id="stat-rmse">—</div>
      <div class="stat-lbl">Latest RMSE</div>
    </div>
    <div class="stat s-purple">
      <div class="stat-val" id="stat-workers">2</div>
      <div class="stat-lbl">Active Workers</div>
    </div>
  </div>

  <div class="grid2">

    <!-- Loss Chart -->
    <div class="panel">
      <div class="panel-title">LOSS CURVE</div>
      <div id="loss-chart"></div>
    </div>

    <!-- Network Viz -->
    <div class="panel">
      <div class="panel-title">GOSSIP NETWORK</div>
      <div class="network">
        <svg class="net" viewBox="0 0 300 180">
          <!-- connection line -->
          <line id="conn-line" x1="75" y1="90" x2="225" y2="90"
                stroke="#112240" stroke-width="2" stroke-dasharray="6,4"/>
          <!-- animated gossip pulse -->
          <circle id="pulse1" r="5" fill="none" stroke="var(--teal)" stroke-width="1.5" opacity="0">
            <animate id="anim1" attributeName="cx" from="75" to="225" dur="1s" repeatCount="indefinite" begin="indefinite"/>
            <animate attributeName="cy" from="90" to="90" dur="1s" repeatCount="indefinite" begin="anim1.begin"/>
            <animate attributeName="opacity" values="0;1;1;0" dur="1s" repeatCount="indefinite" begin="anim1.begin"/>
          </circle>
          <!-- worker nodes -->
          <circle cx="75" cy="90" r="32" fill="#0d1828" stroke="#112240" stroke-width="1.5" id="node0"/>
          <circle cx="225" cy="90" r="32" fill="#0d1828" stroke="#112240" stroke-width="1.5" id="node1"/>
          <text x="75" y="86" text-anchor="middle" fill="#c8d8ee" font-family="Syne" font-size="11" font-weight="700">W0</text>
          <text x="225" y="86" text-anchor="middle" fill="#c8d8ee" font-family="Syne" font-size="11" font-weight="700">W1</text>
          <text x="75"  y="102" text-anchor="middle" fill="#3d5470" font-family="Space Mono" font-size="7" id="rmse0">rmse=—</text>
          <text x="225" y="102" text-anchor="middle" fill="#3d5470" font-family="Space Mono" font-size="7" id="rmse1">rmse=—</text>
          <!-- port labels -->
          <text x="75"  y="136" text-anchor="middle" fill="#1a3a5c" font-family="Space Mono" font-size="7">:60000</text>
          <text x="225" y="136" text-anchor="middle" fill="#1a3a5c" font-family="Space Mono" font-size="7">:60001</text>
          <!-- gossip label -->
          <text x="150" y="78" text-anchor="middle" fill="#3d5470" font-family="Space Mono" font-size="7" id="gossip-label">⇄ weights</text>
        </svg>
      </div>
    </div>

  </div>

  <div class="grid2">

    <!-- Workers -->
    <div class="panel">
      <div class="panel-title">WORKER NODES</div>
      <div class="workers">
        <div class="worker" id="w0card">
          <div class="worker-dot" id="w0dot"></div>
          <div class="worker-id" style="color:var(--teal)">⚙ Worker 0</div>
          <div class="worker-rmse" id="w0rmse">RMSE: —</div>
          <div class="worker-gossips" id="w0gossips">exchanges: 0</div>
          <div style="font-size:0.62rem;color:var(--muted);margin-top:6px;">shard_0.csv · port :60000</div>
        </div>
        <div class="worker" id="w1card">
          <div class="worker-dot" id="w1dot"></div>
          <div class="worker-id" style="color:var(--cyan)">⚙ Worker 1</div>
          <div class="worker-rmse" id="w1rmse">RMSE: —</div>
          <div class="worker-gossips" id="w1gossips">exchanges: 0</div>
          <div style="font-size:0.62rem;color:var(--muted);margin-top:6px;">shard_1.csv · port :60001</div>
        </div>
      </div>
    </div>

    <!-- Gossip Event Log -->
    <div class="panel">
      <div class="panel-title">GOSSIP EVENT LOG</div>
      <div class="event-log" id="event-log">
        <div style="color:var(--muted);font-size:0.7rem;">Waiting for gossip events...</div>
      </div>
    </div>

  </div>

  <div class="grid2">

    <!-- Predict -->
    <div class="panel">
      <div class="panel-title">PREDICT MAGNITUDE</div>
      <div class="predict-form">
        <div class="inp-group">
          <div class="inp-row3">
            <div>
              <label>Depth (km)</label>
              <input type="number" id="inp-depth" value="10" step="0.1"/>
            </div>
            <div>
              <label>Latitude</label>
              <input type="number" id="inp-lat" value="35.0" step="0.01"/>
            </div>
            <div>
              <label>Longitude</label>
              <input type="number" id="inp-lon" value="140.0" step="0.01"/>
            </div>
          </div>
        </div>
        <button class="btn btn-start" onclick="predict()" style="align-self:flex-start">🔮 PREDICT</button>
        <div id="predict-result" style="display:none">
          <div class="predict-result" id="pred-val">—</div>
          <div class="predict-sub">Predicted Magnitude (averaged gossip weights)</div>
        </div>
      </div>
    </div>

    <!-- Comparison Table -->
    <div class="panel">
      <div class="panel-title">GOSSIP VS PARAMETER SERVER</div>
      <table class="compare-table">
        <thead>
          <tr><th>Property</th><th>Gossip</th><th>PS</th></tr>
        </thead>
        <tbody>
          <tr><td>Architecture</td><td class="tag-good">Decentralized</td><td class="tag-warn">Centralized</td></tr>
          <tr><td>Single Point of Failure</td><td class="tag-good">No ✅</td><td class="tag-warn">Yes ⚠️</td></tr>
          <tr><td>Scalability</td><td class="tag-good">Linear</td><td class="tag-warn">PS bottleneck</td></tr>
          <tr><td>Convergence</td><td class="tag-warn">Slower</td><td class="tag-good">Faster</td></tr>
          <tr><td>Communication</td><td class="tag-neutral">Peer-to-peer</td><td class="tag-neutral">Hub-spoke</td></tr>
          <tr><td>Real-world use</td><td class="tag-neutral">Bitcoin, Cassandra</td><td class="tag-neutral">TF ParameterServer</td></tr>
        </tbody>
      </table>
    </div>

  </div>

</main>

<script>
let running = false;

async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    updateUI(d);
  } catch(e) {}
}

function updateUI(d) {
  // Stats
  document.getElementById('stat-round').textContent = d.round;
  document.getElementById('stat-gossips').textContent = d.workers[0]?.gossips || 0;
  if (d.loss_history.length) {
    const last = d.loss_history[d.loss_history.length-1].loss;
    document.getElementById('stat-rmse').textContent = last.toFixed(4);
  }

  // Badge
  const badge = document.getElementById('badge');
  if (d.running) {
    badge.textContent = '● GOSSIP ACTIVE';
    badge.className = 'badge';
  } else {
    badge.textContent = '● STOPPED';
    badge.className = 'badge stopped';
  }

  // Loss chart
  const history = d.loss_history;
  if (history.length) {
    const maxL = Math.max(...history.map(h=>h.loss));
    document.getElementById('loss-chart').innerHTML = history.map(h => `
      <div class="loss-row">
        <span class="loss-round">R${h.round}</span>
        <div class="bar-wrap"><div class="bar-fill" style="width:${Math.max(2,(h.loss/maxL*100)).toFixed(0)}%"></div></div>
        <span class="loss-num">${h.loss.toFixed(4)}</span>
      </div>`).join('');
  }

  // Workers
  for (let i = 0; i < 2; i++) {
    const w = d.workers[i];
    if (!w) continue;
    const active = d.running;
    document.getElementById(`w${i}card`).className = 'worker' + (active ? ' active' : '');
    document.getElementById(`w${i}dot`).className = 'worker-dot' + (active ? ' active' : '');
    document.getElementById(`w${i}rmse`).textContent = `RMSE: ${w.rmse ?? '—'}`;
    document.getElementById(`w${i}gossips`).textContent = `exchanges: ${w.gossips}`;
    document.getElementById(`rmse${i}`).textContent = `rmse=${w.rmse ?? '—'}`;
  }

  // Node colors when active
  const c = d.running ? '#0d2a1e' : '#0d1828';
  const s = d.running ? 'rgba(0,229,200,0.4)' : '#112240';
  document.getElementById('node0').setAttribute('fill', c);
  document.getElementById('node1').setAttribute('fill', c);
  document.getElementById('node0').setAttribute('stroke', s);
  document.getElementById('node1').setAttribute('stroke', s);
  document.getElementById('conn-line').setAttribute('stroke', d.running ? '#1a4a3a' : '#112240');

  // Gossip label
  if (d.round > 0) {
    document.getElementById('gossip-label').textContent = `⇄ round ${d.round}`;
  }

  // Event log
  const events = d.gossip_events;
  if (events.length) {
    document.getElementById('event-log').innerHTML = [...events].reverse().map(e => `
      <div class="event-row">
        <span class="event-round">R${e.round}</span>
        <span>W${e.from}</span>
        <span class="event-arrow">⇄</span>
        <span>W${e.to}</span>
        <span class="event-delta">Δ${e.delta}</span>
      </div>`).join('');
  }
}

async function startTraining() {
  await fetch('/api/start', {method:'POST'});
  running = true;
  // animate gossip pulse
  document.getElementById('anim1').beginElement();
}

async function stopTraining() {
  await fetch('/api/stop', {method:'POST'});
}

async function predict() {
  const depth = parseFloat(document.getElementById('inp-depth').value);
  const lat   = parseFloat(document.getElementById('inp-lat').value);
  const lon   = parseFloat(document.getElementById('inp-lon').value);
  const r = await fetch('/api/predict', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({depth, lat, lon})
  });
  const d = await r.json();
  if (d.prediction !== null) {
    document.getElementById('predict-result').style.display = 'block';
    document.getElementById('pred-val').textContent = d.prediction.toFixed(2) + ' M';
  }
}

// Poll every second
setInterval(fetchStatus, 1000);
fetchStatus();
</script>
</body>
</html>"""

# ── Routes ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify({
            "running": state["running"],
            "round": state["round"],
            "workers": state["workers"],
            "loss_history": state["loss_history"],
            "gossip_events": state["gossip_events"],
        })

@app.route("/api/start", methods=["POST"])
def api_start():
    with state_lock:
        state["running"] = True
        state["start_time"] = time.time()
    return jsonify({"ok": True})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state_lock:
        state["running"] = False
    return jsonify({"ok": True})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    depth = float(data.get("depth", 10))
    lat   = float(data.get("lat", 35.0))
    lon   = float(data.get("lon", 140.0))

    with state_lock:
        w0     = np.array(state["weights"][0])
        w1     = np.array(state["weights"][1])
        x_mean = np.array(state.get("x_mean", [30.0, 0.0, 0.0]))
        x_std  = np.array(state.get("x_std",  [50.0, 45.0, 90.0]))

    # average of both gossip workers
    w_avg = (w0 + w1) / 2
    if np.all(w_avg == 0):
        return jsonify({"prediction": None})

    X = np.array([depth, lat, lon], dtype=np.float32)
    X = (X - x_mean) / (x_std + 1e-8)
    prediction = float(np.dot(X, w_avg[:3]) + w_avg[3])
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    port = int(os.environ.get("GOSSIP_PORT", 5001))
    print(f"\n{'='*50}")
    print(f"  Gossip Protocol Dashboard")
    print(f"  http://localhost:{port}")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
