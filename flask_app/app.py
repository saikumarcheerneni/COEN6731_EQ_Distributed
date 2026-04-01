"""
flask_app/app.py
Live Earthquake Distributed Training Dashboard
Enhanced: CSV Upload + Azure Blob Storage + Training Analytics Charts
"""
from flask import Flask, render_template_string, jsonify, request, redirect, url_for
import json, socket, os, time, csv, math
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

PS_HOST         = os.environ.get("PS_HOST", "localhost")
PS_PORT         = int(os.environ.get("PS_PORT", 50051))
AZURE_CONN_STR  = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
CONTAINER_NAME  = "eq-data"
UPLOAD_DIR      = Path(os.environ.get("UPLOAD_DIR", "uploaded_data"))
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_SHARDS_DIR = Path("data_shards")
OUTPUTS_DIR     = Path("outputs")

def ps_request(msg):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect((PS_HOST, PS_PORT))
            s.sendall(json.dumps(msg).encode())
            data = b""
            while True:
                chunk = s.recv(65536)
                if not chunk: break
                data += chunk
                try: return json.loads(data.decode())
                except json.JSONDecodeError: continue
    except Exception as e:
        return {"error": str(e)}

def upload_shard_to_blob(shard_df, shard_id):
    try:
        from azure.storage.blob import BlobServiceClient
        client    = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        csv_bytes = shard_df.to_csv(index=False).encode()
        blob      = client.get_blob_client(container=CONTAINER_NAME, blob=f"shard_{shard_id}.csv")
        blob.upload_blob(csv_bytes, overwrite=True)
        return True, f"shard_{shard_id}.csv uploaded to Azure Blob"
    except Exception as e:
        return False, str(e)

def load_csv_history(path):
    epochs, rmses, losses = [], [], []
    if not path.exists():
        return epochs, rmses, losses
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row.get("epoch", 0)))
            rmses.append(float(row.get("rmse", 0)))
            losses.append(float(row.get("loss", 0)))
    return epochs, rmses, losses

def load_all_worker_histories():
    """Load history — download from Azure Blob first, then read local files."""
    # Try to download latest history from Azure Blob
    if AZURE_CONN_STR:
        try:
            from azure.storage.blob import BlobServiceClient
            client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
            OUTPUTS_DIR.mkdir(exist_ok=True)
            for wid in range(4):
                blob_name = f"worker_{wid}_history.csv"
                try:
                    blob = client.get_blob_client(
                        container=CONTAINER_NAME, blob=blob_name)
                    local_path = OUTPUTS_DIR / blob_name
                    with open(local_path, "wb") as f:
                        f.write(blob.download_blob().readall())
                except Exception:
                    pass  
        except Exception:
            pass

    workers = {}
    for wid in range(4):
        p = OUTPUTS_DIR / f"worker_{wid}_history.csv"
        if p.exists():
            epochs, rmses, losses = load_csv_history(p)
            if epochs:
                workers[wid] = {"epochs": epochs, "rmses": rmses, "losses": losses}
    return workers

def load_ps_loss_history():
    """Load PS loss history from get_status."""
    status = ps_request({"type": "get_status"})
    if "data" in status:
        history = status["data"].get("loss_history", [])
        rounds  = [h["round"] for h in history]
        losses  = [h["loss"]  for h in history]
        return rounds, losses
    return [], []

# ── Shared CSS ───────────────────────────────────────────────────
SHARED_CSS = """
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#060810;color:#cdd6f4;font-family:'Outfit',sans-serif;min-height:100vh;}
header{background:rgba(12,17,32,0.97);border-bottom:1px solid #1a2a45;padding:16px 32px;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;}
.logo{font-size:1.4rem;font-weight:900;background:linear-gradient(90deg,#00e5ff,#00ff9d);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
nav{display:flex;gap:8px;}
nav a{font-family:'JetBrains Mono',monospace;font-size:0.75rem;padding:6px 16px;
  border-radius:8px;border:1px solid #1a2a45;color:#94a3b8;text-decoration:none;transition:all 0.2s;}
nav a:hover,nav a.active{border-color:#00e5ff;color:#00e5ff;background:rgba(0,229,255,0.06);}
.status-badge{font-family:'JetBrains Mono',monospace;font-size:0.75rem;
  padding:5px 14px;border-radius:20px;border:1px solid #1a2a45;background:#111827;}
.status-badge.online{border-color:#00ff9d;color:#00ff9d;}
.status-badge.offline{border-color:#ff1744;color:#ff1744;}
.container{max-width:1100px;margin:0 auto;padding:32px 24px;}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px;}
.card{background:#111827;border:1px solid #1a2a45;border-radius:14px;padding:20px;text-align:center;}
.card-val{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;margin-bottom:6px;}
.card-lbl{font-size:0.75rem;color:#45556e;text-transform:uppercase;letter-spacing:0.08em;}
.cyan{color:#00e5ff;}.green{color:#00ff9d;}.purple{color:#a78bfa;}.yellow{color:#ffc107;}
.orange{color:#ff7043;}.red{color:#ff1744;}.teal{color:#00e5c8;}
.panel{background:#111827;border:1px solid #1a2a45;border-radius:14px;padding:24px;margin-bottom:20px;}
.panel-title{font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#45556e;
  letter-spacing:0.1em;margin-bottom:16px;}
.btn{display:inline-block;padding:12px 28px;border-radius:10px;border:none;cursor:pointer;
  font-family:'JetBrains Mono',monospace;font-size:0.8rem;font-weight:700;text-decoration:none;margin:6px 6px 6px 0;}
.btn-start{background:linear-gradient(135deg,#0097a7,#00695c);color:#fff;box-shadow:0 0 20px rgba(0,229,255,0.2);}
.btn-stop{background:rgba(255,23,68,0.12);border:1px solid #ff1744;color:#ff1744;}
.btn-reset{background:rgba(255,255,255,0.05);border:1px solid #1a2a45;color:#cdd6f4;}
.btn-upload{background:linear-gradient(135deg,#6d28d9,#7c3aed);color:#fff;}
.btn-run{background:linear-gradient(135deg,#f97316,#ea580c);color:#fff;}
input[type=number],input[type=file],select{background:#111827;border:1px solid #1a2a45;color:#cdd6f4;
  padding:10px 14px;border-radius:8px;font-family:'JetBrains Mono',monospace;font-size:0.85rem;}
input[type=number],select{width:140px;}
input[type=file]{width:100%;margin-bottom:12px;}
.success-msg{background:rgba(0,255,157,0.08);border:1px solid #00ff9d;border-radius:10px;
  padding:14px 20px;margin-bottom:20px;color:#00ff9d;font-size:0.85rem;}
.error-msg{background:rgba(255,23,68,0.08);border:1px solid #ff1744;border-radius:10px;
  padding:14px 20px;margin-bottom:20px;color:#ff1744;font-size:0.85rem;}
.info-msg{background:rgba(0,120,212,0.08);border:1px solid #0078d4;border-radius:10px;
  padding:14px 20px;margin-bottom:20px;color:#60a5fa;font-size:0.85rem;}
.azure-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(0,120,212,0.1);
  border:1px solid rgba(0,120,212,0.3);border-radius:6px;padding:4px 10px;
  font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#60a5fa;}
.chart-area{background:#0c1120;border:1px solid #1a2a45;border-radius:12px;padding:16px;margin-bottom:8px;}
.chart-legend{display:flex;gap:24px;flex-wrap:wrap;margin-top:10px;}
.legend-item{display:flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#94a3b8;}
.legend-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
"""

def nav_header(active="dashboard", status=None):
    status_html = ""
    if status:
        if status.get("error"):
            status_html = '<span class="status-badge offline">● PS OFFLINE</span>'
        else:
            mode_label = status.get("mode", "sync").upper()
            done_label = " · DONE" if status.get("done") else ""
            status_html = f'<span class="status-badge online">● PS ONLINE — Round {status.get("round",0)}/{status.get("max_rounds","?")} [{mode_label}{done_label}]</span>'
    return f"""<header>
  <div class="logo">🌍 EQ Distributed Training</div>
  <nav>
    <a href="/" class="{'active' if active=='dashboard' else ''}">Dashboard</a>
    <a href="/upload" class="{'active' if active=='upload' else ''}">Upload Dataset</a>
    <a href="/analytics" class="{'active' if active=='analytics' else ''}">Training Analytics</a>
  </nav>
  <div>{status_html}</div>
</header>"""

# ── Dashboard HTML ───────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html><html><head>
<meta charset="UTF-8"/><title>🌍 EQ Distributed Training</title>
{% if prediction is none %}
<meta http-equiv="refresh" content="5">
{% endif %}
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@400;700;900&display=swap" rel="stylesheet"/>
<style>""" + SHARED_CSS + """
.loss-row{display:flex;align-items:center;gap:12px;margin-bottom:8px;}
.loss-epoch{font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#45556e;width:70px;}
.loss-bar-wrap{flex:1;height:6px;background:rgba(255,255,255,0.06);border-radius:3px;}
.loss-bar{height:100%;border-radius:3px;background:linear-gradient(90deg,#00e5ff,#00ff9d);}
.loss-val{font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#ffc107;width:55px;text-align:right;}
.predict-box{background:#0c1120;border:1px solid #1a2a45;border-radius:10px;padding:20px;}
.inp-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;}
.result-big{font-family:'JetBrains Mono',monospace;font-size:2.5rem;font-weight:700;
  color:#00ff9d;text-align:center;padding:16px;}
.workers-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.worker-card{background:#0c1120;border:1px solid #1a2a45;border-radius:10px;padding:16px;}
.tag{display:inline-block;padding:2px 10px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:0.62rem;font-weight:700;}
.tag-training{background:rgba(0,255,157,0.1);color:#00ff9d;}
.tag-idle{background:rgba(255,255,255,0.05);color:#45556e;}
.tag-straggler{background:rgba(255,193,7,0.1);color:#ffc107;}
.tag-done{background:rgba(0,229,255,0.1);color:#00e5ff;}
</style></head><body>
{{ nav | safe }}
<div class="container">
  <div class="grid">
    <div class="card"><div class="card-val cyan">{{ status.get('round',0) }}</div><div class="card-lbl">Training Round</div></div>
    <div class="card"><div class="card-val green">{{ status.get('total_updates',0) }}</div><div class="card-lbl">Gradient Updates</div></div>
    <div class="card"><div class="card-val purple">{{ status.get('num_workers',2) }}</div><div class="card-lbl">Worker Nodes</div></div>
    <div class="card"><div class="card-val yellow">{{ status.get('uptime',0) }}s</div><div class="card-lbl">PS Uptime</div></div>
  </div>
  <div class="panel">
    <div class="panel-title">// TRAINING CONTROLS</div>
    <a href="/start" class="btn btn-start">▶ START TRAINING</a>
    <a href="/stop" class="btn btn-stop">■ STOP</a>
    <a href="/" class="btn btn-reset">↺ REFRESH</a>
    {% if status.get('stragglers') %}
    <span style="color:#ffc107;font-size:0.8rem;margin-left:12px;">⚠️ Straggler workers: {{ status['stragglers'] }}</span>
    {% endif %}
    {% if status.get('done') %}
    <span style="color:#00e5ff;font-size:0.8rem;margin-left:12px;">✅ Training complete — {{ status.get('round',0) }} rounds</span>
    {% endif %}
    {% if status.get('mode') %}
    <span style="color:#45556e;font-size:0.75rem;font-family:'JetBrains Mono',monospace;margin-left:12px;">MODE: {{ status.get('mode','sync').upper() }}</span>
    {% endif %}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
    <div class="panel">
      <div class="panel-title">// TRAINING LOSS CURVE</div>
      {% set history = status.get('loss_history', []) %}
      {% if history %}
        {% set max_loss = history | map(attribute='loss') | max %}
        {% for item in history %}
        <div class="loss-row">
          <span class="loss-epoch">Round {{ item.round }}</span>
          <div class="loss-bar-wrap">
            <div class="loss-bar" style="width:{{ [[(item.loss/max_loss*100)|int, 2]|max, 100]|min }}%"></div>
          </div>
          <span class="loss-val">{{ "%.4f"|format(item.loss) }}</span>
        </div>
        {% endfor %}
      {% else %}
        <p style="color:#45556e;font-size:0.85rem;">Training not started yet...</p>
      {% endif %}
    </div>
    <div class="panel">
      <div class="panel-title">// PREDICT MAGNITUDE</div>
      <div class="predict-box">
        <form method="POST" action="/predict">
          <div class="inp-row">
            <div>
              <div style="font-size:0.7rem;color:#45556e;margin-bottom:4px;">DEPTH (km)</div>
              <input type="number" name="depth" value="{{ req_depth }}" step="0.1"/>
            </div>
            <div>
              <div style="font-size:0.7rem;color:#45556e;margin-bottom:4px;">LATITUDE</div>
              <input type="number" name="lat" value="{{ req_lat }}" step="0.01"/>
            </div>
            <div>
              <div style="font-size:0.7rem;color:#45556e;margin-bottom:4px;">LONGITUDE</div>
              <input type="number" name="lon" value="{{ req_lon }}" step="0.01"/>
            </div>
          </div>
          <button type="submit" class="btn btn-start" style="margin:0">🔮 PREDICT</button>
        </form>
        {% if prediction is not none %}
        <div class="result-big">{{ "%.2f"|format(prediction) }} M</div>
        <p style="text-align:center;color:#45556e;font-size:0.8rem;">Predicted Magnitude</p>
        {% endif %}
        {% if pred_error %}
        <div class="error-msg" style="margin-top:12px;">{{ pred_error }}</div>
        {% endif %}
      </div>
    </div>
  </div>
  <div class="panel" style="margin-top:0;">
    <div class="panel-title">// WORKER NODES</div>
    <div class="workers-grid">
      {% for i in range(status.get('num_workers',2)) %}
      <div class="worker-card">
        <div style="font-weight:700;margin-bottom:6px;color:#00ff9d;">⚙️ Worker {{ i }}</div>
        <div style="font-size:0.8rem;color:#45556e;margin-bottom:8px;">shard_{{ i }}.csv · VM {{ i+1 }}</div>
        {% if i in status.get('stragglers',[]) %}
          <span class="tag tag-straggler">STRAGGLER</span>
        {% elif status.get('done') %}
          <span class="tag tag-done">COMPLETE</span>
        {% elif status.get('round',0) > 0 %}
          <span class="tag tag-training">TRAINING</span>
        {% else %}
          <span class="tag tag-idle">IDLE</span>
        {% endif %}
      </div>
      {% endfor %}
    </div>
  </div>
</div>
</body></html>
"""

# ── Upload HTML ──────────────────────────────────────────────────
UPLOAD_HTML = """
<!DOCTYPE html><html><head>
<meta charset="UTF-8"/><title>📤 Upload Dataset</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@400;700;900&display=swap" rel="stylesheet"/>
<style>""" + SHARED_CSS + """
.dropzone{border:2px dashed #1a2a45;border-radius:14px;padding:40px;text-align:center;transition:all 0.3s;margin-bottom:20px;}
.dropzone:hover{border-color:#a78bfa;background:rgba(167,139,250,0.03);}
.inp-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px;align-items:flex-end;}
.inp-group{display:flex;flex-direction:column;gap:4px;}
.inp-label{font-size:0.7rem;color:#45556e;font-family:'JetBrains Mono',monospace;text-transform:uppercase;}
.shard-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-top:16px;}
.shard-card{background:#0c1120;border:1px solid #1a2a45;border-radius:10px;padding:14px;}
.storage-info{background:#0c1120;border:1px solid rgba(0,120,212,0.3);border-radius:10px;padding:16px;margin-bottom:20px;}
</style></head><body>
{{ nav | safe }}
<div class="container">
  <h2 style="font-size:1.6rem;margin-bottom:8px;">📤 Upload Earthquake Dataset</h2>
  <p style="color:#45556e;font-size:0.85rem;margin-bottom:24px;">Upload CSV — system auto-splits into shards and stores in Azure Blob Storage for distributed training.</p>
  {% if azure_enabled %}
  <div class="storage-info">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
      <span class="azure-badge">☁ Azure Blob Storage</span>
      <span style="color:#00ff9d;font-size:0.8rem;">Connected</span>
    </div>
    <div style="color:#45556e;font-size:0.8rem;">Shards uploaded to <span style="color:#60a5fa;">eqdistributed / eq-data</span> container.</div>
  </div>
  {% else %}
  <div class="info-msg">ℹ️ Azure Blob Storage not configured. Shards saved locally only.</div>
  {% endif %}
  {% if success_msg %}<div class="success-msg">✅ {{ success_msg }}</div>{% endif %}
  {% if error_msg %}<div class="error-msg">❌ {{ error_msg }}</div>{% endif %}
  {% if blob_results %}
  <div class="success-msg">☁ Azure Blob Upload Results:<br>
    {% for r in blob_results %}&nbsp;&nbsp;{{ r }}<br>{% endfor %}
  </div>
  {% endif %}
  <div class="panel">
    <div class="panel-title">// UPLOAD NEW DATASET</div>
    <form method="POST" action="/upload" enctype="multipart/form-data">
      <div class="dropzone">
        <div style="color:#45556e;font-size:0.9rem;margin-bottom:8px;">Choose a CSV file</div>
        <div style="color:#2d3748;font-size:0.75rem;font-family:'JetBrains Mono',monospace;">Supported: .csv files up to 100MB</div>
        <input type="file" name="dataset" accept=".csv" style="margin-top:16px;"/>
      </div>
      <div class="inp-row">
        <div class="inp-group"><span class="inp-label">Workers</span>
          <select name="num_workers">
            <option value="2" selected>2 Workers</option>
            <option value="3">3 Workers</option>
            <option value="4">4 Workers</option>
          </select>
        </div>
        <div class="inp-group"><span class="inp-label">Max Rows</span>
          <input type="number" name="max_rows" placeholder="All rows" min="1000"/>
        </div>
      </div>
      <button type="submit" class="btn btn-upload">📤 UPLOAD & SPLIT</button>
      {% if azure_enabled %}
      <span style="color:#60a5fa;font-size:0.75rem;font-family:'JetBrains Mono',monospace;margin-left:12px;">☁ Will auto-upload shards to Azure Blob</span>
      {% endif %}
    </form>
  </div>
  <div class="panel">
    <div class="panel-title">// REQUIRED COLUMNS</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
      <div style="background:#0c1120;border:1px solid #1a2a45;border-radius:8px;padding:12px;text-align:center;"><div class="cyan" style="font-weight:700;">Magnitude</div><div style="color:#45556e;font-size:0.72rem;">target · mag</div></div>
      <div style="background:#0c1120;border:1px solid #1a2a45;border-radius:8px;padding:12px;text-align:center;"><div class="green" style="font-weight:700;">Depth</div><div style="color:#45556e;font-size:0.72rem;">km · depth</div></div>
      <div style="background:#0c1120;border:1px solid #1a2a45;border-radius:8px;padding:12px;text-align:center;"><div class="purple" style="font-weight:700;">Latitude</div><div style="color:#45556e;font-size:0.72rem;">coord · lat</div></div>
      <div style="background:#0c1120;border:1px solid #1a2a45;border-radius:8px;padding:12px;text-align:center;"><div class="yellow" style="font-weight:700;">Longitude</div><div style="color:#45556e;font-size:0.72rem;">coord · lon</div></div>
    </div>
  </div>
  {% if shards %}
  <div class="panel">
    <div class="panel-title">// CURRENT SHARDS</div>
    <div class="shard-grid">
      {% for shard in shards %}
      <div class="shard-card">
        <div style="color:#00e5ff;font-family:'JetBrains Mono',monospace;font-size:0.8rem;font-weight:700;">📁 {{ shard.name }}</div>
        <div style="color:#45556e;font-size:0.75rem;margin-top:4px;">{{ shard.rows }} rows · {{ shard.size }}</div>
        {% if azure_enabled %}<div style="margin-top:6px;"><span class="azure-badge">☁ In Azure Blob</span></div>{% endif %}
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}
</div></body></html>
"""

# ── Analytics HTML ───────────────────────────────────────────────
ANALYTICS_HTML = """
<!DOCTYPE html><html><head>
<meta charset="UTF-8"/><title>📊 Training Analytics</title>
<meta http-equiv="refresh" content="10">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@400;700;900&display=swap" rel="stylesheet"/>
<style>""" + SHARED_CSS + """
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}
.stat-row{display:flex;justify-content:space-between;padding:8px 0;
  border-bottom:1px solid rgba(26,42,69,0.5);font-family:'JetBrains Mono',monospace;font-size:0.8rem;}
.stat-label{color:#45556e;}
.stat-val{color:#cdd6f4;}
</style></head><body>
{{ nav | safe }}
<div class="container">
  <h2 style="font-size:1.6rem;margin-bottom:4px;">📊 Training Analytics</h2>
  <p style="color:#45556e;font-size:0.85rem;margin-bottom:24px;">
    Real training results from all worker nodes. Auto-refreshes every 10 seconds.
  </p>

  {% if not has_data %}
  <div class="panel" style="text-align:center;padding:48px;">
    <p style="color:#94a3b8;font-size:1rem;margin-bottom:8px;">No training data yet.</p>
    <p style="color:#45556e;font-size:0.85rem;">Start training from the Dashboard then come back here.</p>
    <a href="/" class="btn btn-start" style="margin-top:20px;display:inline-block;">← Go to Dashboard</a>
  </div>
  {% else %}

  <!-- Summary cards -->
  <div class="grid">
    <div class="card">
      <div class="card-val cyan">{{ num_workers }}</div>
      <div class="card-lbl">Workers Trained</div>
    </div>
    <div class="card">
      <div class="card-val green">{{ total_epochs }}</div>
      <div class="card-lbl">Total Epochs</div>
    </div>
    <div class="card">
      <div class="card-val purple">{{ "%.4f"|format(best_rmse) }}</div>
      <div class="card-lbl">Best RMSE</div>
    </div>
    <div class="card">
      <div class="card-val yellow">{{ "%.4f"|format(final_loss) }}</div>
      <div class="card-lbl">Final PS Loss</div>
    </div>
  </div>

  <!-- Chart 1: RMSE per worker over epochs -->
  <div class="panel">
    <div class="panel-title">// RMSE CONVERGENCE — PER WORKER</div>
    <div class="chart-area">
      <svg width="100%" height="260" viewBox="0 0 800 260" preserveAspectRatio="xMidYMid meet">
        <!-- Grid lines -->
        {% for i in range(5) %}
        <line x1="60" y1="{{ 20+i*55 }}" x2="780" y2="{{ 20+i*55 }}" stroke="#1a2a45" stroke-width="0.5"/>
        <text x="50" y="{{ 24+i*55 }}" fill="#45556e" font-family="JetBrains Mono" font-size="9" text-anchor="end">
          {{ "%.3f"|format(rmse_max - i*(rmse_max-rmse_min)/4) }}
        </text>
        {% endfor %}
        <!-- X axis labels -->
        {% for i in range(0, max_epochs, [1, max_epochs//5]|max) %}
        <text x="{{ 60+i*(720/(max_epochs-1 if max_epochs > 1 else 1)) }}" y="255"
          fill="#45556e" font-family="JetBrains Mono" font-size="9" text-anchor="middle">{{ i+1 }}</text>
        {% endfor %}
        <text x="420" y="275" fill="#45556e" font-family="JetBrains Mono" font-size="9" text-anchor="middle">Epoch</text>
        <!-- Worker 0 line — cyan -->
        {% if w0_rmse %}
        <polyline fill="none" stroke="#00e5ff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"
          points="{% for i in range(w0_rmse|length) %}{{ 60+i*(720/(max_epochs-1 if max_epochs > 1 else 1)) }},{{ 20+(rmse_max-w0_rmse[i])/(rmse_max-rmse_min+0.0001)*220 }} {% endfor %}"/>
        <circle cx="{{ 60+(w0_rmse|length-1)*(720/(max_epochs-1 if max_epochs > 1 else 1)) }}"
          cy="{{ 20+(rmse_max-w0_rmse[-1])/(rmse_max-rmse_min+0.0001)*220 }}" r="4" fill="#00e5ff"/>
        {% endif %}
        <!-- Worker 1 line — green -->
        {% if w1_rmse %}
        <polyline fill="none" stroke="#00ff9d" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"
          stroke-dasharray="8 4"
          points="{% for i in range(w1_rmse|length) %}{{ 60+i*(720/(max_epochs-1 if max_epochs > 1 else 1)) }},{{ 20+(rmse_max-w1_rmse[i])/(rmse_max-rmse_min+0.0001)*220 }} {% endfor %}"/>
        <circle cx="{{ 60+(w1_rmse|length-1)*(720/(max_epochs-1 if max_epochs > 1 else 1)) }}"
          cy="{{ 20+(rmse_max-w1_rmse[-1])/(rmse_max-rmse_min+0.0001)*220 }}" r="4" fill="#00ff9d"/>
        {% endif %}
      </svg>
    </div>
    <div class="chart-legend">
      {% if w0_rmse %}<div class="legend-item"><div class="legend-dot" style="background:#00e5ff;"></div>Worker 0 — Final RMSE: {{ "%.4f"|format(w0_rmse[-1]) }}</div>{% endif %}
      {% if w1_rmse %}<div class="legend-item"><div class="legend-dot" style="background:#00ff9d;"></div>Worker 1 — Final RMSE: {{ "%.4f"|format(w1_rmse[-1]) }}</div>{% endif %}
    </div>
  </div>

  <!-- Chart 2: Loss per worker over epochs -->
  <div class="panel">
    <div class="panel-title">// TRAINING LOSS — PER WORKER</div>
    <div class="chart-area">
      <svg width="100%" height="260" viewBox="0 0 800 260" preserveAspectRatio="xMidYMid meet">
        {% for i in range(5) %}
        <line x1="60" y1="{{ 20+i*55 }}" x2="780" y2="{{ 20+i*55 }}" stroke="#1a2a45" stroke-width="0.5"/>
        <text x="50" y="{{ 24+i*55 }}" fill="#45556e" font-family="JetBrains Mono" font-size="9" text-anchor="end">
          {{ "%.3f"|format(loss_max - i*(loss_max-loss_min)/4) }}
        </text>
        {% endfor %}
        {% for i in range(0, max_epochs, [1, max_epochs//5]|max) %}
        <text x="{{ 60+i*(720/(max_epochs-1 if max_epochs > 1 else 1)) }}" y="255"
          fill="#45556e" font-family="JetBrains Mono" font-size="9" text-anchor="middle">{{ i+1 }}</text>
        {% endfor %}
        <text x="420" y="275" fill="#45556e" font-family="JetBrains Mono" font-size="9" text-anchor="middle">Epoch</text>
        {% if w0_loss %}
        <polyline fill="none" stroke="#f97316" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"
          points="{% for i in range(w0_loss|length) %}{{ 60+i*(720/(max_epochs-1 if max_epochs > 1 else 1)) }},{{ 20+(loss_max-w0_loss[i])/(loss_max-loss_min+0.0001)*220 }} {% endfor %}"/>
        <circle cx="{{ 60+(w0_loss|length-1)*(720/(max_epochs-1 if max_epochs > 1 else 1)) }}"
          cy="{{ 20+(loss_max-w0_loss[-1])/(loss_max-loss_min+0.0001)*220 }}" r="4" fill="#f97316"/>
        {% endif %}
        {% if w1_loss %}
        <polyline fill="none" stroke="#a78bfa" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"
          stroke-dasharray="8 4"
          points="{% for i in range(w1_loss|length) %}{{ 60+i*(720/(max_epochs-1 if max_epochs > 1 else 1)) }},{{ 20+(loss_max-w1_loss[i])/(loss_max-loss_min+0.0001)*220 }} {% endfor %}"/>
        <circle cx="{{ 60+(w1_loss|length-1)*(720/(max_epochs-1 if max_epochs > 1 else 1)) }}"
          cy="{{ 20+(loss_max-w1_loss[-1])/(loss_max-loss_min+0.0001)*220 }}" r="4" fill="#a78bfa"/>
        {% endif %}
      </svg>
    </div>
    <div class="chart-legend">
      {% if w0_loss %}<div class="legend-item"><div class="legend-dot" style="background:#f97316;"></div>Worker 0 Loss — Final: {{ "%.4f"|format(w0_loss[-1]) }}</div>{% endif %}
      {% if w1_loss %}<div class="legend-item"><div class="legend-dot" style="background:#a78bfa;"></div>Worker 1 Loss — Final: {{ "%.4f"|format(w1_loss[-1]) }}</div>{% endif %}
    </div>
  </div>

  <!-- Chart 3: PS loss history (rounds) -->
  {% if ps_rounds %}
  <div class="panel">
    <div class="panel-title">// PARAMETER SERVER LOSS — BY ROUND</div>
    <div class="chart-area">
      <svg width="100%" height="220" viewBox="0 0 800 220" preserveAspectRatio="xMidYMid meet">
        {% set ps_max = ps_losses|max %}
        {% set ps_min = ps_losses|min %}
        {% for i in range(5) %}
        <line x1="60" y1="{{ 10+i*48 }}" x2="780" y2="{{ 10+i*48 }}" stroke="#1a2a45" stroke-width="0.5"/>
        <text x="50" y="{{ 14+i*48 }}" fill="#45556e" font-family="JetBrains Mono" font-size="9" text-anchor="end">
          {{ "%.3f"|format(ps_max - i*(ps_max-ps_min)/4) }}
        </text>
        {% endfor %}
        {% set n = ps_rounds|length %}
        {% for i in range(0, n, [1, n//5]|max) %}
        <text x="{{ 60+i*(720/(n-1 if n > 1 else 1)) }}" y="215"
          fill="#45556e" font-family="JetBrains Mono" font-size="9" text-anchor="middle">{{ ps_rounds[i] }}</text>
        {% endfor %}
        <polyline fill="none" stroke="#ffc107" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
          points="{% for i in range(n) %}{{ 60+i*(720/(n-1 if n > 1 else 1)) }},{{ 10+(ps_max-ps_losses[i])/(ps_max-ps_min+0.0001)*192 }} {% endfor %}"/>
      </svg>
    </div>
    <div class="chart-legend">
      <div class="legend-item"><div class="legend-dot" style="background:#ffc107;"></div>PS Aggregated Loss — {{ ps_rounds|length }} rounds recorded</div>
    </div>
  </div>
  {% endif %}

  <!-- Worker stats table -->
  <div class="two-col">
    {% if w0_rmse %}
    <div class="panel">
      <div class="panel-title">// WORKER 0 STATS</div>
      <div class="stat-row"><span class="stat-label">Epochs completed</span><span class="stat-val">{{ w0_rmse|length }}</span></div>
      <div class="stat-row"><span class="stat-label">Initial RMSE</span><span class="stat-val">{{ "%.4f"|format(w0_rmse[0]) }}</span></div>
      <div class="stat-row"><span class="stat-label">Final RMSE</span><span class="stat-val cyan">{{ "%.4f"|format(w0_rmse[-1]) }}</span></div>
      <div class="stat-row"><span class="stat-label">RMSE improvement</span><span class="stat-val green">{{ "%.1f"|format((1-w0_rmse[-1]/w0_rmse[0])*100) }}%</span></div>
      <div class="stat-row"><span class="stat-label">Initial loss</span><span class="stat-val">{{ "%.4f"|format(w0_loss[0]) }}</span></div>
      <div class="stat-row"><span class="stat-label">Final loss</span><span class="stat-val orange">{{ "%.4f"|format(w0_loss[-1]) }}</span></div>
    </div>
    {% endif %}
    {% if w1_rmse %}
    <div class="panel">
      <div class="panel-title">// WORKER 1 STATS</div>
      <div class="stat-row"><span class="stat-label">Epochs completed</span><span class="stat-val">{{ w1_rmse|length }}</span></div>
      <div class="stat-row"><span class="stat-label">Initial RMSE</span><span class="stat-val">{{ "%.4f"|format(w1_rmse[0]) }}</span></div>
      <div class="stat-row"><span class="stat-label">Final RMSE</span><span class="stat-val cyan">{{ "%.4f"|format(w1_rmse[-1]) }}</span></div>
      <div class="stat-row"><span class="stat-label">RMSE improvement</span><span class="stat-val green">{{ "%.1f"|format((1-w1_rmse[-1]/w1_rmse[0])*100) }}%</span></div>
      <div class="stat-row"><span class="stat-label">Initial loss</span><span class="stat-val">{{ "%.4f"|format(w1_loss[0]) }}</span></div>
      <div class="stat-row"><span class="stat-label">Final loss</span><span class="stat-val purple">{{ "%.4f"|format(w1_loss[-1]) }}</span></div>
    </div>
    {% endif %}
  </div>

  <div style="text-align:center;">
    <a href="/analytics" class="btn btn-reset">↺ REFRESH</a>
    <a href="/" class="btn btn-start">← Dashboard</a>
  </div>
  {% endif %}
</div></body></html>
"""

# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    status = ps_request({"type": "get_status"})
    if "data" in status: status = status["data"]
    elif "error" in status: status = {}
    return render_template_string(DASHBOARD_HTML,
        nav=nav_header("dashboard", status), status=status,
        prediction=None, pred_error=None,
        req_depth=10, req_lat=35.0, req_lon=140.0)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect("/")
    depth = float(request.form.get("depth", 10))
    lat   = float(request.form.get("lat", 35.0))
    lon   = float(request.form.get("lon", 140.0))
    resp   = ps_request({"type": "get_weights"})
    status = ps_request({"type": "get_status"})
    if "data" in status: status = status["data"]
    elif "error" in status: status = {}
    prediction = None
    pred_error = None
    if "weights" in resp:
        w = np.array(resp["weights"])
        X = np.array([depth, lat, lon], dtype=np.float32)
        X = (X - np.array([30.0, 0.0, 0.0])) / (np.array([50.0, 45.0, 90.0]) + 1e-8)
        prediction = float(X @ w[:-1] + w[-1])
    else:
        pred_error = "PS offline or training not started yet. Start workers first!"
    return render_template_string(DASHBOARD_HTML,
        nav=nav_header("dashboard", status), status=status,
        prediction=prediction, pred_error=pred_error,
        req_depth=depth, req_lat=lat, req_lon=lon)

@app.route("/start")
def start_training():
    status = ps_request({"type": "get_status"})
    if "data" in status: status = status["data"]
    elif "error" in status: status = {}
    return render_template_string(DASHBOARD_HTML,
        nav=nav_header("dashboard", status), status=status,
        prediction=None, pred_error=None,
        req_depth=10, req_lat=35.0, req_lon=140.0)

@app.route("/stop")
def stop_training():
    status = ps_request({"type": "get_status"})
    if "data" in status: status = status["data"]
    elif "error" in status: status = {}
    return render_template_string(DASHBOARD_HTML,
        nav=nav_header("dashboard", status), status=status,
        prediction=None, pred_error=None,
        req_depth=10, req_lat=35.0, req_lon=140.0)

@app.route("/status")
def api_status():
    return jsonify(ps_request({"type": "get_status"}))

@app.route("/analytics")
def analytics_page():
    workers   = load_all_worker_histories()
    ps_rounds, ps_losses = load_ps_loss_history()

    if not workers and not ps_rounds:
        return render_template_string(ANALYTICS_HTML,
            nav=nav_header("analytics"), has_data=False,
            num_workers=0, total_epochs=0, best_rmse=0,
            final_loss=0, w0_rmse=[], w1_rmse=[],
            w0_loss=[], w1_loss=[], ps_rounds=[], ps_losses=[],
            max_epochs=1, rmse_max=1, rmse_min=0,
            loss_max=1, loss_min=0)

    w0 = workers.get(0, {})
    w1 = workers.get(1, {})
    w0_rmse  = w0.get("rmses", [])
    w0_loss  = w0.get("losses", [])
    w1_rmse  = w1.get("rmses", [])
    w1_loss  = w1.get("losses", [])

    all_rmse = w0_rmse + w1_rmse
    all_loss = w0_loss + w1_loss

    rmse_max = max(all_rmse) * 1.05 if all_rmse else 1
    rmse_min = min(all_rmse) * 0.95 if all_rmse else 0
    loss_max = max(all_loss) * 1.05 if all_loss else 1
    loss_min = min(all_loss) * 0.95 if all_loss else 0

    max_epochs  = max(len(w0_rmse), len(w1_rmse), 1)
    num_workers = len(workers)
    total_epochs = sum(len(w.get("rmses", [])) for w in workers.values())
    best_rmse   = min(all_rmse) if all_rmse else 0
    final_loss  = ps_losses[-1] if ps_losses else (w0_loss[-1] if w0_loss else 0)

    return render_template_string(ANALYTICS_HTML,
        nav=nav_header("analytics"), has_data=True,
        num_workers=num_workers, total_epochs=total_epochs,
        best_rmse=best_rmse, final_loss=final_loss,
        w0_rmse=w0_rmse, w1_rmse=w1_rmse,
        w0_loss=w0_loss, w1_loss=w1_loss,
        ps_rounds=ps_rounds, ps_losses=ps_losses,
        max_epochs=max_epochs,
        rmse_max=rmse_max, rmse_min=rmse_min,
        loss_max=loss_max, loss_min=loss_min)

@app.route("/api/analytics")
def api_analytics():
    workers = load_all_worker_histories()
    ps_rounds, ps_losses = load_ps_loss_history()
    return jsonify({"workers": workers, "ps_rounds": ps_rounds, "ps_losses": ps_losses})

def get_current_shards():
    shards = []
    if DATA_SHARDS_DIR.exists():
        for f in sorted(DATA_SHARDS_DIR.glob("shard_*.csv")):
            try:
                import pandas as pd
                rows = len(pd.read_csv(f))
            except Exception: rows = "?"
            size_kb  = f.stat().st_size / 1024
            size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
            shards.append({"name": f.name, "rows": rows, "size": size_str})
    return shards

@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template_string(UPLOAD_HTML,
        nav=nav_header("upload"),
        shards=get_current_shards(),
        azure_enabled=bool(AZURE_CONN_STR),
        success_msg=None, error_msg=None, blob_results=None)

@app.route("/upload", methods=["POST"])
def upload_dataset():
    import pandas as pd
    if "dataset" not in request.files or request.files["dataset"].filename == "":
        return render_template_string(UPLOAD_HTML, nav=nav_header("upload"),
            shards=get_current_shards(), azure_enabled=bool(AZURE_CONN_STR),
            success_msg=None, error_msg="No file selected.", blob_results=None)
    file = request.files["dataset"]
    if not file.filename.lower().endswith(".csv"):
        return render_template_string(UPLOAD_HTML, nav=nav_header("upload"),
            shards=get_current_shards(), azure_enabled=bool(AZURE_CONN_STR),
            success_msg=None, error_msg="Only CSV files supported.", blob_results=None)
    try:
        filename = secure_filename(file.filename)
        filepath = UPLOAD_DIR / filename
        file.save(str(filepath))
        df = pd.read_csv(filepath)
        col_map = {
            "Magnitude": ["mag", "magnitude"],
            "Depth":     ["depth"],
            "Latitude":  ["lat", "latitude"],
            "Longitude": ["lon", "long", "longitude"]
        }
        rename = {}
        for target, hints in col_map.items():
            for col in df.columns:
                if any(h in col.lower() for h in hints):
                    rename[col] = target
                    break
        df = df.rename(columns=rename)
        needed  = ["Magnitude", "Depth", "Latitude", "Longitude"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return render_template_string(UPLOAD_HTML, nav=nav_header("upload"),
                shards=get_current_shards(), azure_enabled=bool(AZURE_CONN_STR),
                success_msg=None, blob_results=None,
                error_msg=f"Missing columns: {', '.join(missing)}. Found: {list(df.columns)}")
        df = df[needed].dropna()
        max_rows = request.form.get("max_rows")
        if max_rows and max_rows.strip():
            df = df.iloc[:int(max_rows)]
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        num_workers = int(request.form.get("num_workers", 2))
        DATA_SHARDS_DIR.mkdir(exist_ok=True)
        for old in DATA_SHARDS_DIR.glob("shard_*.csv"):
            old.unlink()
        blob_results = []
        indices = np.array_split(np.arange(len(df)), num_workers)
        shards  = [df.iloc[idx] for idx in indices]
        for i, shard in enumerate(shards):
            local_path = DATA_SHARDS_DIR / f"shard_{i}.csv"
            shard.to_csv(local_path, index=False)
            if AZURE_CONN_STR:
                success, msg = upload_shard_to_blob(shard, i)
                if success:
                    blob_results.append(f"✅ shard_{i}.csv → Azure Blob ({len(shard):,} rows)")
                else:
                    blob_results.append(f"⚠️ shard_{i}.csv saved locally only — Blob error: {msg}")
        success_msg = f"'{filename}' uploaded → {num_workers} shards created ({len(df):,} total rows)"
        return render_template_string(UPLOAD_HTML,
            nav=nav_header("upload"),
            shards=get_current_shards(),
            azure_enabled=bool(AZURE_CONN_STR),
            success_msg=success_msg, error_msg=None,
            blob_results=blob_results if blob_results else None)
    except Exception as e:
        return render_template_string(UPLOAD_HTML, nav=nav_header("upload"),
            shards=get_current_shards(), azure_enabled=bool(AZURE_CONN_STR),
            success_msg=None, error_msg=f"Error: {str(e)}", blob_results=None)

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)