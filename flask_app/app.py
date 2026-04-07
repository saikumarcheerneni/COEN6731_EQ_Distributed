"""
flask_app/app.py
Live Earthquake Distributed Training Dashboard
Enhanced: CSV Upload + Azure Blob Storage + Training Analytics Charts + Latency
"""
from flask import Flask, render_template_string, jsonify, request, redirect, url_for
import json, socket, os, time, csv, math
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

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

def get_shard_count():
    if DATA_SHARDS_DIR.exists():
        return len(list(DATA_SHARDS_DIR.glob("shard_*.csv")))
    return 0

def fix_worker_count(status):
    shard_count = get_shard_count()
    if shard_count > status.get("num_workers", 0):
        status["num_workers"] = shard_count
    return status

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

def cleanup_old_workers(num_workers):
    OUTPUTS_DIR.mkdir(exist_ok=True)
    for wid in range(4):
        if wid >= num_workers:
            old_history = OUTPUTS_DIR / f"worker_{wid}_history.csv"
            if old_history.exists():
                old_history.unlink()
            if AZURE_CONN_STR:
                try:
                    from azure.storage.blob import BlobServiceClient
                    client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
                    blob = client.get_blob_client(container=CONTAINER_NAME, blob=f"worker_{wid}_history.csv")
                    blob.delete_blob()
                except Exception:
                    pass

def load_csv_history(path):
    epochs, rmses, losses, comms = [], [], [], []
    if not path.exists():
        return epochs, rmses, losses, comms
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row.get("epoch", 0)))
            rmses.append(float(row.get("rmse", 0)))
            losses.append(float(row.get("loss", 0)))
            comms.append(float(row.get("avg_comm_ms", 0)))
    return epochs, rmses, losses, comms

def load_all_worker_histories():
    shard_count = get_shard_count()
    max_workers = shard_count if shard_count > 0 else 4
    if AZURE_CONN_STR:
        try:
            from azure.storage.blob import BlobServiceClient
            client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
            OUTPUTS_DIR.mkdir(exist_ok=True)
            for wid in range(max_workers):
                blob_name = f"worker_{wid}_history.csv"
                try:
                    blob = client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
                    local_path = OUTPUTS_DIR / blob_name
                    with open(local_path, "wb") as f:
                        f.write(blob.download_blob().readall())
                except Exception:
                    pass
        except Exception:
            pass
    workers = {}
    for wid in range(max_workers):
        p = OUTPUTS_DIR / f"worker_{wid}_history.csv"
        if p.exists():
            epochs, rmses, losses, comms = load_csv_history(p)
            if epochs:
                workers[wid] = {"epochs": epochs, "rmses": rmses, "losses": losses, "comms": comms}
    return workers

def load_ps_loss_history():
    status = ps_request({"type": "get_status"})
    if "data" in status:
        history = status["data"].get("loss_history", [])
        rounds  = [h["round"] for h in history]
        losses  = [h["loss"]  for h in history]
        return rounds, losses
    return [], []

# ── Shared base styles ───────────────────────────────────────────
BASE_CSS = """
*{margin:0;padding:0;box-sizing:border-box;}
body{
  font-family:'Outfit',sans-serif;
  color:#cdd6f4;
  background:#060810;
  min-height:100vh;
  position:relative;
  overflow-x:hidden;
}
body::before{
  content:'';
  position:fixed;inset:0;
  background-image:
    linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
  background-size:44px 44px;
  pointer-events:none;
  z-index:0;
}
body::after{
  content:'';
  position:fixed;
  top:-200px;left:-150px;
  width:600px;height:600px;
  background:radial-gradient(circle, rgba(0,229,255,0.04) 0%, transparent 65%);
  pointer-events:none;
  z-index:0;
}
.glow-br{
  position:fixed;
  bottom:-150px;right:-100px;
  width:500px;height:500px;
  background:radial-gradient(circle, rgba(0,255,157,0.04) 0%, transparent 65%);
  pointer-events:none;
  z-index:0;
}
header{
  position:relative;z-index:10;
  background:rgba(6,8,16,0.92);
  border-bottom:0.5px solid rgba(0,229,255,0.1);
  padding:14px 32px;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
  backdrop-filter:blur(8px);
}
.logo{
  font-size:1.3rem;font-weight:700;
  background:linear-gradient(90deg,#00e5ff,#00ff9d);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
nav{display:flex;gap:6px;}
nav a{
  font-family:'JetBrains Mono',monospace;font-size:0.72rem;
  padding:5px 14px;border-radius:8px;
  border:0.5px solid rgba(0,229,255,0.12);
  color:rgba(148,163,184,0.7);text-decoration:none;transition:all 0.2s;
}
nav a:hover,nav a.active{
  border-color:rgba(0,229,255,0.4);
  color:#00e5ff;
  background:rgba(0,229,255,0.06);
}
.ps-badge{
  font-family:'JetBrains Mono',monospace;font-size:0.7rem;
  padding:4px 12px;border-radius:20px;
}
.ps-online{border:0.5px solid rgba(0,255,157,0.3);color:#00ff9d;background:rgba(0,255,157,0.06);}
.ps-offline{border:0.5px solid rgba(255,23,68,0.3);color:#ff1744;background:rgba(255,23,68,0.06);}
.container{max-width:1120px;margin:0 auto;padding:28px 24px;position:relative;z-index:1;}
.section-tag{
  display:inline-flex;align-items:center;gap:6px;
  font-family:'JetBrains Mono',monospace;
  font-size:10px;font-weight:500;letter-spacing:0.12em;
  color:rgba(0,229,255,0.5);
  background:rgba(0,229,255,0.06);
  border:0.5px solid rgba(0,229,255,0.12);
  border-radius:4px;padding:3px 8px;
  margin-bottom:12px;
  text-transform:uppercase;
}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:22px;}
.metric{
  background:rgba(12,17,32,0.8);
  border:0.5px solid rgba(0,229,255,0.1);
  border-radius:10px;padding:14px 16px;
  backdrop-filter:blur(6px);
}
.metric-label{font-size:11px;color:rgba(148,163,184,0.6);margin-bottom:6px;font-family:'JetBrains Mono',monospace;}
.metric-val{font-size:22px;font-weight:600;margin-bottom:3px;}
.metric-sub{font-size:10px;color:rgba(148,163,184,0.35);}
.panel{
  background:rgba(12,17,32,0.75);
  border:0.5px solid rgba(0,229,255,0.08);
  border-radius:12px;padding:20px;
  margin-bottom:16px;
  backdrop-filter:blur(6px);
}
.panel-head{
  font-family:'JetBrains Mono',monospace;
  font-size:10px;font-weight:500;letter-spacing:0.1em;
  color:rgba(0,229,255,0.4);
  text-transform:uppercase;
  margin-bottom:14px;
  padding-bottom:10px;
  border-bottom:0.5px solid rgba(0,229,255,0.07);
}
.btn{
  display:inline-block;padding:10px 22px;border-radius:8px;border:none;cursor:pointer;
  font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:600;
  text-decoration:none;margin:4px 4px 4px 0;transition:all 0.2s;
}
.btn-start{background:linear-gradient(135deg,#0097a7,#00695c);color:#fff;}
.btn-stop{background:rgba(255,23,68,0.1);border:0.5px solid rgba(255,23,68,0.3);color:#ff1744;}
.btn-reset{background:rgba(255,255,255,0.04);border:0.5px solid rgba(0,229,255,0.15);color:rgba(148,163,184,0.8);}
.btn-upload{background:linear-gradient(135deg,#6d28d9,#7c3aed);color:#fff;}
input[type=number],input[type=file],select{
  background:rgba(12,17,32,0.9);
  border:0.5px solid rgba(0,229,255,0.15);
  color:#cdd6f4;padding:9px 12px;border-radius:8px;
  font-family:'JetBrains Mono',monospace;font-size:0.82rem;
}
input[type=number],select{width:140px;}
input[type=file]{width:100%;margin-bottom:12px;}
.success-msg{background:rgba(0,255,157,0.06);border:0.5px solid rgba(0,255,157,0.2);border-radius:8px;padding:12px 16px;margin-bottom:16px;color:#00ff9d;font-size:0.82rem;}
.error-msg{background:rgba(255,23,68,0.06);border:0.5px solid rgba(255,23,68,0.2);border-radius:8px;padding:12px 16px;margin-bottom:16px;color:#ff1744;font-size:0.82rem;}
.info-msg{background:rgba(0,120,212,0.06);border:0.5px solid rgba(0,120,212,0.2);border-radius:8px;padding:12px 16px;margin-bottom:16px;color:#60a5fa;font-size:0.82rem;}
.azure-badge{
  display:inline-flex;align-items:center;gap:5px;
  background:rgba(0,120,212,0.08);border:0.5px solid rgba(0,120,212,0.2);
  border-radius:5px;padding:3px 8px;
  font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#60a5fa;
}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:0.58rem;font-weight:600;}
.tag-training{background:rgba(0,255,157,0.08);color:#00ff9d;border:0.5px solid rgba(0,255,157,0.2);}
.tag-idle{background:rgba(255,255,255,0.04);color:rgba(148,163,184,0.5);border:0.5px solid rgba(255,255,255,0.06);}
.tag-done{background:rgba(0,229,255,0.08);color:#00e5ff;border:0.5px solid rgba(0,229,255,0.2);}
.tag-straggler{background:rgba(255,193,7,0.08);color:#ffc107;border:0.5px solid rgba(255,193,7,0.2);}
.chart-wrap{position:relative;}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;}
.leg-item{display:flex;align-items:center;gap:5px;font-size:10px;color:rgba(148,163,184,0.6);font-family:'JetBrains Mono',monospace;}
.leg-dot{width:8px;height:8px;border-radius:2px;flex-shrink:0;}
"""

def nav_header(active="dashboard", status=None):
    status_html = ""
    if status:
        if status.get("error"):
            status_html = '<span class="ps-badge ps-offline">● PS OFFLINE</span>'
        else:
            mode = status.get("mode","sync").upper()
            done = " · DONE" if status.get("done") else ""
            status_html = f'<span class="ps-badge ps-online">● PS ONLINE — Round {status.get("round",0)}/{status.get("max_rounds","?")} [{mode}{done}]</span>'
    return f"""<div class="glow-br"></div>
<header>
  <div class="logo">🌍 EQ Distributed Training</div>
  <nav>
    <a href="/" class="{'active' if active=='dashboard' else ''}">Dashboard</a>
    <a href="/upload" class="{'active' if active=='upload' else ''}">Upload Dataset</a>
    <a href="/analytics" class="{'active' if active=='analytics' else ''}">Training Analytics</a>
  </nav>
  <div>{status_html}</div>
</header>"""

# ── Dashboard HTML ───────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html><html><head>
<meta charset="UTF-8"/><title>EQ Distributed Training</title>
{% if prediction is none %}<meta http-equiv="refresh" content="5">{% endif %}
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>""" + BASE_CSS + """
.loss-row{display:flex;align-items:center;gap:10px;margin-bottom:7px;}
.loss-epoch{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:rgba(148,163,184,0.5);width:68px;}
.loss-bar-wrap{flex:1;height:4px;background:rgba(255,255,255,0.05);border-radius:2px;}
.loss-bar{height:100%;border-radius:2px;background:linear-gradient(90deg,#00e5ff,#00ff9d);}
.loss-val{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#ffc107;width:52px;text-align:right;}
.predict-box{background:rgba(6,8,16,0.6);border:0.5px solid rgba(0,229,255,0.1);border-radius:10px;padding:18px;}
.inp-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;}
.result-big{font-family:'JetBrains Mono',monospace;font-size:2.4rem;font-weight:600;color:#00ff9d;text-align:center;padding:14px;}
.workers-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;}
.worker-card{background:rgba(6,8,16,0.7);border:0.5px solid rgba(0,229,255,0.1);border-radius:10px;padding:14px;}
</style></head><body>
{{ nav | safe }}
<div class="container">
  <div class="metrics">
    <div class="metric">
      <div class="metric-label">training round</div>
      <div class="metric-val" style="color:#00e5ff;">{{ status.get('round',0) }}</div>
      <div class="metric-sub">current</div>
    </div>
    <div class="metric">
      <div class="metric-label">gradient updates</div>
      <div class="metric-val" style="color:#00ff9d;">{{ status.get('total_updates',0) }}</div>
      <div class="metric-sub">total</div>
    </div>
    <div class="metric">
      <div class="metric-label">worker nodes</div>
      <div class="metric-val" style="color:#a78bfa;">{{ status.get('num_workers',2) }}</div>
      <div class="metric-sub">active</div>
    </div>
    <div class="metric">
      <div class="metric-label">PS uptime</div>
      <div class="metric-val" style="color:#ffc107;">{{ status.get('uptime',0) }}s</div>
      <div class="metric-sub">running</div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-head">Training Controls</div>
    <a href="/start" class="btn btn-start">▶ Start Training</a>
    <a href="/stop" class="btn btn-stop">■ Stop</a>
    <a href="/" class="btn btn-reset">↺ Refresh</a>
    {% if status.get('stragglers') %}
    <span style="color:#ffc107;font-size:0.78rem;margin-left:10px;font-family:'JetBrains Mono',monospace;">⚠ Straggler workers: {{ status['stragglers'] }}</span>
    {% endif %}
    {% if status.get('done') %}
    <span style="color:#00e5ff;font-size:0.78rem;margin-left:10px;font-family:'JetBrains Mono',monospace;">✓ Training complete — {{ status.get('round',0) }} rounds</span>
    {% endif %}
    {% if status.get('mode') %}
    <span style="color:rgba(148,163,184,0.4);font-size:0.7rem;margin-left:10px;font-family:'JetBrains Mono',monospace;">MODE: {{ status.get('mode','sync').upper() }}</span>
    {% endif %}
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
    <div class="panel">
      <div class="panel-head">Training Loss Curve</div>
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
        <p style="color:rgba(148,163,184,0.4);font-size:0.82rem;">Training not started yet...</p>
      {% endif %}
    </div>
    <div class="panel">
      <div class="panel-head">Predict Magnitude</div>
      <div class="predict-box">
        <form method="POST" action="/predict">
          <div class="inp-row">
            <div>
              <div style="font-size:0.68rem;color:rgba(148,163,184,0.5);margin-bottom:4px;font-family:'JetBrains Mono',monospace;">DEPTH (km)</div>
              <input type="number" name="depth" value="{{ req_depth }}" step="0.1"/>
            </div>
            <div>
              <div style="font-size:0.68rem;color:rgba(148,163,184,0.5);margin-bottom:4px;font-family:'JetBrains Mono',monospace;">LATITUDE</div>
              <input type="number" name="lat" value="{{ req_lat }}" step="0.01"/>
            </div>
            <div>
              <div style="font-size:0.68rem;color:rgba(148,163,184,0.5);margin-bottom:4px;font-family:'JetBrains Mono',monospace;">LONGITUDE</div>
              <input type="number" name="lon" value="{{ req_lon }}" step="0.01"/>
            </div>
          </div>
          <button type="submit" class="btn btn-start" style="margin:0;">🔮 Predict</button>
        </form>
        {% if prediction is not none %}
        <div class="result-big">{{ "%.2f"|format(prediction) }} M</div>
        <p style="text-align:center;color:rgba(148,163,184,0.4);font-size:0.78rem;">Predicted Magnitude</p>
        {% endif %}
        {% if pred_error %}
        <div class="error-msg" style="margin-top:10px;">{{ pred_error }}</div>
        {% endif %}
      </div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-head">Worker Nodes</div>
    <div class="workers-grid">
      {% for i in range(status.get('num_workers', 2)) %}
      <div class="worker-card">
        <div style="font-weight:600;margin-bottom:5px;color:#00ff9d;font-size:0.9rem;">⚙ Worker {{ i }}</div>
        <div style="font-size:0.75rem;color:rgba(148,163,184,0.4);margin-bottom:8px;font-family:'JetBrains Mono',monospace;">shard_{{ i }}.csv · VM {{ i+1 }}</div>
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
</div></body></html>
"""

# ── Upload HTML ──────────────────────────────────────────────────
UPLOAD_HTML = """<!DOCTYPE html><html><head>
<meta charset="UTF-8"/><title>Upload Dataset</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>""" + BASE_CSS + """
.dropzone{border:1.5px dashed rgba(0,229,255,0.15);border-radius:12px;padding:36px;text-align:center;transition:all 0.3s;margin-bottom:18px;}
.dropzone:hover{border-color:rgba(167,139,250,0.4);background:rgba(167,139,250,0.03);}
.inp-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px;align-items:flex-end;}
.inp-group{display:flex;flex-direction:column;gap:4px;}
.inp-label{font-size:0.68rem;color:rgba(148,163,184,0.5);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:0.08em;}
.shard-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;margin-top:14px;}
.shard-card{background:rgba(6,8,16,0.7);border:0.5px solid rgba(0,229,255,0.1);border-radius:10px;padding:12px;}
.storage-info{background:rgba(0,120,212,0.05);border:0.5px solid rgba(0,120,212,0.15);border-radius:10px;padding:14px;margin-bottom:18px;}
</style></head><body>
{{ nav | safe }}
<div class="container">
  <div class="section-tag">Upload Dataset</div>
  <h2 style="font-size:1.5rem;margin-bottom:6px;font-weight:600;">Upload Earthquake Dataset</h2>
  <p style="color:rgba(148,163,184,0.5);font-size:0.82rem;margin-bottom:22px;">Upload CSV — system auto-splits into shards and distributes across worker nodes.</p>

  {% if azure_enabled %}
  <div class="storage-info">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
      <span class="azure-badge">☁ Azure Blob Storage</span>
      <span style="color:#00ff9d;font-size:0.78rem;">Connected</span>
    </div>
    <div style="color:rgba(148,163,184,0.4);font-size:0.78rem;">Shards uploaded to <span style="color:#60a5fa;">eqdistributed / eq-data</span></div>
  </div>
  {% else %}
  <div class="info-msg">Azure Blob Storage not configured. Shards saved locally only.</div>
  {% endif %}

  {% if success_msg %}<div class="success-msg">✓ {{ success_msg }}</div>{% endif %}
  {% if error_msg %}<div class="error-msg">✕ {{ error_msg }}</div>{% endif %}
  {% if blob_results %}
  <div class="success-msg">☁ Azure Blob Results:<br>{% for r in blob_results %}&nbsp;&nbsp;{{ r }}<br>{% endfor %}</div>
  {% endif %}

  <div class="panel">
    <div class="panel-head">Upload New Dataset</div>
    <form method="POST" action="/upload" enctype="multipart/form-data">
      <div class="dropzone">
        <div style="color:rgba(148,163,184,0.5);font-size:0.88rem;margin-bottom:6px;">Choose a CSV file</div>
        <div style="color:rgba(148,163,184,0.25);font-size:0.72rem;font-family:'JetBrains Mono',monospace;">Supported: .csv files up to 100MB</div>
        <input type="file" name="dataset" accept=".csv" style="margin-top:14px;"/>
      </div>
      <div class="inp-row">
        <div class="inp-group">
          <span class="inp-label">Workers</span>
          <select name="num_workers">
            <option value="2" selected>2 Workers</option>
            <option value="3">3 Workers</option>
            <option value="4">4 Workers</option>
          </select>
        </div>
        <div class="inp-group">
          <span class="inp-label">Max Rows</span>
          <input type="number" name="max_rows" placeholder="All rows" min="1000"/>
        </div>
      </div>
      <button type="submit" class="btn btn-upload">Upload & Split</button>
      {% if azure_enabled %}
      <span style="color:rgba(96,165,250,0.6);font-size:0.7rem;font-family:'JetBrains Mono',monospace;margin-left:10px;">☁ Auto-uploads to Azure Blob</span>
      {% endif %}
    </form>
  </div>

  <div class="panel">
    <div class="panel-head">Required Columns</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
      <div style="background:rgba(6,8,16,0.7);border:0.5px solid rgba(0,229,255,0.1);border-radius:8px;padding:10px;text-align:center;">
        <div style="color:#00e5ff;font-weight:600;font-size:0.85rem;">Magnitude</div>
        <div style="color:rgba(148,163,184,0.35);font-size:0.68rem;margin-top:3px;">target</div>
      </div>
      <div style="background:rgba(6,8,16,0.7);border:0.5px solid rgba(0,229,255,0.1);border-radius:8px;padding:10px;text-align:center;">
        <div style="color:#00ff9d;font-weight:600;font-size:0.85rem;">Depth</div>
        <div style="color:rgba(148,163,184,0.35);font-size:0.68rem;margin-top:3px;">km</div>
      </div>
      <div style="background:rgba(6,8,16,0.7);border:0.5px solid rgba(0,229,255,0.1);border-radius:8px;padding:10px;text-align:center;">
        <div style="color:#a78bfa;font-weight:600;font-size:0.85rem;">Latitude</div>
        <div style="color:rgba(148,163,184,0.35);font-size:0.68rem;margin-top:3px;">coord</div>
      </div>
      <div style="background:rgba(6,8,16,0.7);border:0.5px solid rgba(0,229,255,0.1);border-radius:8px;padding:10px;text-align:center;">
        <div style="color:#ffc107;font-weight:600;font-size:0.85rem;">Longitude</div>
        <div style="color:rgba(148,163,184,0.35);font-size:0.68rem;margin-top:3px;">coord</div>
      </div>
    </div>
  </div>

  {% if shards %}
  <div class="panel">
    <div class="panel-head">Current Shards</div>
    <div class="shard-grid">
      {% for shard in shards %}
      <div class="shard-card">
        <div style="color:#00e5ff;font-family:'JetBrains Mono',monospace;font-size:0.78rem;font-weight:600;">{{ shard.name }}</div>
        <div style="color:rgba(148,163,184,0.4);font-size:0.72rem;margin-top:3px;">{{ shard.rows }} rows · {{ shard.size }}</div>
        {% if azure_enabled %}<div style="margin-top:6px;"><span class="azure-badge">☁ In Azure Blob</span></div>{% endif %}
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}
</div></body></html>
"""

# ── Analytics HTML ───────────────────────────────────────────────
ANALYTICS_HTML = """<!DOCTYPE html><html><head>
<meta charset="UTF-8"/><title>Training Analytics</title>
<meta http-equiv="refresh" content="10">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>""" + BASE_CSS + """
.row2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px;}
.stat-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:0.5px solid rgba(0,229,255,0.06);font-family:'JetBrains Mono',monospace;font-size:0.75rem;}
.stat-row:last-child{border-bottom:none;}
.stat-label{color:rgba(148,163,184,0.45);}
.stat-val{color:#cdd6f4;}
.worker-row{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:0.5px solid rgba(0,229,255,0.06);}
.worker-row:last-child{border-bottom:none;}
.w-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.w-name{font-size:12px;font-family:'JetBrains Mono',monospace;min-width:62px;color:#cdd6f4;}
.w-bar-wrap{flex:1;height:4px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden;}
.w-bar{height:100%;border-radius:2px;}
.w-val{font-size:11px;color:rgba(148,163,184,0.5);min-width:36px;text-align:right;}
.tab-row{display:flex;gap:5px;margin-bottom:12px;flex-wrap:wrap;}
.tab-btn{font-size:10px;padding:3px 9px;border-radius:4px;border:0.5px solid rgba(0,229,255,0.12);cursor:pointer;background:transparent;color:rgba(148,163,184,0.5);font-family:'JetBrains Mono',monospace;transition:all 0.15s;}
.tab-btn.active{background:rgba(0,229,255,0.08);color:#00e5ff;border-color:rgba(0,229,255,0.25);}
.stat-mini{background:rgba(6,8,16,0.7);border:0.5px solid rgba(0,229,255,0.07);border-radius:8px;padding:9px 11px;}
.stat-mini-label{font-size:10px;color:rgba(148,163,184,0.4);font-family:'JetBrains Mono',monospace;margin-bottom:3px;}
.stat-mini-val{font-size:15px;font-weight:600;}
.lat-row{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:0.5px solid rgba(0,229,255,0.06);font-size:11px;}
.lat-row:last-child{border-bottom:none;}
.lat-bar-wrap{width:70px;height:3px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden;}
.lat-bar{height:100%;border-radius:2px;}
</style></head><body>
{{ nav | safe }}
<div class="container">

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:10px;">
    <div>
      <div class="section-tag">Analytics</div>
      <h2 style="font-size:1.5rem;font-weight:600;margin-top:4px;">Training Analytics</h2>
      <p style="color:rgba(148,163,184,0.4);font-size:0.78rem;margin-top:3px;font-family:'JetBrains Mono',monospace;">Auto-refreshes every 10 seconds</p>
    </div>
    <div style="display:flex;gap:8px;">
      <a href="/analytics" class="btn btn-reset" style="font-size:0.7rem;padding:7px 14px;">↺ Refresh</a>
      <a href="/" class="btn btn-start" style="font-size:0.7rem;padding:7px 14px;">← Dashboard</a>
    </div>
  </div>

  {% if not has_data %}
  <div class="panel" style="text-align:center;padding:48px;">
    <p style="color:rgba(148,163,184,0.6);font-size:1rem;margin-bottom:8px;">No training data yet.</p>
    <p style="color:rgba(148,163,184,0.35);font-size:0.82rem;">Start training from the Dashboard then come back here.</p>
    <a href="/" class="btn btn-start" style="margin-top:18px;display:inline-block;">Go to Dashboard</a>
  </div>
  {% else %}

  <div class="metrics">
    <div class="metric">
      <div class="metric-label">workers trained</div>
      <div class="metric-val" style="color:#00ff9d;">{{ num_workers }}</div>
      <div class="metric-sub">this session</div>
    </div>
    <div class="metric">
      <div class="metric-label">total epochs</div>
      <div class="metric-val" style="color:#00e5ff;">{{ total_epochs }}</div>
      <div class="metric-sub">combined</div>
    </div>
    <div class="metric">
      <div class="metric-label">best RMSE</div>
      <div class="metric-val" style="color:#a78bfa;">{{ "%.4f"|format(best_rmse) }}</div>
      <div class="metric-sub">lowest achieved</div>
    </div>
    <div class="metric">
      <div class="metric-label">final PS loss</div>
      <div class="metric-val" style="color:#ffc107;">{{ "%.4f"|format(final_loss) }}</div>
      <div class="metric-sub">last round</div>
    </div>
  </div>

  <div class="row2">
    <div class="panel">
      <div class="panel-head">RMSE Convergence per Worker</div>
      <div class="chart-wrap" style="height:190px;"><canvas id="rmseChart"></canvas></div>
      <div class="legend" id="rmseLegend"></div>
    </div>
    <div class="panel">
      <div class="panel-head">Training Loss per Worker</div>
      <div class="chart-wrap" style="height:190px;"><canvas id="lossChart"></canvas></div>
      <div class="legend" id="lossLegend"></div>
    </div>
  </div>

  <div class="row2">
    <div class="panel">
      <div class="panel-head">RMSE Improvement — Doughnut</div>
      <div class="chart-wrap" style="height:170px;"><canvas id="pieChart"></canvas></div>
      <div class="legend" id="pieLegend"></div>
    </div>
    <div class="panel">
      <div class="panel-head">PS Communication Latency</div>
      <div id="latRows"></div>
      <div style="margin-top:14px;">
        <div style="font-size:10px;color:rgba(0,229,255,0.35);font-family:'JetBrains Mono',monospace;margin-bottom:8px;letter-spacing:0.08em;">LATENCY BAR COMPARISON</div>
        <div class="chart-wrap" style="height:100px;"><canvas id="latChart"></canvas></div>
      </div>
    </div>
  </div>

  <div class="row2">
    <div class="panel">
      <div class="panel-head">Worker Node Status</div>
      <div id="workerRows"></div>
    </div>
    <div class="panel">
      <div class="panel-head">Per-Worker Stats</div>
      <div class="tab-row" id="workerTabs"></div>
      <div id="workerStatContent"></div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-head">Parameter Server Aggregated Loss</div>
    <div class="chart-wrap" style="height:130px;"><canvas id="psChart"></canvas></div>
  </div>

  {% endif %}
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const COLORS = ["#00e5ff","#00ff9d","#f97316","#a78bfa"];
const DASHES = [[]],[5,3],[4,4],[8,3]];
const gridColor = 'rgba(0,229,255,0.06)';
const tickColor = 'rgba(148,163,184,0.45)';

const workerData = {{ worker_chart_data | tojson }};
const psRounds   = {{ ps_rounds | tojson }};
const psLosses   = {{ ps_losses | tojson }};
const maxEpochs  = {{ max_epochs }};

const wids = Object.keys(workerData).map(Number);
const labels = Array.from({length: maxEpochs}, (_,i) => i+1);

function buildLegend(elId) {
  document.getElementById(elId).innerHTML = wids
    .filter(wid => workerData[wid].rmses && workerData[wid].rmses.length)
    .map(wid => `<span class="leg-item"><span class="leg-dot" style="background:${COLORS[wid%4]};"></span>Worker ${wid}</span>`)
    .join('');
}

new Chart(document.getElementById('rmseChart'), {
  type: 'line',
  data: {
    labels,
    datasets: wids.filter(wid=>workerData[wid].rmses&&workerData[wid].rmses.length).map(wid => ({
      label: `Worker ${wid}`,
      data: workerData[wid].rmses,
      borderColor: COLORS[wid%4],
      backgroundColor: 'transparent',
      borderWidth: 1.8,
      pointRadius: 2.5,
      tension: 0.3,
      borderDash: DASHES[wid%4],
    }))
  },
  options: {
    responsive:true, maintainAspectRatio:false,
    plugins:{legend:{display:false}},
    scales:{
      x:{ticks:{font:{size:9},color:tickColor},grid:{color:gridColor}},
      y:{ticks:{font:{size:9},color:tickColor},grid:{color:gridColor}}
    }
  }
});
buildLegend('rmseLegend');

new Chart(document.getElementById('lossChart'), {
  type: 'line',
  data: {
    labels,
    datasets: wids.filter(wid=>workerData[wid].losses&&workerData[wid].losses.length).map(wid => ({
      label: `Worker ${wid}`,
      data: workerData[wid].losses,
      borderColor: COLORS[wid%4],
      backgroundColor: 'transparent',
      borderWidth: 1.8,
      pointRadius: 2.5,
      tension: 0.3,
      borderDash: DASHES[wid%4],
    }))
  },
  options: {
    responsive:true, maintainAspectRatio:false,
    plugins:{legend:{display:false}},
    scales:{
      x:{ticks:{font:{size:9},color:tickColor},grid:{color:gridColor}},
      y:{ticks:{font:{size:9},color:tickColor},grid:{color:gridColor}}
    }
  }
});
buildLegend('lossLegend');

const improvements = wids.filter(wid=>workerData[wid].rmses&&workerData[wid].rmses.length).map(wid=>{
  const r = workerData[wid].rmses;
  return parseFloat(((1 - r[r.length-1]/r[0])*100).toFixed(1));
});
const pieWids = wids.filter(wid=>workerData[wid].rmses&&workerData[wid].rmses.length);
new Chart(document.getElementById('pieChart'), {
  type:'doughnut',
  data:{
    labels: pieWids.map(wid=>`Worker ${wid}`),
    datasets:[{
      data: improvements,
      backgroundColor: pieWids.map(wid=>COLORS[wid%4]),
      borderWidth: 0, hoverOffset: 5
    }]
  },
  options:{
    responsive:true, maintainAspectRatio:false, cutout:'62%',
    plugins:{
      legend:{display:false},
      tooltip:{callbacks:{label:(c)=>`${c.label}: ${c.parsed.toFixed(1)}% improvement`}}
    }
  }
});
document.getElementById('pieLegend').innerHTML = pieWids.map((wid,i)=>
  `<span class="leg-item"><span class="leg-dot" style="background:${COLORS[wid%4]};"></span>Worker ${wid} ${improvements[i]}%</span>`
).join('');

const latWids = wids.filter(wid=>workerData[wid].comms&&workerData[wid].comms.length);
const latVals = latWids.map(wid=>{const c=workerData[wid].comms.filter(x=>x>0);return c.length?(c.reduce((a,b)=>a+b,0)/c.length).toFixed(1):0;});
const maxLat  = Math.max(...latVals.map(Number), 0.1);
document.getElementById('latRows').innerHTML = latWids.map((wid,i)=>`
  <div class="lat-row">
    <span style="color:${COLORS[wid%4]};font-family:'JetBrains Mono',monospace;font-weight:600;min-width:62px;">Worker ${wid}</span>
    <div class="lat-bar-wrap"><div class="lat-bar" style="width:${(latVals[i]/maxLat*100).toFixed(0)}%;background:${COLORS[wid%4]};"></div></div>
    <span style="color:rgba(148,163,184,0.5);min-width:36px;">${latVals[i]}ms</span>
  </div>`).join('');

new Chart(document.getElementById('latChart'), {
  type:'bar',
  data:{
    labels: latWids.map(wid=>`W${wid}`),
    datasets:[{data:latVals.map(Number), backgroundColor:latWids.map(wid=>COLORS[wid%4]), borderRadius:3, borderWidth:0}]
  },
  options:{
    responsive:true, maintainAspectRatio:false,
    plugins:{legend:{display:false}},
    scales:{
      x:{ticks:{font:{size:9},color:tickColor},grid:{display:false}},
      y:{ticks:{font:{size:9},color:tickColor},grid:{color:gridColor},title:{display:true,text:'ms',font:{size:9},color:tickColor}}
    }
  }
});

const maxEp = Math.max(...wids.map(wid=>workerData[wid].rmses?workerData[wid].rmses.length:0));
document.getElementById('workerRows').innerHTML = wids.map(wid=>{
  const w = workerData[wid];
  const ep = w.rmses ? w.rmses.length : 0;
  return `<div class="worker-row">
    <div class="w-dot" style="background:${COLORS[wid%4]};"></div>
    <div class="w-name">Worker ${wid}</div>
    <div class="w-bar-wrap"><div class="w-bar" style="width:${maxEp>0?(ep/maxEp*100).toFixed(0):0}%;background:${COLORS[wid%4]};"></div></div>
    <div class="w-val">${ep} ep</div>
    <span class="tag tag-done" style="font-size:9px;">complete</span>
  </div>`;
}).join('');

const tabsEl = document.getElementById('workerTabs');
const statEl = document.getElementById('workerStatContent');
let activeW = wids[0];

function renderStat(wid) {
  const w = workerData[wid];
  const r = w.rmses||[], l = w.losses||[], c = (w.comms||[]).filter(x=>x>0);
  const imp = r.length>=2 ? ((1-r[r.length-1]/r[0])*100).toFixed(1) : 'N/A';
  const avgLat = c.length ? (c.reduce((a,b)=>a+b,0)/c.length).toFixed(1) : 'N/A';
  statEl.innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;">
      <div class="stat-mini"><div class="stat-mini-label">Epochs</div><div class="stat-mini-val">${r.length}</div></div>
      <div class="stat-mini"><div class="stat-mini-label">Improvement</div><div class="stat-mini-val" style="color:#00ff9d;">${imp}%</div></div>
      <div class="stat-mini"><div class="stat-mini-label">Init RMSE</div><div class="stat-mini-val">${r.length?r[0].toFixed(4):'N/A'}</div></div>
      <div class="stat-mini"><div class="stat-mini-label">Final RMSE</div><div class="stat-mini-val" style="color:${COLORS[wid%4]};">${r.length?r[r.length-1].toFixed(4):'N/A'}</div></div>
      <div class="stat-mini"><div class="stat-mini-label">Init loss</div><div class="stat-mini-val">${l.length?l[0].toFixed(4):'N/A'}</div></div>
      <div class="stat-mini"><div class="stat-mini-label">Final loss</div><div class="stat-mini-val" style="color:${COLORS[wid%4]};">${l.length?l[l.length-1].toFixed(4):'N/A'}</div></div>
      <div class="stat-mini" style="grid-column:1/3;"><div class="stat-mini-label">Avg PS latency</div><div class="stat-mini-val">${avgLat}ms</div></div>
    </div>`;
}

tabsEl.innerHTML = wids.map(wid=>
  `<button class="tab-btn${wid===activeW?' active':''}" data-wid="${wid}" style="border-left:2px solid ${COLORS[wid%4]};">W${wid}</button>`
).join('');
tabsEl.addEventListener('click', e => {
  const btn = e.target.closest('.tab-btn');
  if(!btn) return;
  activeW = parseInt(btn.dataset.wid);
  tabsEl.querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('active',parseInt(b.dataset.wid)===activeW));
  renderStat(activeW);
});
renderStat(activeW);

if(psRounds.length>1){
  new Chart(document.getElementById('psChart'),{
    type:'line',
    data:{
      labels:psRounds,
      datasets:[{
        label:'PS loss',data:psLosses,
        borderColor:'#ffc107',backgroundColor:'rgba(255,193,7,0.05)',
        fill:true,borderWidth:1.5,pointRadius:1.5,tension:0.4
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{font:{size:9},color:tickColor,maxTicksLimit:7},grid:{color:gridColor},title:{display:true,text:'round',font:{size:9},color:tickColor}},
        y:{ticks:{font:{size:9},color:tickColor},grid:{color:gridColor}}
      }
    }
  });
}
</script>
</body></html>
"""

# ── Routes ───────────────────────────────────────────────────────
@app.route("/")
def index():
    status = ps_request({"type": "get_status"})
    if "data" in status: status = status["data"]
    elif "error" in status: status = {}
    status = fix_worker_count(status)
    return render_template_string(DASHBOARD_HTML,
        nav=nav_header("dashboard", status), status=status,
        prediction=None, pred_error=None,
        req_depth=10, req_lat=35.0, req_lon=140.0)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET": return redirect("/")
    depth = float(request.form.get("depth", 10))
    lat   = float(request.form.get("lat", 35.0))
    lon   = float(request.form.get("lon", 140.0))
    resp   = ps_request({"type": "get_weights"})
    status = ps_request({"type": "get_status"})
    if "data" in status: status = status["data"]
    elif "error" in status: status = {}
    status = fix_worker_count(status)
    prediction = None
    pred_error = None
    if "weights" in resp:
        w = np.array(resp["weights"])
        X = np.array([depth, lat, lon], dtype=np.float32)
        X = (X - np.array([30.0, 0.0, 0.0])) / (np.array([50.0, 45.0, 90.0]) + 1e-8)
        prediction = float(X @ w[:-1] + w[-1])
    else:
        pred_error = "PS offline or training not started yet."
    return render_template_string(DASHBOARD_HTML,
        nav=nav_header("dashboard", status), status=status,
        prediction=prediction, pred_error=pred_error,
        req_depth=depth, req_lat=lat, req_lon=lon)

@app.route("/start")
def start_training():
    status = ps_request({"type": "get_status"})
    if "data" in status: status = status["data"]
    elif "error" in status: status = {}
    status = fix_worker_count(status)
    return render_template_string(DASHBOARD_HTML,
        nav=nav_header("dashboard", status), status=status,
        prediction=None, pred_error=None,
        req_depth=10, req_lat=35.0, req_lon=140.0)

@app.route("/stop")
def stop_training():
    status = ps_request({"type": "get_status"})
    if "data" in status: status = status["data"]
    elif "error" in status: status = {}
    status = fix_worker_count(status)
    return render_template_string(DASHBOARD_HTML,
        nav=nav_header("dashboard", status), status=status,
        prediction=None, pred_error=None,
        req_depth=10, req_lat=35.0, req_lon=140.0)

@app.route("/status")
def api_status():
    return jsonify(ps_request({"type": "get_status"}))

@app.route("/analytics")
def analytics_page():
    workers = load_all_worker_histories()
    ps_rounds, ps_losses = load_ps_loss_history()

    if not workers and not ps_rounds:
        return render_template_string(ANALYTICS_HTML,
            nav=nav_header("analytics"), has_data=False,
            num_workers=0, total_epochs=0, best_rmse=0,
            final_loss=0, worker_chart_data={},
            ps_rounds=[], ps_losses=[],
            max_epochs=1, rmse_max=1, rmse_min=0,
            loss_max=1, loss_min=0, comm_max=100, comm_min=0)

    COLORS = ["#00e5ff", "#00ff9d", "#f97316", "#a78bfa"]
    DASHES = ["", "8 4", "4 4", "12 3"]
    worker_chart_data = {}
    all_rmse, all_loss, all_comm = [], [], []

    for wid, wdata in workers.items():
        rmses  = wdata.get("rmses", [])
        losses = wdata.get("losses", [])
        comms  = [c for c in wdata.get("comms", []) if c > 0]
        worker_chart_data[wid] = {
            "rmses": rmses, "losses": losses, "comms": comms,
            "color": COLORS[wid % len(COLORS)],
            "dash":  DASHES[wid % len(DASHES)],
        }
        all_rmse += rmses
        all_loss += losses
        all_comm += comms

    rmse_max = max(all_rmse) * 1.05 if all_rmse else 1
    rmse_min = min(all_rmse) * 0.95 if all_rmse else 0
    loss_max = max(all_loss) * 1.05 if all_loss else 1
    loss_min = min(all_loss) * 0.95 if all_loss else 0
    comm_max = max(all_comm) * 1.1  if all_comm else 100
    comm_min = min(all_comm) * 0.9  if all_comm else 0

    max_epochs   = max((len(w["rmses"]) for w in worker_chart_data.values()), default=1)
    num_workers  = len(workers)
    total_epochs = sum(len(w.get("rmses", [])) for w in workers.values())
    best_rmse    = min(all_rmse) if all_rmse else 0
    final_loss   = ps_losses[-1] if ps_losses else (all_loss[-1] if all_loss else 0)

    return render_template_string(ANALYTICS_HTML,
        nav=nav_header("analytics"), has_data=True,
        num_workers=num_workers, total_epochs=total_epochs,
        best_rmse=best_rmse, final_loss=final_loss,
        worker_chart_data=worker_chart_data,
        ps_rounds=ps_rounds, ps_losses=ps_losses,
        max_epochs=max_epochs,
        rmse_max=rmse_max, rmse_min=rmse_min,
        loss_max=loss_max, loss_min=loss_min,
        comm_max=comm_max, comm_min=comm_min)

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
        cleanup_old_workers(num_workers)
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
                    blob_results.append(f"shard_{i}.csv → Azure Blob ({len(shard):,} rows)")
                else:
                    blob_results.append(f"shard_{i}.csv saved locally only — {msg}")
        ps_request({"type": "set_num_workers", "num_workers": num_workers})
        print(f"[UPLOAD] Notified PS: expecting {num_workers} workers")
        success_msg = f"'{filename}' uploaded — {num_workers} shards created ({len(df):,} rows)"
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