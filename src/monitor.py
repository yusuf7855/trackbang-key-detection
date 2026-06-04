"""
monitor.py — Scraper + Eğitim Dashboard
Kullanım: python src/monitor.py
Tarayıcı: http://localhost:8765
"""

import re
import json
import os
import glob
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT          = 8765
SCRAPER_LOG   = "/tmp/scraper.log"
TRAIN_LOG     = "/tmp/train.log"
CACHE_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "cache")

ANSI_RE   = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
METRIC_RE = re.compile(r'([\w_]+):\s*([\d.]+(?:e[+-]?\d+)?)')


def clean(raw: str) -> str:
    s = ANSI_RE.sub('', raw)
    while '\x08' in s:
        prev = s
        s = re.sub(r'[^\x08]\x08', '', s)
        if s == prev: break
    return s.replace('\x08', '').strip()


def pid_alive(pattern: str) -> bool:
    try:
        out = subprocess.check_output(['pgrep', '-f', pattern], text=True)
        return bool(out.strip())
    except Exception:
        return False


# ── Scraper log parse ─────────────────────────────────────────────────────────

def parse_scraper(log_path: str = SCRAPER_LOG) -> dict:
    if not os.path.exists(log_path):
        return {"ok": False, "total_queries": 0, "done_queries": 0,
                "new_samples": 0, "dataset_size": 0, "current_query": "—",
                "recent": [], "alive": False}

    total_queries = 0
    done_queries  = 0
    new_samples   = 0
    dataset_size  = 0
    current_query = "—"
    recent        = []

    with open(log_path, 'r', errors='ignore') as f:
        for raw in f:
            line = clean(raw)

            m = re.search(r'Toplam arama sorgusu:\s*(\d+)', line)
            if m:
                total_queries = int(m.group(1))

            m = re.search(r'Mevcut etiket sayısı:\s*(\d+)', line)
            if m:
                dataset_size = int(m.group(1))

            m = re.search(r"Arama:\s*'(.+?)'\s*\(", line)
            if m:
                current_query = m.group(1)
                done_queries += 1

            if line.startswith('+ '):
                new_samples += 1
                recent.append(line[2:])

    # Cache'ten gerçek sayıyı al
    try:
        cache_count = len(glob.glob(os.path.join(CACHE_DIR, "*.npy")))
    except Exception:
        cache_count = dataset_size

    return {
        "ok":            True,
        "total_queries": total_queries,
        "done_queries":  done_queries,
        "new_samples":   new_samples,
        "dataset_size":  cache_count,
        "current_query": current_query,
        "recent":        recent[-10:],
        "alive":         pid_alive("scraper.py"),
    }


# ── Train log parse ───────────────────────────────────────────────────────────

def parse_train(log_path: str = TRAIN_LOG) -> dict:
    # En son retrain log'unu bul
    cycle_logs = sorted(glob.glob("/tmp/train_cycle*.log"))
    if cycle_logs:
        # En yeni cycle log'u kullan eğer train.log'dan daha yeniyse
        latest_cycle = cycle_logs[-1]
        if os.path.exists(log_path):
            if os.path.getmtime(latest_cycle) > os.path.getmtime(log_path):
                log_path = latest_cycle
        else:
            log_path = latest_cycle

    if not os.path.exists(log_path):
        return {"ok": False, "epochs": [], "current": {}, "total_epochs": 100, "alive": False}

    epochs       = {}
    current      = {}
    total_epochs = 100
    total_batches= 59
    cur_epoch    = 0

    with open(log_path, 'r', errors='ignore') as f:
        for raw in f:
            line = clean(raw)

            m = re.match(r'Epoch (\d+)/(\d+)', line)
            if m:
                cur_epoch    = int(m.group(1))
                total_epochs = int(m.group(2))
                current = {"epoch": cur_epoch, "batch": 0,
                           "total_batches": total_batches, "eta": "--", "metrics": {}}
                continue

            m = re.search(r'Epoch (\d+): val_key_output_accuracy improved from [\d.\-inf]+ to ([\d.]+)', line)
            if m:
                ep = int(m.group(1))
                epochs.setdefault(ep, {})['val_key_output_accuracy'] = float(m.group(2))
                epochs[ep]['_improved'] = True
                continue

            m = re.search(r'Epoch (\d+): val_key_output_accuracy did not improve from ([\d.]+)', line)
            if m:
                ep = int(m.group(1))
                epochs.setdefault(ep, {})['val_key_output_accuracy'] = float(m.group(2))
                epochs[ep]['_improved'] = False
                continue

            m = re.match(r'\s*(\d+)/(\d+)\s+[━\s]+\s*([\d:s.]+)\s+[\d.]+s/step(.+)', line)
            if m and cur_epoch:
                batch = int(m.group(1)); tb = int(m.group(2))
                eta   = m.group(3);      rest = m.group(4)
                metrics = {k: float(v) for k, v in METRIC_RE.findall(rest)}
                total_batches = tb
                if batch == tb:
                    epochs.setdefault(cur_epoch, {}).update(metrics)
                current = {"epoch": cur_epoch, "batch": batch,
                           "total_batches": tb, "eta": eta, "metrics": metrics}

    epoch_list = [{"epoch": e, **epochs[e]} for e in sorted(epochs)]
    return {
        "ok":           True,
        "total_epochs": total_epochs,
        "total_batches":total_batches,
        "epochs":       epoch_list,
        "current":      current,
        "alive":        pid_alive("train.py"),
    }


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TrackBang — Key Detection Pipeline</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f0f13; color: #e8e8f0; min-height: 100vh; }

header { background: #1a1a24; border-bottom: 1px solid #2e2e42;
         padding: 14px 24px; display: flex; align-items: center; gap: 12px; }
header h1 { font-size: 17px; font-weight: 600; }
.badge { font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
.badge-live { background: #14532d; color: #22c55e; }
.badge-dead { background: #3f1717; color: #f87171; }
.last-update { margin-left: auto; font-size: 12px; color: #6b6b8a; }

/* ── Sections ── */
.section { border-bottom: 1px solid #2e2e42; }
.section-title { background: #1a1a24; padding: 10px 24px;
                 font-size: 11px; font-weight: 700; letter-spacing: 1px;
                 text-transform: uppercase; color: #6b6b8a; display: flex; align-items: center; gap: 8px; }

/* ── Progress bar ── */
.prog-wrap { padding: 16px 24px; }
.prog-row  { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.prog-label{ font-size: 12px; color: #6b6b8a; width: 90px; }
.prog-bar  { flex: 1; height: 10px; background: #2e2e42; border-radius: 5px; overflow: hidden; }
.prog-fill { height: 100%; border-radius: 5px; transition: width 0.5s; }
.fill-scraper { background: linear-gradient(90deg, #f59e0b, #ef4444); }
.fill-epoch   { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.fill-batch   { background: linear-gradient(90deg, #0ea5e9, #22c55e); }
.prog-pct  { font-size: 12px; color: #c8c8e0; width: 80px; text-align: right; font-variant-numeric: tabular-nums; }

/* ── Cards ── */
.cards { display: grid; gap: 1px; background: #2e2e42; }
.cards-4 { grid-template-columns: repeat(4, 1fr); }
.cards-3 { grid-template-columns: repeat(3, 1fr); }
.card { background: #15151f; padding: 18px 22px; }
.card-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.8px; color: #6b6b8a; margin-bottom: 6px; }
.card-value { font-size: 28px; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }
.card-sub   { font-size: 12px; color: #9b9bb8; margin-top: 5px; }
.green  { color: #22c55e; }
.yellow { color: #fbbf24; }
.blue   { color: #60a5fa; }
.orange { color: #fb923c; }

/* ── Charts ── */
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: #2e2e42; }
.chart-card  { background: #15151f; padding: 18px 20px; }
.chart-title { font-size: 12px; font-weight: 600; margin-bottom: 12px; color: #c8c8e0; }
canvas { max-height: 200px; }

/* ── Log tail ── */
.log { background: #0a0a0f; padding: 12px 24px; font-family: 'SF Mono','Fira Code',monospace;
       font-size: 11px; color: #4a4a6a; max-height: 100px; overflow: hidden; }
.log p { line-height: 1.7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.log p.new { color: #9b9bb8; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
</style>
</head>
<body>

<header>
  <h1>TrackBang — Key Detection Pipeline</h1>
  <span class="badge" id="scraperBadge">SCRAPER</span>
  <span class="badge" id="trainBadge">EĞİTİM</span>
  <span class="last-update" id="lastUpdate">—</span>
</header>

<!-- ── SCRAPER BÖLÜMÜ ── -->
<div class="section">
  <div class="section-title">
    <span style="color:#fb923c">▶</span> SCRAPER — SoundCloud Veri Toplama
  </div>

  <div class="prog-wrap">
    <div class="prog-row">
      <span class="prog-label">Sorgular</span>
      <div class="prog-bar"><div class="prog-fill fill-scraper" id="scraperBar" style="width:0%"></div></div>
      <span class="prog-pct" id="scraperPct">0 / 0</span>
    </div>
    <div style="font-size:12px; color:#6b6b8a; margin-top:4px" id="currentQuery">Yükleniyor…</div>
  </div>

  <div class="cards cards-4">
    <div class="card">
      <div class="card-label">Toplam Cache</div>
      <div class="card-value orange" id="cacheCount">—</div>
      <div class="card-sub">eğitimde kullanılabilir</div>
    </div>
    <div class="card">
      <div class="card-label">Bu Turda Eklenen</div>
      <div class="card-value green" id="newSamples">—</div>
      <div class="card-sub">yeni .npy dosyası</div>
    </div>
    <div class="card">
      <div class="card-label">Sorgular</div>
      <div class="card-value blue" id="doneQueries">—</div>
      <div class="card-sub" id="totalQueries">/ — toplam</div>
    </div>
    <div class="card">
      <div class="card-label">Son Track</div>
      <div class="card-value" style="color:#c8c8e0; font-size:13px; margin-top:4px" id="lastTrack">—</div>
      <div class="card-sub">key — BPM</div>
    </div>
  </div>

  <div class="log" id="scraperLog">
    <p style="color:#3a3a5a">Scraper log bekleniyor…</p>
  </div>
</div>

<!-- ── EĞİTİM BÖLÜMÜ ── -->
<div class="section">
  <div class="section-title">
    <span style="color:#8b5cf6">▶</span> EĞİTİM — CNN Model
  </div>

  <div class="prog-wrap">
    <div class="prog-row">
      <span class="prog-label">Epoch</span>
      <div class="prog-bar"><div class="prog-fill fill-epoch" id="epochBar" style="width:0%"></div></div>
      <span class="prog-pct" id="epochPct">0 / 0</span>
    </div>
    <div class="prog-row">
      <span class="prog-label">Batch</span>
      <div class="prog-bar"><div class="prog-fill fill-batch" id="batchBar" style="width:0%"></div></div>
      <span class="prog-pct" id="batchPct">0 / 0</span>
    </div>
    <div style="font-size:12px; color:#6b6b8a; margin-top:4px" id="etaRow">—</div>
  </div>

  <div class="cards cards-3">
    <div class="card">
      <div class="card-label">Train Key Accuracy</div>
      <div class="card-value green" id="trainAcc">—</div>
      <div class="card-sub">anlık batch</div>
    </div>
    <div class="card">
      <div class="card-label">Val Key Accuracy</div>
      <div class="card-value yellow" id="valAcc">—</div>
      <div class="card-sub" id="valAccSub">en iyi epoch</div>
    </div>
    <div class="card">
      <div class="card-label">Val BPM MAE</div>
      <div class="card-value blue" id="valBpm">—</div>
      <div class="card-sub">gerçek BPM hatası</div>
    </div>
  </div>

  <div class="charts">
    <div class="chart-card">
      <div class="chart-title">Key Accuracy (%)</div>
      <canvas id="accChart"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">Loss</div>
      <canvas id="lossChart"></canvas>
    </div>
  </div>
</div>

<script>
const accChart  = mkChart('accChart',  ['Train Acc','Val Acc'],  ['#22c55e','#fbbf24']);
const lossChart = mkChart('lossChart', ['Train Loss','Val Loss'],['#60a5fa','#f87171']);

function mkChart(id, labels, colors) {
  return new Chart(document.getElementById(id).getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: labels.map((lbl, i) => ({
        label: lbl, data: [], borderColor: colors[i],
        backgroundColor: colors[i] + '18', fill: true,
        tension: 0.35, pointRadius: 2, borderWidth: 2,
      }))
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      animation: { duration: 250 },
      plugins: { legend: { labels: { color: '#9b9bb8', font: { size: 11 } } } },
      scales: {
        x: { ticks: { color:'#5a5a7a', font:{size:10} }, grid:{color:'#1e1e2e'} },
        y: { ticks: { color:'#5a5a7a', font:{size:10} }, grid:{color:'#1e1e2e'} },
      }
    }
  });
}

function badge(id, alive, label) {
  const el = document.getElementById(id);
  el.textContent = label + (alive ? ' ▶' : ' ■');
  el.className   = 'badge ' + (alive ? 'badge-live' : 'badge-dead');
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function refresh() {
  try {
    const [sr, tr] = await Promise.all([fetch('/api/scraper'), fetch('/api/train')]);
    const s = await sr.json();
    const t = await tr.json();

    document.getElementById('lastUpdate').textContent =
      new Date().toLocaleTimeString('tr-TR');

    // ── Scraper ──
    badge('scraperBadge', s.alive, 'SCRAPER');

    const sqPct = s.total_queries > 0
      ? Math.round(s.done_queries / s.total_queries * 100) : 0;
    document.getElementById('scraperBar').style.width = sqPct + '%';
    document.getElementById('scraperPct').textContent =
      s.done_queries + ' / ' + s.total_queries + '  (' + sqPct + '%)';
    document.getElementById('currentQuery').textContent =
      'Şu an: ' + s.current_query;
    document.getElementById('cacheCount').textContent  = s.dataset_size.toLocaleString('tr-TR');
    document.getElementById('newSamples').textContent  = '+' + s.new_samples;
    document.getElementById('doneQueries').textContent = s.done_queries;
    document.getElementById('totalQueries').textContent= '/ ' + s.total_queries + ' toplam';

    if (s.recent && s.recent.length > 0) {
      const last = s.recent[s.recent.length - 1];
      // "G# Minor   123 BPM  Funky House..." → key + title
      document.getElementById('lastTrack').textContent = last.substring(0, 40);
    }

    // Scraper log tail
    const lr = await fetch('/log/scraper');
    const lt = await lr.text();
    const lines = lt.split('\n').filter(l => l.trim()).slice(-5);
    document.getElementById('scraperLog').innerHTML =
      lines.map((l,i) =>
        `<p class="${i===lines.length-1?'new':''}">${esc(l)}</p>`
      ).join('');

    // ── Eğitim ──
    badge('trainBadge', t.alive, 'EĞİTİM');

    const cur  = t.current || {};
    const ep   = cur.epoch || 0;
    const bat  = cur.batch || 0;
    const tBat = cur.total_batches || t.total_batches || 59;
    const tEp  = t.total_epochs || 100;
    const m    = cur.metrics || {};

    const epPct  = tEp  > 0 ? Math.round(ep  / tEp  * 100) : 0;
    const batPct = tBat > 0 ? Math.round(bat / tBat * 100) : 0;
    document.getElementById('epochBar').style.width  = epPct  + '%';
    document.getElementById('batchBar').style.width  = batPct + '%';
    document.getElementById('epochPct').textContent  = ep + ' / ' + tEp;
    document.getElementById('batchPct').textContent  = bat + ' / ' + tBat;
    document.getElementById('etaRow').textContent    =
      `Epoch ${ep}/${tEp} — Batch ${bat}/${tBat} — ETA: ${cur.eta||'—'}`;

    if (m.key_output_accuracy !== undefined)
      document.getElementById('trainAcc').textContent =
        (m.key_output_accuracy * 100).toFixed(1) + '%';

    const epochs = t.epochs || [];
    if (epochs.length > 0) {
      const best = epochs.reduce((a,b) =>
        (b.val_key_output_accuracy||0) > (a.val_key_output_accuracy||0) ? b : a);
      const last = epochs[epochs.length-1];
      document.getElementById('valAcc').textContent    =
        ((best.val_key_output_accuracy||0)*100).toFixed(1) + '%';
      document.getElementById('valAccSub').textContent =
        'en iyi: epoch ' + best.epoch;
      if (last.val_bpm_output_mae !== undefined)
        document.getElementById('valBpm').textContent =
          (last.val_bpm_output_mae*200).toFixed(1) + ' BPM';

      const labels = epochs.map(e => 'E'+e.epoch);
      accChart.data.labels  = lossChart.data.labels = labels;
      accChart.data.datasets[0].data  = epochs.map(e => +((e.key_output_accuracy||0)*100).toFixed(1));
      accChart.data.datasets[1].data  = epochs.map(e => +((e.val_key_output_accuracy||0)*100).toFixed(1));
      lossChart.data.datasets[0].data = epochs.map(e => +((e.loss||0).toFixed(3)));
      lossChart.data.datasets[1].data = epochs.map(e => +((e.val_loss||0).toFixed(3)));
      accChart.update(); lossChart.update();
    }

  } catch(e) { console.error(e); }
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path == '/':
            self._send(200, 'text/html; charset=utf-8', HTML.encode())
        elif self.path == '/api/scraper':
            self._send(200, 'application/json',
                       json.dumps(parse_scraper(), ensure_ascii=False).encode())
        elif self.path == '/api/train':
            self._send(200, 'application/json',
                       json.dumps(parse_train(), ensure_ascii=False).encode())
        elif self.path == '/log/scraper':
            self._send_tail(SCRAPER_LOG)
        elif self.path == '/log/train':
            self._send_tail(TRAIN_LOG)
        else:
            self._send(404, 'text/plain', b'Not found')

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def _send_tail(self, path):
        try:
            with open(path, 'r', errors='ignore') as f:
                lines = f.readlines()
            tail = [clean(l) for l in lines[-20:] if clean(l)]
            body = '\n'.join(tail).encode('utf-8', errors='replace')
        except Exception:
            body = b'(log okunamadi)'
        self._send(200, 'text/plain; charset=utf-8', body)


def main():
    print(f"\n{'='*50}")
    print(f"  Pipeline Monitörü — http://localhost:{PORT}")
    print(f"  Scraper log : {SCRAPER_LOG}")
    print(f"  Train log   : {TRAIN_LOG}")
    print(f"  Çıkmak için  Ctrl+C")
    print(f"{'='*50}\n")
    server = HTTPServer(('127.0.0.1', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMonitör kapatıldı.")


if __name__ == '__main__':
    main()
