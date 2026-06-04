#!/usr/bin/env python3
"""
Key Detection DNN — Real-time Training Dashboard
Kullanım: python dashboard.py
Tarayıcıda açılır: http://localhost:7860
"""
import os, time, threading, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_output.log')
PORT = 7860

# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Key Detection — Training Dashboard</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #07060f;
    --card:    #0e0c1a;
    --border:  rgba(124,58,237,.22);
    --purple:  #7c3aed;
    --violet:  #a78bfa;
    --cyan:    #06b6d4;
    --green:   #10b981;
    --yellow:  #f59e0b;
    --red:     #ef4444;
    --text:    #e9e5ff;
    --muted:   rgba(255,255,255,.38);
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    min-height: 100vh;
    padding: 28px 24px 40px;
  }

  /* ── Header ── */
  .header {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 28px;
  }
  .header-logo {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #6d28d9, #9333ea);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-weight: 800; color: #fff;
    box-shadow: 0 0 18px rgba(124,58,237,.45);
  }
  .header-title { font-size: 18px; font-weight: 700; letter-spacing: -.02em; }
  .header-sub   { font-size: 12px; color: var(--muted); margin-top: 2px; }

  .phase-badge {
    margin-left: auto;
    padding: 5px 14px; border-radius: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
    border: 1px solid;
    transition: all .3s;
  }
  .phase-badge.cache    { background:rgba(245,158,11,.1);  border-color:rgba(245,158,11,.4);  color:#f59e0b; }
  .phase-badge.training { background:rgba(124,58,237,.12); border-color:rgba(124,58,237,.4);  color:#a78bfa; }
  .phase-badge.done     { background:rgba(16,185,129,.1);  border-color:rgba(16,185,129,.4);  color:#10b981; }
  .phase-badge.idle     { background:rgba(255,255,255,.05);border-color:rgba(255,255,255,.12);color:var(--muted); }

  /* ── Grid ── */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin-bottom: 16px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 16px; }
  @media(max-width:900px){ .grid-4{grid-template-columns:repeat(2,1fr)} }
  @media(max-width:700px){ .grid-2,.grid-3{grid-template-columns:1fr} }

  /* ── Card ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
  }
  .card-title {
    font-size: 10.5px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 14px;
  }

  /* ── Big Progress Bar ── */
  .prog-wrap { margin-bottom: 16px; }
  .prog-meta { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px; }
  .prog-label { font-size: 13px; color: var(--muted); }
  .prog-pct   { font-size: 22px; font-weight: 800; color: var(--violet); }
  .prog-track {
    height: 10px; border-radius: 6px;
    background: rgba(255,255,255,.06);
    overflow: hidden;
  }
  .prog-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #6d28d9, #a78bfa);
    box-shadow: 0 0 12px rgba(124,58,237,.5);
    transition: width .6s cubic-bezier(.16,1,.3,1);
  }
  .prog-fill.green { background: linear-gradient(90deg,#059669,#10b981); box-shadow:0 0 12px rgba(16,185,129,.4); }

  /* ── Stat card ── */
  .stat-val  { font-size: 28px; font-weight: 800; line-height: 1; letter-spacing: -.03em; }
  .stat-unit { font-size: 13px; color: var(--muted); margin-left: 4px; }
  .stat-sub  { font-size: 11px; color: var(--muted); margin-top: 6px; }

  .stat-val.purple { color: var(--violet); }
  .stat-val.cyan   { color: var(--cyan);   }
  .stat-val.green  { color: var(--green);  }
  .stat-val.yellow { color: var(--yellow); }
  .stat-val.red    { color: var(--red);    }

  /* ── ETA ring ── */
  .eta-card {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 24px 20px;
  }
  .eta-ring-wrap { position: relative; width: 130px; height: 130px; margin-bottom: 12px; }
  .eta-ring-wrap svg { position:absolute;top:0;left:0; }
  .eta-ring-bg  { stroke: rgba(255,255,255,.06); }
  .eta-ring-fg  { stroke: var(--violet); stroke-linecap: round; transition: stroke-dashoffset .8s cubic-bezier(.16,1,.3,1); }
  .eta-ring-fg.green { stroke: var(--green); }
  .eta-center {
    position:absolute; inset:0; display:flex;
    flex-direction:column; align-items:center; justify-content:center;
  }
  .eta-time  { font-size: 19px; font-weight: 800; letter-spacing: -.02em; }
  .eta-label { font-size: 10px; color: var(--muted); letter-spacing: .04em; margin-top: 2px; }

  /* ── Mini chart ── */
  .mini-chart { width:100%; height:60px; }

  /* ── Metric rows ── */
  .metric-row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid rgba(255,255,255,.05); }
  .metric-row:last-child { border-bottom:none; }
  .metric-name  { font-size:12px; color:var(--muted); }
  .metric-value { font-size:13px; font-weight:700; }

  /* ── Log ── */
  .log-box {
    background: #050408;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 10px;
    padding: 14px 16px;
    height: 160px;
    overflow-y: auto;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 11px;
    line-height: 1.7;
    color: rgba(255,255,255,.45);
  }
  .log-box::-webkit-scrollbar { width:4px; }
  .log-box::-webkit-scrollbar-thumb { background:rgba(124,58,237,.3); border-radius:2px; }
  .log-line { white-space: pre-wrap; word-break: break-all; }
  .log-line.hi-cache    { color: #f59e0b; }
  .log-line.hi-epoch    { color: #a78bfa; font-weight: 700; }
  .log-line.hi-metric   { color: #6ee7b7; }
  .log-line.hi-err      { color: #f87171; }
  .log-line.hi-done     { color: #34d399; font-weight: 700; }

  /* ── Pulse dot ── */
  .pulse-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:var(--green); margin-right:8px;
    animation: pulse-anim 1.5s ease-in-out infinite;
  }
  @keyframes pulse-anim {
    0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,.6)}
    50%    {box-shadow:0 0 0 6px rgba(16,185,129,0)}
  }

  /* ── Skeleton shimmer ── */
  .shimmer {
    background: linear-gradient(90deg, rgba(255,255,255,.04) 25%, rgba(255,255,255,.08) 50%, rgba(255,255,255,.04) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.8s ease infinite;
    border-radius: 6px;
    height: 28px;
  }
  @keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

  /* ── Connections indicator ── */
  .conn { font-size:11px; color:var(--muted); display:flex; align-items:center; gap:6px; margin-bottom:20px; }
  .conn-dot { width:7px; height:7px; border-radius:50%; background:var(--green); animation:pulse-anim 2s ease infinite; }
  .conn-dot.dead { background:var(--red); animation:none; }
</style>
</head>
<body>

<div class="header">
  <div class="header-logo">K</div>
  <div>
    <div class="header-title">Key Detection DNN — Training</div>
    <div class="header-sub">Gerçek zamanlı eğitim takibi</div>
  </div>
  <div class="phase-badge idle" id="phaseBadge">Bekleniyor</div>
</div>

<div class="conn" id="connStatus">
  <div class="conn-dot" id="connDot"></div>
  <span id="connText">Log dosyasına bağlanıyor...</span>
</div>

<!-- ── CACHE SECTION ── -->
<div id="cacheSection">
  <div class="grid-2" style="margin-bottom:16px">
    <!-- Progress -->
    <div class="card" style="grid-column:1/-1">
      <div class="card-title">Cache Oluşturma İlerlemesi (HPSS)</div>
      <div class="prog-wrap">
        <div class="prog-meta">
          <span class="prog-label" id="cacheCountLabel">0 / 3817 dosya</span>
          <span class="prog-pct" id="cachePct">0%</span>
        </div>
        <div class="prog-track"><div class="prog-fill" id="cacheFill" style="width:0%"></div></div>
      </div>
    </div>
  </div>

  <div class="grid-4">
    <div class="card">
      <div class="card-title">İşlenen</div>
      <div class="stat-val purple" id="cacheProcessed">0</div>
      <div class="stat-sub">dosya</div>
    </div>
    <div class="card">
      <div class="card-title">Kalan</div>
      <div class="stat-val yellow" id="cacheRemaining">3817</div>
      <div class="stat-sub">dosya</div>
    </div>
    <div class="card">
      <div class="card-title">Hız</div>
      <div class="stat-val cyan" id="cacheSpeed">—</div>
      <div class="stat-sub">dosya/sn</div>
    </div>
    <div class="card eta-card">
      <div class="card-title">Tahmini Bitiş</div>
      <div class="eta-ring-wrap">
        <svg viewBox="0 0 130 130" width="130" height="130">
          <circle class="eta-ring-bg" cx="65" cy="65" r="54" fill="none" stroke-width="8"/>
          <circle class="eta-ring-fg" id="cacheRing" cx="65" cy="65" r="54" fill="none" stroke-width="8"
            stroke-dasharray="339.3" stroke-dashoffset="339.3" transform="rotate(-90 65 65)"/>
        </svg>
        <div class="eta-center">
          <div class="eta-time" id="cacheETA">—</div>
          <div class="eta-label">KALAN</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ── TRAINING SECTION ── -->
<div id="trainSection" style="display:none">
  <div class="grid-2" style="margin-bottom:16px">
    <!-- Epoch progress -->
    <div class="card" style="grid-column:1/-1">
      <div class="card-title">Epoch İlerlemesi</div>
      <div class="prog-wrap">
        <div class="prog-meta">
          <span class="prog-label" id="epochLabel">Epoch 0 / 120</span>
          <span class="prog-pct" id="epochPct">0%</span>
        </div>
        <div class="prog-track"><div class="prog-fill green" id="epochFill" style="width:0%"></div></div>
      </div>
    </div>
  </div>

  <div class="grid-4">
    <div class="card eta-card">
      <div class="card-title">Kalan Süre</div>
      <div class="eta-ring-wrap">
        <svg viewBox="0 0 130 130" width="130" height="130">
          <circle class="eta-ring-bg" cx="65" cy="65" r="54" fill="none" stroke-width="8"/>
          <circle class="eta-ring-fg green" id="trainRing" cx="65" cy="65" r="54" fill="none" stroke-width="8"
            stroke-dasharray="339.3" stroke-dashoffset="339.3" transform="rotate(-90 65 65)"/>
        </svg>
        <div class="eta-center">
          <div class="eta-time" id="trainETA">—</div>
          <div class="eta-label">KALAN</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Key Accuracy</div>
      <div class="stat-val green" id="valKeyAcc">—</div>
      <div class="stat-sub">val set</div>
      <div style="margin-top:8px">
        <div class="stat-val purple" style="font-size:18px" id="trainKeyAcc">—</div>
        <div class="stat-sub">train set</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">BPM MAE</div>
      <div class="stat-val cyan" id="valBpmMAE">—</div>
      <div class="stat-sub">val set</div>
      <div style="margin-top:8px">
        <div class="stat-val purple" style="font-size:18px" id="trainBpmMAE">—</div>
        <div class="stat-sub">train set</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Loss</div>
      <div class="stat-val yellow" id="valLoss">—</div>
      <div class="stat-sub">val loss</div>
      <div style="margin-top:8px">
        <div class="stat-val purple" style="font-size:18px" id="trainLoss">—</div>
        <div class="stat-sub">train loss</div>
      </div>
    </div>
  </div>

  <!-- Charts row -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Key Accuracy Geçmişi</div>
      <canvas class="mini-chart" id="accChart"></canvas>
    </div>
    <div class="card">
      <div class="card-title">Loss Geçmişi</div>
      <canvas class="mini-chart" id="lossChart"></canvas>
    </div>
  </div>
</div>

<!-- ── LOG ── -->
<div class="card" style="margin-top:16px">
  <div class="card-title" style="display:flex;align-items:center;gap:6px">
    <span class="pulse-dot" id="logDot"></span>
    Canlı Log
  </div>
  <div class="log-box" id="logBox"></div>
</div>

<script>
const MAX_LOG = 200;
const CIRC = 339.3; // 2π × 54

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  phase: 'idle',          // 'idle' | 'cache' | 'training' | 'done'
  cache: { pct:0, cur:0, total:3817, speed:0, eta:'—' },
  train: {
    epoch:0, totalEpochs:120,
    trainAcc:null, valAcc:null,
    trainLoss:null, valLoss:null,
    trainMAE:null, valMAE:null,
    epochTime:null,
    accHistory:[], valAccHistory:[],
    lossHistory:[], valLossHistory:[],
  },
  logLines: [],
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Charts ───────────────────────────────────────────────────────────────────
function drawChart(canvas, data1, data2, color1, color2, label) {
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth; const H = canvas.offsetHeight;
  canvas.width = W; canvas.height = H;
  ctx.clearRect(0,0,W,H);
  if (!data1.length) return;

  const all = [...data1, ...data2].filter(v => v != null);
  const mn = Math.min(...all); const mx = Math.max(...all);
  const range = mx - mn || 1;

  function drawLine(data, color) {
    if (!data.length) return;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    data.forEach((v,i) => {
      const x = (i/(data.length-1||1))*W;
      const y = H - ((v-mn)/range)*(H-8) - 4;
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();

    // glow
    ctx.shadowColor = color; ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  drawLine(data1, color1);
  if (data2.length) drawLine(data2, color2);
}

function refreshCharts() {
  const t = state.train;
  const acc = $('accChart');
  const los = $('lossChart');
  if (acc && acc.offsetWidth) {
    drawChart(acc, t.accHistory, t.valAccHistory, '#7c3aed', '#10b981', 'acc');
  }
  if (los && los.offsetWidth) {
    drawChart(los, t.lossHistory, t.valLossHistory, '#f59e0b', '#ef4444', 'loss');
  }
}

// ── UI update ────────────────────────────────────────────────────────────────
function updateUI() {
  const s = state;

  // Phase badge
  const badges = { idle:'Bekleniyor', cache:'Cache Oluşturuluyor', training:'Eğitim', done:'Tamamlandı' };
  $('phaseBadge').textContent = badges[s.phase] || 'Bekleniyor';
  $('phaseBadge').className = 'phase-badge ' + s.phase;

  // Cache section
  if (s.phase === 'cache' || (s.phase === 'idle' && s.cache.cur > 0)) {
    $('cacheSection').style.display = '';
    const c = s.cache;
    $('cachePct').textContent = c.pct + '%';
    $('cacheFill').style.width = c.pct + '%';
    $('cacheCountLabel').textContent = c.cur.toLocaleString() + ' / ' + c.total.toLocaleString() + ' dosya';
    $('cacheProcessed').textContent = c.cur.toLocaleString();
    $('cacheRemaining').textContent = (c.total - c.cur).toLocaleString();
    $('cacheSpeed').textContent = c.speed ? c.speed.toFixed(2) : '—';
    $('cacheETA').textContent = c.eta;

    // ring
    const offset = CIRC * (1 - c.pct/100);
    $('cacheRing').style.strokeDashoffset = offset;
    if (c.pct >= 100) $('cacheRing').classList.add('green');
  }

  // Training section
  if (s.phase === 'training' || s.phase === 'done') {
    $('trainSection').style.display = '';
    $('cacheSection').style.display = 'none';
    const t = s.train;
    const pct = Math.round((t.epoch / t.totalEpochs) * 100);
    $('epochPct').textContent = pct + '%';
    $('epochFill').style.width = pct + '%';
    $('epochLabel').textContent = 'Epoch ' + t.epoch + ' / ' + t.totalEpochs;
    $('trainRing').style.strokeDashoffset = CIRC * (1 - pct/100);

    // ETA
    if (t.epochTime && t.epoch > 0) {
      const remaining = (t.totalEpochs - t.epoch) * t.epochTime;
      $('trainETA').textContent = fmtSecs(remaining);
    }

    // Metrics
    $('valKeyAcc').textContent  = t.valAcc  != null ? (t.valAcc*100).toFixed(1)+'%' : '—';
    $('trainKeyAcc').textContent= t.trainAcc!= null ? (t.trainAcc*100).toFixed(1)+'%' : '—';
    $('valBpmMAE').textContent  = t.valMAE  != null ? t.valMAE.toFixed(3) : '—';
    $('trainBpmMAE').textContent= t.trainMAE!= null ? t.trainMAE.toFixed(3) : '—';
    $('valLoss').textContent    = t.valLoss != null ? t.valLoss.toFixed(4) : '—';
    $('trainLoss').textContent  = t.trainLoss!= null ? t.trainLoss.toFixed(4) : '—';

    refreshCharts();
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function fmtSecs(s) {
  s = Math.round(s);
  if (s >= 3600) {
    const h = Math.floor(s/3600); const m = Math.floor((s%3600)/60);
    return h+'s '+(m<10?'0':'')+m+'d';
  }
  if (s >= 60) {
    const m = Math.floor(s/60); const sec = s%60;
    return m+'d '+(sec<10?'0':'')+sec+'s';
  }
  return s+'s';
}

function parseDuration(str) {
  // "1:23:45" or "45:12" or "5:12"
  const parts = str.split(':').map(Number);
  if (parts.length === 3) return parts[0]*3600 + parts[1]*60 + parts[2];
  if (parts.length === 2) return parts[0]*60 + parts[1];
  return parts[0];
}

// ── Parse log lines ──────────────────────────────────────────────────────────
function parseLine(raw) {
  // strip color/control codes and non-ASCII block chars
  const line = raw.replace(/\x1b\[[0-9;]*m/g,'').replace(/[^\x00-\x7F▏▎▍▌▋▊▉█░▒▓]/g,'').trim();
  if (!line) return;

  // ── Cache tqdm ──
  // Cache:  42%|████      | 1604/3817 [23:45<32:18,  1.09dosya/s]
  const cacheM = line.match(/Cache:\s+(\d+)%\|.*\|\s*(\d+)\/(\d+)\s*\[([0-9:]+)<([0-9:?]+),\s*([\d.?]+)/);
  if (cacheM) {
    state.phase = 'cache';
    state.cache.pct   = parseInt(cacheM[1]);
    state.cache.cur   = parseInt(cacheM[2]);
    state.cache.total = parseInt(cacheM[3]);
    state.cache.eta   = cacheM[5] === '?' ? '—' : fmtSecs(parseDuration(cacheM[5]));
    state.cache.speed = cacheM[6] === '?' ? 0 : parseFloat(cacheM[6]);
    updateUI(); return;
  }

  // ── Cache done ──
  if (line.includes('cache') && line.includes('tamamland')) {
    state.cache.pct = 100; state.cache.cur = state.cache.total;
    updateUI(); return;
  }

  // ── Epoch start: "Epoch 12/120" ──
  const epochM = line.match(/^Epoch\s+(\d+)\/(\d+)/);
  if (epochM) {
    state.phase = 'training';
    state.train.epoch       = parseInt(epochM[1]);
    state.train.totalEpochs = parseInt(epochM[2]);
    updateUI(); return;
  }

  // ── Keras metrics line: ends with step count info + metrics ──
  // 1905/1905 ━━━━━━━━━━ 45s 24ms/step - loss: 2.9055 - key_output_accuracy: 0.2153 ...
  const metricsM = line.match(/(\d+)\/\1.*?(\d+)s\s+[\d.]+ms\/step\s+(.*)/);
  if (metricsM) {
    const raw2 = metricsM[3];
    const epochSecs = parseInt(metricsM[2]);
    if (epochSecs > 0) state.train.epochTime = epochSecs;

    function extractVal(key) {
      const m = raw2.match(new RegExp(key+':\\s*([\\d.]+)'));
      return m ? parseFloat(m[1]) : null;
    }

    const trainAcc  = extractVal('key_output_accuracy');
    const valAcc    = extractVal('val_key_output_accuracy');
    const trainLoss = extractVal('(?:^|\\s)loss');
    const valLoss   = extractVal('val_loss');
    const trainMAE  = extractVal('bpm_output_mae');
    const valMAE    = extractVal('val_bpm_output_mae');

    if (trainAcc  != null) { state.train.trainAcc  = trainAcc;  state.train.accHistory.push(trainAcc); }
    if (valAcc    != null) { state.train.valAcc     = valAcc;    state.train.valAccHistory.push(valAcc); }
    if (trainLoss != null) { state.train.trainLoss  = trainLoss; state.train.lossHistory.push(trainLoss); }
    if (valLoss   != null) { state.train.valLoss    = valLoss;   state.train.valLossHistory.push(valLoss); }
    if (trainMAE  != null) { state.train.trainMAE   = trainMAE; }
    if (valMAE    != null) { state.train.valMAE     = valMAE; }

    updateUI(); return;
  }

  // ── Done ──
  if (line.toLowerCase().includes('eğitim tamamland') || line.toLowerCase().includes('training complete')) {
    state.phase = 'done';
    updateUI();
  }
}

// ── Log display ──────────────────────────────────────────────────────────────
function classifyLine(line) {
  if (line.match(/Cache:\s+\d+%/)) return 'hi-cache';
  if (line.match(/^Epoch\s+\d+/))  return 'hi-epoch';
  if (line.match(/loss:|accuracy:|mae:/i)) return 'hi-metric';
  if (line.match(/error|traceback|exception/i)) return 'hi-err';
  if (line.match(/tamamland|complete|saved/i)) return 'hi-done';
  return '';
}

function addLog(line) {
  state.logLines.push(line);
  if (state.logLines.length > MAX_LOG) state.logLines.shift();
  const box = $('logBox');
  const div = document.createElement('div');
  div.className = 'log-line ' + classifyLine(line);
  div.textContent = line;
  box.appendChild(div);
  if (box.children.length > MAX_LOG) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}

// ── SSE ──────────────────────────────────────────────────────────────────────
let evtSource;
let reconnectTimer;
let lastLineBuffer = '';

function connect() {
  evtSource = new EventSource('/events');

  evtSource.onopen = () => {
    $('connDot').classList.remove('dead');
    $('connText').textContent = 'Bağlandı — log izleniyor...';
    clearTimeout(reconnectTimer);
  };

  evtSource.onmessage = (e) => {
    const raw = e.data;
    // Split on \r to get last real update in tqdm lines
    const parts = raw.split('\r');
    const last = parts[parts.length - 1].trim();
    if (!last) return;
    parseLine(last);
    // Only add to log if it's a "real" line (not every tqdm tick)
    if (parts.length === 1 || last.match(/Cache:\s+\d+%.*\|\s*\d+\/\d+/) ) {
      // for tqdm: only log every ~1% or on first
      const pct = state.cache.pct;
      if (!raw.includes('Cache:') || pct % 5 === 0 || state.train.epoch > 0) {
        addLog(last);
      }
    }
  };

  evtSource.onerror = () => {
    $('connDot').classList.add('dead');
    $('connText').textContent = 'Bağlantı kesildi — yeniden deneniyor...';
    evtSource.close();
    reconnectTimer = setTimeout(connect, 3000);
  };
}

connect();

// Redraw charts on resize
window.addEventListener('resize', refreshCharts);
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass   # suppress access logs

    def do_GET(self):
        if self.path == '/':
            body = HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == '/events':
            self.send_response(200)
            self.send_header('Content-Type',  'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection',    'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            pos = 0
            # First send everything already in the file
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'rb') as f:
                    pos = 0

            while True:
                try:
                    if not os.path.exists(LOG_FILE):
                        time.sleep(1); continue

                    with open(LOG_FILE, 'r', errors='replace') as f:
                        f.seek(pos)
                        chunk = f.read(65536)
                        if chunk:
                            # Split chunk: lines separated by \n, within a line \r for tqdm
                            lines = chunk.split('\n')
                            for line in lines:
                                if not line.strip(): continue
                                msg = 'data: ' + line + '\n\n'
                                self.wfile.write(msg.encode('utf-8', errors='replace'))
                            self.wfile.flush()
                        pos = f.tell()
                    time.sleep(0.4)
                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception:
                    break
        else:
            self.send_response(404); self.end_headers()


def open_browser():
    time.sleep(1.2)
    webbrowser.open(f'http://localhost:{PORT}')


if __name__ == '__main__':
    print(f'\n  Key Detection Training Dashboard')
    print(f'  ─────────────────────────────────')
    print(f'  Log dosyası : {LOG_FILE}')
    print(f'  Dashboard   : http://localhost:{PORT}')
    print(f'  Durdurmak   : Ctrl+C\n')

    threading.Thread(target=open_browser, daemon=True).start()

    server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    server.serve_forever()
