"""Serve a browser-based viewer for single-channel EEG .npz recordings.

Examples (run from the repository root):
    python tools/view_eeg_npz.py --dataset ANPHY
    python tools/view_eeg_npz.py --dataset DOD-H --window-seconds 60
    python tools/view_eeg_npz.py --data-dir path/to/folder/with/npz

Open the printed URL in a browser. In VS Code Remote, forward the printed port
from the Ports panel if VS Code does not forward it automatically.

What you get in the browser:
  * the EEG signal for the current time window,
  * a strip with the sleep stage of every epoch in that window,
  * an overview of the whole recording (hypnogram) that doubles as a scrollbar:
    drag the blue box, or click anywhere to jump there,
  * a legend - click a stage (for example "Not defined") to highlight every
    matching section in the signal and in the overview, then use
    Previous / Next to jump from one matching section to the next.
"""

import argparse
import functools
import glob
import json
import math
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np


DATASET_CHANNELS = {
    "ANPHY": "C3-TP10",
    "DOD-H": "C3-RPA",
}

# Stage codes used by the viewer: 0=W, 1=N1, 2=N2, 3=N3, 4=REM, -1=not defined.
UNDEFINED = -1
STAGE_ALIASES = {
    "0": 0, "W": 0, "WAKE": 0, "SLEEP STAGE W": 0,
    "1": 1, "N1": 1, "S1": 1, "SLEEP STAGE 1": 1, "SLEEP STAGE N1": 1,
    "2": 2, "N2": 2, "S2": 2, "SLEEP STAGE 2": 2, "SLEEP STAGE N2": 2,
    "3": 3, "N3": 3, "S3": 3, "S4": 3, "N4": 3,
    "SLEEP STAGE 3": 3, "SLEEP STAGE 4": 3, "SLEEP STAGE N3": 3,
    "4": 4, "R": 4, "REM": 4, "SLEEP STAGE R": 4, "SLEEP STAGE REM": 4,
}
MAX_POINTS = 6000          # points sent to the browser per window
CACHED_RECORDINGS = 4      # recordings kept in memory between requests

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EEG NPZ Viewer</title>
<style>
  :root { color-scheme: light; font: 15px system-ui, sans-serif; }
  body { max-width: 1280px; margin: 20px auto; padding: 0 18px; color: #17212b; }
  h1 { font-size: 1.3rem; margin: 0 0 2px; }
  #status { color: #526170; min-height: 1.5em; }
  #status.error { color: #c53030; font-weight: 600; }
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; margin: 12px 0; }
  .controls .spacer { flex: 1; }
  button, select { font: inherit; padding: 5px 10px; }
  button { cursor: pointer; }
  button:disabled { cursor: default; }
  #recording { min-width: 220px; max-width: 420px; }
  canvas { display: block; width: 100%; box-sizing: border-box; touch-action: none; }
  .plot { position: relative; }
  #signal { height: min(52vh, 460px); border: 1px solid #cbd4dc; background: #fff; cursor: grab; }
  #signal.dragging { cursor: grabbing; }
  #readout { position: absolute; top: 6px; right: 22px; padding: 2px 8px; border-radius: 4px;
             background: rgba(255,255,255,.9); color: #334155; font-size: .85rem; pointer-events: none; }
  #stages { height: 30px; margin-top: 4px; }
  .nav-title { display: flex; justify-content: space-between; gap: 12px; margin: 12px 0 3px;
               color: #526170; font-size: .88rem; }
  #time-label { color: #17212b; font-weight: 600; font-variant-numeric: tabular-nums; }
  #navigator { height: 112px; border: 1px solid #cbd4dc; background: #fff; cursor: pointer; }
  #navigator:focus-visible { outline: 2px solid #3182ce; outline-offset: 1px; }
  .legend-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin: 12px 0 6px; }
  #legend { display: flex; flex-wrap: wrap; gap: 6px 8px; }
  .chip { display: inline-flex; align-items: center; gap: 6px; border: 1px solid #d5dde5; border-radius: 6px;
          padding: 4px 9px; background: #f8fafc; color: inherit; }
  .chip:hover:not(:disabled) { border-color: #64748b; }
  .chip[aria-pressed="true"] { border-color: var(--stage); box-shadow: 0 0 0 2px var(--stage); background: #fff; }
  .chip:disabled { opacity: .45; }
  .chip .swatch { width: 13px; height: 13px; border-radius: 3px; background: var(--stage); }
  .chip .share { color: #64748b; font-size: .85rem; font-variant-numeric: tabular-nums; }
  .jump { display: inline-flex; align-items: center; gap: 6px; }
  #occurrence { color: #526170; font-size: .9rem; min-width: 9em; text-align: center; }
  .help { color: #526170; font-size: .9rem; }
  .help summary { cursor: pointer; }
  kbd { border: 1px solid #cbd4dc; border-bottom-width: 2px; border-radius: 4px; padding: 0 5px;
        background: #f8fafc; font: .85em ui-monospace, monospace; }
</style>
</head>
<body>
<h1 id="heading">EEG NPZ Viewer</h1>
<div id="status">Loading recording list...</div>

<div class="controls">
  <button id="previous" type="button" title="Previous recording (p)">&#9664; Previous</button>
  <select id="recording" aria-label="Recording"></select>
  <button id="next" type="button" title="Next recording (n)">Next &#9654;</button>
  <span class="spacer"></span>
  <label for="window">Window</label>
  <select id="window">
    <option value="10">10 seconds</option>
    <option value="30">30 seconds</option>
    <option value="60">1 minute</option>
    <option value="120">2 minutes</option>
    <option value="300">5 minutes</option>
    <option value="900">15 minutes</option>
    <option value="1800">30 minutes</option>
    <option value="3600">1 hour</option>
    <option value="all">Full recording</option>
  </select>
  <label for="yscale">Amplitude</label>
  <select id="yscale" title="Auto rescales every window. Fixed keeps one scale for the whole recording, so windows are comparable.">
    <option value="auto">Auto (fit window)</option>
    <option value="fixed">Fixed (whole recording)</option>
  </select>
</div>

<div class="plot">
  <canvas id="signal"></canvas>
  <div id="readout" hidden></div>
</div>
<canvas id="stages" role="img" aria-label="Sleep stage of each epoch in the current window"></canvas>

<div class="nav-title">
  <span>Whole recording &mdash; drag the blue box or click anywhere to move</span>
  <span id="time-label"></span>
</div>
<canvas id="navigator" tabindex="0" role="scrollbar" aria-orientation="horizontal"
        aria-label="Position in recording" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0"></canvas>

<div class="legend-row">
  <div id="legend" aria-label="Click a sleep stage to highlight it"></div>
  <div class="jump">
    <button id="prev-occ" type="button" title="Previous highlighted section ( [ )" disabled>&#9664;</button>
    <span id="occurrence">Click a stage to highlight it</span>
    <button id="next-occ" type="button" title="Next highlighted section ( ] )" disabled>&#9654;</button>
    <button id="clear" type="button" title="Clear highlights (Esc)" disabled>Clear</button>
  </div>
</div>

<details class="help">
  <summary>Mouse and keyboard shortcuts</summary>
  <p>
    <b>Move:</b> drag the signal, scroll over it, or use the overview bar.
    <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> move 10% of the window,
    <kbd>Page Up</kbd>/<kbd>Page Down</kbd> one full window,
    <kbd>Home</kbd>/<kbd>End</kbd> jump to start/end.<br>
    <b>Zoom:</b> <kbd>+</kbd>/<kbd>-</kbd> change the window length.
    <b>Recordings:</b> <kbd>n</kbd>/<kbd>p</kbd>.
    <b>Highlights:</b> <kbd>]</kbd>/<kbd>[</kbd> next/previous highlighted section, <kbd>Esc</kbd> clears.
  </p>
</details>

<script>
const CONFIG = __CONFIG__;
const STAGES = [
  {code: 0,  short: "W",   name: "Wake",        color: "#3182ce", row: 0},
  {code: 1,  short: "N1",  name: "N1 sleep",    color: "#38a169", row: 2},
  {code: 2,  short: "N2",  name: "N2 sleep",    color: "#d69e2e", row: 3},
  {code: 3,  short: "N3",  name: "N3 sleep",    color: "#e53e3e", row: 4},
  {code: 4,  short: "REM", name: "REM sleep",   color: "#805ad5", row: 1},
  {code: -1, short: "?",   name: "Not defined", color: "#ed8936", row: 5},
];
const STAGE = new Map(STAGES.map(stage => [stage.code, stage]));
const PAD = {left: 64, right: 16, top: 14, bottom: 30};   // shared by all three canvases
const TIME_STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400];

const $ = id => document.getElementById(id);
const el = {
  heading: $("heading"), status: $("status"), recording: $("recording"),
  window: $("window"), yscale: $("yscale"), signal: $("signal"), readout: $("readout"),
  stages: $("stages"), nav: $("navigator"), timeLabel: $("time-label"),
  legend: $("legend"), occurrence: $("occurrence"), prevOcc: $("prev-occ"),
  nextOcc: $("next-occ"), clear: $("clear"),
};
const state = {
  files: [], index: 0, info: null, data: null,
  start: 0, lastWindow: 30,
  selected: new Set(), runs: [],
  loading: false, pending: false,
};

// ---------- helpers ----------
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

function formatTime(seconds, decimals = false) {
  if (!Number.isFinite(seconds)) return "";
  const whole = Math.floor(seconds + 1e-9);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  let secs = String(whole % 60).padStart(2, "0");
  if (decimals) secs += "." + Math.floor((seconds - whole) * 10 + 1e-9);
  return `${hours}:${String(minutes).padStart(2, "0")}:${secs}`;
}

function formatDuration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(seconds < 600 ? 1 : 0)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

function rgba(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const pixelW = Math.round(width * ratio), pixelH = Math.round(height * ratio);
  if (canvas.width !== pixelW || canvas.height !== pixelH) {
    canvas.width = pixelW; canvas.height = pixelH;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = "12px system-ui, sans-serif";
  return {ctx, width, height};
}

function timeTicks(from, to, maxTicks) {
  const span = to - from;
  const step = TIME_STEPS.find(s => span / s <= maxTicks) || TIME_STEPS[TIME_STEPS.length - 1];
  const ticks = [];
  for (let t = Math.ceil(from / step - 1e-9) * step; t <= to + 1e-9; t += step) ticks.push(t);
  return {ticks, step};
}

function windowLength() {
  if (!state.info) return 30;
  if (el.window.value === "all") return state.info.duration;
  return Math.min(Number(el.window.value), state.info.duration);
}
const maxStart = () => state.info ? Math.max(0, state.info.duration - windowLength()) : 0;

function showStatus(text, isError = false) {
  el.status.textContent = text;
  el.status.classList.toggle("error", isError);
}

// ---------- drawing: signal ----------
function drawSignal() {
  const {ctx, width, height} = setupCanvas(el.signal);
  const data = state.data, info = state.info;
  if (!data || !info) { drawStageStrip(); return; }
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const t0 = data.start, t1 = data.end > data.start ? data.end : data.start + 1;
  const xOf = t => PAD.left + plotW * (t - t0) / (t1 - t0);

  let low, high;
  if (el.yscale.value === "fixed") {
    [low, high] = info.y_range;
  } else {
    low = Infinity; high = -Infinity;
    for (const v of data.values) {
      if (v === null) continue;
      if (v < low) low = v;
      if (v > high) high = v;
    }
    if (!Number.isFinite(low)) { low = -1; high = 1; }
    if (low === high) { low -= 1; high += 1; }
    const margin = (high - low) * 0.08;
    low -= margin; high += margin;
  }
  const yOf = v => PAD.top + plotH * (high - v) / (high - low);

  // Highlighted stages (drawn first so the trace stays readable on top).
  forEachRun(info, t0, t1, (code, from, to) => {
    if (!state.selected.has(code)) return;
    const x0 = xOf(Math.max(t0, from)), x1 = xOf(Math.min(t1, to));
    ctx.fillStyle = rgba(STAGE.get(code).color, 0.24);
    ctx.fillRect(x0, PAD.top, Math.max(1, x1 - x0), plotH);
  });

  // Grid and axis labels.
  ctx.strokeStyle = "#e3e8ed"; ctx.fillStyle = "#526170"; ctx.lineWidth = 1;
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + plotH * i / 4;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(width - PAD.right, y); ctx.stroke();
    ctx.fillText(Number((high - (high - low) * i / 4).toPrecision(3)), PAD.left - 8, y + 4);
  }
  const {ticks, step} = timeTicks(t0, t1, Math.max(3, Math.floor(plotW / 110)));
  ctx.textAlign = "center";
  for (const t of ticks) {
    const x = xOf(t);
    ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + plotH); ctx.stroke();
    ctx.fillText(formatTime(t, step < 1), x, height - 10);
  }

  // The EEG trace; null values (NaN in the file) leave a gap.
  ctx.save();
  ctx.beginPath(); ctx.rect(PAD.left, PAD.top, plotW, plotH); ctx.clip();
  ctx.beginPath();
  let penDown = false;
  for (let i = 0; i < data.values.length; i++) {
    const v = data.values[i];
    if (v === null) { penDown = false; continue; }
    const x = xOf(data.times[i]), y = yOf(v);
    if (penDown) ctx.lineTo(x, y); else { ctx.moveTo(x, y); penDown = true; }
  }
  ctx.strokeStyle = "#111820"; ctx.lineWidth = 1; ctx.stroke();
  ctx.restore();
  ctx.strokeStyle = "#94a3b8";
  ctx.strokeRect(PAD.left + .5, PAD.top + .5, plotW - 1, plotH - 1);
  drawStageStrip();
}

// Calls fn(code, startSeconds, endSeconds) for each run of equal stages overlapping [t0, t1].
function forEachRun(info, t0, t1, fn) {
  const es = info.epoch_seconds, stages = info.stages;
  if (!es || !stages.length) return;
  const first = Math.max(0, Math.floor(t0 / es));
  const last = Math.min(stages.length, Math.ceil(t1 / es));
  for (let i = first; i < last;) {
    let j = i;
    while (j + 1 < last && stages[j + 1] === stages[i]) j++;
    fn(stages[i], i * es, (j + 1) * es);
    i = j + 1;
  }
}

// ---------- drawing: stage strip for the current window ----------
function drawStageStrip() {
  const {ctx, width, height} = setupCanvas(el.stages);
  const data = state.data, info = state.info;
  if (!data || !info) return;
  ctx.fillStyle = "#526170"; ctx.textAlign = "right";
  ctx.fillText("Stage", PAD.left - 8, height / 2 + 4);
  const plotW = width - PAD.left - PAD.right;
  if (!info.stages.length) {
    ctx.textAlign = "left";
    ctx.fillText("No stage labels in this file", PAD.left, height / 2 + 4);
    return;
  }
  const t0 = data.start, t1 = data.end > data.start ? data.end : data.start + 1;
  const xOf = t => PAD.left + plotW * (t - t0) / (t1 - t0);
  ctx.textAlign = "center";
  forEachRun(info, t0, t1, (code, from, to) => {
    const stage = STAGE.get(code);
    const dimmed = state.selected.size && !state.selected.has(code);
    const x0 = xOf(Math.max(t0, from)), x1 = xOf(Math.min(t1, to));
    const w = Math.max(1, x1 - x0);
    ctx.fillStyle = dimmed ? rgba(stage.color, 0.22) : stage.color;
    ctx.fillRect(x0, 2, w, height - 4);
    const text = w > 110 ? `${stage.short} – ${stage.name}` : stage.short;
    if (w > 26) {
      ctx.fillStyle = dimmed ? "#475569" : "#fff";
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.fillText(text, x0 + w / 2, height / 2 + 4);
    }
  });
}

// ---------- drawing: whole-recording overview / scrollbar ----------
function drawNavigator() {
  const {ctx, width, height} = setupCanvas(el.nav);
  const info = state.info;
  if (!info) return;
  const top = 8, axisH = 22;
  const rowsH = height - top - axisH;
  const plotW = width - PAD.left - PAD.right;
  const xOf = t => PAD.left + plotW * t / info.duration;
  const es = info.epoch_seconds, stages = info.stages;

  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(PAD.left, top, plotW, rowsH);

  if (es && stages.length) {
    const rowH = rowsH / STAGES.length;
    ctx.textAlign = "right";
    for (const stage of STAGES) {
      ctx.fillStyle = state.selected.has(stage.code) ? stage.color : "#64748b";
      ctx.font = (state.selected.has(stage.code) ? "700 " : "") + "11px system-ui, sans-serif";
      ctx.fillText(stage.short, PAD.left - 8, top + rowH * (stage.row + 0.5) + 4);
    }
    // Full-height bands for highlighted stages, then the hypnogram bars on top.
    forEachRun(info, 0, info.duration, (code, from, to) => {
      if (!state.selected.has(code)) return;
      ctx.fillStyle = rgba(STAGE.get(code).color, 0.28);
      ctx.fillRect(xOf(from), top, Math.max(1, xOf(to) - xOf(from)), rowsH);
    });
    forEachRun(info, 0, info.duration, (code, from, to) => {
      const stage = STAGE.get(code);
      const dimmed = state.selected.size && !state.selected.has(code);
      ctx.fillStyle = dimmed ? "#cbd5e1" : stage.color;
      ctx.fillRect(xOf(from), top + rowH * stage.row + 1, Math.max(1, xOf(to) - xOf(from)), rowH - 2);
    });
  } else {
    ctx.fillStyle = "#64748b"; ctx.textAlign = "left";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("No stage labels in this file – the bar still scrolls through the signal.",
      PAD.left + 10, top + rowsH / 2 + 4);
  }

  // Time axis.
  ctx.font = "11px system-ui, sans-serif";
  ctx.strokeStyle = "#94a3b8"; ctx.fillStyle = "#526170"; ctx.lineWidth = 1;
  const {ticks} = timeTicks(0, info.duration, Math.max(3, Math.floor(plotW / 90)));
  for (const t of ticks) {
    const x = xOf(t);
    ctx.beginPath(); ctx.moveTo(x, top + rowsH); ctx.lineTo(x, top + rowsH + 5); ctx.stroke();
    ctx.textAlign = x - PAD.left < 20 ? "left" : x > width - PAD.right - 20 ? "right" : "center";
    ctx.fillText(formatTime(t), x, height - 4);
  }

  // The current window ("scroll thumb").
  let x0 = xOf(state.start), x1 = xOf(state.start + windowLength());
  if (x1 - x0 < 8) { const mid = (x0 + x1) / 2; x0 = mid - 4; x1 = mid + 4; }
  ctx.fillStyle = "rgba(49, 130, 206, 0.16)";
  ctx.fillRect(x0, top - 4, x1 - x0, rowsH + 8);
  ctx.strokeStyle = "#2b6cb0"; ctx.lineWidth = 2;
  ctx.strokeRect(x0, top - 4, x1 - x0, rowsH + 8);

  el.nav.setAttribute("aria-valuemax", String(Math.round(maxStart())));
  el.nav.setAttribute("aria-valuenow", String(Math.round(state.start)));
  el.nav.setAttribute("aria-valuetext", formatTime(state.start));
}

function drawAll() { drawSignal(); drawNavigator(); }

// ---------- navigation ----------
function updateTimeLabel() {
  if (!state.info) { el.timeLabel.textContent = ""; return; }
  const end = Math.min(state.info.duration, state.start + windowLength());
  el.timeLabel.textContent =
    `${formatTime(state.start)} – ${formatTime(end)}  of  ${formatTime(state.info.duration)}`;
}

function setStart(time) {
  if (!state.info) return;
  state.start = clamp(time, 0, maxStart());
  drawNavigator();
  updateTimeLabel();
  updateOccurrence();
  saveHash();
  requestData();
}

// Only one request runs at a time; while dragging, the newest position wins.
async function requestData() {
  if (!state.info) return;
  if (state.loading) { state.pending = true; return; }
  state.loading = true;
  const index = state.index, start = state.start;
  const windowValue = el.window.value === "all" ? "all" : windowLength();
  try {
    const query = new URLSearchParams({index, start, window: windowValue});
    const response = await fetch(`/api/data?${query}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    if (index === state.index) { state.data = data; drawSignal(); }
  } catch (error) {
    showStatus(`Could not load signal: ${error.message}`, true);
  } finally {
    state.loading = false;
    if (state.pending) { state.pending = false; requestData(); }
  }
}

async function loadRecording(index, start = 0) {
  if (!state.files.length) return;
  index = (index + state.files.length) % state.files.length;
  state.index = index; state.info = null; state.data = null;
  el.recording.value = String(index);
  showStatus(`Loading ${state.files[index]} ...`);
  try {
    const response = await fetch(`/api/info?index=${index}`);
    const info = await response.json();
    if (!response.ok) throw new Error(info.error || response.statusText);
    if (index !== state.index) return;        // the user already moved on
    state.info = info;
    const epochs = info.stages.length
      ? ` · ${info.stages.length} epochs of ${Number(info.epoch_seconds.toFixed(2))} s` : " · no stage labels";
    showStatus(`${info.filename} · ${index + 1}/${state.files.length} · ${info.fs} Hz · ` +
      `${formatTime(info.duration)} long${epochs}`);
    buildLegend();
    updateRuns();
    drawAll();
    setStart(start);
  } catch (error) {
    drawAll();
    showStatus(`Could not load ${state.files[index]}: ${error.message}`, true);
  }
}

function changeWindow(value) {
  const center = state.start + state.lastWindow / 2;
  el.window.value = value;
  state.lastWindow = windowLength();
  setStart(center - state.lastWindow / 2);    // keep the same moment centred
}

function zoom(direction) {
  const options = [...el.window.options];
  const next = el.window.selectedIndex + direction;
  if (next >= 0 && next < options.length) changeWindow(options[next].value);
}

// ---------- legend, highlighting and jumping ----------
function buildLegend() {
  const info = state.info;
  const counts = new Map();
  for (const code of info.stages) counts.set(code, (counts.get(code) || 0) + 1);
  el.legend.replaceChildren();
  for (const stage of STAGES) {
    const count = counts.get(stage.code) || 0;
    const button = document.createElement("button");
    button.type = "button"; button.className = "chip";
    button.style.setProperty("--stage", stage.color);
    button.dataset.code = String(stage.code);
    button.disabled = count === 0;
    button.setAttribute("aria-pressed", String(state.selected.has(stage.code)));
    let title = count
      ? `${count} epochs, ${formatDuration(count * info.epoch_seconds)} – click to highlight`
      : "Not present in this recording";
    if (stage.code === -1 && info.undefined_labels.length) {
      title += `\nLabel values in the file: ${info.undefined_labels.join(", ")}`;
    }
    button.title = title;
    const swatch = document.createElement("span"); swatch.className = "swatch";
    const text = document.createElement("span");
    text.textContent = stage.code === -1 ? stage.name : `${stage.short} – ${stage.name}`;
    const share = document.createElement("span"); share.className = "share";
    share.textContent = info.stages.length ? `${(100 * count / info.stages.length).toFixed(count && count * 10 < info.stages.length ? 1 : 0)}%` : "";
    button.append(swatch, text, share);
    button.addEventListener("click", () => toggleStage(stage.code));
    el.legend.append(button);
  }
}

function toggleStage(code) {
  if (state.selected.has(code)) state.selected.delete(code); else state.selected.add(code);
  selectionChanged();
}

function selectionChanged() {
  for (const button of el.legend.children) {
    button.setAttribute("aria-pressed", String(state.selected.has(Number(button.dataset.code))));
  }
  updateRuns();
  drawAll();
}

// Continuous sections (in seconds) covered by any highlighted stage.
function updateRuns() {
  state.runs = [];
  const info = state.info;
  if (info && info.epoch_seconds && state.selected.size) {
    let open = null;
    info.stages.forEach((code, i) => {
      const hit = state.selected.has(code);
      if (hit && !open) { open = {start: i * info.epoch_seconds}; state.runs.push(open); }
      if (hit) open.end = (i + 1) * info.epoch_seconds;
      else open = null;
    });
  }
  updateOccurrence();
}

function updateOccurrence() {
  const runs = state.runs, some = state.selected.size > 0;
  el.clear.disabled = !some;
  el.prevOcc.disabled = el.nextOcc.disabled = runs.length === 0;
  if (!some) { el.occurrence.textContent = "Click a stage to highlight it"; return; }
  if (!runs.length) { el.occurrence.textContent = "Not in this recording"; return; }
  const viewEnd = state.start + windowLength();
  const visible = runs.findIndex(run => run.end > state.start && run.start < viewEnd);
  const total = runs.reduce((sum, run) => sum + run.end - run.start, 0);
  el.occurrence.textContent = (visible >= 0 ? `Section ${visible + 1} of ${runs.length}` : `${runs.length} sections`) +
    ` · ${formatDuration(total)}`;
}

function jumpToRun(direction) {
  if (!state.runs.length) return;
  const lead = el.window.value === "all" ? 0 : windowLength() * 0.1;   // a little context before
  const targets = state.runs.map(run => clamp(run.start - lead, 0, maxStart()));
  const target = direction > 0
    ? targets.find(t => t > state.start + 0.01)
    : targets.filter(t => t < state.start - 0.01).pop();
  if (target !== undefined) setStart(target);
}

// ---------- remember position in the URL ----------
let hashTimer = 0;
function saveHash() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    const hash = new URLSearchParams({file: state.files[state.index], t: state.start.toFixed(1), w: el.window.value});
    history.replaceState(null, "", "#" + hash);
  }, 300);
}

// ---------- events ----------
el.recording.addEventListener("change", () => loadRecording(Number(el.recording.value)));
el.window.addEventListener("change", () => changeWindow(el.window.value));
el.yscale.addEventListener("change", drawSignal);
$("previous").addEventListener("click", () => loadRecording(state.index - 1));
$("next").addEventListener("click", () => loadRecording(state.index + 1));
el.prevOcc.addEventListener("click", () => jumpToRun(-1));
el.nextOcc.addEventListener("click", () => jumpToRun(1));
el.clear.addEventListener("click", () => { state.selected.clear(); selectionChanged(); });

// Overview bar: drag the box, or click anywhere to centre the window there.
function navTime(event) {
  const rect = el.nav.getBoundingClientRect();
  const plotW = rect.width - PAD.left - PAD.right;
  return clamp((event.clientX - rect.left - PAD.left) / plotW, 0, 1) * state.info.duration;
}
let navDrag = null;
el.nav.addEventListener("pointerdown", event => {
  if (!state.info) return;
  el.nav.setPointerCapture(event.pointerId);
  const time = navTime(event), length = windowLength();
  if (time >= state.start && time <= state.start + length) {
    navDrag = {offset: time - state.start};
  } else {
    navDrag = {offset: length / 2};
    setStart(time - length / 2);
  }
  el.nav.style.cursor = "grabbing";
});
el.nav.addEventListener("pointermove", event => {
  if (!state.info) return;
  const time = navTime(event);
  if (navDrag) setStart(time - navDrag.offset);
  else el.nav.style.cursor = time >= state.start && time <= state.start + windowLength() ? "grab" : "pointer";
});
for (const type of ["pointerup", "pointercancel"]) {
  el.nav.addEventListener(type, () => { navDrag = null; el.nav.style.cursor = ""; });
}

// Signal: drag to pan, hover to read time and stage.
let signalDrag = null;
el.signal.addEventListener("pointerdown", event => {
  if (!state.info) return;
  el.signal.setPointerCapture(event.pointerId);
  signalDrag = {x: event.clientX, start: state.start};
  el.signal.classList.add("dragging");
});
el.signal.addEventListener("pointermove", event => {
  if (!state.info) return;
  const rect = el.signal.getBoundingClientRect();
  const plotW = rect.width - PAD.left - PAD.right;
  if (signalDrag) {
    setStart(signalDrag.start - (event.clientX - signalDrag.x) / plotW * windowLength());
    return;
  }
  const data = state.data;
  const fraction = (event.clientX - rect.left - PAD.left) / plotW;
  if (!data || fraction < 0 || fraction > 1) { el.readout.hidden = true; return; }
  const time = data.start + fraction * (data.end - data.start);
  const es = state.info.epoch_seconds;
  const epoch = es ? Math.floor(time / es) : -1;
  const code = state.info.stages[epoch];
  el.readout.textContent = formatTime(time, true) +
    (code === undefined ? "" : ` · ${STAGE.get(code).name} · epoch ${epoch + 1}`);
  el.readout.hidden = false;
});
for (const type of ["pointerup", "pointercancel"]) {
  el.signal.addEventListener(type, () => { signalDrag = null; el.signal.classList.remove("dragging"); });
}
el.signal.addEventListener("pointerleave", () => { el.readout.hidden = true; });

function onWheel(event) {
  event.preventDefault();
  if (!state.info) return;
  let delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
  if (event.deltaMode === 1) delta *= 33;
  setStart(state.start + windowLength() * clamp(delta / 500, -0.5, 0.5));
}
el.signal.addEventListener("wheel", onWheel, {passive: false});
el.nav.addEventListener("wheel", onWheel, {passive: false});

document.addEventListener("keydown", event => {
  if (event.target.matches("input, select, textarea") || event.ctrlKey || event.metaKey || event.altKey) return;
  const length = windowLength();
  const actions = {
    ArrowRight: () => setStart(state.start + length * 0.1),
    ArrowLeft: () => setStart(state.start - length * 0.1),
    PageDown: () => setStart(state.start + length),
    PageUp: () => setStart(state.start - length),
    Home: () => setStart(0),
    End: () => setStart(Infinity),
    n: () => loadRecording(state.index + 1),
    p: () => loadRecording(state.index - 1),
    "]": () => jumpToRun(1),
    "[": () => jumpToRun(-1),
    "+": () => zoom(-1), "=": () => zoom(-1), "-": () => zoom(1),
    Escape: () => { state.selected.clear(); selectionChanged(); },
  };
  const action = actions[event.key] || actions[event.key.toLowerCase()];
  if (action) { event.preventDefault(); action(); }
});
window.addEventListener("resize", drawAll);

// ---------- start ----------
async function initialize() {
  el.heading.textContent = `${CONFIG.dataset} EEG${CONFIG.channel ? " (" + CONFIG.channel + ")" : ""}`;
  const saved = new URLSearchParams(location.hash.slice(1));
  const wanted = saved.get("w") || String(CONFIG.window);
  if (![...el.window.options].some(option => option.value === wanted) && Number(wanted) > 0) {
    const option = document.createElement("option");
    option.value = wanted; option.textContent = `${wanted} seconds`;
    const later = [...el.window.options].find(o => o.value === "all" || Number(o.value) > Number(wanted));
    el.window.insertBefore(option, later);
  }
  el.window.value = wanted;
  if (el.window.selectedIndex < 0) el.window.value = "30";
  try {
    const response = await fetch("/api/list");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    state.files = data.files;
    if (!state.files.length) throw new Error("No .npz recordings found.");
    state.files.forEach((name, i) => el.recording.append(new Option(name, String(i))));
    const index = Math.max(0, state.files.indexOf(saved.get("file")));
    await loadRecording(index, Number(saved.get("t")) || 0);
    state.lastWindow = windowLength();
  } catch (error) {
    showStatus(`Could not load recordings: ${error.message}`, true);
  }
}
initialize();
</script>
</body>
</html>
"""


class Recording:
    """One .npz file, flattened to a continuous single-channel signal."""

    def __init__(self, path, signal, fs, stages, epoch_samples, undefined_labels):
        self.path = path
        self.signal = signal
        self.fs = fs
        self.stages = stages
        self.epoch_samples = epoch_samples
        self.undefined_labels = undefined_labels
        self.duration = signal.size / fs
        self.y_range = robust_range(signal)

    def info(self):
        return {
            "filename": os.path.basename(self.path),
            "fs": self.fs,
            "duration": self.duration,
            "epoch_seconds": self.epoch_samples / self.fs if self.epoch_samples else None,
            "stages": self.stages,
            "undefined_labels": self.undefined_labels,
            "y_range": self.y_range,
        }


def robust_range(signal):
    """Amplitude range that ignores rare artefacts (0.5th to 99.5th percentile)."""
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return [-1.0, 1.0]
    low, high = (float(v) for v in np.percentile(finite, [0.5, 99.5]))
    if low == high:
        low, high = low - 1.0, high + 1.0
    margin = (high - low) * 0.25
    return [low - margin, high + margin]


def normalize_labels(labels):
    """Map raw labels to viewer stage codes; anything unknown becomes UNDEFINED.

    Returns (codes, undefined_labels) where undefined_labels lists the raw
    values that could not be mapped, so the browser can show them.
    """
    codes, unknown = [], set()
    for value in labels.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            value = int(value)
        key = str(value).strip().upper()
        code = STAGE_ALIASES.get(key, UNDEFINED)
        if code == UNDEFINED:
            unknown.add(str(value).strip() or "(empty)")
        codes.append(code)
    return codes, sorted(unknown)[:12]


def load_recording(path, default_fs, channel_index):
    with np.load(path, allow_pickle=False) as data:
        if "x" not in data:
            raise ValueError(f"{path} does not contain an 'x' EEG array")
        x = np.asarray(data["x"], dtype=np.float32)
        labels = np.asarray(data["y"]).reshape(-1) if "y" in data else np.array([])
        fs = float(np.asarray(data["fs"]).item()) if "fs" in data else default_fs

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"{path} has an invalid sampling rate: {fs}")
    if x.ndim == 3:
        if not 0 <= channel_index < x.shape[1]:
            raise ValueError(
                f"Channel index {channel_index} is out of range for {path}; "
                f"the file contains {x.shape[1]} channels"
            )
        x = x[:, channel_index, :]
    elif x.ndim not in (1, 2):
        raise ValueError(f"{path} has unsupported x shape {x.shape}")
    elif channel_index != 0:
        raise ValueError(f"{path} has no explicit channel dimension; use channel index 0")

    if x.size == 0:
        raise ValueError(f"{path} contains an empty EEG signal")
    if x.ndim == 2:
        if labels.size and x.shape[0] != labels.size:
            raise ValueError(f"{path} has different numbers of signal epochs and labels")
        epoch_samples = x.shape[1]
    elif labels.size and x.size % labels.size == 0:
        epoch_samples = x.size // labels.size
    elif labels.size:
        raise ValueError(f"{path} signal length cannot be divided into labelled epochs")
    else:
        epoch_samples = None

    stages, undefined_labels = normalize_labels(labels)
    return Recording(
        path, np.ascontiguousarray(x.reshape(-1)), fs, stages,
        epoch_samples if stages else None, undefined_labels,
    )


def downsample_positions(values, max_points=MAX_POINTS):
    """Indices that keep the first, last, min and max sample of each bucket.

    This keeps spikes visible however far the view is zoomed out.
    """
    n = values.size
    if n <= max_points:
        return np.arange(n)
    bucket = int(np.ceil(4 * n / max_points))
    buckets = int(np.ceil(n / bucket))
    padded = np.full(buckets * bucket, np.nan, dtype=values.dtype)
    padded[:n] = values
    blocks = padded.reshape(buckets, bucket)
    missing = ~np.isfinite(blocks)
    lowest = np.where(missing, np.inf, blocks).argmin(axis=1)
    highest = np.where(missing, -np.inf, blocks).argmax(axis=1)
    base = np.arange(buckets) * bucket
    positions = np.concatenate([base, base + lowest, base + highest, base + bucket - 1])
    return np.unique(np.clip(positions, 0, n - 1))


def create_handler(files, default_fs, channel_index, dataset, window_seconds):
    @functools.lru_cache(maxsize=CACHED_RECORDINGS)
    def cached(index, modified):  # `modified` reloads a file that changed on disk
        return load_recording(files[index], default_fs, channel_index)

    def get_recording(query):
        index = int(query.get("index", ["0"])[0])
        if not 0 <= index < len(files):
            raise ValueError("Recording index is out of range")
        return cached(index, os.path.getmtime(files[index]))

    config = json.dumps({
        "dataset": dataset,
        "channel": DATASET_CHANNELS.get(dataset, ""),
        "window": int(window_seconds) if float(window_seconds).is_integer() else window_seconds,
    })
    page = PAGE.replace("__CONFIG__", config).encode("utf-8")

    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path)
            query = parse_qs(route.query)
            try:
                if route.path == "/":
                    self.respond(200, page, "text/html; charset=utf-8")
                elif route.path == "/api/list":
                    self.respond_json(200, {"files": [os.path.basename(p) for p in files]})
                elif route.path == "/api/info":
                    self.respond_json(200, get_recording(query).info())
                elif route.path == "/api/data":
                    self.respond_json(200, self.window_data(get_recording(query), query))
                else:
                    self.respond_json(404, {"error": "Not found"})
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.respond_json(400, {"error": str(error)})
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser cancelled the request

        @staticmethod
        def window_data(recording, query):
            signal, fs, duration = recording.signal, recording.fs, recording.duration
            window_request = query.get("window", [str(window_seconds)])[0]
            start = float(query.get("start", ["0"])[0])
            window = duration if window_request == "all" else float(window_request)
            if not np.isfinite(start) or not np.isfinite(window) or window <= 0:
                raise ValueError("Start time and window must be finite; window must be positive")
            window = min(window, duration)
            start = min(max(0.0, start), duration - window)
            end = start + window
            first = min(signal.size - 1, int(start * fs))
            last = min(signal.size, max(first + 1, int(np.ceil(end * fs)) + 1))
            values = signal[first:last]
            positions = downsample_positions(values)
            picked = values[positions].astype(np.float64)
            return {
                "start": start,
                "end": end,
                "times": ((first + positions) / fs).round(4).tolist(),
                # NaN/inf cannot be sent as JSON; null draws as a gap.
                "values": [v if math.isfinite(v) else None for v in picked.tolist()],
            }

        def respond_json(self, status, result):
            body = json.dumps(result, allow_nan=False).encode("utf-8")
            self.respond(status, body, "application/json; charset=utf-8")

        def respond(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return ViewerHandler


def has_desktop():
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) or (
        hasattr(os, "uname") and os.uname().sysname == "Darwin"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Open a browser-based viewer for single-channel EEG .npz files."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_CHANNELS), default="ANPHY")
    parser.add_argument("--data-dir", help="Folder containing .npz files")
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--fs", type=float, default=100.0,
                        help="Sampling rate used when a file has no 'fs' entry")
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=0, help="Local port (default: choose one)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args()
    if not args.window_seconds > 0 or not args.fs > 0:
        parser.error("--window-seconds and --fs must be greater than zero")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")

    channel = DATASET_CHANNELS[args.dataset]
    data_dir = args.data_dir or os.path.join("dset", args.dataset, "npz", channel)
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        parser.error(f"No .npz files found in {os.path.abspath(data_dir)}")

    handler = create_handler(
        files, args.fs, args.channel_index, args.dataset, args.window_seconds
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"EEG viewer ready: {url}", flush=True)
    print(f"Data folder: {os.path.abspath(data_dir)} ({len(files)} recordings)", flush=True)
    if args.no_browser:
        pass
    elif has_desktop():
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    else:
        print(
            "No desktop display detected. Open the URL in your browser; in VS Code "
            "Remote, forward this port in the Ports panel if needed.",
            flush=True,
        )
    print("Press Ctrl+C here to stop the viewer.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping EEG viewer.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()