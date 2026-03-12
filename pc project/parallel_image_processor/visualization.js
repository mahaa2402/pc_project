/**
 * Phase 1: Parallel Image Processor visualization.
 * Loads metrics_rank*.csv and threshold_parallel.ppm from a run folder,
 * shows decomposition (rank bands on image), per-stage timing bars, and summary.
 */

(function () {
  'use strict';

  const STAGE_KEYS = ['scatter_s', 'gray_s', 'blur_s', 'sobel_s', 'threshold_s', 'gather_s'];
  const STAGE_LABELS = ['Scatter', 'Grayscale', 'Blur', 'Sobel', 'Threshold', 'Gather'];
  const STAGE_COLORS = [
    '#4a90d9', '#7ed321', '#f5a623', '#bd10e0', '#50e3c2', '#d0021b'
  ];

  let metrics = [];      // array of { rank, rows_start, rows_end, rows_local, ...times }
  let imageData = null;  // { width, height, data: Uint8ClampedArray RGB }
  let imageWidth = 0;
  let imageHeight = 0;

  function $(id) { return document.getElementById(id); }
  function setStatus(msg, isError = false) {
    const el = $('load-status');
    if (!el) return;
    el.textContent = msg;
    el.className = 'status ' + (isError ? 'error' : '');
  }

  function getBaseUrl() {
    const path = window.location.pathname;
    const last = path.lastIndexOf('/');
    if (last === -1) return '';
    return path.slice(0, last + 1);
  }

  function getStemFromQuery() {
    const params = new URLSearchParams(window.location.search);
    return params.get('stem') || '';
  }

  /**
   * Parse CSV text (one header row, one data row per rank file).
   */
  function parseMetricsCsv(text) {
    const lines = text.trim().split(/\r?\n/);
    if (lines.length < 2) return null;
    const headers = lines[0].split(',').map(h => h.trim());
    const row = lines[1].split(',').map(c => c.trim());
    const obj = {};
    headers.forEach((h, i) => {
      const v = row[i];
      obj[h] = isNaN(Number(v)) ? v : Number(v);
    });
    return obj;
  }

  /**
   * Load all metrics_rank*.csv by fetching rank 0, 1, 2, ... until 404.
   */
  async function loadMetricsFromStem(stem) {
    const base = getBaseUrl();
    const prefix = stem ? stem + '/' : '';
    const out = [];
    for (let r = 0; ; r++) {
      const url = base + prefix + 'metrics_rank' + r + '.csv';
      try {
        const res = await fetch(url);
        if (!res.ok) break;
        const text = await res.text();
        const row = parseMetricsCsv(text);
        if (row) out.push(row);
      } catch (_) {
        break;
      }
    }
    return out;
  }

  /**
   * Load threshold_parallel.ppm from stem (fetch as arraybuffer).
   */
  async function loadPpmFromStem(stem) {
    const base = getBaseUrl();
    const prefix = stem ? stem + '/' : '';
    const url = base + prefix + 'threshold_parallel.ppm';
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to load threshold_parallel.ppm: ' + res.status);
    const buf = await res.arrayBuffer();
    return parsePpmP6(new Uint8Array(buf));
  }

  /**
   * Parse P6 PPM: "P6" [whitespace/comments] width height [whitespace/comments] 255 [whitespace] binary RGB.
   */
  function parsePpmP6(bytes) {
    let i = 0;
    function skipWhitespaceAndComments() {
      while (i < bytes.length) {
        if (bytes[i] === 0x23) {
          while (i < bytes.length && bytes[i] !== 0x0a) i++;
          continue;
        }
        if (bytes[i] === 0x20 || bytes[i] === 0x09 || bytes[i] === 0x0a || bytes[i] === 0x0d) {
          i++;
          continue;
        }
        break;
      }
    }
    function readToken() {
      skipWhitespaceAndComments();
      const start = i;
      while (i < bytes.length && bytes[i] !== 0x20 && bytes[i] !== 0x09 && bytes[i] !== 0x0a && bytes[i] !== 0x0d && bytes[i] !== 0x23) i++;
      return start;
    }
    if (bytes[0] !== 0x50 || bytes[1] !== 0x36) throw new Error('Not P6 PPM');
    i = 2;
    skipWhitespaceAndComments();
    const wStart = readToken();
    const width = parseInt(String.fromCharCode.apply(null, bytes.subarray(wStart, i)), 10);
    skipWhitespaceAndComments();
    const hStart = readToken();
    const height = parseInt(String.fromCharCode.apply(null, bytes.subarray(hStart, i)), 10);
    skipWhitespaceAndComments();
    const maxStart = readToken();
    const maxval = parseInt(String.fromCharCode.apply(null, bytes.subarray(maxStart, i)), 10);
    if (maxval !== 255) throw new Error('Unsupported PPM maxval: ' + maxval);
    skipWhitespaceAndComments();
    const dataStart = i;
    const pixelCount = width * height;
    if (dataStart + pixelCount * 3 > bytes.length) throw new Error('PPM data too short');
    const data = new Uint8ClampedArray(pixelCount * 4);
    for (let p = 0; p < pixelCount; p++) {
      const src = dataStart + p * 3;
      data[p * 4]     = bytes[src];
      data[p * 4 + 1] = bytes[src + 1];
      data[p * 4 + 2] = bytes[src + 2];
      data[p * 4 + 3] = 255;
    }
    return { width, height, data };
  }

  /**
   * Load run from stem (URL / server): fetch CSVs and PPM.
   */
  async function loadFromStem(stem) {
    setStatus('Loading…');
    try {
      const [m, img] = await Promise.all([
        loadMetricsFromStem(stem),
        loadPpmFromStem(stem).catch(() => null)
      ]);
      if (!m.length) {
        setStatus('No metrics_rank*.csv found in "' + (stem || '.') + '"', true);
        return;
      }
      metrics = m.sort((a, b) => a.rank - b.rank);
      imageData = img;
      if (img) {
        imageWidth = img.width;
        imageHeight = img.height;
      } else {
        imageWidth = 0;
        imageHeight = 0;
        setStatus('Loaded ' + metrics.length + ' rank(s). Image not found.');
      }
      if (imageData) setStatus('Loaded ' + metrics.length + ' rank(s) and image.');
      render();
      $('main-content').classList.remove('hidden');
    } catch (e) {
      setStatus('Error: ' + e.message, true);
      console.error(e);
    }
  }

  /**
   * Load from File objects (folder picker): files = { 'metrics_rank0.csv': File, ..., 'threshold_parallel.ppm': File }.
   */
  async function loadFromFiles(files) {
    setStatus('Loading from folder…');
    try {
      metrics = [];
      const rankFiles = Object.keys(files).filter(k => /^metrics_rank\d+\.csv$/.test(k));
      for (const name of rankFiles.sort()) {
        const text = await files[name].text();
        const row = parseMetricsCsv(text);
        if (row) metrics.push(row);
      }
      metrics.sort((a, b) => a.rank - b.rank);
      if (!metrics.length) {
        setStatus('No metrics_rank*.csv in selected folder.', true);
        return;
      }
      imageData = null;
      imageWidth = 0;
      imageHeight = 0;
      if (files['threshold_parallel.ppm']) {
        const buf = await files['threshold_parallel.ppm'].arrayBuffer();
        imageData = parsePpmP6(new Uint8Array(buf));
        imageWidth = imageData.width;
        imageHeight = imageData.height;
      }
      setStatus('Loaded ' + metrics.length + ' rank(s)' + (imageData ? ' and image.' : ', no image.'));
      render();
      $('main-content').classList.remove('hidden');
    } catch (e) {
      setStatus('Error: ' + e.message, true);
      console.error(e);
    }
  }

  function render() {
    renderDecomposition();
    renderTimingBars();
    renderSummary();
  }

  const MAX_DISPLAY_WIDTH = 900;
  const MAX_DISPLAY_HEIGHT = 600;

  function renderDecomposition() {
    const canvas = $('image-canvas');
    const bandsCanvas = $('bands-canvas');
    const labelsDiv = $('band-labels');
    if (!canvas || !bandsCanvas || !labelsDiv) return;

    const maxEnd = metrics.length ? Math.max(...metrics.map(m => m.rows_end)) : 0;
    const srcWidth = imageWidth || 600;
    const srcHeight = imageHeight || Math.max(maxEnd, 400);
    const scale = Math.min(1, MAX_DISPLAY_WIDTH / srcWidth, MAX_DISPLAY_HEIGHT / srcHeight);
    const displayWidth = Math.round(srcWidth * scale);
    const displayHeight = Math.round(srcHeight * scale);

    canvas.width = displayWidth;
    canvas.height = displayHeight;
    bandsCanvas.width = displayWidth;
    bandsCanvas.height = displayHeight;

    const ctx = canvas.getContext('2d');
    const bandCtx = bandsCanvas.getContext('2d');

    if (imageData && imageData.data) {
      const tmp = document.createElement('canvas');
      tmp.width = srcWidth;
      tmp.height = srcHeight;
      const tmpCtx = tmp.getContext('2d');
      const imageDataObj = tmpCtx.createImageData(srcWidth, srcHeight);
      imageDataObj.data.set(imageData.data);
      tmpCtx.putImageData(imageDataObj, 0, 0);
      ctx.drawImage(tmp, 0, 0, srcWidth, srcHeight, 0, 0, displayWidth, displayHeight);
    } else {
      ctx.fillStyle = '#2a2a2a';
      ctx.fillRect(0, 0, displayWidth, displayHeight);
      ctx.fillStyle = '#666';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No image (threshold_parallel.ppm not found)', displayWidth / 2, displayHeight / 2);
    }

    bandCtx.clearRect(0, 0, displayWidth, displayHeight);
    labelsDiv.innerHTML = '';
    if (!metrics.length) return;

    const refHeight = imageHeight || maxEnd || 1;
    const rowToY = (row) => (row / refHeight) * displayHeight;
    const colors = ['rgba(74,144,217,0.35)', 'rgba(126,211,33,0.35)', 'rgba(245,166,35,0.35)', 'rgba(189,16,224,0.35)', 'rgba(80,227,194,0.35)', 'rgba(208,2,27,0.35)'];
    metrics.forEach((m, idx) => {
      const y0 = rowToY(m.rows_start);
      const y1 = rowToY(m.rows_end);
      bandCtx.fillStyle = colors[idx % colors.length];
      bandCtx.fillRect(0, y0, displayWidth, y1 - y0);
      bandCtx.strokeStyle = 'rgba(255,255,255,0.8)';
      bandCtx.lineWidth = 1;
      bandCtx.strokeRect(0, y0, displayWidth, y1 - y0);
      const label = document.createElement('div');
      label.className = 'band-label';
      label.textContent = 'Rank ' + m.rank + ': rows ' + m.rows_start + '–' + (m.rows_end - 1);
      label.style.top = ((y0 / displayHeight) * 100) + '%';
      label.style.left = '8px';
      labelsDiv.appendChild(label);
    });
  }

  function renderTimingBars() {
    const legendEl = $('timing-legend');
    if (legendEl) {
      legendEl.innerHTML = STAGE_LABELS.map((l, i) =>
        '<span class="legend-item" style="--c:' + STAGE_COLORS[i] + '"><i></i>' + l + '</span>'
      ).join('');
    }
    const container = $('timing-bars');
    if (!container) return;
    container.innerHTML = '';

    const maxTotal = Math.max(...metrics.map(m => m.total_s), 1e-9);
    const barHeight = 28;
    const gap = 8;
    const stageLabels = STAGE_LABELS;

    metrics.forEach(m => {
      const row = document.createElement('div');
      row.className = 'timing-row';
      const label = document.createElement('span');
      label.className = 'timing-rank-label';
      label.textContent = 'Rank ' + m.rank + ' (total: ' + m.total_s.toFixed(3) + ' s)';
      row.appendChild(label);
      const barWrap = document.createElement('div');
      barWrap.className = 'timing-bar-wrap';
      let left = 0;
      STAGE_KEYS.forEach((key, i) => {
        const t = m[key] || 0;
        const w = (t / maxTotal) * 100;
        const seg = document.createElement('div');
        seg.className = 'timing-segment';
        seg.style.width = w + '%';
        seg.style.left = left + '%';
        seg.style.backgroundColor = STAGE_COLORS[i];
        seg.title = stageLabels[i] + ': ' + t.toFixed(4) + ' s';
        barWrap.appendChild(seg);
        left += w;
      });
      row.appendChild(barWrap);
      container.appendChild(row);
    });
  }

  function renderSummary() {
    const tbody = $('summary-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    const rows = [
      ['Number of ranks', String(metrics.length)],
      ['Image size', imageWidth && imageHeight ? imageWidth + ' × ' + imageHeight : '—'],
      ['Total time (max across ranks)', metrics.length ? (Math.max(...metrics.map(m => m.total_s))).toFixed(4) + ' s' : '—']
    ];
    STAGE_KEYS.forEach((key, i) => {
      const max = metrics.length ? Math.max(...metrics.map(m => m[key] || 0)) : 0;
      rows.push(['Max ' + STAGE_LABELS[i].toLowerCase(), max.toFixed(4) + ' s']);
    });
    rows.forEach(([label, value]) => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + label + '</td><td>' + value + '</td>';
      tbody.appendChild(tr);
    });
  }

  function init() {
    const stemInput = $('stem-input');
    const loadFromStemBtn = $('load-from-stem');
    const pickFolderBtn = $('pick-folder');

    const stem = getStemFromQuery();
    if (stem) {
      if (stemInput) stemInput.value = stem;
      loadFromStem(stem);
    }

    if (loadFromStemBtn && stemInput) {
      loadFromStemBtn.addEventListener('click', () => loadFromStem(stemInput.value.trim()));
    }

    if (pickFolderBtn) {
      pickFolderBtn.addEventListener('click', async () => {
        if (!('showDirectoryPicker' in window)) {
          setStatus('Folder picker not supported in this browser. Use stem + Load or run server from parallel_image_processor/ and open ?stem=...', true);
          return;
        }
        try {
          const dir = await window.showDirectoryPicker();
          const files = {};
          for await (const [name, handle] of dir.entries()) {
            if (handle.kind === 'file' && (name.match(/^metrics_rank\d+\.csv$/) || name === 'threshold_parallel.ppm')) {
              files[name] = await handle.getFile();
            }
          }
          await loadFromFiles(files);
        } catch (e) {
          if (e.name !== 'AbortError') setStatus('Error: ' + e.message, true);
        }
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
