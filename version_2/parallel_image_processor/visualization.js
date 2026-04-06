/**
 * Phase 1: Parallel Image Processor visualization.
 * Loads metrics_rank*.csv and threshold_parallel.ppm from a run folder,
 * shows decomposition (rank bands on image), per-stage timing bars, and summary.
 */

(function () {
  'use strict' ;

  const STAGE_KEYS = ['scatter_s', 'gray_s', 'blur_s', 'sobel_s', 'threshold_s', 'gather_s'];
  const STAGE_LABELS = ['Scatter', 'Grayscale', 'Blur', 'Sobel', 'Threshold', 'Gather'];
  const STAGE_COLORS = [
    '#4a90d9', '#7ed321', '#f5a623', '#bd10e0', '#50e3c2', '#d0021b'
  ];

  /**
   * Bar segment widths use time^POWER (not raw seconds) so 0.001s stages stay visible next to ~0.1s blur.
   * Chip text and tooltips still show exact seconds. ~0.4 is between sqrt and linear on small values.
   */
  const TIMING_BAR_WIDTH_POWER = 0.4;

  function timingBarVisualWeight(seconds) {
    const t = Math.max(Number(seconds) || 0, 1e-15);
    return Math.pow(t, TIMING_BAR_WIDTH_POWER);
  }

  /** Matches C++ intermediate filenames: rank_XXX_<key>.ppm */
  const PIPELINE_STAGES = [
    { key: '01_scatter_rgb', label: 'After scatter (local RGB strip)' },
    { key: '02_grayscale', label: 'Grayscale' },
    { key: '03_blur', label: 'Gaussian blur' },
    { key: '04_sobel', label: 'Sobel edges' },
    { key: '05_threshold', label: 'Threshold' }
  ];

  let metrics = [];      // array of { rank, rows_start, rows_end, rows_local, ...times }
  let imageData = null;  // { width, height, data: Uint8ClampedArray RGB }
  let imageWidth = 0;
  let imageHeight = 0;

  let loadedStem = '';
  let runFilesMap = null;
  /** rankIdx = index into sorted metrics (0 .. n-1), not necessarily MPI rank id */
  let pipelineReplay = { rankIdx: 0 };
  let intermediateReq = 0;
  const intermediateCache = new Map();

  function $(id) { return document.getElementById(id); }
  function setStatus(msg, isError = false) {
    const el = $('load-status');
    if (!el) return;
    el.textContent = msg;
    el.className = 'status ' + (isError ? 'error' : '');
  }

  function setUploadStatus(msg, isError = false) {
    const el = $('upload-status');
    if (!el) return;
    el.textContent = msg;
    el.className = 'status upload-status-line ' + (isError ? 'error' : '');
  }

  function getBaseUrl() {
    const path = window.location.pathname;
    const last = path.lastIndexOf('/');
    if (last === -1) return '';
    return path.slice(0, last + 1);
  }

  function getUploadUrl() {
    return getBaseUrl() + 'upload';
  }

  async function uploadImageFile(file) {
    setUploadStatus('Uploading, converting to PPM, running MPI pipeline…');
    const fd = new FormData();
    fd.append('image', file, file.name);
    let res;
    try {
      res = await fetch(getUploadUrl(), { method: 'POST', body: fd });
    } catch (e) {
      setUploadStatus('Network error: ' + e.message, true);
      return;
    }
    let data;
    try {
      data = await res.json();
    } catch (_) {
      setUploadStatus('Upload failed (' + res.status + ').', true);
      return;
    }
    if (!res.ok || !data.ok) {
      setUploadStatus(data.error || ('Upload failed (' + res.status + ').'), true);
      return;
    }
    let msg = 'Saved as ' + (data.saved_as || file.name);
    if (data.ppm_saved_as) msg += '; PPM: ' + data.ppm_saved_as;
    if (data.pipeline_ok) {
      msg += '; pipeline complete.';
      setUploadStatus(msg);
      if (data.stem) {
        const stemInput = $('stem-input');
        if (stemInput) stemInput.value = data.stem;
        await loadFromStem(data.stem);
      }
      return;
    }
    if (data.pipeline_error) {
      msg += '. MPI failed: ' + data.pipeline_error;
    } else {
      msg += '. Pipeline did not finish.';
    }
    setUploadStatus(msg, true);
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
   * Load metrics_rank*.csv. Uses num_ranks from rank0 CSV when present (no extra 404s).
   */
  async function loadMetricsFromStem(stem) {
    const base = getBaseUrl();
    const prefix = stem ? stem + '/' : '';
    const url0 = base + prefix + 'metrics_rank0.csv';
    let res0;
    try {
      res0 = await fetch(url0);
    } catch (_) {
      return [];
    }
    if (!res0.ok) return [];
    const t0 = await res0.text();
    const row0 = parseMetricsCsv(t0);
    if (!row0) return [];
    const out = [row0];
    const nDeclared = typeof row0.num_ranks === 'number' && row0.num_ranks >= 1
      ? Math.min(1024, Math.floor(row0.num_ranks))
      : null;
    if (nDeclared !== null && nDeclared > 1) {
      for (let r = 1; r < nDeclared; r++) {
        const res = await fetch(base + prefix + 'metrics_rank' + r + '.csv');
        if (!res.ok) break;
        const text = await res.text();
        const row = parseMetricsCsv(text);
        if (row) out.push(row);
      }
      return out;
    }
    for (let r = 1; ; r++) {
      const res = await fetch(base + prefix + 'metrics_rank' + r + '.csv');
      if (!res.ok) break;
      const text = await res.text();
      const row = parseMetricsCsv(text);
      if (row) out.push(row);
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

  function pathBasename(p) {
    const norm = p.replace(/\\/g, '/');
    const i = norm.lastIndexOf('/');
    return i === -1 ? norm : norm.slice(i + 1);
  }

  function rankTagPadded(rank) {
    return 'rank_' + String(rank).padStart(3, '0');
  }

  /**
   * Rasterize PPM to a display canvas. For thin horizontal strips, minDisplayHeight upscales
   * so row-wise pipeline stages stay visible (may exceed native resolution).
   */
  function ppmImageDataToCanvas(img, maxW, maxH, opts) {
    opts = opts || {};
    const minDH = opts.minDisplayHeight || 0;
    const maxUpscale = opts.maxUpscale == null ? (minDH > 0 ? 8 : 1) : opts.maxUpscale;
    let s = Math.min(1, maxW / img.width, maxH / img.height);
    if (minDH > 0 && img.height > 0 && img.height * s < minDH) {
      s = Math.max(s, Math.min(minDH / img.height, maxUpscale));
    }
    if (img.width * s > maxW) s = maxW / img.width;
    if (img.height * s > maxH) s = maxH / img.height;
    const dw = Math.max(1, Math.round(img.width * s));
    const dh = Math.max(1, Math.round(img.height * s));
    const c = document.createElement('canvas');
    c.width = dw;
    c.height = dh;
    const tmp = document.createElement('canvas');
    tmp.width = img.width;
    tmp.height = img.height;
    const tctx = tmp.getContext('2d');
    const imageDataObj = tctx.createImageData(img.width, img.height);
    imageDataObj.data.set(img.data);
    tctx.putImageData(imageDataObj, 0, 0);
    const octx = c.getContext('2d');
    octx.imageSmoothingEnabled = s <= 1;
    octx.drawImage(tmp, 0, 0, img.width, img.height, 0, 0, dw, dh);
    return c;
  }

  const INTERMEDIATE_THUMB_MAX_W = 520;
  const INTERMEDIATE_THUMB_MAX_H = 520;
  const INTERMEDIATE_THUMB_MIN_H = 240;

  function ppmIntermediateThumbnail(parsed) {
    return ppmImageDataToCanvas(parsed, INTERMEDIATE_THUMB_MAX_W, INTERMEDIATE_THUMB_MAX_H, {
      minDisplayHeight: INTERMEDIATE_THUMB_MIN_H,
      maxUpscale: 8
    });
  }

  async function fetchIntermediatePpm(stem, filesMap, rank, stageKey) {
    const name = rankTagPadded(rank) + '_' + stageKey + '.ppm';
    const rel = 'intermediate/' + name;
    if (filesMap && filesMap[rel]) {
      const buf = await filesMap[rel].arrayBuffer();
      return new Uint8Array(buf);
    }
    if (!stem) return null;
    const base = getBaseUrl();
    const url = base + stem + '/' + rel;
    const res = await fetch(url);
    if (!res.ok) return null;
    const buf = await res.arrayBuffer();
    return new Uint8Array(buf);
  }

  function clearIntermediateCache() {
    intermediateCache.clear();
  }

  function clampRankIndex() {
    if (!metrics.length) return;
    if (pipelineReplay.rankIdx < 0) pipelineReplay.rankIdx = 0;
    if (pipelineReplay.rankIdx >= metrics.length) pipelineReplay.rankIdx = metrics.length - 1;
  }

  function openPpmLightbox(img, caption) {
    const box = $('ppm-lightbox');
    const cv = $('ppm-lightbox-canvas');
    const cap = $('ppm-lightbox-caption');
    if (!box || !cv || !img || !img.data) return;
    cv.width = img.width;
    cv.height = img.height;
    const ctx = cv.getContext('2d');
    const id = ctx.createImageData(img.width, img.height);
    id.data.set(img.data);
    ctx.putImageData(id, 0, 0);
    if (cap) cap.textContent = caption || '';
    box.classList.remove('hidden');
    document.body.classList.add('lightbox-open');
  }

  function closePpmLightbox() {
    const box = $('ppm-lightbox');
    if (box) box.classList.add('hidden');
    document.body.classList.remove('lightbox-open');
  }

  async function collectRunFolderFiles(dirHandle, relPrefix) {
    const out = {};
    for await (const [name, handle] of dirHandle.entries()) {
      const rel = relPrefix ? relPrefix + '/' + name : name;
      if (handle.kind === 'directory') {
        Object.assign(out, await collectRunFolderFiles(handle, rel));
      } else if (handle.kind === 'file') {
        const base = pathBasename(rel);
        const lc = base.toLowerCase();
        if (/^metrics_rank\d+\.csv$/.test(base) || base === 'threshold_parallel.ppm' ||
            (rel.startsWith('intermediate/') && lc.endsWith('.ppm'))) {
          out[rel] = await handle.getFile();
        }
      }
    }
    return out;
  }

  async function refreshIntermediateGrid() {
    const grid = $('intermediate-grid');
    const lbl = $('rank-index-label');
    if (!grid || !metrics.length) return;
    clampRankIndex();
    const sorted = metrics.slice().sort((a, b) => a.rank - b.rank);
    const mr = sorted[pipelineReplay.rankIdx];
    if (!mr) return;
    const myReq = ++intermediateReq;
    if (lbl) {
      lbl.textContent = 'Rank ' + mr.rank + ' (' + (pipelineReplay.rankIdx + 1) + '/' + sorted.length + ') — rows '
        + mr.rows_start + '–' + (mr.rows_end - 1);
    }
    grid.innerHTML = '<p class="intermediate-loading">Loading strips…</p>';
    const cells = [];
    for (const stage of PIPELINE_STAGES) {
      const cacheKey = mr.rank + '_' + stage.key;
      let bytes;
      if (intermediateCache.has(cacheKey)) {
        bytes = intermediateCache.get(cacheKey);
      } else {
        bytes = await fetchIntermediatePpm(loadedStem, runFilesMap, mr.rank, stage.key);
        intermediateCache.set(cacheKey, bytes);
      }
      if (myReq !== intermediateReq) return;
      if (!bytes) {
        cells.push({ stage, error: true });
        continue;
      }
      try {
        cells.push({ stage, parsed: parsePpmP6(bytes) });
      } catch (_) {
        cells.push({ stage, error: true });
      }
    }
    if (myReq !== intermediateReq) return;
    grid.innerHTML = '';
    let anyOk = false;
    for (const c of cells) {
      const fig = document.createElement('figure');
      fig.className = 'intermediate-cell';
      const cap = document.createElement('figcaption');
      cap.textContent = c.stage.label;
      fig.appendChild(cap);
      if (c.error || !c.parsed) {
        const p = document.createElement('p');
        p.className = 'intermediate-miss';
        p.textContent = 'No strip';
        fig.appendChild(p);
      } else {
        anyOk = true;
        fig.classList.add('intermediate-cell-clickable');
        const thumb = ppmIntermediateThumbnail(c.parsed);
        const capText = 'Rank ' + mr.rank + ' — ' + c.stage.label + ' (' + c.parsed.width + '×' + c.parsed.height + ')';
        fig.addEventListener('click', () => openPpmLightbox(c.parsed, capText));
        fig.appendChild(thumb);
      }
      grid.appendChild(fig);
    }
    if (!anyOk && cells.length) {
      const note = document.createElement('p');
      note.className = 'intermediate-miss';
      note.textContent = 'No intermediate PPMs found. Rebuild and re-run parallel_ppm to populate intermediate/.';
      grid.appendChild(note);
    }
  }

  /**
   * Load run from stem (URL / server): fetch CSVs and PPM.
   */
  async function loadFromStem(stem) {
    clearIntermediateCache();
    loadedStem = stem;
    runFilesMap = null;
    setStatus('Loading…');
    try {
      const [m, img] = await Promise.all([
        loadMetricsFromStem(stem),
        loadPpmFromStem(stem).catch(() => null)
      ]);
      if (!m.length) {
        setStatus('No metrics_rank*.csv found in "' + (stem || '.') + '"', true);
        const ig = $('intermediate-grid');
        if (ig) ig.innerHTML = '';
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
      pipelineReplay.rankIdx = 0;
      try {
        const u = new URL(window.location.href);
        if (stem) u.searchParams.set('stem', stem);
        history.replaceState(null, '', u.pathname + u.search + u.hash);
      } catch (_) { /* ignore */ }
      await refreshIntermediateGrid();
      $('intermediate-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      setStatus('Error: ' + e.message, true);
      console.error(e);
    }
  }

  /**
   * Load from File objects (folder picker): flat or nested keys including intermediate/*.ppm.
   */
  async function loadFromFiles(files) {
    clearIntermediateCache();
    loadedStem = '';
    runFilesMap = files;
    setStatus('Loading from folder…');
    try {
      metrics = [];
      const rankFiles = Object.keys(files).filter(k => /^metrics_rank\d+\.csv$/.test(pathBasename(k)));
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
      const threshKey = Object.keys(files).find(k => pathBasename(k) === 'threshold_parallel.ppm');
      if (threshKey) {
        const buf = await files[threshKey].arrayBuffer();
        imageData = parsePpmP6(new Uint8Array(buf));
        imageWidth = imageData.width;
        imageHeight = imageData.height;
      }
      setStatus('Loaded ' + metrics.length + ' rank(s)' + (imageData ? ' and image.' : ', no image.'));
      render();
      $('main-content').classList.remove('hidden');
      pipelineReplay.rankIdx = 0;
      await refreshIntermediateGrid();
      $('intermediate-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      setStatus('Error: ' + e.message, true);
      console.error(e);
    }
  }

  function render() {
    renderDecomposition();
    renderTimingBars();
    renderSummary();
    void renderSerialCompareSection();
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

    const stageLabels = STAGE_LABELS;

    metrics.forEach(m => {
      const row = document.createElement('div');
      row.className = 'timing-row';
      const label = document.createElement('span');
      label.className = 'timing-rank-label';
      label.textContent = 'Rank ' + m.rank + ' (total: ' + m.total_s.toFixed(3) + ' s)';
      row.appendChild(label);
      const stack = document.createElement('div');
      stack.className = 'timing-bar-stack';
      const barWrap = document.createElement('div');
      barWrap.className = 'timing-bar-wrap';
      const rowTotal = Math.max(m.total_s, 1e-12);
      const weights = STAGE_KEYS.map(key => timingBarVisualWeight(m[key] || 0));
      barWrap.style.gridTemplateColumns = weights.map(w => w + 'fr').join(' ');
      STAGE_KEYS.forEach((key, i) => {
        const t = m[key] || 0;
        const seg = document.createElement('div');
        seg.className = 'timing-segment';
        seg.style.backgroundColor = STAGE_COLORS[i];
        const pct = (t / rowTotal) * 100;
        seg.title = stageLabels[i] + ': ' + t.toFixed(4) + ' s (' + pct.toFixed(1) + '% of rank total). '
          + 'Bar width is scaled (t^' + TIMING_BAR_WIDTH_POWER + ') so small times stay visible; value is exact.';
        barWrap.appendChild(seg);
      });
      stack.appendChild(barWrap);
      const breakdown = document.createElement('div');
      breakdown.className = 'timing-breakdown';
      STAGE_KEYS.forEach((key, i) => {
        const t = m[key] || 0;
        const chip = document.createElement('span');
        chip.className = 'timing-chip';
        const sw = document.createElement('span');
        sw.className = 'timing-chip-swatch';
        sw.style.backgroundColor = STAGE_COLORS[i];
        const name = document.createElement('span');
        name.className = 'timing-chip-name';
        name.textContent = stageLabels[i];
        const val = document.createElement('span');
        val.className = 'timing-chip-val';
        val.textContent = t.toFixed(3);
        const unit = document.createElement('span');
        unit.className = 'timing-chip-unit';
        unit.textContent = 's';
        chip.appendChild(sw);
        chip.appendChild(name);
        chip.appendChild(val);
        chip.appendChild(unit);
        breakdown.appendChild(chip);
      });
      stack.appendChild(breakdown);
      row.appendChild(stack);
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

  let serialBenchmarksCache = null;

  async function ensureSerialBenchmarks() {
    if (serialBenchmarksCache) return serialBenchmarksCache;
    try {
      const res = await fetch(getBaseUrl() + 'serial_benchmarks.json');
      if (!res.ok) return null;
      serialBenchmarksCache = await res.json();
      return serialBenchmarksCache;
    } catch (_) {
      return null;
    }
  }

  function linearFitPixelsToMs(runs) {
    if (!runs || runs.length < 2) return null;
    const pts = runs.map(r => ({ x: r.pixels, y: r.total_ms }));
    const n = pts.length;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (let i = 0; i < n; i++) {
      sx += pts[i].x;
      sy += pts[i].y;
      sxx += pts[i].x * pts[i].x;
      sxy += pts[i].x * pts[i].y;
    }
    const denom = n * sxx - sx * sx;
    if (Math.abs(denom) < 1) return null;
    const slope = (n * sxy - sx * sy) / denom;
    const intercept = (sy - slope * sx) / n;
    return {
      slope,
      intercept,
      predictMs(pixels) {
        return Math.max(0, slope * pixels + intercept);
      }
    };
  }

  function formatDurationMs(ms) {
    if (!Number.isFinite(ms) || ms < 0) return '—';
    if (ms < 1000) return ms.toFixed(0) + ' ms';
    const s = ms / 1000;
    if (s < 120) return s.toFixed(2) + ' s';
    const m = Math.floor(s / 60);
    const rs = s - m * 60;
    return m + ' min ' + rs.toFixed(1) + ' s';
  }

  function setupCompareCanvas(canvas, cssW, cssH) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function drawCompareBars(canvas, serialMs, parallelMs, hasEstimate) {
    const W = 640;
    const H = 200;
    const ctx = setupCompareCanvas(canvas, W, H);
    ctx.fillStyle = '#1a1b26';
    ctx.fillRect(0, 0, W, H);
    const padL = 150;
    const padR = 24;
    const barH = 36;
    const gap = 28;
    const y0 = 48;
    const innerW = W - padL - padR;
    const maxVal = Math.max(hasEstimate ? serialMs : 0, parallelMs, 1) * 1.08;
    function bar(y, label, ms, color) {
      const frac = Math.min(1, ms / maxVal);
      const bw = innerW * frac;
      ctx.fillStyle = '#24283b';
      ctx.fillRect(padL, y, innerW, barH);
      ctx.fillStyle = color;
      ctx.fillRect(padL, y, Math.max(bw, ms > 0 ? 4 : 0), barH);
      ctx.fillStyle = '#c0caf5';
      ctx.font = '600 13px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, padL - 12, y + barH / 2);
      ctx.textAlign = 'left';
      ctx.fillText(formatDurationMs(ms), padL + bw + 8, y + barH / 2);
    }
    if (hasEstimate) {
      bar(y0, 'Serial (est.)', serialMs, '#f7768e');
      bar(y0 + barH + gap, 'Parallel (run)', parallelMs, '#9ece6a');
    } else {
      bar(y0 + (barH + gap) / 2, 'Parallel (run)', parallelMs, '#9ece6a');
      ctx.fillStyle = '#565f89';
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.fillText('Serial estimate needs image dimensions (load threshold_parallel.ppm).', padL, y0 + 8);
    }
    ctx.fillStyle = '#565f89';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0', padL, H - 14);
    ctx.textAlign = 'right';
    ctx.fillText(formatDurationMs(maxVal), padL + innerW, H - 14);
  }

  function drawCompareScatter(canvas, runs, fit, curMp, curEstMs, curParMs, hasCurrent) {
    const W = 640;
    const H = 320;
    const ctx = setupCompareCanvas(canvas, W, H);
    ctx.fillStyle = '#1a1b26';
    ctx.fillRect(0, 0, W, H);
    const pad = { l: 56, r: 20, t: 20, b: 48 };
    const iw = W - pad.l - pad.r;
    const ih = H - pad.t - pad.b;
    const refPts = runs.map(r => ({ mp: r.pixels / 1e6, ms: r.total_ms }));
    let maxMp = Math.max(...refPts.map(p => p.mp), hasCurrent ? curMp : 0, 0.1) * 1.12;
    let maxMs = Math.max(...refPts.map(p => p.ms), hasCurrent ? Math.max(curEstMs, curParMs) : 0, 1000);
    if (fit) {
      maxMs = Math.max(maxMs, fit.predictMs(maxMp * 1e6), fit.predictMs(0));
    }
    maxMs *= 1.1;
    const xOf = (mp) => pad.l + (mp / maxMp) * iw;
    const yOf = (ms) => {
      const c = Math.min(Math.max(ms, 0), maxMs);
      return pad.t + ih - (c / maxMs) * ih;
    };
    ctx.strokeStyle = '#3b4261';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t + ih);
    ctx.lineTo(pad.l + iw, pad.t + ih);
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, pad.t + ih);
    ctx.stroke();
    if (fit) {
      const p0 = 0;
      const p1 = maxMp * 1e6;
      const ms0 = fit.predictMs(p0);
      const ms1 = fit.predictMs(p1);
      ctx.strokeStyle = '#7aa2f7';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(xOf(0), yOf(ms0));
      ctx.lineTo(xOf(maxMp), yOf(Math.min(ms1, maxMs * 1.5)));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    for (let i = 0; i < refPts.length; i++) {
      const p = refPts[i];
      ctx.fillStyle = '#565f89';
      ctx.beginPath();
      ctx.arc(xOf(p.mp), yOf(p.ms), 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#a9b1d6';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    if (hasCurrent && fit) {
      ctx.fillStyle = '#f7768e';
      ctx.beginPath();
      ctx.moveTo(xOf(curMp), yOf(curEstMs) - 7);
      ctx.lineTo(xOf(curMp) + 7, yOf(curEstMs));
      ctx.lineTo(xOf(curMp), yOf(curEstMs) + 7);
      ctx.lineTo(xOf(curMp) - 7, yOf(curEstMs));
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = '#9ece6a';
      ctx.beginPath();
      ctx.arc(xOf(curMp), yOf(curParMs), 7, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = '#c0caf5';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Megapixels', pad.l + iw / 2, H - 18);
    ctx.save();
    ctx.translate(16, pad.t + ih / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Serial wall time (ms)', 0, 0);
    ctx.restore();
    ctx.fillStyle = '#565f89';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('◇ est. serial  ● parallel', pad.l, 14);
  }

  function renderBenchmarkTable(wrap, runs) {
    wrap.innerHTML = '';
    const table = document.createElement('table');
    table.className = 'serial-benchmark-table';
    table.innerHTML = '<thead><tr><th>Label</th><th>Resolution</th><th>Pixels</th><th>Serial total</th></tr></thead>';
    const tb = document.createElement('tbody');
    runs.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + r.label + '</td><td>' + r.width + ' × ' + r.height + '</td><td>' +
        r.pixels.toLocaleString() + '</td><td>' + formatDurationMs(r.total_ms) + '</td>';
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    wrap.appendChild(table);
  }

  async function renderSerialCompareSection() {
    const narrative = $('serial-compare-narrative');
    const tableWrap = $('serial-benchmark-table-wrap');
    const cBar = $('chart-compare-bars');
    const cScat = $('chart-compare-scatter');
    const section = $('serial-compare-section');
    if (!narrative || !tableWrap || !cBar || !cScat || !section) return;
    if (!metrics.length) {
      section.classList.add('hidden');
      return;
    }
    section.classList.remove('hidden');
    const parallelSec = Math.max(...metrics.map(m => m.total_s));
    const parallelMs = parallelSec * 1000;
    const bench = await ensureSerialBenchmarks();
    if (!bench || !bench.runs || !bench.runs.length) {
      narrative.innerHTML = '<p class="serial-compare-warn">Could not load <code>serial_benchmarks.json</code>. Place it next to <code>visualization.html</code>.</p>';
      tableWrap.innerHTML = '';
      const ectx = setupCompareCanvas(cBar, 640, 200);
      ectx.fillStyle = '#1a1b26';
      ectx.fillRect(0, 0, 640, 200);
      const ectx2 = setupCompareCanvas(cScat, 640, 320);
      ectx2.fillStyle = '#1a1b26';
      ectx2.fillRect(0, 0, 640, 320);
      return;
    }
    const runs = bench.runs;
    const fit = linearFitPixelsToMs(runs);
    renderBenchmarkTable(tableWrap, runs);
    const pixels = imageWidth > 0 && imageHeight > 0 ? imageWidth * imageHeight : 0;
    const mp = pixels / 1e6;
    const hasDim = pixels > 0;
    const estSerialMs = hasDim && fit ? fit.predictMs(pixels) : null;
    const hasEst = hasDim && fit && estSerialMs != null;
    const speedup = hasEst && parallelMs > 0 ? estSerialMs / parallelMs : null;

    let html = '<div class="serial-narrative-box"><h3 class="serial-inline-h">What this means</h3><ul class="serial-narrative-list">';
    html += '<li><strong>Parallel run</strong> (this page): <strong>' + parallelSec.toFixed(3) + ' s</strong> wall time using <strong>' + metrics.length + '</strong> MPI ranks (slowest rank).</li>';
    if (hasEst) {
      html += '<li><strong>Your image</strong>: <strong>' + imageWidth + ' × ' + imageHeight + '</strong> (' + mp.toFixed(2) + ' MP, ' + pixels.toLocaleString() + ' pixels).</li>';
      html += '<li><strong>Estimated serial time</strong> (linear fit on ' + runs.length + ' benchmark runs): <strong>' + formatDurationMs(estSerialMs) + '</strong> (~' + (estSerialMs / 1000).toFixed(1) + ' s).</li>';
      if (speedup != null && speedup >= 1) {
        html += '<li><strong>Approx. speedup</strong> vs that serial estimate: <strong>' + speedup.toFixed(1) + '×</strong> faster with this parallel run.</li>';
      } else if (speedup != null) {
        html += '<li>Estimated serial &lt; parallel (noise, different hardware, or fit out of range) — treat ratio as qualitative.</li>';
      }
      html += '<li>Large images dominate preprocessing cost; doing this <strong>before</strong> an LLM avoids slow serial bottlenecks and keeps latency predictable when you scale resolution.</li>';
    } else {
      html += '<li>Load <code>threshold_parallel.ppm</code> (or upload a full run) so we know width × height for a serial-time estimate.</li>';
      html += '<li>Parallel wall time for this run: <strong>' + parallelSec.toFixed(3) + ' s</strong>.</li>';
    }
    html += '</ul></div>';
    if (fit) {
      html += '<p class="serial-fit-eq">Fit on benchmarks: <code>time_ms ≈ ' + fit.slope.toExponential(4) + ' × pixels + ' +
        fit.intercept.toFixed(1) + '</code></p>';
    }
    narrative.innerHTML = html;

    drawCompareBars(cBar, hasEst ? estSerialMs : 0, parallelMs, hasEst);
    drawCompareScatter(cScat, runs, fit, mp, hasEst ? estSerialMs : 0, parallelMs, hasEst);
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
          const files = await collectRunFolderFiles(dir, '');
          await loadFromFiles(files);
        } catch (e) {
          if (e.name !== 'AbortError') setStatus('Error: ' + e.message, true);
        }
      });
    }

    const imageFileInput = $('image-file-input');
    const chooseImageBtn = $('choose-image-btn');
    const uploadImageBtn = $('upload-image-btn');
    const imageFileLabel = $('image-file-label');

    function syncUploadButtonState() {
      const has = imageFileInput && imageFileInput.files && imageFileInput.files.length > 0;
      if (uploadImageBtn) uploadImageBtn.disabled = !has;
    }

    if (chooseImageBtn && imageFileInput) {
      chooseImageBtn.addEventListener('click', () => imageFileInput.click());
    }
    if (imageFileInput) {
      imageFileInput.addEventListener('change', () => {
        const f = imageFileInput.files && imageFileInput.files[0];
        if (imageFileLabel) imageFileLabel.textContent = f ? f.name : 'No file selected';
        syncUploadButtonState();
        if (f) setUploadStatus('');
      });
    }
    if (uploadImageBtn && imageFileInput) {
      uploadImageBtn.disabled = true;
      uploadImageBtn.addEventListener('click', () => {
        const f = imageFileInput.files && imageFileInput.files[0];
        if (f) uploadImageFile(f);
      });
    }

    const rankPrev = $('rank-prev');
    const rankNext = $('rank-next');
    if (rankPrev) {
      rankPrev.addEventListener('click', async () => {
        if (!metrics.length) return;
        pipelineReplay.rankIdx = Math.max(0, pipelineReplay.rankIdx - 1);
        await refreshIntermediateGrid();
      });
    }
    if (rankNext) {
      rankNext.addEventListener('click', async () => {
        if (!metrics.length) return;
        pipelineReplay.rankIdx = Math.min(metrics.length - 1, pipelineReplay.rankIdx + 1);
        await refreshIntermediateGrid();
      });
    }

    const lbClose = $('ppm-lightbox-close');
    const lbBack = $('ppm-lightbox-backdrop');
    if (lbClose) lbClose.addEventListener('click', closePpmLightbox);
    if (lbBack) lbBack.addEventListener('click', closePpmLightbox);
    document.addEventListener('keydown', (e) => {
      const lb = $('ppm-lightbox');
      if (e.key === 'Escape' && lb && !lb.classList.contains('hidden')) closePpmLightbox();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
