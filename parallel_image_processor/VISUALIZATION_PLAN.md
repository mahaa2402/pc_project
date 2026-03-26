# Plan: Parallelism Visualization (Web UI)

**Goal:** Let the teacher (and you) **see parallelism visually**: which rows went to which rank, how each rank’s chunk is transformed at each stage (blur, Sobel, etc.), and how the logged metrics/benchmarks look — all in one small web page.

**Scope:** Only code under `parallel_image_processor/`. The existing `main.cpp` already uses `../image_utils.hpp`; we keep that and add new files here.

---

## What You Already Have (No Code Changes)

- **Per-rank metrics:** `metrics_rank0.csv`, `metrics_rank1.csv`, … in `<stem>/` with:
  - `rank`, `rows_start`, `rows_end`, `rows_local`
  - `scatter_s`, `gray_s`, `blur_s`, `sobel_s`, `threshold_s`, `gather_s`, `total_s`
- **Final image:** `<stem>/threshold_parallel.ppm` (full image after gather).
- **Pipeline:** Scatter → Grayscale → Blur (with halo) → Sobel (with halo) → Threshold → Gather.

The C++ does **not** currently write per-rank or per-stage intermediate images (grayscale, blur, sobel); only the final threshold image and CSVs are written.

---

## Plan in Two Phases

### Phase 1: Web viewer using existing outputs only

Build a **single-page web app** (e.g. `visualization.html` + optional `visualization.js` / `visualization.css`) inside `parallel_image_processor/` that:

1. **Loads data**
   - Reads all `metrics_rank*.csv` from a chosen run (e.g. from `<stem>/`).
   - Loads the final image `threshold_parallel.ppm` from the same folder (or a copy placed next to the HTML for demo).  
   - **How to choose folder:** Either (A) user runs a tiny static server (e.g. `python3 -m http.server` from `parallel_image_processor/<stem>/`) and opens `../visualization.html?stem=jack-mccracken-...` so the page can fetch `metrics_rank*.csv` and `threshold_parallel.ppm` from that stem, or (B) use the File System Access API / file inputs so the user picks the folder and the page reads CSVs + PPM from it (works with `file://` or local server).

2. **Visualizations**
   - **Decomposition / “who got which rows”**
     - Show the final image (or a placeholder if no image) with **horizontal bands** and labels: e.g. “Rank 0: rows 0–942”, “Rank 1: rows 942–1883”, etc., using `rows_start` and `rows_end` from the CSVs. This makes “the part that was sent to that processor” visible at a glance.
   - **Per-stage timings (benchmarks)**
     - For each rank, show  a **stacked bar** or **timeline** of the 7 stages (scatter, gray, blur, sobel, threshold, gather). Optionally show **total_s** per rank. This shows parallelism (all ranks do the same stages) and load balance (e.g. rank 0’s scatter is heavier).
   - **Summary metrics**
     - Table or cards: e.g. max time per stage across ranks, total time, number of ranks, image size. All from the existing CSV columns.

3. **Tech**
   - Pure HTML + JS (and optional CSS). No backend; data = existing files in `<stem>/`.
   - PPM P6: either decode in JavaScript (read binary, parse P6 header, draw into `<canvas>`) or add a small optional step that converts PPM → PNG in C++ and the page loads PNG. For “directory only” and minimal C++ change, decoding PPM in JS is enough.

**Deliverable:** Open the page → select or point to a run’s folder (e.g. `jack-mccracken-5NcG8CjL7Sc-unsplash/`) → see rank bands on the image + timing bars + summary. No change to `main.cpp` required.

---x

### Phase 2 (Optional): Per-rank, per-stage image chunks for “live” transformation view

To show **how the image has been transformed** at each step **for the part that was sent to each processor**, we need per-rank, per-stage images. Right now only the final stencil is saved.

1. **C++ changes (only in `parallel_image_processor/main.cpp`)**
   - Add an **optional mode** (e.g. environment variable `DUMP_STAGES=1` or command-line flag `--dump-stages`) so that when set:
     - After each stage (after grayscale, after blur, after sobel, after threshold), each rank writes its **local chunk** (the rows it owns) to a file in `<stem>/`, e.g.:
       - `stage_grayscale_rank0.ppm`, `stage_grayscale_rank1.ppm`, …
       - `stage_blur_rank0.ppm`, …
       - `stage_sobel_rank0.ppm`, …
       - `stage_threshold_rank0.ppm`, … (or reuse final image by row range)
     - Each file is a small PPM (width × local_rows only). Use existing `PPMHandler::savePPM` on an `Image` that holds only that rank’s rows.
   - No change to the actual pipeline logic; only extra I/O when the flag is set.

2. **Web viewer extension**
   - When per-rank, per-stage PPMs exist, the same page (or a second view) shows a **grid**:
     - **Rows = ranks**, **columns = stages** (Scatter input → Grayscale → Blur → Sobel → Threshold).
     - Each cell shows the image chunk for that rank at that stage (load the small PPM, draw on canvas). This gives the “live” feel: “this is what rank 2 had after blur.”
   - Still use the same `metrics_rank*.csv` for row boundaries and timings; the new PPMs only add the visual content per cell.

**Deliverable:** Run with `DUMP_STAGES=1` (or `--dump-stages`) → same folder gets `stage_*_rank*.ppm` → refresh the web page → see the grid of per-rank, per-stage images plus the existing decomposition and timing views.

---

## File Layout (all under `parallel_image_processor/`)

- `main.cpp` — existing; optionally add dump-stage logic (Phase 2).
- `visualization.html` — single entry page (Phase 1; extend in Phase 2).
- `visualization.js` — load CSVs, optional PPM decode, draw decomposition + timing bars + optional stage grid.
- `visualization.css` — layout and styling (optional).
- `VISUALIZATION_PLAN.md` — this plan.

Output directory (e.g. `jack-mccracken-5NcG8CjL7Sc-unsplash/`) continues to contain:

- `metrics_rank0.csv`, … (existing)
- `threshold_parallel.ppm` (existing)
- Optionally: `stage_grayscale_rank0.ppm`, … (Phase 2 only when dump enabled).

---

## How the teacher “sees” parallelism

1. **Decomposition view:** One image with bands labeled by rank and row range → “this is the part sent to each processor.”
2. **Timeline / bar chart:** One bar per rank, segments = stages → “all ranks do the same stages in parallel” and “we can see who was slowest in which stage.”
3. **Stage grid (Phase 2):** One small image per (rank, stage) → “this is how that processor’s chunk looked after blur / sobel / threshold.”

All of this uses only the `parallel_image_processor` directory plus the existing `../image_utils` for PPM and image ops. No new languages or runtimes beyond a browser and (if you want to avoid CORS) a trivial static HTTP server.

---

## Next step

Start with **Phase 1**: implement `visualization.html` (and JS/CSS) that:

- Reads all `metrics_rank*.csv` from a given run folder.
- Loads and displays `threshold_parallel.ppm` with rank bands overlaid.
- Draws per-rank, per-stage timing bars and a short summary table.

Then we can add Phase 2 (optional dump + grid) when you want the per-rank, per-stage image view.
