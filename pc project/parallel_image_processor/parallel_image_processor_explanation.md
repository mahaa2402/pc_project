## MPI Parallel Image Processor – Detailed Explanation

This document explains in detail what the `parallel_image_processor` program does, how it is structured, how the image is decomposed across MPI processes, and how the results and timings are produced.

The main file is `parallel_image_processor/main.cpp`, and it reuses the image utilities in `image_utils.hpp` / `image_utils.cpp`.

---

## High‑level goal

The program implements a **parallel image processing pipeline** for PPM (P6) images using **MPI**. It performs the following stages:

1. Load a color RGB image from disk (rank 0).
2. Split the image by **rows** across multiple MPI processes.
3. On each process, run a sequence of operations on its part of the image:
   - Convert to grayscale.
   - Apply an 11x11 Gaussian blur (noise reduction) with **halo exchange**.
   - Apply Sobel edge detection (3x3 kernels) with **halo exchange**.
   - Apply a binary threshold to get a black‑and‑white edge map.
4. Gather the processed rows back on rank 0 and save a **single final output image**.
5. Record **timing metrics** per process in CSV files for performance analysis.

All final results are written **inside `parallel_image_processor/<image_stem>/`**, _not_ in `serial_image_processor/`.

---

## Image utilities (`image_utils.*`)

These utilities implement the core image operations and are shared between the serial and parallel versions:

- `Image` struct:
  - Holds `width`, `height`, `max_val`, and a `pixels` vector.
  - Pixels are in interleaved RGB order: `R,G,B,R,G,B,...`.

- `PPMHandler::loadPPM` / `savePPM`:
  - Load and save P6 (binary) PPM images.
  - Robust header parsing, handles `#` comments and flexible whitespace.

- `convertToGrayscale`:
  - Converts RGB to grayscale using luminosity:
    \[
      Y = 0.299R + 0.587G + 0.114B
    \]
  - Still stores 3 identical channels per pixel (R=G=B=Y) for simplicity.

- `applyFilter`:
  - Generic N×N convolution (odd‑sized kernels like 3×3 or 11×11).
  - Clamps results to \[0, 255\].
  - Borders are left as **zero** (implicit black padding).

- `applyGaussianBlur`:
  - Uses a prebuilt **11×11 Gaussian kernel** (`makeGaussian11x11`, sigma ≈ 2).
  - Heavy smoothing – substantially more work per pixel than a 3×3 filter.

- `applySobel`:
  - Uses standard 3×3 Sobel kernels in X and Y to compute gradient magnitude.
  - Produces a grayscale edge‑intensity image.

- `applyThreshold`:
  - Turns the image into pure black/white:
    - If pixel value > threshold (50 by default) → 255 (white).
    - Else → 0 (black).

The **parallel code** in `main.cpp` calls these utilities on local chunks of the image.

---

## Data distribution and decomposition

### 1. MPI initialization and argument parsing

In `main`:

- `MPI_Init` starts MPI.
- `MPI_Comm_size` and `MPI_Comm_rank` get:
  - `world_size` = total number of processes.
  - `rank` = ID of the current process (0, 1, ..., `world_size - 1`).
- `t_global_start = MPI_Wtime()` records a global start time.

The program expects:

```bash
./parallel_ppm ../images_ppm/<input.ppm>
```

If no argument is given, rank 0 prints a usage message and the program exits.

### 2. Output directory selection

The input path (e.g. `../images_ppm/jack-mccracken-5NcG8CjL7Sc-unsplash.ppm`) is used to compute:

- `image_stem` = filename without extension (e.g. `jack-mccracken-5NcG8CjL7Sc-unsplash`).
- `output_dir = "./" / image_stem`, relative to `parallel_image_processor/`.

Rank 0:

- calls `std::filesystem::create_directories(output_dir)` to create
  `parallel_image_processor/<image_stem>/`.

All ranks then `MPI_Barrier` to ensure the directory is ready before writing results.

### 3. Loading the full image on rank 0

Only **rank 0** loads the full PPM:

- `full_image = PPMHandler::loadPPM(input_path);`
- `width`, `height`, `max_val` are filled from the loaded image.

These dimensions are then **broadcast** to all ranks:

- `MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);`
- `MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);`
- `MPI_Bcast(&max_val, 1, MPI_INT, 0, MPI_COMM_WORLD);`

If dimensions are invalid, rank 0 prints an error and the program exits.

### 4. Row‑based decomposition (`compute_decomposition`)

To spread work across ranks, the image is split by **rows**:

- `width` and `height` define the global image size.
- Each pixel row is `row_bytes = width * 3` bytes (RGB).

The function `compute_decomposition(width, height, world_size, rank)`:

- Computes:
  - `base_rows = height / world_size`
  - `remainder = height % world_size`
- Assigns:
  - `rows_r = base_rows + 1` for the first `remainder` ranks.
  - `rows_r = base_rows` for the others.
- Fills:
  - `counts_bytes[r] = rows_r * row_bytes`
  - `displs_bytes[r]` = byte offset of rank r’s first row in the global buffer.
- Sets:
  - `local_rows` for the **current** rank.

As a result:

- The image is partitioned into `world_size` contiguous blocks of rows.
- The blocks are as equal as possible (difference ≤ 1 row between ranks).
- `global_row_start = displs_bytes[rank] / row_bytes`
  and `global_row_end = global_row_start + local_rows` describe each rank’s global span.

If `local_rows == 0` (too many processes for the number of rows), the program exits with an error.

---

## Data movement: scatter and gather

### 1. Scatter (sending RGB rows from rank 0 to all ranks)

Rank 0 owns `full_image.pixels` (size = `width * height * 3` bytes). All ranks allocate:

- `std::vector<unsigned char> local_rgb(local_rows * row_bytes);`

Then:

- `MPI_Scatterv` is used:
  - Send buffer on rank 0: `full_image.pixels.data()`.
  - Counts per rank: `decomp.counts_bytes`.
  - Displacements per rank: `decomp.displs_bytes`.
  - Receive buffer on each rank: `local_rgb.data()`.

Result:

- Each rank r receives exactly its block of RGB rows.
- **No rank other than 0 stores the full image.**

The code measures:

- `t_scatter_start` / `t_scatter_end` to compute `scatter_time`.

### 2. Gather (collecting final rows back to rank 0)

After processing, each rank has a local thresholded image `stencil_local` with:

- `width` columns.
- `local_rows` rows.

Rank 0 allocates `final_image.pixels` of size `width * height * 3`, and then:

- Uses `MPI_Gatherv` with:
  - Send buffer on each rank: `stencil_local.pixels.data()`.
  - Receive buffer on rank 0: `final_image.pixels.data()`.
  - Same `counts_bytes` and `displs_bytes` as used in the scatter.

Result:

- Rank 0’s `final_image` contains all thresholded rows stacked back in the correct order.

The gather phase is timed:

- `t_gather_start` / `t_gather_end` → `gather_time`.

---

## Local processing pipeline on each rank

After scattering, each rank works independently on its own rows.

### 1. Grayscale conversion

Each rank builds:

- `Image local_color;`
  - `width` = global width.
  - `height` = `local_rows`.
  - `pixels` = moved from `local_rgb`.

Then calls:

- `Image local_gray = convertToGrayscale(local_color);`

This stage is timed:

- `t_gray_start` / `t_gray_end` → `gray_time`.

### 2. Gaussian blur with halo exchange

The Gaussian blur uses an **11×11 kernel**, radius `gauss_radius = 5`. To correctly blur rows at block boundaries, each rank must know some neighbor rows from adjacent ranks.

Approach:

1. Allocate `gray_with_halo` with:
   - `width` = global width.
   - `height` = `local_rows + 2 * gauss_radius`.
   - Central region `[gauss_radius, gauss_radius + local_rows)` is for local rows.
   - Top and bottom `gauss_radius` rows are **halo** regions.

2. Copy local grayscale rows into the central region:
   - For each local row r:
     - Copy row `r` from `local_gray` into row `r + gauss_radius` of `gray_with_halo`.

3. Call `exchange_halo_rows(gray_with_halo, local_rows, gauss_radius, rank, world_size, row_bytes)`:
   - `up_rank = rank - 1` (or `MPI_PROC_NULL` for rank 0).
   - `down_rank = rank + 1` (or `MPI_PROC_NULL` for last rank).
   - For radius > 0 and `local_rows > 0`:
     - Send the **first `gauss_radius` owned rows** to `up_rank`, receive into top halo.
     - Send the **last `gauss_radius` owned rows** to `down_rank`, receive into bottom halo.
   - Border ranks use `MPI_PROC_NULL` so halos remain zeros, matching the serial version’s zero padding at image borders.

4. Run the blur:
   - `Image blurred_with_halo = applyGaussianBlur(gray_with_halo);`
   - Because the convolution ignores outer borders (based on kernel size), the halos ensure that interior pixels near rank boundaries see the correct neighbor data.

5. Extract blurred local rows:
   - Create `blurred_local` with `height = local_rows`.
   - Copy rows `[gauss_radius, gauss_radius + local_rows)` from `blurred_with_halo` into `blurred_local`.

This entire Gaussian stage is timed:

- `t_blur_start` / `t_blur_end` → `blur_time`.

### 3. Sobel edge detection with halo exchange

The Sobel operator uses 3×3 kernels, so radius `sobel_radius = 1`.

Steps (similar to Gaussian):

1. Allocate `blur_with_halo` with height `local_rows + 2 * sobel_radius`.
2. Copy blurred local rows into central rows `[sobel_radius, sobel_radius + local_rows)`.
3. Call `exchange_halo_rows(blur_with_halo, local_rows, sobel_radius, rank, world_size, row_bytes)`.
4. Apply Sobel:
   - `Image edges_with_halo = applySobel(blur_with_halo);`
5. Extract local edges:
   - Allocate `edges_local` with `height = local_rows`.
   - Copy rows `[sobel_radius, sobel_radius + local_rows)` from `edges_with_halo` into `edges_local`.

Timing:

- `t_sobel_start` / `t_sobel_end` → `sobel_time`.

### 4. Thresholding

Thresholding is purely **local**, no halo or communication is needed:

- `constexpr unsigned char EDGE_THRESHOLD = 50;`
- `Image stencil_local = applyThreshold(edges_local, EDGE_THRESHOLD);`

Timing:

- `t_thresh_start` / `t_thresh_end` → `thresh_time`.

At this point, each rank has its final black‑and‑white edge map for its rows.

---

## Output files and directory layout

All output paths are based on the input image name.

### 1. Final thresholded image

After `MPI_Gatherv`, rank 0 has the full `final_image`. It writes:

- `output_dir / "threshold_parallel.ppm"`

Example:

- Input path: `../images_ppm/jack-mccracken-5NcG8CjL7Sc-unsplash.ppm`
- Output directory (relative to `parallel_image_processor/`):
  - `jack-mccracken-5NcG8CjL7Sc-unsplash/`
- Final image:
  - `jack-mccracken-5NcG8CjL7Sc-unsplash/threshold_parallel.ppm`

### 2. Per‑rank metrics CSV

Every rank writes a CSV file:

- `metrics_rank<rank>.csv` inside the **same** `output_dir`.

Each CSV file has header:

```text
rank,rows_start,rows_end,rows_local,
scatter_s,gray_s,blur_s,sobel_s,threshold_s,gather_s,total_s
```

And one data row containing:

- `rank`: MPI rank ID.
- `rows_start`, `rows_end`: global row range \[start, end) handled by this rank.
- `rows_local`: number of rows on this rank.
- Timings in seconds for each stage:
  - `scatter_s`   – time for `MPI_Scatterv`.
  - `gray_s`      – local grayscale conversion.
  - `blur_s`      – Gaussian blur + halo exchange.
  - `sobel_s`     – Sobel edge detection + halo exchange.
  - `threshold_s` – thresholding.
  - `gather_s`    – `MPI_Gatherv`.
  - `total_s`     – `t_global_end - t_global_start`, full pipeline time.

These CSV files can be opened in Excel, Python, etc., for performance analysis.

---

## Console output and global timing summary

### 1. Per‑rank detailed logs

To make console output readable, the code loops over ranks 0..`world_size-1` and has each rank print in turn, separated by `MPI_Barrier`.

For each rank, the output includes:

- Row ownership:

```text
[rank r] rows [global_row_start,global_row_end) local_rows=...
```

- Per‑stage timings:

```text
  scatter_s   = ...
  grayscale_s = ...
  blur_s      = ...
  sobel_s     = ...
  threshold_s = ...
  gather_s    = ...
  total_s     = ...
```

### 2. Global max timings (rank 0 summary)

For each timing metric, the program calls `MPI_Reduce` with `MPI_MAX` to find the **slowest** rank:

- `max_scatter`, `max_gray`, `max_blur`, `max_sobel`, `max_thresh`, `max_gather`, `max_total`.

Only rank 0 prints the summary:

- Number of processes.
- Image size.
- Threshold value.
- Output image path.
- Max timing per stage across all ranks.

This provides a clear picture of the **bottleneck** stages and total parallel runtime.

---

## How everything fits together

1. **Rank 0** loads the full PPM image and broadcasts dimensions.
2. All ranks compute how many rows they own and where those rows lie globally.
3. **Scatter**: rank 0 distributes RGB rows to all ranks using `MPI_Scatterv`.
4. Each rank:
   - Converts its rows to grayscale.
   - Builds halo‑padded images for Gaussian blur and Sobel.
   - Uses `exchange_halo_rows` to get boundary data from neighbors.
   - Runs `applyGaussianBlur`, `applySobel`, and `applyThreshold` locally.
5. **Gather**: thresholded local rows return to rank 0 via `MPI_Gatherv`.
6. **Rank 0**:
   - Writes `threshold_parallel.ppm` into `parallel_image_processor/<image_stem>/`.
7. **All ranks**:
   - Write their own `metrics_rank<r>.csv` into the same directory.
8. Program prints per‑rank timing logs and a global timing summary before finalizing MPI.

This design ensures:

- The heavy computation (blur + Sobel + threshold) scales with the number of MPI ranks.
- Communication is limited to:
  - One scatter and one gather.
  - Two halo exchanges per convolution stage (Gaussian and Sobel).
- All results and metrics are contained neatly within the `parallel_image_processor` folder structure.

