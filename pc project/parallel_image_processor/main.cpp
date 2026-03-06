/**
 * MPI Parallel PPM Image Processor (row-based decomposition)
 *
 * Row decomposition & halo exchange:
 * - Rank 0 loads the full PPM (P6) image from ../images_ppm using PPMHandler.
 * - The image is split by rows and distributed across ranks with MPI_Scatterv.
 *   Each rank r owns a contiguous block of rows [row_start_r, row_end_r).
 *
 * - Convolution kernels (11x11 Gaussian, 3x3 Sobel) need neighbour rows. For
 *   each convolution stage we build a local Image that contains:
 *      [ halo_top | owned_rows | halo_bottom ]
 *   where halo_top and halo_bottom have radius R rows (R=5 for Gaussian,
 *   R=1 for Sobel). Neighbouring ranks exchange those boundary rows using
 *   MPI_Sendrecv:
 *      - send top    R owned rows to rank-1, receive into halo_top
 *      - send bottom R owned rows to rank+1, receive into halo_bottom
 *   Ranks at the global top/bottom use MPI_PROC_NULL so their outer halos
 *   stay zero, matching the serial implementation's implicit zero padding
 *   where border pixels are left as 0.
 *
 * Per-rank pipeline:
 *   1. Receive its subset of RGB rows via MPI_Scatterv.
 *   2. Convert those rows to grayscale with convertToGrayscale.
 *   3. Build grayscale+halo image, exchange halos, apply 11x11 Gaussian.
 *   4. Extract blurred local rows, build blur+halo image, exchange, apply Sobel.
 *   5. Apply binary threshold locally.
 *   6. Gather final thresholded rows back to rank 0 with MPI_Gatherv.
 *
 * Output layout:
 *   - For an input ../images_ppm/foo.ppm, all results live under:
 *         parallel_image_processor/foo/
 *   - Rank 0 writes the full final threshold image:
 *         foo/threshold_parallel.ppm
 *   - Each rank r writes a metrics CSV:
 *         foo/metrics_rank<r>.csv
 *     containing per-stage times and the row range handled by that rank.
 *
 * Timing:
 *   - MPI_Wtime() is used to measure:
 *       scatter, grayscale, blur, sobel, threshold, gather, total
 *   - Each rank prints its own timings and the global row range it owns.
 *   - Rank 0 also prints the maximum time over all ranks for each stage.
 *
 * Usage (run from parallel_image_processor/):
 *   mpicxx -std=c++17 main.cpp ../image_utils.cpp -O2 -o parallel_ppm
 *   mpirun -np 4 ./parallel_ppm ../images_ppm/jack-mccracken-5NcG8CjL7Sc-unsplash.ppm
 */

#include "../image_utils.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct DecompInfo {
    int width{0};
    int height{0};
    int max_val{255};
    int local_rows{0};
    int row_bytes{0};
    std::vector<int> counts_bytes;   // send/recv counts in bytes per rank
    std::vector<int> displs_bytes;   // displacements in bytes per rank
};

DecompInfo compute_decomposition(int global_width, int global_height,
                                 int world_size, int rank) {
    DecompInfo info;
    info.width = global_width;
    info.height = global_height;
    info.max_val = 255;
    info.row_bytes = global_width * 3;
    info.counts_bytes.resize(world_size);
    info.displs_bytes.resize(world_size);

    const int base_rows = global_height / world_size;
    const int remainder = global_height % world_size;

    int offset_bytes = 0;
    for (int r = 0; r < world_size; ++r) {
        const int rows_r = base_rows + (r < remainder ? 1 : 0);
        const int bytes_r = rows_r * info.row_bytes;
        info.counts_bytes[r] = bytes_r;
        info.displs_bytes[r] = offset_bytes;
        offset_bytes += bytes_r;
        if (r == rank) {
            info.local_rows = rows_r;
        }
    }

    return info;
}

/**
 * Exchange halo rows with neighbours for a grayscale (or single-channel) image.
 *
 * Layout:
 *   - height = local_rows + 2*R
 *   - rows [R, R + local_rows) are this rank's owned rows
 *   - rows [0, R) and [R + local_rows, R + local_rows + R) are halo rows
 *
 * Uses MPI_Sendrecv to:
 *   - send top    R owned rows to rank-1, receive into top halo
 *   - send bottom R owned rows to rank+1, receive into bottom halo
 */
void exchange_halo_rows(Image& img_with_halo,
                        int local_rows,
                        int radius,
                        int rank,
                        int world_size,
                        int row_bytes) {
    const int up_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int down_rank = (rank < world_size - 1) ? rank + 1 : MPI_PROC_NULL;

    unsigned char* base = img_with_halo.pixels.data();

    if (radius > 0 && local_rows > 0) {
        // Top exchange: send first R owned rows, receive into top halo.
        MPI_Sendrecv(
            /* sendbuf   */ base + radius * row_bytes,
            /* sendcount */ radius * row_bytes,
            MPI_UNSIGNED_CHAR,
            up_rank,
            /* sendtag   */ 0,
            /* recvbuf   */ base,
            /* recvcount */ radius * row_bytes,
            MPI_UNSIGNED_CHAR,
            up_rank,
            /* recvtag   */ 1,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        // Bottom exchange: send last R owned rows, receive into bottom halo.
        MPI_Sendrecv(
            /* sendbuf   */ base + (radius + local_rows - radius) * row_bytes,
            /* sendcount */ radius * row_bytes,
            MPI_UNSIGNED_CHAR,
            down_rank,
            /* sendtag   */ 1,
            /* recvbuf   */ base + (radius + local_rows) * row_bytes,
            /* recvcount */ radius * row_bytes,
            MPI_UNSIGNED_CHAR,
            down_rank,
            /* recvtag   */ 0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const double t_global_start = MPI_Wtime();

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage (run from parallel_image_processor/):\n"
                      << "  mpirun -np <P> ./parallel_ppm ../images_ppm/<input.ppm>\n";
        }
        MPI_Finalize();
        return 1;
    }

    const std::string input_path = argv[1]; // e.g., ../images_ppm/foo.ppm

    // All ranks will agree on the output directory name based on the input stem.
    fs::path input_fs_path(input_path);
    const std::string image_stem = input_fs_path.stem().string();
    const fs::path    output_dir = fs::path(".") / image_stem;

    if (rank == 0) {
        try {
            fs::create_directories(output_dir);
        } catch (const std::exception& e) {
            std::cerr << "Failed to create output directory '" << output_dir.string()
                      << "': " << e.what() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    Image full_image;
    int width = 0;
    int height = 0;
    int max_val = 255;

    if (rank == 0) {
        try {
            full_image = PPMHandler::loadPPM(input_path);
            width = full_image.width;
            height = full_image.height;
            max_val = full_image.max_val;
        } catch (const std::exception& e) {
            std::cerr << "Error loading input image: " << e.what() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast image dimensions so every rank can compute its row slice.
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_val, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (width <= 0 || height <= 0) {
        if (rank == 0) {
            std::cerr << "Invalid image dimensions broadcast: " << width << "x" << height << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    const DecompInfo decomp = compute_decomposition(width, height, world_size, rank);
    const int local_rows = decomp.local_rows;
    const int row_bytes = decomp.row_bytes;

    if (local_rows == 0) {
        if (rank == 0) {
            std::cerr << "World size is larger than number of rows; some ranks have 0 rows.\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Determine global row range handled by this rank for logging.
    const int global_row_start = decomp.displs_bytes[rank] / row_bytes;
    const int global_row_end   = global_row_start + local_rows; // exclusive

    std::vector<unsigned char> local_rgb(static_cast<std::size_t>(local_rows) * row_bytes);

    const double t_scatter_start = MPI_Wtime();

    MPI_Scatterv(
        rank == 0 ? full_image.pixels.data() : nullptr,
        decomp.counts_bytes.data(),
        decomp.displs_bytes.data(),
        MPI_UNSIGNED_CHAR,
        local_rgb.data(),
        static_cast<int>(local_rgb.size()),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD);

    const double t_scatter_end = MPI_Wtime();

    // Local grayscale conversion (no halo needed for this step).
    const double t_gray_start = MPI_Wtime();

    Image local_color;
    local_color.width = width;
    local_color.height = local_rows;
    local_color.max_val = max_val;
    local_color.pixels = std::move(local_rgb);

    Image local_gray = convertToGrayscale(local_color);

    const double t_gray_end = MPI_Wtime();

    // Gaussian blur with 11x11 kernel (radius 5) using halo exchange.
    const int gauss_radius = 5;

    const double t_blur_start = MPI_Wtime();

    Image gray_with_halo;
    gray_with_halo.width = width;
    gray_with_halo.height = local_rows + 2 * gauss_radius;
    gray_with_halo.max_val = local_gray.max_val;
    gray_with_halo.pixels.assign(
        static_cast<std::size_t>(gray_with_halo.width) * gray_with_halo.height * 3,
        0);

    // Copy owned grayscale rows into central region.
    for (int r = 0; r < local_rows; ++r) {
        std::memcpy(
            &gray_with_halo.pixels[static_cast<std::size_t>(r + gauss_radius) * row_bytes],
            &local_gray.pixels[static_cast<std::size_t>(r) * row_bytes],
            static_cast<std::size_t>(row_bytes));
    }

    exchange_halo_rows(gray_with_halo, local_rows, gauss_radius, rank, world_size, row_bytes);

    Image blurred_with_halo = applyGaussianBlur(gray_with_halo);

    // Extract blurred rows that correspond to the owned rows (strip halos).
    Image blurred_local;
    blurred_local.width = width;
    blurred_local.height = local_rows;
    blurred_local.max_val = blurred_with_halo.max_val;
    blurred_local.pixels.resize(static_cast<std::size_t>(local_rows) * row_bytes);

    for (int r = 0; r < local_rows; ++r) {
        std::memcpy(
            &blurred_local.pixels[static_cast<std::size_t>(r) * row_bytes],
            &blurred_with_halo.pixels[static_cast<std::size_t>(r + gauss_radius) * row_bytes],
            static_cast<std::size_t>(row_bytes));
    }

    const double t_blur_end = MPI_Wtime();

    // Sobel edge detection (3x3 kernel, radius 1) with halo exchange.
    const int sobel_radius = 1;

    const double t_sobel_start = MPI_Wtime();

    Image blur_with_halo;
    blur_with_halo.width = width;
    blur_with_halo.height = local_rows + 2 * sobel_radius;
    blur_with_halo.max_val = blurred_local.max_val;
    blur_with_halo.pixels.assign(
        static_cast<std::size_t>(blur_with_halo.width) * blur_with_halo.height * 3,
        0);

    for (int r = 0; r < local_rows; ++r) {
        std::memcpy(
            &blur_with_halo.pixels[static_cast<std::size_t>(r + sobel_radius) * row_bytes],
            &blurred_local.pixels[static_cast<std::size_t>(r) * row_bytes],
            static_cast<std::size_t>(row_bytes));
    }

    exchange_halo_rows(blur_with_halo, local_rows, sobel_radius, rank, world_size, row_bytes);

    Image edges_with_halo = applySobel(blur_with_halo);

    Image edges_local;
    edges_local.width = width;
    edges_local.height = local_rows;
    edges_local.max_val = edges_with_halo.max_val;
    edges_local.pixels.resize(static_cast<std::size_t>(local_rows) * row_bytes);

    for (int r = 0; r < local_rows; ++r) {
        std::memcpy(
            &edges_local.pixels[static_cast<std::size_t>(r) * row_bytes],
            &edges_with_halo.pixels[static_cast<std::size_t>(r + sobel_radius) * row_bytes],
            static_cast<std::size_t>(row_bytes));
    }

    const double t_sobel_end = MPI_Wtime();

    // Thresholding is purely local, no halo exchange required.
    const double t_thresh_start = MPI_Wtime();

    constexpr unsigned char EDGE_THRESHOLD = 50;
    Image stencil_local = applyThreshold(edges_local, EDGE_THRESHOLD);

    const double t_thresh_end = MPI_Wtime();

    // Gather all thresholded rows back to rank 0.
    const double t_gather_start = MPI_Wtime();

    Image final_image;
    if (rank == 0) {
        final_image.width = width;
        final_image.height = height;
        final_image.max_val = 255;
        final_image.pixels.resize(static_cast<std::size_t>(width) * height * 3);
    }

    MPI_Gatherv(
        stencil_local.pixels.data(),
        static_cast<int>(stencil_local.pixels.size()),
        MPI_UNSIGNED_CHAR,
        rank == 0 ? final_image.pixels.data() : nullptr,
        decomp.counts_bytes.data(),
        decomp.displs_bytes.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD);

    const double t_gather_end = MPI_Wtime();

    // Rank 0 saves the final image inside parallel_image_processor/<stem>/.
    if (rank == 0) {
        const fs::path output_image_path = output_dir / "threshold_parallel.ppm";
        try {
            PPMHandler::savePPM(final_image, output_image_path.string());
        } catch (const std::exception& e) {
            std::cerr << "Error saving output image: " << e.what() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    const double t_global_end = MPI_Wtime();

    // --- Timing per rank ---
    const double scatter_time = t_scatter_end - t_scatter_start;
    const double gray_time    = t_gray_end    - t_gray_start;
    const double blur_time    = t_blur_end    - t_blur_start;
    const double sobel_time   = t_sobel_end   - t_sobel_start;
    const double thresh_time  = t_thresh_end  - t_thresh_start;
    const double gather_time  = t_gather_end  - t_gather_start;
    const double total_time   = t_global_end  - t_global_start;

    // Write per-rank metrics CSV into the same output directory.
    {
        fs::path metrics_path = output_dir / ("metrics_rank" + std::to_string(rank) + ".csv");
        std::ofstream mf(metrics_path);
        if (mf) {
            mf << "rank,rows_start,rows_end,rows_local,"
               << "scatter_s,gray_s,blur_s,sobel_s,threshold_s,gather_s,total_s\n";
            mf << rank << ","
               << global_row_start << ","
               << global_row_end << ","
               << local_rows << ","
               << scatter_time << ","
               << gray_time << ","
               << blur_time << ","
               << sobel_time << ","
               << thresh_time << ","
               << gather_time << ","
               << total_time << "\n";
        }
    }

    // Print detailed per-rank timings to the terminal in rank order so output is readable.
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < world_size; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == rank) {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "[rank " << rank << "] rows [" << global_row_start
                      << "," << global_row_end << ")"
                      << " local_rows=" << local_rows << "\n";
            std::cout << "  scatter_s   = " << scatter_time << "\n";
            std::cout << "  grayscale_s = " << gray_time << "\n";
            std::cout << "  blur_s      = " << blur_time << "\n";
            std::cout << "  sobel_s     = " << sobel_time << "\n";
            std::cout << "  threshold_s = " << thresh_time << "\n";
            std::cout << "  gather_s    = " << gather_time << "\n";
            std::cout << "  total_s     = " << total_time << "\n\n";
        }
    }

    // --- Global max timings (rank 0 summary) ---
    double max_scatter = 0.0, max_gray = 0.0, max_blur = 0.0, max_sobel = 0.0;
    double max_thresh  = 0.0, max_gather = 0.0, max_total = 0.0;

    MPI_Reduce(&scatter_time, &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gray_time,    &max_gray,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&blur_time,    &max_blur,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sobel_time,   &max_sobel,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&thresh_time,  &max_thresh,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gather_time,  &max_gather,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time,   &max_total,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "=== MPI parallel PPM pipeline summary ===\n";
        std::cout << "  processes     : " << world_size << "\n";
        std::cout << "  image         : " << width << "x" << height << "\n";
        std::cout << "  threshold     : " << static_cast<int>(EDGE_THRESHOLD) << "\n";
        std::cout << "  output image  : " << (output_dir / "threshold_parallel.ppm").string() << "\n\n";

        std::cout << "Max timings across ranks (seconds):\n";
        std::cout << "  scatter/load  : " << max_scatter << "\n";
        std::cout << "  grayscale     : " << max_gray << "\n";
        std::cout << "  gaussian blur : " << max_blur << "\n";
        std::cout << "  sobel         : " << max_sobel << "\n";
        std::cout << "  threshold     : " << max_thresh << "\n";
        std::cout << "  gather        : " << max_gather << "\n";
        std::cout << "  total         : " << max_total << "\n";
    }

    MPI_Finalize();
    return 0;
}

