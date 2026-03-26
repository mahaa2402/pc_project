#include "image_utils.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

constexpr unsigned char EDGE_THRESHOLD = 50;

const char* STEP_LOAD           = "load";
const char* STEP_GRAYSCALE      = "grayscale";
const char* STEP_SAVE_GRAY      = "save_grayscale";
const char* STEP_GAUSSIAN_BLUR  = "gaussian_blur";
const char* STEP_SAVE_BLUR      = "save_gaussian_blur";
const char* STEP_SOBEL          = "sobel";
const char* STEP_SAVE_SOBEL     = "save_sobel";
const char* STEP_THRESHOLD      = "threshold";
const char* STEP_SAVE_THRESHOLD = "save_threshold";

struct StepRecord {
    std::string step_name;
    Clock::time_point start;
    Clock::time_point end;
    double duration_ms{0};
    int width{0};
    int height{0};
    std::size_t pixel_count{0};
    std::string output_path;
};

struct ImageRecord {
    std::string input_path;
    std::string filename;  // basename for logging
    int index{0};
    std::string output_dir;
    std::vector<StepRecord> steps;
    double total_ms() const {
        return std::accumulate(steps.begin(), steps.end(), 0.0,
            [](double sum, const StepRecord& s) { return sum + s.duration_ms; });
    }
};

static std::string basename(const std::string& path) {
    return fs::path(path).filename().string();
}

static void log_step(const ImageRecord& rec, const StepRecord& step,
                     Clock::time_point run_start, std::ostream& out) {
    const double start_ms = Ms(step.start - run_start).count();
    const double end_ms   = Ms(step.end   - run_start).count();
    out << "  [img " << rec.index << "] " << rec.filename
        << " step=" << step.step_name
        << " start_ms=" << std::fixed << std::setprecision(2) << start_ms
        << " end_ms=" << end_ms
        << " duration_ms=" << step.duration_ms
        << " " << step.width << "x" << step.height
        << " px=" << step.pixel_count;
    if (!step.output_path.empty())
        out << " out=" << fs::path(step.output_path).filename().string();
    out << "\n";
}

static void write_image_csv(const std::string& out_dir, const ImageRecord& rec) {
    std::string csv_path = out_dir + "/metrics.csv";
    std::ofstream f(csv_path);
    if (!f) return;
    f << "step,duration_ms,width,height,pixel_count,megapixels_per_sec\n";
    for (const StepRecord& s : rec.steps) {
        double mp_per_sec = (s.duration_ms > 0 && s.pixel_count > 0)
            ? (s.pixel_count / 1e6) / (s.duration_ms / 1000.0) : 0;
        f << s.step_name << "," << std::fixed << std::setprecision(3) << s.duration_ms
          << "," << s.width << "," << s.height << "," << s.pixel_count
          << "," << mp_per_sec << "\n";
    }
    f << "total," << rec.total_ms() << ",,,,\n";
}

static std::vector<std::string> collect_input_paths(int argc, char* argv[]) {
    std::vector<std::string> paths;
    if (argc < 2) {
        std::cerr << "Usage:\n  " << argv[0] << " --folder <dir>   # all .ppm in dir\n"
                  << "  " << argv[0] << " <f1.ppm> [f2.ppm ...]   # listed files\n";
        return paths;
    }
    std::string first(argv[1]);
    if (first == "--folder") {
        if (argc < 3) { std::cerr << "Error: --folder needs <dir>\n"; return paths; }
        fs::path dir(argv[2]);
        if (!fs::is_directory(dir)) { std::cerr << "Error: not a dir: " << dir << "\n"; return paths; }
        for (const auto& e : fs::directory_iterator(dir)) {
            if (e.is_regular_file() && e.path().extension() == ".ppm")
                paths.push_back(e.path().string());
        }
        std::sort(paths.begin(), paths.end());
    } else {
        for (int i = 1; i < argc; ++i) paths.push_back(argv[i]);
    }
    return paths;
}

static std::string output_dir_for(const std::string& input_path) {
    return fs::path(input_path).stem().string();
}

int main(int argc, char* argv[]) {
    std::vector<std::string> input_paths = collect_input_paths(argc, argv);
    if (input_paths.empty()) return 1;

    const Clock::time_point wall_start = Clock::now();

    std::cout << "=== Serial pipeline: " << input_paths.size() << " image(s) | threshold="
              << static_cast<int>(EDGE_THRESHOLD) << " ===\n\n";

    std::vector<ImageRecord> all_records;

    for (std::size_t idx = 0; idx < input_paths.size(); ++idx) {
        const std::string& input_path = input_paths[idx];
        const int image_index = static_cast<int>(idx) + 1;
        ImageRecord rec;
        rec.input_path = input_path;
        rec.filename   = basename(input_path);
        rec.index      = image_index;
        rec.output_dir = output_dir_for(input_path);

        fs::create_directories(rec.output_dir);

        auto record_step = [&](const char* name, int w, int h, const std::string& out_path) -> StepRecord& {
            rec.steps.emplace_back();
            StepRecord& s = rec.steps.back();
            s.step_name   = name;
            s.start       = Clock::now();
            s.width      = w;
            s.height     = h;
            s.pixel_count = static_cast<std::size_t>(w) * std::max(0, h);
            s.output_path = out_path;
            return s;
        };
        auto finish_step = [](StepRecord& s) {
            s.end = Clock::now();
            s.duration_ms = Ms(s.end - s.start).count();
        };

        try {
            StepRecord& s_load = record_step(STEP_LOAD, 0, 0, "");
            Image img = PPMHandler::loadPPM(input_path);
            s_load.width = img.width;
            s_load.height = img.height;
            s_load.pixel_count = img.pixels.size() / 3;
            finish_step(s_load);
            log_step(rec, s_load, wall_start, std::cout);

            StepRecord& s_gr = record_step(STEP_GRAYSCALE, img.width, img.height, "");
            Image gray = convertToGrayscale(img);
            finish_step(s_gr);
            log_step(rec, s_gr, wall_start, std::cout);

            std::string gray_path = rec.output_dir + "/grayscale.ppm";
            StepRecord& s_sg = record_step(STEP_SAVE_GRAY, gray.width, gray.height, gray_path);
            PPMHandler::savePPM(gray, gray_path);
            finish_step(s_sg);
            log_step(rec, s_sg, wall_start, std::cout);

            StepRecord& s_blur = record_step(STEP_GAUSSIAN_BLUR, gray.width, gray.height, "");
            Image blurred = applyGaussianBlur(gray);
            finish_step(s_blur);
            log_step(rec, s_blur, wall_start, std::cout);

            std::string blur_path = rec.output_dir + "/gaussian_blur.ppm";
            StepRecord& s_sb = record_step(STEP_SAVE_BLUR, blurred.width, blurred.height, blur_path);
            PPMHandler::savePPM(blurred, blur_path);
            finish_step(s_sb);
            log_step(rec, s_sb, wall_start, std::cout);

            StepRecord& s_sob = record_step(STEP_SOBEL, blurred.width, blurred.height, "");
            Image edges = applySobel(blurred);
            finish_step(s_sob);
            log_step(rec, s_sob, wall_start, std::cout);

            std::string sobel_path = rec.output_dir + "/sobel.ppm";
            StepRecord& s_ss = record_step(STEP_SAVE_SOBEL, edges.width, edges.height, sobel_path);
            PPMHandler::savePPM(edges, sobel_path);
            finish_step(s_ss);
            log_step(rec, s_ss, wall_start, std::cout);

            StepRecord& s_thr = record_step(STEP_THRESHOLD, edges.width, edges.height, "");
            Image stencil = applyThreshold(edges, EDGE_THRESHOLD);
            finish_step(s_thr);
            log_step(rec, s_thr, wall_start, std::cout);

            std::string thresh_path = rec.output_dir + "/threshold.ppm";
            StepRecord& s_st = record_step(STEP_SAVE_THRESHOLD, stencil.width, stencil.height, thresh_path);
            PPMHandler::savePPM(stencil, thresh_path);
            finish_step(s_st);
            log_step(rec, s_st, wall_start, std::cout);

            write_image_csv(rec.output_dir, rec);

            double img_total = rec.total_ms();
            double img_mp = rec.steps[0].pixel_count / 1e6;
            double img_mp_per_sec = (img_total > 0) ? (img_mp / (img_total / 1000.0)) : 0;
            std::cout << "  --- image " << rec.index << " " << rec.filename
                      << " total_ms=" << std::fixed << std::setprecision(2) << img_total
                      << " megapixels=" << std::setprecision(2) << img_mp
                      << " throughput_mp/s=" << img_mp_per_sec
                      << " csv=" << rec.output_dir << "/metrics.csv\n\n";

        } catch (const std::exception& e) {
            std::cerr << "Error image " << image_index << " " << rec.filename << ": " << e.what() << "\n";
            continue;
        }

        all_records.push_back(std::move(rec));
    }

    const Clock::time_point wall_end = Clock::now();
    const double elapsed_ms = Ms(wall_end - wall_start).count();
    const double elapsed_sec = elapsed_ms / 1000.0;

    std::vector<std::string> step_names = {
        STEP_LOAD, STEP_GRAYSCALE, STEP_SAVE_GRAY, STEP_GAUSSIAN_BLUR, STEP_SAVE_BLUR,
        STEP_SOBEL, STEP_SAVE_SOBEL, STEP_THRESHOLD, STEP_SAVE_THRESHOLD
    };

    std::cout << "=== Per-step aggregate ===\n";
    std::cout << std::fixed << std::setprecision(2);
    double total_step_ms = 0;
    for (const std::string& name : step_names) {
        double sum = 0, min_v = 1e9, max_v = 0;
        int n = 0;
        for (const ImageRecord& r : all_records) {
            for (const StepRecord& s : r.steps) {
                if (s.step_name != name) continue;
                sum += s.duration_ms;
                min_v = std::min(min_v, s.duration_ms);
                max_v = std::max(max_v, s.duration_ms);
                ++n;
            }
        }
        if (n == 0) continue;
        total_step_ms += sum;
        std::cout << "  " << std::setw(20) << name << " total_ms=" << std::setw(10) << sum
                  << " mean=" << (sum / n) << " min=" << min_v << " max=" << max_v << " n=" << n << "\n";
    }

    std::cout << "\n=== Per-image total ===\n";
    for (const ImageRecord& r : all_records)
        std::cout << "  " << r.index << " " << r.filename << " total_ms=" << r.total_ms() << "\n";

    std::size_t total_pixels = 0;
    for (const ImageRecord& r : all_records) {
        if (!r.steps.empty() && r.steps[0].step_name == STEP_LOAD)
            total_pixels += r.steps[0].pixel_count;
    }
    const int n_img = static_cast<int>(all_records.size());
    const double megapixels = total_pixels / 1e6;
    const double mp_per_sec = (elapsed_sec > 0) ? (megapixels / elapsed_sec) : 0;
    const double img_per_sec = (elapsed_sec > 0) ? (n_img / elapsed_sec) : 0;

    std::cout << "\n=== Summary ===\n";
    std::cout << "  elapsed_sec=" << elapsed_sec << "  (wall clock from start to end)\n";
    std::cout << "  images_processed=" << n_img << " total_pixels=" << total_pixels
              << " megapixels=" << megapixels << "\n";
    std::cout << "  throughput_megapixels_per_sec=" << mp_per_sec
              << " throughput_images_per_sec=" << img_per_sec << "\n";

    std::cout << "\n=== Per-stage % of total step time ===\n";
    for (const std::string& name : step_names) {
        double sum = 0;
        for (const ImageRecord& r : all_records)
            for (const StepRecord& s : r.steps)
                if (s.step_name == name) sum += s.duration_ms;
        if (total_step_ms <= 0) continue;
        std::cout << "  " << std::setw(20) << name << " " << (100.0 * sum / total_step_ms) << "%\n";
    }

    std::cout << "\nTotal elapsed: " << elapsed_sec << " s\n";
    return 0;
}
