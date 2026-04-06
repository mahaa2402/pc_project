#pragma once

#include <string>
#include <vector>
#include <cstddef>

/** 3x3 convolution kernel (row-major: kernel[row][col]). */
using Kernel = std::vector<std::vector<float>>;

/**
 * Simple image representation for PPM P6 format.
 * Pixels are stored as interleaved RGB: R,G,B,R,G,B,...
 */
struct Image {
    int width{0};
    int height{0};
    int max_val{255};
    std::vector<unsigned char> pixels;
};

/**
 * Load and save PPM P6 (binary) images with robust header parsing.
 * Handles comments (#) and varying whitespace in headers.
 */
class PPMHandler {
public:
    /** Load a P6 PPM from file. Skips # comment lines in header. */
    static Image loadPPM(const std::string& filename);

    /** Save image as P6 PPM (header + raw binary pixel data). */
    static void savePPM(const Image& img, const std::string& filename);
};

/** Convert RGB image to grayscale using luminosity: Y = 0.299R + 0.587G + 0.114B. */
Image convertToGrayscale(const Image& input);

/** Generic convolution (any odd-sized kernel, e.g. 3x3 or 11x11). Border set to 0. Result clamped to [0, 255]. */
Image applyFilter(const Image& input, const Kernel& kernel);

/** Sobel edge detection using applyFilter (Gx, Gy) then magnitude = sqrt(Gx^2 + Gy^2). */
Image applySobel(const Image& grayImg);

/** 11x11 Gaussian blur (reduces noise; ~15x more work per pixel than 3x3). */
Image applyGaussianBlur(const Image& input);

/** Binary threshold: pixel > threshold -> 255 (white), else 0 (black). */
Image applyThreshold(const Image& input, unsigned char threshold = 50);
