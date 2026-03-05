#include "image_utils.hpp"
#include <fstream>
#include <stdexcept>
#include <cctype>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace {

/** Read next decimal number from stream, skipping lines that start with #. */
int getNextHeaderNumber(std::istream& in) {
    std::string token;
    while (in >> token) {
        if (token.empty()) continue;
        if (token[0] == '#') {
            in.ignore(65536, '\n');
            continue;
        }
        try {
            return std::stoi(token);
        } catch (...) {
            throw std::runtime_error("PPM header: invalid number '" + token + "'");
        }
    }
    throw std::runtime_error("PPM header: unexpected end while reading number");
}

/** Skip whitespace and optional comment lines, then read magic "P6". */
void expectP6(std::istream& in) {
    char c;
    while (in.get(c)) {
        if (c == '#') {
            in.ignore(65536, '\n');
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(c))) {
            in.putback(c);
            break;
        }
    }
    std::string magic(2, '\0');
    if (!in.get(magic[0]) || !in.get(magic[1]))
        throw std::runtime_error("PPM: could not read magic number");
    if (magic != "P6")
        throw std::runtime_error("PPM: expected P6, got '" + magic + "'");
}

} // namespace

Image PPMHandler::loadPPM(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open file: " + filename);

    expectP6(in);

    int width  = getNextHeaderNumber(in);
    int height = getNextHeaderNumber(in);
    int max_val = getNextHeaderNumber(in);

    if (width <= 0 || height <= 0)
        throw std::runtime_error("PPM: invalid dimensions " + std::to_string(width) + "x" + std::to_string(height));
    if (max_val <= 0 || max_val > 65535)
        throw std::runtime_error("PPM: invalid max_val " + std::to_string(max_val));

    // P6 spec: single whitespace (e.g. newline) after max_val, then raw bytes
    char after_max;
    if (!in.get(after_max) || !std::isspace(static_cast<unsigned char>(after_max)))
        throw std::runtime_error("PPM: missing newline after max_val");

    const std::size_t num_bytes = static_cast<std::size_t>(width) * height * 3;
    Image img;
    img.width   = width;
    img.height  = height;
    img.max_val = max_val;
    img.pixels.resize(num_bytes);

    if (!in.read(reinterpret_cast<char*>(img.pixels.data()), static_cast<std::streamsize>(num_bytes)))
        throw std::runtime_error("PPM: failed to read pixel data (expected " + std::to_string(num_bytes) + " bytes)");

    return img;
}

void PPMHandler::savePPM(const Image& img, const std::string& filename) {
    if (img.width <= 0 || img.height <= 0)
        throw std::runtime_error("Cannot save image: invalid dimensions");
    const std::size_t expected = static_cast<std::size_t>(img.width) * img.height * 3;
    if (img.pixels.size() != expected)
        throw std::runtime_error("Cannot save image: pixel buffer size mismatch");

    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot create file: " + filename);

    out << "P6\n" << img.width << ' ' << img.height << '\n' << img.max_val << '\n';
    out.write(reinterpret_cast<const char*>(img.pixels.data()), static_cast<std::streamsize>(img.pixels.size()));
    if (!out)
        throw std::runtime_error("Failed to write pixel data to " + filename);
}

// --- Grayscale & Convolution (Vision Engine) ---------------------------------

Image convertToGrayscale(const Image& input) {
    if (input.pixels.size() != static_cast<std::size_t>(input.width) * input.height * 3)
        throw std::runtime_error("convertToGrayscale: invalid image");

    Image out;
    out.width   = input.width;
    out.height  = input.height;
    out.max_val = input.max_val;
    out.pixels.resize(static_cast<std::size_t>(input.width) * input.height * 3);

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * input.width + x) * 3;
            const float r = static_cast<float>(input.pixels[i]);
            const float g = static_cast<float>(input.pixels[i + 1]);
            const float b = static_cast<float>(input.pixels[i + 2]);
            const float Y = 0.299f * r + 0.587f * g + 0.114f * b;
            const float clamped = std::max(0.f, std::min(255.f, Y));
            const auto gray = static_cast<unsigned char>(std::round(clamped));
            out.pixels[i]     = gray;
            out.pixels[i + 1] = gray;
            out.pixels[i + 2] = gray;
        }
    }
    return out;
}

namespace {

/** Returns 3x3 convolution sum at (x,y). (x,y) must be interior. */
float convolveAt3x3(const Image& input, const Kernel& kernel, int x, int y) {
    const int w = input.width;
    float sum = 0.f;
    for (int dy = 0; dy < 3; ++dy) {
        for (int dx = 0; dx < 3; ++dx) {
            float gray = static_cast<float>(input.pixels[(static_cast<std::size_t>(y + dy - 1) * w + (x + dx - 1)) * 3]);
            sum += kernel[dy][dx] * gray;
        }
    }
    return sum;
}

/** Returns convolution sum at (x,y) for KxK kernel. (x,y) must allow full kernel inside image. */
float convolveAtNxN(const Image& input, const Kernel& kernel, int x, int y) {
    const int w = input.width;
    const int K = static_cast<int>(kernel.size());
    const int half = K / 2;
    float sum = 0.f;
    for (int dy = 0; dy < K; ++dy) {
        for (int dx = 0; dx < K; ++dx) {
            int ny = y + dy - half;
            int nx = x + dx - half;
            float gray = static_cast<float>(input.pixels[(static_cast<std::size_t>(ny) * w + nx) * 3]);
            sum += kernel[dy][dx] * gray;
        }
    }
    return sum;
}

const Kernel SobelX = {
    { -1.f,  0.f,  1.f },
    { -2.f,  0.f,  2.f },
    { -1.f,  0.f,  1.f }
};
const Kernel SobelY = {
    {  1.f,  2.f,  1.f },
    {  0.f,  0.f,  0.f },
    { -1.f, -2.f, -1.f }
};

/** Build 11x11 Gaussian kernel (sigma ~ 2). Sum normalized to 1. */
Kernel makeGaussian11x11() {
    const int K = 11;
    const float sigma = 2.0f;
    const int half = K / 2;
    Kernel k(K, std::vector<float>(K));
    float sum = 0.f;
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            float x = static_cast<float>(i - half);
            float y = static_cast<float>(j - half);
            float v = std::exp(-(x * x + y * y) / (2.f * sigma * sigma));
            k[i][j] = v;
            sum += v;
        }
    }
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j)
            k[i][j] /= sum;
    return k;
}

const Kernel GaussianBlurKernel11x11 = makeGaussian11x11();

} // namespace

Image applyFilter(const Image& input, const Kernel& kernel) {
    if (input.pixels.size() != static_cast<std::size_t>(input.width) * input.height * 3)
        throw std::runtime_error("applyFilter: invalid image");
    const int Ky = static_cast<int>(kernel.size());
    if (Ky == 0) throw std::runtime_error("applyFilter: empty kernel");
    const int Kx = static_cast<int>(kernel[0].size());
    if (Kx == 0 || (Ky % 2 == 0) || (Kx % 2 == 0))
        throw std::runtime_error("applyFilter: kernel must be odd-sized");
    const int half_y = Ky / 2;
    const int half_x = Kx / 2;

    const int w = input.width;
    const int h = input.height;
    Image out;
    out.width   = w;
    out.height  = h;
    out.max_val = 255;
    out.pixels.assign(static_cast<std::size_t>(w) * h * 3, 0);

    const bool use3x3 = (Ky == 3 && Kx == 3);
    for (int y = half_y; y < h - half_y; ++y) {
        for (int x = half_x; x < w - half_x; ++x) {
            float sum = use3x3 ? convolveAt3x3(input, kernel, x, y) : convolveAtNxN(input, kernel, x, y);
            const float clamped = std::max(0.f, std::min(255.f, sum));
            const auto val = static_cast<unsigned char>(std::round(clamped));
            const std::size_t i = (static_cast<std::size_t>(y) * w + x) * 3;
            out.pixels[i]     = val;
            out.pixels[i + 1] = val;
            out.pixels[i + 2] = val;
        }
    }
    return out;
}

Image applyGaussianBlur(const Image& input) {
    return applyFilter(input, GaussianBlurKernel11x11);
}

Image applySobel(const Image& grayImg) {
    if (grayImg.pixels.size() != static_cast<std::size_t>(grayImg.width) * grayImg.height * 3)
        throw std::runtime_error("applySobel: invalid image");

    const int w = grayImg.width;
    const int h = grayImg.height;
    Image out;
    out.width   = w;
    out.height  = h;
    out.max_val = 255;
    out.pixels.assign(static_cast<std::size_t>(w) * h * 3, 0);

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const float Gx = convolveAt3x3(grayImg, SobelX, x, y);
            const float Gy = convolveAt3x3(grayImg, SobelY, x, y);
            const float mag = std::sqrt(Gx * Gx + Gy * Gy);
            const float clamped = std::max(0.f, std::min(255.f, mag));
            const auto val = static_cast<unsigned char>(std::round(clamped));
            const std::size_t i = (static_cast<std::size_t>(y) * w + x) * 3;
            out.pixels[i]     = val;
            out.pixels[i + 1] = val;
            out.pixels[i + 2] = val;
        }
    }
    return out;
}

Image applyThreshold(const Image& input, unsigned char threshold) {
    if (input.pixels.size() != static_cast<std::size_t>(input.width) * input.height * 3)
        throw std::runtime_error("applyThreshold: invalid image");

    const int w = input.width;
    const int h = input.height;
    Image out;
    out.width   = w;
    out.height  = h;
    out.max_val = 255;
    out.pixels.resize(static_cast<std::size_t>(w) * h * 3);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * w + x) * 3;
            unsigned char val = (input.pixels[i] > threshold) ? 255 : 0;
            out.pixels[i]     = val;
            out.pixels[i + 1] = val;
            out.pixels[i + 2] = val;
        }
    }
    return out;
}
