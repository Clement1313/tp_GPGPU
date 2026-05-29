#include <cassert>
#include <spdlog/spdlog.h>

#include "render.hpp"

__device__ uchar4 heat_lut_color(float x);

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
    cudaError_t err = cudaGetLastError();
    spdlog::error("{} ({}, line: {})", msg, fname, line);
    spdlog::error("Error {}: {}", cudaGetErrorName(err),
                  cudaGetErrorString(err));
    std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

struct rgba8_t
{
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
    std::uint8_t a;
};

///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
__global__ void apply_LUT(char* buffer, int width, int height, size_t pitch, int max_iter, const uchar4* LUT) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int* iter_line = reinterpret_cast<int*>(buffer + y * pitch);
    uchar4* color_line = reinterpret_cast<uchar4*>(buffer + y * pitch);

    int iteration = iter_line[x];
    if (iteration < 0)
        iteration = 0;
    if (iteration > max_iter)
        iteration = max_iter;

    color_line[x] = LUT[iteration];
}

/// This function is single thread for now!
///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
/// \param LUT Output look-up table
__global__ void compute_LUT(const char* buffer, int width, int height,
                            size_t pitch, int max_iter, uchar4* LUT)
{
    (void)buffer;
    (void)width;
    (void)height;
    (void)pitch;

    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0)
        return;

    if (max_iter <= 0)
    {
        LUT[0] = make_uchar4(0, 0, 0, 255);
        return;
    }

    for (int iteration = 0; iteration <= max_iter; ++iteration)
    {
        float t = (float)iteration / (float)max_iter;
        LUT[iteration] = heat_lut_color(t);
    }
}

// Compute the number or iteration of the fractal per pixel and store the result
// in *buffer*.
/// Note that a 32-bits location can be used to store an integer (int32) or a
/// color (uchar4).
///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
__global__ void compute_iter(char* buffer, int width, int height, size_t pitch,
                             int max_iter)
{
    // float denum = width * width + height * height;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int* lineptr = reinterpret_cast<int*>(buffer + y * pitch);

    float halfW = (float)width * 0.5f;
    float halfH = (float)height * 0.5f;
    float nx = ((float)x - halfW) / halfW;
    float ny = ((float)y - halfH) / halfH;
    const float scaleX = 1.5f;
    const float scaleY = scaleX * ((float)height / (float)width);
    float mx0 = -0.5f + nx * scaleX;
    float my0 = -ny * scaleY;

    float mx = 0.0f, my = 0.0f;
    int iteration = 0;

    while ((mx * mx + my * my) < 4.0f && iteration < max_iter)
    {
        float mxtemp = mx * mx - my * my + mx0;
        my = 2.0f * mx * my + my0;
        mx = mxtemp;
        ++iteration;
    }

    lineptr[x] = iteration;
}

rgba8_t heat_lut(float x)
{
    assert(0 <= x && x <= 1);
    float x0 = 1.f / 4.f;
    float x1 = 2.f / 4.f;
    float x2 = 3.f / 4.f;

    if (x < x0)
    {
        auto g = static_cast<std::uint8_t>(x / x0 * 255);
        return rgba8_t{ 0, g, 255, 255 };
    }
    else if (x < x1)
    {
        auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
        return rgba8_t{ 0, 255, b, 255 };
    }
    else if (x < x2)
    {
        auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
        return rgba8_t{ r, 255, 0, 255 };
    }
    else if (x < 1.0)
    {
        auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
        return rgba8_t{ 255, b, 0, 255 };
    }
    else
    {
        return rgba8_t{ 0, 0, 0, 255 };
    }
}

__device__ uchar4 heat_lut_color(float x)
{
    if (x < 0.25f && 0.0f <= x)
        return make_uchar4(0, (unsigned char)(x / 0.25f * 255), 255, 255);
    if (x < 0.5f && 0.25f <= x)
        return make_uchar4(0, 255, (unsigned char)((0.5f - x) / 0.25f * 255),
                           255);
    if (x < 0.75f && 0.5f <= x)
        return make_uchar4((unsigned char)((x - 0.5f) / 0.25f * 255), 255, 0,
                           255);
    if (x < 1.0f && 0.75f <= x)
        return make_uchar4(255, (unsigned char)((1.0f - x) / 0.25f * 255), 0,
                           255);

    return make_uchar4(0, 0, 0, 255);
}

// Device code
__global__ void mykernel(char* buffer, int width, int height, size_t pitch)
{
    float denum = width * width + height * height;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    uchar4* lineptr = (uchar4*)(buffer + y * pitch);
    float v = (x * x + y * y) / denum;
    uint8_t grayv = v * 255;

    lineptr[x] = { grayv, grayv, grayv, 255 };

    float halfW = (float)width * 0.5f;
    float halfH = (float)height * 0.5f;
    float nx = ((float)x - halfW) / halfW;
    float ny = ((float)y - halfH) / halfH;
    const float scaleX = 1.5f;
    const float scaleY = scaleX * ((float)height / (float)width);
    float mx0 = -0.5f + nx * scaleX;
    float my0 = -ny * scaleY;

    float mx = 0.0f, my = 0.0f;
    int iteration = 0;
    const int maxIter = 100;

    while ((mx * mx + my * my) < 4.0f && iteration < maxIter)
    {
        float mxtemp = mx * mx - my * my + mx0;
        my = 2.0f * mx * my + my0;
        mx = mxtemp;
        ++iteration;
    }

    float t = (float)iteration / (float)maxIter;
    uchar4 color = heat_lut_color(t);

    lineptr[x] = color;
}

void render(char* hostBuffer, int width, int height, std::ptrdiff_t stride,
            int n_iterations)
{
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    char* devBuffer;
    size_t pitch;
    uchar4* devLUT = nullptr;

    rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(rgba8_t), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMalloc(&devLUT, (n_iterations + 1) * sizeof(uchar4));
    if (rc)
        abortError("Fail LUT allocation");
    {
        int bsize = 32;
        int w = std::ceil((float)width / bsize);
        int h = std::ceil((float)height / bsize);

        spdlog::debug("running kernels of size ({},{})", w, h);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);

        compute_iter<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch,
                                            n_iterations);
        if (cudaPeekAtLastError())
            abortError("compute_iter error");

        compute_LUT<<<1, 1>>>(devBuffer, width, height, pitch, n_iterations,
                              devLUT);
        if (cudaPeekAtLastError())
            abortError("compute_LUT error");

        apply_LUT<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch,
                                         n_iterations, devLUT);

        if (cudaPeekAtLastError())
            abortError("apply_LUT error");
    }

    // Copy back to main memory
    rc = cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch,
                      width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(devBuffer);
    if (rc)
        abortError("Unable to free memory");

    rc = cudaFree(devLUT);
    if (rc)
        abortError("Unable to free LUT memory");
}
