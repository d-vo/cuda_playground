// Libary
#include <cuda_runtime.h>
#include "cudaResize.cuh"

#include <opencv4/opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        std::cout << "Usage: ./ResizeDemo lena.yuv" << std::endl;
        return EXIT_FAILURE;
    }

    std::string lenaPath(argv[1]);
    if (!std::filesystem::exists(lenaPath)) {
        std::cout << "Could not find demo image. \nUsage: ./ResizeDemo lena.yuv" << std::endl;
        return EXIT_FAILURE;
    }
    std::ifstream input(lenaPath, std::ios::binary);

    // copies all data into buffer
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

    int srcWidth = 512;
    int srcHeight = 512;
    int srcYuv420size = srcWidth * srcHeight * 1.5;

    if (buffer.size() != srcYuv420size) {
        std::cout << "Wrong image dimensions" << std::endl;
        return EXIT_FAILURE;
    }

    float scale = 0.5;
    int dstWidth = srcWidth * scale;
    int dstHeight = srcHeight * scale;
    int dstYuv420size = dstWidth * dstHeight * 1.5;

    // Allocate Unified Memory – accessible from CPU or GPU
    unsigned char *src, *dst;
    cudaMallocManaged(&src, srcYuv420size * sizeof(unsigned char));
    cudaMallocManaged(&dst, dstYuv420size * sizeof(unsigned char));

    // Load image
    std::memcpy(src, buffer.data(), srcYuv420size);
    std::memset(dst, 0, dstYuv420size * sizeof(unsigned char));

    // Resize
    if (!resize(src, srcWidth, srcHeight, dst, dstWidth, dstHeight)) {
        std::cout << "Could not resize" << std::endl;
        return EXIT_FAILURE;
    }

    // Sync
    cudaDeviceSynchronize();

    // Validate
    const int srcUOffset = srcWidth * srcHeight;
    const int srcVOffset = srcWidth * srcHeight + srcWidth * srcHeight / 4;
    std::array<int, 3> srcWidths = {srcWidth, srcWidth / 2, srcWidth / 2};
    std::array<int, 3> srcHeights = {srcHeight, srcHeight / 2, srcHeight / 2};
    std::array<unsigned char *, 3> srcPointers = {src, &src[srcUOffset], &src[srcVOffset]};

    const int dstUOffset = dstWidth * dstHeight;
    const int dstVOffset = dstWidth * dstHeight + dstWidth * dstHeight / 4;
    std::array<int, 3> dstWidths = {dstWidth, dstWidth / 2, dstWidth / 2};
    std::array<int, 3> dstHeights = {dstHeight, dstHeight / 2, dstHeight / 2};
    std::array<unsigned char *, 3> dstPointers = {dst, &dst[dstUOffset], &dst[dstVOffset]};

    for (size_t i = 0; i < 3; i++) {
        cv::Mat srcMat(srcHeights[i], srcWidths[i], CV_8U, srcPointers[i]);
        cv::Mat dstMat(dstHeights[i], dstWidths[i], CV_8U, dstPointers[i]);

        cv::imshow("Src" + std::to_string(i), srcMat);
        cv::imshow("Dst" + std::to_string(i), dstMat);
    }


    cv::waitKey(0);

    cudaFree(src);
    cudaFree(dst);
    return EXIT_SUCCESS;
}
