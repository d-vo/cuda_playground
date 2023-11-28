#include "cudaResize.cuh"
#include <nppi_geometry_transforms.h>

#include <array>


bool
resizeYuv420(const unsigned char *src, int srcWidth, int srcHeight, unsigned char *dst, int dstWidth,
             int dstHeight, cudaStream_t stream) {
    if (srcWidth % 2 != 0 or srcHeight % 2 != 0 or dstWidth % 2 != 0 or dstHeight % 2 != 0) {
        return false;
    }
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = stream;

    // Source
    const int srcUOffset = srcWidth * srcHeight;
    const int srcVOffset = srcWidth * srcHeight + srcWidth * srcHeight / 4;
    std::array<int, 3> srcWidths = {srcWidth, srcWidth / 2, srcWidth / 2};
    std::array<int, 3> srcHeights = {srcHeight, srcHeight / 2, srcHeight / 2};
    std::array<const unsigned char *, 3> srcPointers = {src, &src[srcUOffset], &src[srcVOffset]};

    // Destination
    const int dstUOffset = dstWidth * dstHeight;
    const int dstVOffset = dstWidth * dstHeight + dstWidth * dstHeight / 4;
    std::array<int, 3> dstWidths = {dstWidth, dstWidth / 2, dstWidth / 2};
    std::array<int, 3> dstHeights = {dstHeight, dstHeight / 2, dstHeight / 2};
    std::array<unsigned char *, 3> dstPointers = {dst, &dst[dstUOffset], &dst[dstVOffset]};

    NppStatus st;
    for (size_t i = 0; i < 3; ++i) {
        NppiSize srcSize = {srcWidths[i], srcHeights[i]};
        NppiRect srcRoi = {0, 0, srcSize.width, srcSize.height};
        NppiSize dstSize = {dstWidths[i], dstHeights[i]};
        NppiRect dstRoi = {0, 0, dstSize.width, dstSize.height};

        st = nppiResize_8u_C1R_Ctx(srcPointers[i], srcWidths[i], srcSize, srcRoi,
                                   dstPointers[i], dstWidths[i], dstSize, dstRoi, NPPI_INTER_LANCZOS,
                                   nppStreamCtx);
    }

    if (st != NPP_SUCCESS) {
        return false;
    }

    return true;
}


bool
resize(const unsigned char *src, int srcWidth, int srcHeight, unsigned char *dst, int dstWidth, int dstHeight,
       Format format, cudaStream_t stream) {
    if (format == Format::YUV420) {
        return resizeYuv420(src, srcWidth, srcHeight, dst, dstWidth,
                            dstHeight, stream);

    }

    return false;
}
