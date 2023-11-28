#ifndef CUDA_CUDARESIZE_CUH
#define CUDA_CUDARESIZE_CUH

enum class Format {
    YUV420 = 0
};

bool
resize(const unsigned char *src, int srcWidth, int srcHeight, unsigned char *dst, int dstWidth, int dstHeight,
       Format format = Format::YUV420, cudaStream_t stream = nullptr);


#endif //CUDA_CUDARESIZE_CUH
