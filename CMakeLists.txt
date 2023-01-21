cmake_minimum_required(VERSION 3.22)
project(cuda CUDA CXX)


set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)


add_library(cudaResize cudaResize.cu)
set_target_properties(cudaResize PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudaResize PRIVATE CUDA::nppig)


add_executable(ResizeDemo resizeDemo.cpp)
target_link_libraries(ResizeDemo PRIVATE cudaResize ${OpenCV_LIBS})
target_include_directories(ResizeDemo PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
