cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(ext)

set(CMAKE_CUDA_COMPILER /home/justanhduc/cuda/cuda-11.0/bin/nvcc)
enable_language(CUDA)

set(Torch_DIR /home/justanhduc/Documents/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(ext main.cpp)
target_link_libraries(ext "${TORCH_LIBRARIES}")
target_compile_features(ext PRIVATE cxx_range_for)
