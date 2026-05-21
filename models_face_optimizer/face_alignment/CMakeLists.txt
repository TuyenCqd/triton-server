cmake_minimum_required(VERSION 3.18)
project(face_alignment_backend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC")

# CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

# Triton includes
set(TRITON_INCLUDE_PATHS 
    /opt/tritonserver/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# TensorRT
find_package(TensorRT REQUIRED)

# CUDA
find_package(CUDA REQUIRED)

# OpenCV (for CPU fallback if needed)
find_package(OpenCV REQUIRED)

# Source files
set(BACKEND_SRCS
    src/face_alignment_backend.cc
    src/face_alignment_kernel.cu
)

# Create backend library
add_library(face_alignment_backend SHARED ${BACKEND_SRCS})

target_include_directories(face_alignment_backend
    PUBLIC
    ${TRITON_INCLUDE_PATHS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(face_alignment_backend
    PUBLIC
    tritonserver
    ${TENSORRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    cudart
)

# Set output directory
set_target_properties(face_alignment_backend
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
