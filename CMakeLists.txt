cmake_minimum_required(VERSION 3.0)

project(talkingface)

# set(CMAKE_CXX_FLAGS "-O2")
add_definitions(-std=c++14)
add_definitions("-Wall" -g)
add_definitions(-w)
add_definitions(-DUNICODE -D_UNICODE)

# path
set(CUDA_DIR "" CACHE PATH "CUDA toolkit root")
set(TENSORRT_DIR "" CACHE PATH "TensorRT root")
set(OPENCV_CUDA_DIR "" CACHE PATH "OpenCV install prefix (with CUDA)")
set(FFMPEG_DIR "" CACHE PATH "FFmpeg install prefix")
set(NVCODEC_HEADERS_DIR "" CACHE PATH "nv-codec-headers source prefix (optional)")
option(USE_NVCODEC_HEADERS "Add nv-codec-headers include path (only if your code includes ffnvcodec headers)" OFF)

# Allow overriding via environment variables (so CMakeLists.txt has zero hard-coded machine paths)
if(NOT CUDA_DIR AND DEFINED ENV{LIPSYNC_CUDA_DIR})
    set(CUDA_DIR $ENV{LIPSYNC_CUDA_DIR})
endif()
if(NOT TENSORRT_DIR AND DEFINED ENV{LIPSYNC_TENSORRT_DIR})
    set(TENSORRT_DIR $ENV{LIPSYNC_TENSORRT_DIR})
endif()
if(NOT OPENCV_CUDA_DIR AND DEFINED ENV{LIPSYNC_OPENCV_DIR})
    set(OPENCV_CUDA_DIR $ENV{LIPSYNC_OPENCV_DIR})
endif()
if(NOT FFMPEG_DIR AND DEFINED ENV{LIPSYNC_FFMPEG_DIR})
    set(FFMPEG_DIR $ENV{LIPSYNC_FFMPEG_DIR})
endif()
if(NOT NVCODEC_HEADERS_DIR AND DEFINED ENV{LIPSYNC_NVCODEC_HEADERS_DIR})
    set(NVCODEC_HEADERS_DIR $ENV{LIPSYNC_NVCODEC_HEADERS_DIR})
endif()

if(NOT CUDA_DIR)
    message(FATAL_ERROR "CUDA_DIR is not set. Provide -DCUDA_DIR=... or set env LIPSYNC_CUDA_DIR")
endif()
if(NOT TENSORRT_DIR)
    message(FATAL_ERROR "TENSORRT_DIR is not set. Provide -DTENSORRT_DIR=... or set env LIPSYNC_TENSORRT_DIR")
endif()
if(NOT OPENCV_CUDA_DIR)
    message(FATAL_ERROR "OPENCV_CUDA_DIR is not set. Provide -DOPENCV_CUDA_DIR=... or set env LIPSYNC_OPENCV_DIR")
endif()
if(NOT FFMPEG_DIR)
    message(FATAL_ERROR "FFMPEG_DIR is not set. Provide -DFFMPEG_DIR=... or set env LIPSYNC_FFMPEG_DIR")
endif()

# OpenCV with CUDA
set(OpenCV_DIR ${OPENCV_CUDA_DIR}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_CUDA_DIR}/include/opencv4)

if(OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "opencv Not Found!")
endif(OpenCV_FOUND)

include_directories(${OpenCV_INCLUDE_DIRS})

# FFmpeg
include_directories(${FFMPEG_DIR}/include)
link_directories(${FFMPEG_DIR}/lib)
set(FFMPEG_LIBRARIES avcodec avformat avutil swscale swresample)

if(USE_NVCODEC_HEADERS)
    message(STATUS "Using nv-codec-headers from: ${NVCODEC_HEADERS_DIR}")
    if(NOT NVCODEC_HEADERS_DIR)
        message(FATAL_ERROR "USE_NVCODEC_HEADERS=ON but NVCODEC_HEADERS_DIR is not set. Provide -DNVCODEC_HEADERS_DIR=... or env LIPSYNC_NVCODEC_HEADERS_DIR")
    endif()
    include_directories(${NVCODEC_HEADERS_DIR}/include)
endif()

# cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)

# common
include_directories(${TENSORRT_DIR}/samples/common)
link_directories(${TENSORRT_DIR}/samples/common)

# tensorrt
include_directories(${TENSORRT_DIR}/include/)
link_directories(${TENSORRT_DIR}/lib/)

# ------------------------------------------------------
# trt_function lib
# ------------------------------------------------------
set(TRT_FUNCTION trt_function)

# 启用CUDA语言支持
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)

file(GLOB TRT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/trt_function/*/*.cpp)
file(GLOB TRT_HDR ${CMAKE_CURRENT_SOURCE_DIR}/trt_function/*/*.h)
file(GLOB TRT_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/trt_function/*/*.cu)
set(COMPILE_TRT_SRC ${TRT_SRC} ${TRT_HDR} ${TRT_CUDA})

add_library(${TRT_FUNCTION} SHARED ${COMPILE_TRT_SRC}) #构建

target_compile_options( ${TRT_FUNCTION}
	PRIVATE -Wall -g
)
target_compile_features( ${TRT_FUNCTION}
	PRIVATE cxx_std_11
)
target_include_directories( ${TRT_FUNCTION}
	PRIVATE ${CERES_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/trt_function/src
)
target_include_directories( ${TRT_FUNCTION}
	PRIVATE ${CERES_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/trt_function/core
)
target_link_libraries( ${TRT_FUNCTION} PRIVATE
	${OpenCV_LIBRARIES}
)

# target_link_libraries(${TRT_FUNCTION} PRIVATE ${TENSORRT_DIR}/lib/)
target_link_libraries(${TRT_FUNCTION} PRIVATE ${TENSORRT_DIR}/lib/libnvinfer_plugin.so)

# ------------------------------------------------------
#  Java JNI library
# ------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/java-8-openjdk-amd64)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/java-8-openjdk-amd64/linux)

# ------------------------------------------------------
# Additions
# ------------------------------------------------------

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/debug)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/trt_function/core)


# ------------------------------------------------------
# Base
# ------------------------------------------------------

file(GLOB SRC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/audio/*.cpp)
file(GLOB SRC_HDR ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h 
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/audio/*.h)
set(COMPILE_SRC ${SRC_SRC} ${SRC_HDR})


# ------------------------------------------------------
# Build
# ------------------------------------------------------

add_library(talkingface SHARED ${COMPILE_SRC})
target_link_libraries(talkingface
    ${OpenCV_LIBS}
    ${TRT_FUNCTION}
    ${FFMPEG_LIBRARIES}
    nvinfer
    cudart
    -lpthread -lm
    -fopenmp
)

# ------------------------------------------------------
# Unit Test
# ------------------------------------------------------

add_executable(testGpuDecoder 
    unitTest/test_gpu_decoder.cpp
    src/gpu_decoder.cpp
)
target_link_libraries(testGpuDecoder
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
    cudart
    -lpthread
)

# SCRFD GPU测试
add_executable(testScrfdGpu 
    TEST/test_scrfd_gpu.cpp
)
target_link_libraries(testScrfdGpu
    ${OpenCV_LIBS}
    ${TRT_FUNCTION}
    nvinfer
    cudart
    -lpthread -lm
)

# detectLandmarkGPU测试
add_executable(testDetectLandmarkGpu 
    TEST/test_detect_landmark_gpu.cpp
)
target_link_libraries(testDetectLandmarkGpu
    ${OpenCV_LIBS}
    ${TRT_FUNCTION}
    nvinfer
    cudart
    -lpthread -lm
)

# PIPNet GPU测试
add_executable(testPipnetGpu 
    TEST/test_pipnet_gpu.cpp
)
target_link_libraries(testPipnetGpu
    ${OpenCV_LIBS}
    ${TRT_FUNCTION}
    nvinfer
    cudart
    -lpthread -lm
)

# add_executable(testDet unitTest/test_facedet.cpp)
# target_link_libraries(testDet
#     ${OpenCV_LIBS}
#     ${TRT_FUNCTION}
#     nvinfer
#     cudart
#     -lpthread -lm
#     -fopenmp
# )

# add_executable(testDetMT unitTest/test_facedet_mt.cpp)
# target_link_libraries(testDetMT
#     ${OpenCV_LIBS}
#     ${TRT_FUNCTION}
#     nvinfer
#     cudart
#     -lpthread -lm
#     -fopenmp
# )

# add_executable(testRead unitTest/test_read.cpp)
# target_link_libraries(testRead
#     ${OpenCV_LIBS}
#     ${TRT_FUNCTION}
#     nvinfer
#     cudart
#     -lpthread -lm
#     -fopenmp
# )


# add_executable(testProcess ${COMPILE_SRC}
#     unitTest/test_process.cpp)
# target_link_libraries(testProcess
#     ${OpenCV_LIBS}
#     ${TRT_FUNCTION}
#     nvinfer
#     cudart
#     -lpthread -lm
#     -fopenmp
# )

# add_executable(testRender ${COMPILE_SRC}
#     unitTest/test_render.cpp)
# target_link_libraries(testRender
#     ${OpenCV_LIBS}
#     ${TRT_FUNCTION}
#     nvinfer
#     cudart
#     -lpthread -lm
#     -fopenmp
# )

# add_executable(testDetMT unitTest/facedet.cpp)
# target_link_libraries(testDetMT
#     ${OpenCV_LIBS}
#     ${TRT_FUNCTION}
#     nvinfer
#     cudart
#     -lpthread -lm
#     -fopenmp
# )

# add_executable(testShutUp ${COMPILE_SRC}
#     unitTest/test_shutup.cpp)
# target_link_libraries(testShutUp
#     ${OpenCV_LIBS}
#     ${TRT_FUNCTION}
#     nvinfer
#     cudart
#     -lpthread -lm
#     -fopenmp
# )
