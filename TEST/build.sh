#!/bin/bash

# 视频拼接测试
g++ -o concat_video concat_video.cpp -std=c++11
echo "Build done: ./concat_video"

# GPU预处理测试
OPENCV_CUDA_DIR=/mnt/data/vision-devel/zhangyiwei/opencv-install-cuda
CUDA_DIR=/usr/local/cuda-11.2

# 先编译CUDA文件
nvcc -c ../trt_function/src/gpu_kernels.cu -o gpu_kernels.o \
    -I${CUDA_DIR}/include \
    -std=c++14

# 编译主程序并链接
g++ -o test_gpu_preprocess test_gpu_preprocess.cpp \
    ../trt_function/src/gpu_preprocess.cpp \
    gpu_kernels.o \
    -I${OPENCV_CUDA_DIR}/include/opencv4 \
    -I${CUDA_DIR}/include \
    -L${OPENCV_CUDA_DIR}/lib \
    -L${CUDA_DIR}/lib64 \
    -lopencv_core -lopencv_imgproc -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudaarithm \
    -lcudart \
    -std=c++14 \
    -Wl,-rpath,${OPENCV_CUDA_DIR}/lib

echo "Build done: ./test_gpu_preprocess"
