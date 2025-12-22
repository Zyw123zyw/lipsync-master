#!/bin/bash

# 测试 resize 函数对比
# 编译并运行测试

echo "=========================================="
echo "  编译 resize 对比测试"
echo "=========================================="

# 设置路径
TRT_INCLUDE="/home/vcm/3party/TensorRT-8.6.1.6/include"
TRT_LIB="/home/vcm/3party/TensorRT-8.6.1.6/lib"
CUDA_INCLUDE="/usr/local/cuda/include"
CUDA_LIB="/usr/local/cuda/lib64"

# OpenCV 路径（默认使用自定义路径）
if [ -z "$LIPSYNC_OPENCV_DIR" ]; then
    LIPSYNC_OPENCV_DIR="/workspace/opencv-install-cuda"
fi

echo "使用 OpenCV: $LIPSYNC_OPENCV_DIR"

if [ ! -d "$LIPSYNC_OPENCV_DIR" ]; then
    echo "错误: OpenCV 目录不存在: $LIPSYNC_OPENCV_DIR"
    exit 1
fi

OPENCV_INCLUDE="$LIPSYNC_OPENCV_DIR/include/opencv4"
OPENCV_LIB="$LIPSYNC_OPENCV_DIR/lib"
OPENCV_FLAGS="-I${OPENCV_INCLUDE} -L${OPENCV_LIB} -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_cudaarithm -lopencv_cudawarping -Xlinker -rpath -Xlinker ${OPENCV_LIB}"

# 编译
echo "开始编译..."
nvcc -o test_resize_compare \
    test_resize_compare.cpp \
    ../trt_function/src/gpu_kernels.cu \
    -I${TRT_INCLUDE} \
    -I${CUDA_INCLUDE} \
    -I../include \
    -L${TRT_LIB} \
    -L${CUDA_LIB} \
    ${OPENCV_FLAGS} \
    -lcudart \
    -std=c++14 \
    -O3 \
    -Xcompiler -fPIC

if [ $? -ne 0 ]; then
    echo "编译失败！"
    exit 1
fi

echo ""
echo "编译完成！"
echo ""
echo "=========================================="
echo "  运行测试"
echo "=========================================="
echo ""

# 运行测试
# 参数: <图像路径> <目标宽度> <目标高度>
# 如果不提供参数，会生成测试图像并 resize 到 810x1440

if [ $# -eq 0 ]; then
    echo "使用默认参数：生成测试图像，resize 到 810x1440"
    ./test_resize_compare
else
    echo "使用自定义参数: $@"
    ./test_resize_compare "$@"
fi

echo ""
echo "=========================================="
echo "  测试完成"
echo "=========================================="
