#ifndef GPU_KERNELS_CUH
#define GPU_KERNELS_CUH

#include <cuda_runtime.h>

namespace Function {

//=============================================================================
// GPU Resize - 双线性插值
//=============================================================================

/**
 * GPU双线性插值resize (uint8版本)
 * 
 * 实现和OpenCV CPU cv::resize(INTER_LINEAR)一致的逻辑
 * 
 * @param src       输入图像数据 (HWC, BGR, uint8)
 * @param dst       输出图像数据 (HWC, BGR, uint8)
 * @param src_w     输入宽度
 * @param src_h     输入高度
 * @param dst_w     输出宽度
 * @param dst_h     输出高度
 * @param channels  通道数 (通常为3)
 * @param src_step  输入图像每行字节数 (考虑padding)
 * @param dst_step  输出图像每行字节数
 * @param stream    CUDA流
 */
void gpuResize(const unsigned char* src, unsigned char* dst,
               int src_w, int src_h, int dst_w, int dst_h,
               int channels, int src_step, int dst_step,
               cudaStream_t stream = nullptr);

/**
 * GPU双线性插值resize (float版本)
 */
void gpuResizeFloat(const float* src, float* dst,
                    int src_w, int src_h, int dst_w, int dst_h,
                    int channels, int src_step, int dst_step,
                    cudaStream_t stream = nullptr);

//=============================================================================
// 未来可以在这里添加更多GPU kernel函数
// 例如: NV12转BGR, 颜色空间转换, 图像融合等
//=============================================================================

} // namespace Function

#endif // GPU_KERNELS_CUH
