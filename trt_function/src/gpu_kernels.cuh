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
// GPU Resize Unscale - 保持宽高比resize + padding
//=============================================================================

/**
 * resize_unscale的缩放参数
 */
struct GPUScaleParams {
    float ratio;    // 缩放比例
    int dw;         // 左侧padding
    int dh;         // 顶部padding
    int new_w;      // resize后的宽度（不含padding）
    int new_h;      // resize后的高度（不含padding）
};

/**
 * GPU版本的resize_unscale
 * 
 * 保持宽高比resize，然后padding到目标尺寸（黑色填充）
 * 和SCRFD的CPU版本resize_unscale逻辑一致
 * 
 * @param src           输入图像数据 (HWC, BGR, uint8)
 * @param dst           输出图像数据 (HWC, BGR, uint8)，需要预先分配
 * @param src_w         输入宽度
 * @param src_h         输入高度
 * @param dst_w         目标宽度
 * @param dst_h         目标高度
 * @param channels      通道数
 * @param src_step      输入每行字节数
 * @param dst_step      输出每行字节数
 * @param scale_params  输出缩放参数（用于后处理坐标还原）
 * @param stream        CUDA流
 */
void gpuResizeUnscale(const unsigned char* src, unsigned char* dst,
                      int src_w, int src_h, int dst_w, int dst_h,
                      int channels, int src_step, int dst_step,
                      GPUScaleParams& scale_params,
                      cudaStream_t stream = nullptr);

} // namespace Function

#endif // GPU_KERNELS_CUH
