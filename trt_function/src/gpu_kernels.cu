#include "gpu_kernels.cuh"

namespace Function {

//=============================================================================
// GPU Resize - 双线性插值实现
//=============================================================================

/**
 * CUDA kernel: 双线性插值resize (uint8版本)
 * 
 * OpenCV的INTER_LINEAR实现逻辑:
 * 1. 计算目标像素在源图像中的浮点坐标
 * 2. 使用双线性插值计算像素值
 * 
 * 坐标映射公式 (和OpenCV一致):
 *   src_x = (dst_x + 0.5) * scale_x - 0.5
 *   src_y = (dst_y + 0.5) * scale_y - 0.5
 * 其中 scale_x = src_w / dst_w, scale_y = src_h / dst_h
 */
__global__ void resizeBilinearKernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    int channels,
    int src_step, int dst_step,
    float scale_x, float scale_y)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_w || dst_y >= dst_h) return;
    
    // 计算源图像中的浮点坐标 (OpenCV的坐标映射方式)
    float src_xf = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_yf = (dst_y + 0.5f) * scale_y - 0.5f;
    
    // 计算四个邻近像素的整数坐标
    int x0 = (int)floorf(src_xf);
    int y0 = (int)floorf(src_yf);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // 计算插值权重
    float wx = src_xf - x0;
    float wy = src_yf - y0;
    
    // 边界处理 (clamp到有效范围)
    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));
    
    // 计算四个邻近像素的地址
    const unsigned char* p00 = src + y0 * src_step + x0 * channels;
    const unsigned char* p01 = src + y0 * src_step + x1 * channels;
    const unsigned char* p10 = src + y1 * src_step + x0 * channels;
    const unsigned char* p11 = src + y1 * src_step + x1 * channels;
    
    // 输出地址
    unsigned char* out = dst + dst_y * dst_step + dst_x * channels;
    
    // 对每个通道进行双线性插值
    for (int c = 0; c < channels; c++) {
        float v00 = p00[c];
        float v01 = p01[c];
        float v10 = p10[c];
        float v11 = p11[c];
        
        // 双线性插值
        float v0 = v00 * (1.0f - wx) + v01 * wx;  // 上边
        float v1 = v10 * (1.0f - wx) + v11 * wx;  // 下边
        float v = v0 * (1.0f - wy) + v1 * wy;     // 最终值
        
        // 四舍五入并clamp到[0, 255]
        out[c] = (unsigned char)min(255.0f, max(0.0f, v + 0.5f));
    }
}

/**
 * CUDA kernel: 双线性插值resize (float版本)
 */
__global__ void resizeBilinearFloatKernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    int channels,
    int src_step, int dst_step,
    float scale_x, float scale_y)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_w || dst_y >= dst_h) return;
    
    // 计算源图像中的浮点坐标
    float src_xf = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_yf = (dst_y + 0.5f) * scale_y - 0.5f;
    
    // 计算四个邻近像素的整数坐标
    int x0 = (int)floorf(src_xf);
    int y0 = (int)floorf(src_yf);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // 计算插值权重
    float wx = src_xf - x0;
    float wy = src_yf - y0;
    
    // 边界处理
    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));
    
    // step是字节数，需要转换为float元素数
    int src_stride = src_step / sizeof(float);
    int dst_stride = dst_step / sizeof(float);
    
    // 计算四个邻近像素的地址
    const float* p00 = src + y0 * src_stride + x0 * channels;
    const float* p01 = src + y0 * src_stride + x1 * channels;
    const float* p10 = src + y1 * src_stride + x0 * channels;
    const float* p11 = src + y1 * src_stride + x1 * channels;
    
    // 输出地址
    float* out = dst + dst_y * dst_stride + dst_x * channels;
    
    // 对每个通道进行双线性插值
    for (int c = 0; c < channels; c++) {
        float v00 = p00[c];
        float v01 = p01[c];
        float v10 = p10[c];
        float v11 = p11[c];
        
        float v0 = v00 * (1.0f - wx) + v01 * wx;
        float v1 = v10 * (1.0f - wx) + v11 * wx;
        out[c] = v0 * (1.0f - wy) + v1 * wy;
    }
}

void gpuResize(const unsigned char* src, unsigned char* dst,
               int src_w, int src_h, int dst_w, int dst_h,
               int channels, int src_step, int dst_step,
               cudaStream_t stream)
{
    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;
    
    // 使用16x16的block
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);
    
    resizeBilinearKernel<<<grid, block, 0, stream>>>(
        src, dst, src_w, src_h, dst_w, dst_h,
        channels, src_step, dst_step, scale_x, scale_y);
}

void gpuResizeFloat(const float* src, float* dst,
                    int src_w, int src_h, int dst_w, int dst_h,
                    int channels, int src_step, int dst_step,
                    cudaStream_t stream)
{
    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;
    
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);
    
    resizeBilinearFloatKernel<<<grid, block, 0, stream>>>(
        src, dst, src_w, src_h, dst_w, dst_h,
        channels, src_step, dst_step, scale_x, scale_y);
}

} // namespace Function
