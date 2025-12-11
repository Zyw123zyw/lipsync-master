#ifndef GPU_PREPROCESS_H
#define GPU_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>

namespace Function {

/**
 * GPU预处理类
 * 
 * 功能：在GPU上完成 resize + normalize + HWC→CHW 转换
 * 输出直接写入TensorRT的输入buffer，避免H2D传输
 */
class GPUPreprocess {
public:
    GPUPreprocess();
    ~GPUPreprocess();

    /**
     * GPU预处理：resize + normalize + HWC→CHW
     * 
     * @param input       输入GpuMat (BGR, CV_8UC3)
     * @param output      输出设备指针 (TensorRT输入buffer)
     * @param target_size 目标尺寸 (正方形)
     * @param mean        均值 [B, G, R] (3个值)
     * @param norm        归一化系数 [B, G, R] (3个值)
     * @param stream      CUDA流 (可选)
     */
    void process(const cv::cuda::GpuMat& input, 
                 float* output,
                 int target_size,
                 const float* mean,
                 const float* norm,
                 cudaStream_t stream = nullptr);

    /**
     * GPU预处理（带ROI裁剪）
     * 
     * @param input       输入GpuMat (BGR, CV_8UC3)
     * @param roi         裁剪区域
     * @param output      输出设备指针
     * @param target_size 目标尺寸
     * @param mean        均值
     * @param norm        归一化系数
     * @param stream      CUDA流
     */
    void processWithROI(const cv::cuda::GpuMat& input,
                        const cv::Rect& roi,
                        float* output,
                        int target_size,
                        const float* mean,
                        const float* norm,
                        cudaStream_t stream = nullptr);

private:
    // 内部缓存，避免重复分配
    cv::cuda::GpuMat resized_;
    cv::cuda::GpuMat float_mat_;
    cv::cuda::GpuMat normalized_;
    
    // 分离的通道
    std::vector<cv::cuda::GpuMat> channels_;
    
    // CUDA流
    cv::cuda::Stream cv_stream_;
};

} // namespace Function

#endif // GPU_PREPROCESS_H
