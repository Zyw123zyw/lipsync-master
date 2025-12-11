#include "gpu_preprocess.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace Function {

GPUPreprocess::GPUPreprocess() {
    channels_.resize(3);
}

GPUPreprocess::~GPUPreprocess() {
    resized_.release();
    float_mat_.release();
    for (auto& ch : channels_) {
        ch.release();
    }
}

void GPUPreprocess::process(const cv::cuda::GpuMat& input,
                            float* output,
                            int target_size,
                            const float* mean,
                            const float* norm,
                            cudaStream_t stream) {
    if (input.empty()) return;
    
    // 设置CUDA流
    if (stream) {
        cv_stream_ = cv::cuda::StreamAccessor::wrapStream(stream);
    }
    
    // 1. Resize到目标尺寸
    cv::cuda::resize(input, resized_, cv::Size(target_size, target_size), 
                     0, 0, cv::INTER_LINEAR, cv_stream_);
    
    // 2. 转换为float类型 (0-255 -> 0.0-255.0)
    resized_.convertTo(float_mat_, CV_32FC3, 1.0, 0, cv_stream_);
    
    // 3. 先分离通道，再对每个通道单独做normalize
    // 这样可以避免cv::Scalar在CUDA操作中的通道顺序问题
    cv::cuda::split(float_mat_, channels_, cv_stream_);
    
    // 4. 对每个通道做 (pixel - mean) * norm
    for (int c = 0; c < 3; c++) {
        cv::cuda::subtract(channels_[c], cv::Scalar(mean[c]), channels_[c], cv::noArray(), -1, cv_stream_);
        cv::cuda::multiply(channels_[c], cv::Scalar(norm[c]), channels_[c], 1.0, -1, cv_stream_);
    }
    
    // 5. 拷贝到输出buffer (CHW格式)
    // 输出布局: [C, H, W] = [3, target_size, target_size]
    size_t channel_size = target_size * target_size * sizeof(float);
    
    // 等待前面的操作完成
    cv_stream_.waitForCompletion();
    
    // 按CHW顺序拷贝：B, G, R (OpenCV是BGR顺序)
    for (int c = 0; c < 3; c++) {
        if (channels_[c].isContinuous()) {
            cudaMemcpy(output + c * target_size * target_size, 
                       channels_[c].ptr<float>(), 
                       channel_size, 
                       cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpy2D(output + c * target_size * target_size,
                         target_size * sizeof(float),
                         channels_[c].ptr<float>(),
                         channels_[c].step,
                         target_size * sizeof(float),
                         target_size,
                         cudaMemcpyDeviceToDevice);
        }
    }
}

void GPUPreprocess::processWithROI(const cv::cuda::GpuMat& input,
                                   const cv::Rect& roi,
                                   float* output,
                                   int target_size,
                                   const float* mean,
                                   const float* norm,
                                   cudaStream_t stream) {
    if (input.empty()) return;
    
    // 裁剪ROI区域
    cv::cuda::GpuMat cropped = input(roi);
    
    // 调用通用处理
    process(cropped, output, target_size, mean, norm, stream);
}

} // namespace Function
