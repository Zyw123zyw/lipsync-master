/**
 * GPU预处理测试
 * 
 * 测试 GPUPreprocess 类的功能
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include "../trt_function/src/gpu_preprocess.h"

using namespace Function;

// CPU版本预处理（对照，和原模型一致）
std::vector<float> cpuPreprocess(const cv::Mat& mat, int target_size, 
                                  const float* mean, const float* norm) {
    cv::Mat canva = mat.clone();
    cv::resize(canva, canva, cv::Size(target_size, target_size));
    
    // normalize (和原模型的normalize_inplace一致)
    canva.convertTo(canva, CV_32FC3);
    for (int i = 0; i < canva.rows; i++) {
        cv::Vec3f* p = canva.ptr<cv::Vec3f>(i);
        for (int j = 0; j < canva.cols; j++) {
            p[j][0] = (p[j][0] - mean[0]) * norm[0];  // B
            p[j][1] = (p[j][1] - mean[1]) * norm[1];  // G
            p[j][2] = (p[j][2] - mean[2]) * norm[2];  // R
        }
    }
    
    // HWC -> CHW (和原模型的trans2chw一致)
    std::vector<cv::Mat> channels(3);
    cv::split(canva, channels);
    
    std::vector<float> result(target_size * target_size * 3);
    for (int c = 0; c < 3; c++) {
        memcpy(result.data() + c * target_size * target_size,
               channels[c].data,
               target_size * target_size * sizeof(float));
    }
    
    return result;
}

int main() {
    std::cout << "=== GPU Preprocess Test ===" << std::endl;
    
    // 创建测试图像
    cv::Mat test_img(480, 640, CV_8UC3);
    cv::randu(test_img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    // 预处理参数（使用简单参数便于调试）
    // 使用GCFSR的参数：mean=128, norm=1/128
    int target_size = 256;
    float mean[3] = {128.f, 128.f, 128.f};
    float norm[3] = {1.f/128.f, 1.f/128.f, 1.f/128.f};
    
    std::cout << "Input size: " << test_img.cols << "x" << test_img.rows << std::endl;
    std::cout << "Target size: " << target_size << std::endl;
    
    // ========== CPU预处理 ==========
    std::cout << "\n--- CPU Preprocess ---" << std::endl;
    double t0 = (double)cv::getTickCount();
    
    std::vector<float> cpu_result = cpuPreprocess(test_img, target_size, mean, norm);
    
    double t1 = (double)cv::getTickCount();
    double cpu_time = (t1 - t0) / cv::getTickFrequency() * 1000;
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    
    // ========== GPU预处理 ==========
    std::cout << "\n--- GPU Preprocess ---" << std::endl;
    
    // 上传到GPU
    cv::cuda::GpuMat gpu_img;
    gpu_img.upload(test_img);
    
    // 分配GPU输出buffer
    float* gpu_output;
    size_t output_size = target_size * target_size * 3 * sizeof(float);
    cudaMalloc(&gpu_output, output_size);
    
    // GPU预处理
    GPUPreprocess preprocessor;
    
    // Warmup（第一次调用有初始化开销）
    preprocessor.process(gpu_img, gpu_output, target_size, mean, norm);
    cudaDeviceSynchronize();
    
    t0 = (double)cv::getTickCount();
    preprocessor.process(gpu_img, gpu_output, target_size, mean, norm);
    cudaDeviceSynchronize();
    t1 = (double)cv::getTickCount();
    double gpu_time = (t1 - t0) / cv::getTickFrequency() * 1000;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    
    // 下载GPU结果进行对比
    std::vector<float> gpu_result(target_size * target_size * 3);
    cudaMemcpy(gpu_result.data(), gpu_output, output_size, cudaMemcpyDeviceToHost);
    
    // ========== 对比结果 ==========
    std::cout << "\n--- Compare Results ---" << std::endl;
    
    double max_diff = 0;
    double sum_diff = 0;
    for (size_t i = 0; i < cpu_result.size(); i++) {
        double diff = std::abs(cpu_result[i] - gpu_result[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
    }
    double avg_diff = sum_diff / cpu_result.size();
    
    std::cout << "Max diff: " << max_diff << std::endl;
    std::cout << "Avg diff: " << avg_diff << std::endl;
    
    if (max_diff < 1e-3) {
        std::cout << "\n✓ Results match!" << std::endl;
    } else {
        std::cout << "\n✗ Results differ!" << std::endl;
    }
    
    // 打印前几个值对比
    std::cout << "\nFirst 10 values:" << std::endl;
    std::cout << "CPU: ";
    for (int i = 0; i < 10; i++) std::cout << cpu_result[i] << " ";
    std::cout << std::endl;
    std::cout << "GPU: ";
    for (int i = 0; i < 10; i++) std::cout << gpu_result[i] << " ";
    std::cout << std::endl;
    
    // 清理
    cudaFree(gpu_output);
    
    std::cout << "\n=== Test Done ===" << std::endl;
    return 0;
}
