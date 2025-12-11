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

// CPU版本预处理（对照）
std::vector<float> cpuPreprocess(const cv::Mat& mat, int target_size, 
                                  const float* mean, const float* norm) {
    cv::Mat canva = mat.clone();
    cv::resize(canva, canva, cv::Size(target_size, target_size));
    
    // normalize
    canva.convertTo(canva, CV_32FC3);
    std::vector<cv::Mat> channels(3);
    cv::split(canva, channels);
    
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) * norm[i];
    }
    
    // HWC -> CHW
    std::vector<float> result(target_size * target_size * 3);
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < target_size; h++) {
            for (int w = 0; w < target_size; w++) {
                result[c * target_size * target_size + h * target_size + w] = 
                    channels[c].at<float>(h, w);
            }
        }
    }
    
    return result;
}

int main() {
    std::cout << "=== GPU Preprocess Test ===" << std::endl;
    
    // 创建测试图像
    cv::Mat test_img(480, 640, CV_8UC3);
    cv::randu(test_img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    // 预处理参数（模拟PIPNet的参数）
    int target_size = 256;
    float mean[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    float norm[3] = {1.0f / (0.229f * 255.f), 1.0f / (0.224f * 255.f), 1.0f / (0.225f * 255.f)};
    
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
    
    t0 = (double)cv::getTickCount();
    preprocessor.process(gpu_img, gpu_output, target_size, mean, norm);
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
