/**
 * GPU预处理测试
 * 
 * 测试 GPUPreprocess 类的功能
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include "../trt_function/src/gpu_preprocess.h"
#include "../trt_function/src/gpu_kernels.cuh"

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
    
    // 创建测试图像 - 使用不同尺寸来测试resize
    cv::Mat test_img(480, 640, CV_8UC3);
    cv::RNG rng(12345);  // 固定种子
    rng.fill(test_img, cv::RNG::UNIFORM, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    // 预处理参数
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
    
    // ========== 分步验证 ==========
    std::cout << "\n--- Step-by-step Verification ---" << std::endl;
    
    // 验证0: 上传后的数据是否一致
    cv::Mat uploaded_back;
    gpu_img.download(uploaded_back);
    double upload_diff = cv::norm(test_img, uploaded_back, cv::NORM_INF);
    std::cout << "Upload verification diff: " << upload_diff << std::endl;
    std::cout << "Original [0,0]: " << test_img.at<cv::Vec3b>(0,0) << std::endl;
    std::cout << "Uploaded [0,0]: " << uploaded_back.at<cv::Vec3b>(0,0) << std::endl;
    
    // 验证1: resize结果对比 - 使用自定义gpuResize
    cv::cuda::GpuMat gpu_img_fresh;
    gpu_img_fresh.upload(test_img);
    
    cv::Mat cpu_resized;
    cv::resize(test_img, cpu_resized, cv::Size(target_size, target_size));
    
    // 使用自定义的gpuResize（和GPUPreprocess内部一致）
    cv::cuda::GpuMat gpu_resized(target_size, target_size, CV_8UC3);
    Function::gpuResize(gpu_img_fresh.ptr<unsigned char>(), gpu_resized.ptr<unsigned char>(),
                        gpu_img_fresh.cols, gpu_img_fresh.rows, target_size, target_size,
                        3, gpu_img_fresh.step, gpu_resized.step, nullptr);
    cudaDeviceSynchronize();
    
    cv::Mat gpu_resized_cpu;
    gpu_resized.download(gpu_resized_cpu);
    
    double resize_diff = cv::norm(cpu_resized, gpu_resized_cpu, cv::NORM_INF);
    std::cout << "Custom gpuResize max diff: " << resize_diff << std::endl;
    
    // 打印resize后的前几个像素
    std::cout << "CPU resized [0,0]: " << cpu_resized.at<cv::Vec3b>(0,0) << std::endl;
    std::cout << "GPU resized [0,0]: " << gpu_resized_cpu.at<cv::Vec3b>(0,0) << std::endl;
    
    // ========== 对比结果 ==========
    std::cout << "\n--- Compare Final Results ---" << std::endl;
    
    double max_diff = 0;
    double sum_diff = 0;
    int max_diff_idx = 0;
    for (size_t i = 0; i < cpu_result.size(); i++) {
        double diff = std::abs(cpu_result[i] - gpu_result[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        sum_diff += diff;
    }
    double avg_diff = sum_diff / cpu_result.size();
    
    std::cout << "Max diff: " << max_diff << " at index " << max_diff_idx << std::endl;
    std::cout << "Avg diff: " << avg_diff << std::endl;
    
    // 允许resize插值带来的误差（通常在1-2之间）
    if (max_diff < 0.1) {
        std::cout << "\n✓ Results match!" << std::endl;
    } else if (max_diff < 2.0) {
        std::cout << "\n⚠ Results have minor differences (likely due to resize interpolation)" << std::endl;
    } else {
        std::cout << "\n✗ Results differ significantly!" << std::endl;
    }
    
    // 打印前几个值对比
    std::cout << "\nFirst 10 values (Channel 0 / B):" << std::endl;
    std::cout << "CPU: ";
    for (int i = 0; i < 10; i++) std::cout << cpu_result[i] << " ";
    std::cout << std::endl;
    std::cout << "GPU: ";
    for (int i = 0; i < 10; i++) std::cout << gpu_result[i] << " ";
    std::cout << std::endl;
    
    // 打印对应位置的原始像素值
    std::cout << "\nCorresponding resized pixels [0,0-9] B channel:" << std::endl;
    std::cout << "CPU: ";
    for (int i = 0; i < 10; i++) std::cout << (int)cpu_resized.at<cv::Vec3b>(0,i)[0] << " ";
    std::cout << std::endl;
    std::cout << "GPU: ";
    for (int i = 0; i < 10; i++) std::cout << (int)gpu_resized_cpu.at<cv::Vec3b>(0,i)[0] << " ";
    std::cout << std::endl;
    
    // 清理
    cudaFree(gpu_output);
    
    std::cout << "\n=== Test Done ===" << std::endl;
    return 0;
}
