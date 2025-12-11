/**
 * GPU resize_unscale 测试
 * 
 * 对比 CPU 版本和 GPU 版本的 resize_unscale 结果
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include "../trt_function/src/gpu_kernels.cuh"

using namespace Function;

// CPU版本的resize_unscale（从SCRFD复制）
struct CPUScaleParams {
    float ratio;
    int dw;
    int dh;
};

void cpuResizeUnscale(const cv::Mat &mat, cv::Mat &mat_rs,
                      int target_height, int target_width,
                      CPUScaleParams &scale_params)
{
    if (mat.empty()) return;
    int img_height = mat.rows;
    int img_width = mat.cols;

    // 创建黑色画布
    mat_rs = cv::Mat(target_height, target_width, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // 计算缩放比例
    float w_r = (float)target_width / (float)img_width;
    float h_r = (float)target_height / (float)img_height;
    float r = std::min(w_r, h_r);
    
    // 计算resize后的尺寸
    int new_unpad_w = static_cast<int>((float)img_width * r);
    int new_unpad_h = static_cast<int>((float)img_height * r);
    
    // 计算padding
    int pad_w = target_width - new_unpad_w;
    int pad_h = target_height - new_unpad_h;
    int dw = pad_w / 2;
    int dh = pad_h / 2;

    // resize并贴到画布中央
    cv::Mat new_unpad_mat;
    cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // 保存参数
    scale_params.ratio = r;
    scale_params.dw = dw;
    scale_params.dh = dh;
}

int main() {
    std::cout << "=== GPU Resize Unscale Test ===" << std::endl;
    
    // 创建测试图像（模拟不同宽高比的输入）
    cv::Mat test_img(480, 640, CV_8UC3);  // 4:3 宽高比
    cv::RNG rng(12345);
    rng.fill(test_img, cv::RNG::UNIFORM, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    int target_size = 640;  // SCRFD的目标尺寸
    
    std::cout << "Input size: " << test_img.cols << "x" << test_img.rows << std::endl;
    std::cout << "Target size: " << target_size << "x" << target_size << std::endl;
    
    // ========== CPU版本 ==========
    std::cout << "\n--- CPU resize_unscale ---" << std::endl;
    cv::Mat cpu_result;
    CPUScaleParams cpu_params;
    
    double t0 = (double)cv::getTickCount();
    cpuResizeUnscale(test_img, cpu_result, target_size, target_size, cpu_params);
    double t1 = (double)cv::getTickCount();
    double cpu_time = (t1 - t0) / cv::getTickFrequency() * 1000;
    
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "CPU params: ratio=" << cpu_params.ratio 
              << ", dw=" << cpu_params.dw << ", dh=" << cpu_params.dh << std::endl;
    
    // ========== GPU版本 ==========
    std::cout << "\n--- GPU resize_unscale ---" << std::endl;
    
    // 上传到GPU
    cv::cuda::GpuMat gpu_input;
    gpu_input.upload(test_img);
    
    // 分配输出
    cv::cuda::GpuMat gpu_output(target_size, target_size, CV_8UC3);
    GPUScaleParams gpu_params;
    
    // Warmup
    gpuResizeUnscale(gpu_input.ptr<unsigned char>(), gpu_output.ptr<unsigned char>(),
                     gpu_input.cols, gpu_input.rows, target_size, target_size,
                     3, gpu_input.step, gpu_output.step, gpu_params, nullptr);
    cudaDeviceSynchronize();
    
    t0 = (double)cv::getTickCount();
    gpuResizeUnscale(gpu_input.ptr<unsigned char>(), gpu_output.ptr<unsigned char>(),
                     gpu_input.cols, gpu_input.rows, target_size, target_size,
                     3, gpu_input.step, gpu_output.step, gpu_params, nullptr);
    cudaDeviceSynchronize();
    t1 = (double)cv::getTickCount();
    double gpu_time = (t1 - t0) / cv::getTickFrequency() * 1000;
    
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "GPU params: ratio=" << gpu_params.ratio 
              << ", dw=" << gpu_params.dw << ", dh=" << gpu_params.dh 
              << ", new_w=" << gpu_params.new_w << ", new_h=" << gpu_params.new_h << std::endl;
    
    // 下载GPU结果
    cv::Mat gpu_result;
    gpu_output.download(gpu_result);
    
    // ========== 对比结果 ==========
    std::cout << "\n--- Compare Results ---" << std::endl;
    
    // 检查参数是否一致
    bool params_match = (std::abs(cpu_params.ratio - gpu_params.ratio) < 1e-6) &&
                        (cpu_params.dw == gpu_params.dw) &&
                        (cpu_params.dh == gpu_params.dh);
    std::cout << "Scale params match: " << (params_match ? "YES" : "NO") << std::endl;
    
    // 检查图像差异
    double max_diff = cv::norm(cpu_result, gpu_result, cv::NORM_INF);
    cv::Mat diff;
    cv::absdiff(cpu_result, gpu_result, diff);
    double avg_diff = cv::mean(diff)[0];
    
    std::cout << "Max pixel diff: " << max_diff << std::endl;
    std::cout << "Avg pixel diff: " << avg_diff << std::endl;
    
    // 检查padding区域是否都是黑色
    bool padding_ok = true;
    // 检查顶部padding
    if (gpu_params.dh > 0) {
        cv::Mat top_padding = gpu_result(cv::Rect(0, 0, target_size, gpu_params.dh));
        if (cv::countNonZero(top_padding.reshape(1)) != 0) {
            padding_ok = false;
            std::cout << "Top padding not all zeros!" << std::endl;
        }
    }
    // 检查左侧padding
    if (gpu_params.dw > 0) {
        cv::Mat left_padding = gpu_result(cv::Rect(0, 0, gpu_params.dw, target_size));
        if (cv::countNonZero(left_padding.reshape(1)) != 0) {
            padding_ok = false;
            std::cout << "Left padding not all zeros!" << std::endl;
        }
    }
    std::cout << "Padding areas correct: " << (padding_ok ? "YES" : "NO") << std::endl;
    
    // 打印中心区域的像素对比
    int cx = target_size / 2;
    int cy = target_size / 2;
    std::cout << "\nCenter pixel [" << cy << "," << cx << "]:" << std::endl;
    std::cout << "CPU: " << cpu_result.at<cv::Vec3b>(cy, cx) << std::endl;
    std::cout << "GPU: " << gpu_result.at<cv::Vec3b>(cy, cx) << std::endl;
    
    // 总结
    if (max_diff <= 1 && params_match && padding_ok) {
        std::cout << "\n✓ Results match!" << std::endl;
    } else {
        std::cout << "\n✗ Results differ!" << std::endl;
    }
    
    std::cout << "\n=== Test Done ===" << std::endl;
    return 0;
}
