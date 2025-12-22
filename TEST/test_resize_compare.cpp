#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <cmath>
#include "../trt_function/src/gpu_kernels.cuh"

using namespace std;

/**
 * 计算两个图像的差异
 */
double calculateDifference(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        cerr << "图像尺寸或类型不一致！" << endl;
        return -1;
    }
    
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    
    // 计算平均绝对差
    cv::Scalar meanDiff = cv::mean(diff);
    double avgDiff = (meanDiff[0] + meanDiff[1] + meanDiff[2]) / 3.0;
    
    // 计算最大差异
    double minVal, maxVal;
    cv::minMaxLoc(diff, &minVal, &maxVal);
    
    cout << "平均差异: " << avgDiff << " (0-255)" << endl;
    cout << "最大差异: " << maxVal << " (0-255)" << endl;
    
    return avgDiff;
}

/**
 * 逐像素比较
 */
void pixelWiseComparison(const cv::Mat& img1, const cv::Mat& img2, int numSamples = 10) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        return;
    }
    
    cout << "\n随机采样 " << numSamples << " 个像素进行对比：" << endl;
    cout << "格式: (x, y) -> OpenCV: [B,G,R] | Custom: [B,G,R]" << endl;
    
    cv::RNG rng(12345);
    for (int i = 0; i < numSamples; i++) {
        int x = rng.uniform(0, img1.cols);
        int y = rng.uniform(0, img1.rows);
        
        cv::Vec3b pix1 = img1.at<cv::Vec3b>(y, x);
        cv::Vec3b pix2 = img2.at<cv::Vec3b>(y, x);
        
        cout << "(" << x << ", " << y << ") -> "
             << "OpenCV: [" << (int)pix1[0] << "," << (int)pix1[1] << "," << (int)pix1[2] << "] | "
             << "Custom: [" << (int)pix2[0] << "," << (int)pix2[1] << "," << (int)pix2[2] << "]";
        
        int diff = abs(pix1[0] - pix2[0]) + abs(pix1[1] - pix2[1]) + abs(pix1[2] - pix2[2]);
        if (diff > 3) {
            cout << " <-- 差异较大！";
        }
        cout << endl;
    }
}

int main(int argc, char** argv) {
    cout << "========================================" << endl;
    cout << "  Resize 函数对比测试" << endl;
    cout << "========================================" << endl;
    
    // 参数解析
    string input_path = (argc > 1) ? argv[1] : "../data/test.jpg";
    int target_width = (argc > 2) ? atoi(argv[2]) : 810;
    int target_height = (argc > 3) ? atoi(argv[3]) : 1440;
    
    // 1. 读取测试图像
    cout << "\n[1] 读取测试图像: " << input_path << endl;
    cv::Mat cpu_img = cv::imread(input_path);
    if (cpu_img.empty()) {
        cerr << "错误: 无法读取图像 " << input_path << endl;
        cerr << "请提供有效的图像路径，或生成测试图像" << endl;
        
        // 生成测试图像（彩色渐变）
        cout << "\n生成 1080x1920 的测试图像..." << endl;
        cpu_img = cv::Mat(1920, 1080, CV_8UC3);
        for (int y = 0; y < cpu_img.rows; y++) {
            for (int x = 0; x < cpu_img.cols; x++) {
                cpu_img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    (x * 255) / cpu_img.cols,          // B
                    (y * 255) / cpu_img.rows,          // G
                    ((x + y) * 255) / (cpu_img.cols + cpu_img.rows) // R
                );
            }
        }
    }
    
    cout << "原始图像尺寸: " << cpu_img.cols << "x" << cpu_img.rows << endl;
    cout << "目标尺寸: " << target_width << "x" << target_height << endl;
    
    // 2. 上传到 GPU
    cout << "\n[2] 上传图像到 GPU..." << endl;
    cv::cuda::GpuMat gpu_src;
    gpu_src.upload(cpu_img);
    
    // 2.5 预热 GPU (避免第一次调用的初始化开销)
    cout << "\n[2.5] 预热 GPU..." << endl;
    cv::cuda::GpuMat gpu_warmup;
    // OpenCV 预热
    cv::cuda::resize(gpu_src, gpu_warmup, cv::Size(target_width, target_height));
    cudaDeviceSynchronize();
    // 自定义 kernel 预热
    cv::cuda::GpuMat gpu_warmup2;
    gpu_warmup2.create(target_height, target_width, gpu_src.type());
    Function::gpuResize(
        gpu_src.data, gpu_warmup2.data,
        gpu_src.cols, gpu_src.rows,
        target_width, target_height,
        gpu_src.channels(),
        gpu_src.step, gpu_warmup2.step,
        nullptr
    );
    cudaDeviceSynchronize();
    cout << "预热完成" << endl;
    
    // 3. 方法1: 使用 cv::cuda::resize
    cout << "\n[3] 测试 cv::cuda::resize (预热后)..." << endl;
    cv::cuda::GpuMat gpu_dst_opencv;
    
    const int TEST_RUNS = 10;  // 测试 10 次取平均
    double total_time_opencv = 0;
    
    for (int i = 0; i < TEST_RUNS; i++) {
        auto start1 = cv::getTickCount();
        cv::cuda::resize(gpu_src, gpu_dst_opencv, cv::Size(target_width, target_height));
        cudaDeviceSynchronize();  // 确保完成
        auto end1 = cv::getTickCount();
        total_time_opencv += (end1 - start1) * 1000.0 / cv::getTickFrequency();
    }
    double time_opencv = total_time_opencv / TEST_RUNS;
    
    cout << "cv::cuda::resize 平均耗时 (" << TEST_RUNS << " 次): " << time_opencv << " ms" << endl;
    
    // 4. 方法2: 使用自定义 gpuResize
    cout << "\n[4] 测试 Function::gpuResize..." << endl;
    cv::cuda::GpuMat gpu_dst_custom;
    gpu_dst_custom.create(target_height, target_width, gpu_src.type());
    
    double total_time_custom = 0;
    
    for (int i = 0; i < TEST_RUNS; i++) {
        auto start2 = cv::getTickCount();
        Function::gpuResize(
            gpu_src.data, gpu_dst_custom.data,
            gpu_src.cols, gpu_src.rows,
            target_width, target_height,
            gpu_src.channels(),
            gpu_src.step, gpu_dst_custom.step,
            nullptr
        );
        cudaDeviceSynchronize();  // 确保完成
        auto end2 = cv::getTickCount();
        total_time_custom += (end2 - start2) * 1000.0 / cv::getTickFrequency();
    }
    double time_custom = total_time_custom / TEST_RUNS;
    
    cout << "Function::gpuResize 平均耗时 (" << TEST_RUNS << " 次): " << time_custom << " ms" << endl;
    
    // 5. 下载结果到 CPU 进行对比
    cout << "\n[5] 下载结果并对比..." << endl;
    cv::Mat cpu_result_opencv, cpu_result_custom;
    gpu_dst_opencv.download(cpu_result_opencv);
    gpu_dst_custom.download(cpu_result_custom);
    
    // 6. 计算差异
    cout << "\n[6] 计算差异..." << endl;
    double avgDiff = calculateDifference(cpu_result_opencv, cpu_result_custom);
    
    // 7. 逐像素对比
    pixelWiseComparison(cpu_result_opencv, cpu_result_custom, 20);
    
    // 8. 性能对比
    cout << "\n========================================" << endl;
    cout << "  性能对比" << endl;
    cout << "========================================" << endl;
    cout << "cv::cuda::resize     : " << time_opencv << " ms" << endl;
    cout << "Function::gpuResize  : " << time_custom << " ms" << endl;
    cout << "加速比               : " << (time_opencv / time_custom) << "x" << endl;
    
    // 9. 结果判定
    cout << "\n========================================" << endl;
    cout << "  结果判定" << endl;
    cout << "========================================" << endl;
    if (avgDiff < 1.0) {
        cout << "✅ 结果一致！平均差异 < 1.0，可以安全替换。" << endl;
    } else if (avgDiff < 2.0) {
        cout << "✅ 结果基本一致！平均差异 < 2.0，差异在可接受范围内。" << endl;
    } else {
        cout << "⚠️  结果存在差异！平均差异 = " << avgDiff << "，需要检查实现。" << endl;
    }
    
    // 10. 保存对比图像（可选）
    cout << "\n[7] 保存结果图像..." << endl;
    cv::imwrite("resize_opencv.jpg", cpu_result_opencv);
    cv::imwrite("resize_custom.jpg", cpu_result_custom);
    
    cv::Mat diff_img;
    cv::absdiff(cpu_result_opencv, cpu_result_custom, diff_img);
    diff_img *= 10;  // 放大差异以便观察
    cv::imwrite("resize_diff.jpg", diff_img);
    
    cout << "已保存: resize_opencv.jpg, resize_custom.jpg, resize_diff.jpg" << endl;
    
    return 0;
}
