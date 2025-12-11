/**
 * SCRFD GPU检测测试
 * 
 * 对比 CPU detect() 和 GPU detectGPU() 的结果
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include "../trt_function/src/scrfd.h"

using namespace Function;

int main(int argc, char* argv[]) {
    std::cout << "=== SCRFD GPU Detection Test ===" << std::endl;
    
    // 模型路径
    std::string model_path = "/mnt/data/vision-devel/zhangyiwei/lipsync-master/models/scrfd_2.5g_shape640x640.engine";
    if (argc > 1) {
        model_path = argv[1];
    }
    std::cout << "Model: " << model_path << std::endl;
    
    // 测试图像路径（可以用真实图像）
    std::string image_path = "";
    if (argc > 2) {
        image_path = argv[2];
    }
    
    // 创建测试图像
    cv::Mat test_img;
    if (!image_path.empty()) {
        test_img = cv::imread(image_path);
        if (test_img.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return -1;
        }
    } else {
        // 使用随机图像测试（主要测试流程是否正确）
        test_img = cv::Mat(480, 640, CV_8UC3);
        cv::RNG rng(12345);
        rng.fill(test_img, cv::RNG::UNIFORM, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        std::cout << "Using random test image (no real faces)" << std::endl;
    }
    std::cout << "Image size: " << test_img.cols << "x" << test_img.rows << std::endl;
    
    // 初始化SCRFD
    std::cout << "\nInitializing SCRFD..." << std::endl;
    SCRFD detector(model_path);
    detector.initialize_handler();
    detector.warmup();
    std::cout << "SCRFD initialized." << std::endl;
    
    // 上传图像到GPU
    cv::cuda::GpuMat gpu_img;
    gpu_img.upload(test_img);
    
    float score_threshold = 0.5f;
    float iou_threshold = 0.45f;
    unsigned int topk = 5;
    
    // ========== CPU检测 ==========
    std::cout << "\n--- CPU detect() ---" << std::endl;
    std::vector<DetectionBox> cpu_boxes;
    
    double t0 = (double)cv::getTickCount();
    detector.detect(test_img, cpu_boxes, score_threshold, iou_threshold, topk);
    double t1 = (double)cv::getTickCount();
    double cpu_time = (t1 - t0) / cv::getTickFrequency() * 1000;
    
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "CPU detected " << cpu_boxes.size() << " faces" << std::endl;
    for (size_t i = 0; i < cpu_boxes.size(); i++) {
        std::cout << "  Face " << i << ": [" << cpu_boxes[i].x1 << ", " << cpu_boxes[i].y1 
                  << ", " << cpu_boxes[i].w << ", " << cpu_boxes[i].h 
                  << "] score=" << cpu_boxes[i].score << std::endl;
    }
    
    // ========== GPU检测 ==========
    std::cout << "\n--- GPU detectGPU() ---" << std::endl;
    std::vector<DetectionBox> gpu_boxes;
    
    // Warmup
    detector.detectGPU(gpu_img, gpu_boxes, score_threshold, iou_threshold, topk);
    gpu_boxes.clear();
    
    t0 = (double)cv::getTickCount();
    detector.detectGPU(gpu_img, gpu_boxes, score_threshold, iou_threshold, topk);
    t1 = (double)cv::getTickCount();
    double gpu_time = (t1 - t0) / cv::getTickFrequency() * 1000;
    
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "GPU detected " << gpu_boxes.size() << " faces" << std::endl;
    for (size_t i = 0; i < gpu_boxes.size(); i++) {
        std::cout << "  Face " << i << ": [" << gpu_boxes[i].x1 << ", " << gpu_boxes[i].y1 
                  << ", " << gpu_boxes[i].w << ", " << gpu_boxes[i].h 
                  << "] score=" << gpu_boxes[i].score << std::endl;
    }
    
    // ========== 对比结果 ==========
    std::cout << "\n--- Compare Results ---" << std::endl;
    
    bool match = true;
    if (cpu_boxes.size() != gpu_boxes.size()) {
        std::cout << "Detection count mismatch: CPU=" << cpu_boxes.size() 
                  << " GPU=" << gpu_boxes.size() << std::endl;
        match = false;
    } else {
        for (size_t i = 0; i < cpu_boxes.size(); i++) {
            float diff_x = std::abs(cpu_boxes[i].x1 - gpu_boxes[i].x1);
            float diff_y = std::abs(cpu_boxes[i].y1 - gpu_boxes[i].y1);
            float diff_w = std::abs(cpu_boxes[i].w - gpu_boxes[i].w);
            float diff_h = std::abs(cpu_boxes[i].h - gpu_boxes[i].h);
            float diff_score = std::abs(cpu_boxes[i].score - gpu_boxes[i].score);
            
            // 允许小的数值差异（由于浮点精度）
            if (diff_x > 1.0f || diff_y > 1.0f || diff_w > 1.0f || diff_h > 1.0f || diff_score > 0.01f) {
                std::cout << "Face " << i << " differs:" << std::endl;
                std::cout << "  CPU: [" << cpu_boxes[i].x1 << ", " << cpu_boxes[i].y1 
                          << ", " << cpu_boxes[i].w << ", " << cpu_boxes[i].h << "]" << std::endl;
                std::cout << "  GPU: [" << gpu_boxes[i].x1 << ", " << gpu_boxes[i].y1 
                          << ", " << gpu_boxes[i].w << ", " << gpu_boxes[i].h << "]" << std::endl;
                match = false;
            }
        }
    }
    
    if (match) {
        std::cout << "\n✓ Results match!" << std::endl;
    } else if (cpu_boxes.empty() && gpu_boxes.empty()) {
        std::cout << "\n✓ Both detected 0 faces (expected for random image)" << std::endl;
    } else {
        std::cout << "\n✗ Results differ!" << std::endl;
    }
    
    std::cout << "\nSpeedup: " << cpu_time / gpu_time << "x" << std::endl;
    
    std::cout << "\n=== Test Done ===" << std::endl;
    return 0;
}
