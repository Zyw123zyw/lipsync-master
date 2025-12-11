/**
 * PIPNet GPU vs CPU 对比测试
 * 
 * 测试内容：
 * 1. 加载一张图片，模拟人脸区域
 * 2. 分别调用 CPU predict() 和 GPU predictGPU()
 * 3. 对比68个关键点坐标是否一致
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "../trt_function/src/pipnet.h"

using namespace Function;

int main(int argc, char** argv)
{
    std::string model_path = "/root/models_8.9/pipnet.engine";
    std::string image_path = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/input/onepeople.png";
    
    if (argc > 1) model_path = argv[1];
    if (argc > 2) image_path = argv[2];
    
    std::cout << "=== PIPNet GPU vs CPU Test ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    
    // 加载图片
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    
    // 模拟人脸区域（取图片中心区域作为人脸）
    int face_size = std::min(img.cols, img.rows) / 2;
    int x = (img.cols - face_size) / 2;
    int y = (img.rows - face_size) / 2;
    cv::Rect face_rect(x, y, face_size, face_size);
    
    std::cout << "Face rect: [" << face_rect.x << ", " << face_rect.y 
              << ", " << face_rect.width << ", " << face_rect.height << "]" << std::endl;
    
    // 裁剪人脸区域
    cv::Mat face_img = img(face_rect).clone();
    
    // 上传到GPU
    cv::cuda::GpuMat gpu_face;
    gpu_face.upload(face_img);
    
    // 加载模型
    std::cout << "\nLoading PIPNet model..." << std::endl;
    PIPNet pipnet(model_path);
    pipnet.initialize_handler();  // 必须先初始化TensorRT
    pipnet.warmup();
    std::cout << "Model loaded." << std::endl;
    
    // Warmup runs
    std::cout << "\n--- Warmup (5 runs each) ---" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::vector<cv::Point2i> tmp;
        pipnet.predict(face_img, face_rect, tmp);
        tmp.clear();
        pipnet.predictGPU(gpu_face, face_rect, tmp);
    }
    std::cout << "Warmup done." << std::endl;
    
    // CPU预测（多次取平均）
    std::cout << "\n--- CPU Predict (10 runs) ---" << std::endl;
    std::vector<cv::Point2i> landmarks_cpu;
    const int num_runs = 10;
    
    double t0 = (double)cv::getTickCount();
    for (int i = 0; i < num_runs; i++) {
        landmarks_cpu.clear();
        pipnet.predict(face_img, face_rect, landmarks_cpu);
    }
    double cpu_time = ((double)cv::getTickCount() - t0) / cv::getTickFrequency() * 1000 / num_runs;
    
    std::cout << "CPU avg time: " << cpu_time << " ms" << std::endl;
    std::cout << "Landmarks count: " << landmarks_cpu.size() << std::endl;
    
    // GPU预测（多次取平均）
    std::cout << "\n--- GPU Predict (10 runs) ---" << std::endl;
    std::vector<cv::Point2i> landmarks_gpu;
    
    t0 = (double)cv::getTickCount();
    for (int i = 0; i < num_runs; i++) {
        landmarks_gpu.clear();
        pipnet.predictGPU(gpu_face, face_rect, landmarks_gpu);
    }
    double gpu_time = ((double)cv::getTickCount() - t0) / cv::getTickFrequency() * 1000 / num_runs;
    
    std::cout << "GPU avg time: " << gpu_time << " ms" << std::endl;
    std::cout << "Landmarks count: " << landmarks_gpu.size() << std::endl;
    
    // 对比结果
    std::cout << "\n--- Compare Results ---" << std::endl;
    
    if (landmarks_cpu.size() != landmarks_gpu.size()) {
        std::cerr << "ERROR: Landmark count mismatch!" << std::endl;
        return -1;
    }
    
    int max_diff_x = 0, max_diff_y = 0;
    int total_diff_x = 0, total_diff_y = 0;
    int diff_count = 0;
    
    for (size_t i = 0; i < landmarks_cpu.size(); i++) {
        int dx = std::abs(landmarks_cpu[i].x - landmarks_gpu[i].x);
        int dy = std::abs(landmarks_cpu[i].y - landmarks_gpu[i].y);
        
        max_diff_x = std::max(max_diff_x, dx);
        max_diff_y = std::max(max_diff_y, dy);
        total_diff_x += dx;
        total_diff_y += dy;
        
        if (dx > 0 || dy > 0) {
            diff_count++;
        }
    }
    
    float avg_diff_x = (float)total_diff_x / landmarks_cpu.size();
    float avg_diff_y = (float)total_diff_y / landmarks_cpu.size();
    
    std::cout << "Max diff: x=" << max_diff_x << ", y=" << max_diff_y << std::endl;
    std::cout << "Avg diff: x=" << avg_diff_x << ", y=" << avg_diff_y << std::endl;
    std::cout << "Different points: " << diff_count << "/" << landmarks_cpu.size() << std::endl;
    
    // 打印前10个关键点对比
    std::cout << "\nFirst 10 landmarks comparison:" << std::endl;
    std::cout << "Index\tCPU(x,y)\t\tGPU(x,y)\t\tDiff" << std::endl;
    for (int i = 0; i < 10 && i < (int)landmarks_cpu.size(); i++) {
        int dx = landmarks_cpu[i].x - landmarks_gpu[i].x;
        int dy = landmarks_cpu[i].y - landmarks_gpu[i].y;
        std::cout << i << "\t(" << landmarks_cpu[i].x << "," << landmarks_cpu[i].y << ")\t\t"
                  << "(" << landmarks_gpu[i].x << "," << landmarks_gpu[i].y << ")\t\t"
                  << "(" << dx << "," << dy << ")" << std::endl;
    }
    
    // 判断结果
    std::cout << "\n--- Result ---" << std::endl;
    if (max_diff_x <= 2 && max_diff_y <= 2) {
        std::cout << "✓ PASS: GPU and CPU results match (diff <= 2 pixels)" << std::endl;
    } else {
        std::cout << "✗ FAIL: GPU and CPU results differ significantly" << std::endl;
    }
    
    std::cout << "\nSpeedup: " << cpu_time / gpu_time << "x" << std::endl;
    
    // 可视化（可选）
    cv::Mat vis_cpu = img.clone();
    cv::Mat vis_gpu = img.clone();
    
    for (const auto& pt : landmarks_cpu) {
        cv::circle(vis_cpu, pt, 2, cv::Scalar(0, 255, 0), -1);
    }
    for (const auto& pt : landmarks_gpu) {
        cv::circle(vis_gpu, pt, 2, cv::Scalar(0, 0, 255), -1);
    }
    
    cv::imwrite("pipnet_cpu.jpg", vis_cpu);
    cv::imwrite("pipnet_gpu.jpg", vis_gpu);
    std::cout << "\nVisualization saved to pipnet_cpu.jpg and pipnet_gpu.jpg" << std::endl;
    
    std::cout << "\n=== Test Done ===" << std::endl;
    return 0;
}
