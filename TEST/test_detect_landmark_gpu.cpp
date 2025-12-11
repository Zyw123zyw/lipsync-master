/**
 * detectLandmarkGPU 测试
 * 
 * 对比 CPU detectLandmark() 和 GPU detectLandmarkGPU() 的结果
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include "../trt_function/src/scrfd.h"
#include "../trt_function/src/pipnet.h"

using namespace Function;

// 简化版的detectLandmark（CPU版本）
void detectLandmarkCPU(SCRFD* detector, PIPNet* landmarker,
                       const cv::Mat &frame,
                       const cv::Rect2i &roi_rect,
                       cv::Rect2i &face_bbox,
                       std::vector<cv::Point2i> &face_landmark,
                       float score_threshold)
{
    cv::Mat roi_img = frame(roi_rect).clone();

    std::vector<DetectionBox> faceboxes;
    detector->detect(roi_img, faceboxes, score_threshold);
    
    if (!faceboxes.empty())
    {
        DetectionBox detect_box = faceboxes[0];
        detect_box.x1 += roi_rect.x;
        detect_box.y1 += roi_rect.y;

        cv::Rect2i lmsdet_rect;
        detector->expand_box_for_pipnet(frame.size(), detect_box, lmsdet_rect, 1.2);
        cv::Mat kp_img = frame(lmsdet_rect).clone();
        
        std::vector<cv::Point2i> landmark;
        landmarker->predict(kp_img, lmsdet_rect, landmark);

        face_bbox.x = detect_box.x1;
        face_bbox.y = detect_box.y1;
        face_bbox.width = detect_box.w;
        face_bbox.height = detect_box.h;
        face_landmark = landmark;
    }
}

// 简化版的detectLandmarkGPU（GPU版本）
void detectLandmarkGPU(SCRFD* detector, PIPNet* landmarker,
                       const cv::cuda::GpuMat &gpu_frame,
                       const cv::Rect2i &roi_rect,
                       cv::Rect2i &face_bbox,
                       std::vector<cv::Point2i> &face_landmark,
                       float score_threshold)
{
    // GPU上裁剪ROI
    cv::cuda::GpuMat gpu_roi = gpu_frame(roi_rect);

    // GPU检测
    std::vector<DetectionBox> faceboxes;
    detector->detectGPU(gpu_roi, faceboxes, score_threshold);
    
    if (!faceboxes.empty())
    {
        DetectionBox detect_box = faceboxes[0];
        detect_box.x1 += roi_rect.x;
        detect_box.y1 += roi_rect.y;

        cv::Rect2i lmsdet_rect;
        detector->expand_box_for_pipnet(
            cv::Size(gpu_frame.cols, gpu_frame.rows), detect_box, lmsdet_rect, 1.2);
        
        // 只下载人脸区域
        cv::Mat kp_img;
        gpu_frame(lmsdet_rect).download(kp_img);
        
        std::vector<cv::Point2i> landmark;
        landmarker->predict(kp_img, lmsdet_rect, landmark);

        face_bbox.x = detect_box.x1;
        face_bbox.y = detect_box.y1;
        face_bbox.width = detect_box.w;
        face_bbox.height = detect_box.h;
        face_landmark = landmark;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== detectLandmarkGPU Test ===" << std::endl;
    
    // 模型路径
    std::string scrfd_path = "/root/models_8.9/scrfd_2.5g_shape640x640.engine";
    std::string pipnet_path = "/root/models_8.9/pipnet.engine";
    std::string image_path = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/input/onepeople.png";
    
    if (argc > 1) scrfd_path = argv[1];
    if (argc > 2) pipnet_path = argv[2];
    if (argc > 3) image_path = argv[3];
    
    std::cout << "SCRFD: " << scrfd_path << std::endl;
    std::cout << "PIPNet: " << pipnet_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    
    // 加载图像
    cv::Mat test_img = cv::imread(image_path);
    if (test_img.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }
    std::cout << "Image size: " << test_img.cols << "x" << test_img.rows << std::endl;
    
    // 初始化模型
    std::cout << "\nInitializing models..." << std::endl;
    SCRFD detector(scrfd_path);
    detector.initialize_handler();
    detector.warmup();
    
    PIPNet landmarker(pipnet_path);
    landmarker.initialize_handler();
    landmarker.warmup();
    std::cout << "Models initialized." << std::endl;
    
    // 上传到GPU
    cv::cuda::GpuMat gpu_img;
    gpu_img.upload(test_img);
    
    // ROI设为整张图
    cv::Rect2i roi(0, 0, test_img.cols, test_img.rows);
    float score_threshold = 0.5f;
    
    // ========== CPU版本 ==========
    std::cout << "\n--- CPU detectLandmark ---" << std::endl;
    cv::Rect2i cpu_bbox;
    std::vector<cv::Point2i> cpu_landmark;
    
    double t0 = (double)cv::getTickCount();
    detectLandmarkCPU(&detector, &landmarker, test_img, roi, cpu_bbox, cpu_landmark, score_threshold);
    double t1 = (double)cv::getTickCount();
    double cpu_time = (t1 - t0) / cv::getTickFrequency() * 1000;
    
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "CPU bbox: [" << cpu_bbox.x << ", " << cpu_bbox.y 
              << ", " << cpu_bbox.width << ", " << cpu_bbox.height << "]" << std::endl;
    std::cout << "CPU landmarks: " << cpu_landmark.size() << " points" << std::endl;
    if (!cpu_landmark.empty()) {
        std::cout << "  First 5: ";
        for (int i = 0; i < std::min(5, (int)cpu_landmark.size()); i++) {
            std::cout << "(" << cpu_landmark[i].x << "," << cpu_landmark[i].y << ") ";
        }
        std::cout << std::endl;
    }
    
    // ========== GPU版本 ==========
    std::cout << "\n--- GPU detectLandmarkGPU ---" << std::endl;
    cv::Rect2i gpu_bbox;
    std::vector<cv::Point2i> gpu_landmark;
    
    // Warmup
    detectLandmarkGPU(&detector, &landmarker, gpu_img, roi, gpu_bbox, gpu_landmark, score_threshold);
    gpu_bbox = cv::Rect2i();
    gpu_landmark.clear();
    
    t0 = (double)cv::getTickCount();
    detectLandmarkGPU(&detector, &landmarker, gpu_img, roi, gpu_bbox, gpu_landmark, score_threshold);
    t1 = (double)cv::getTickCount();
    double gpu_time = (t1 - t0) / cv::getTickFrequency() * 1000;
    
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "GPU bbox: [" << gpu_bbox.x << ", " << gpu_bbox.y 
              << ", " << gpu_bbox.width << ", " << gpu_bbox.height << "]" << std::endl;
    std::cout << "GPU landmarks: " << gpu_landmark.size() << " points" << std::endl;
    if (!gpu_landmark.empty()) {
        std::cout << "  First 5: ";
        for (int i = 0; i < std::min(5, (int)gpu_landmark.size()); i++) {
            std::cout << "(" << gpu_landmark[i].x << "," << gpu_landmark[i].y << ") ";
        }
        std::cout << std::endl;
    }
    
    // ========== 对比结果 ==========
    std::cout << "\n--- Compare Results ---" << std::endl;
    
    bool match = true;
    
    // 对比bbox
    int bbox_diff = std::abs(cpu_bbox.x - gpu_bbox.x) + std::abs(cpu_bbox.y - gpu_bbox.y) +
                    std::abs(cpu_bbox.width - gpu_bbox.width) + std::abs(cpu_bbox.height - gpu_bbox.height);
    if (bbox_diff > 4) {  // 允许每个值差1
        std::cout << "Bbox differs!" << std::endl;
        match = false;
    }
    
    // 对比landmark数量
    if (cpu_landmark.size() != gpu_landmark.size()) {
        std::cout << "Landmark count differs!" << std::endl;
        match = false;
    } else if (!cpu_landmark.empty()) {
        // 对比landmark位置
        int max_diff = 0;
        for (size_t i = 0; i < cpu_landmark.size(); i++) {
            int diff = std::abs(cpu_landmark[i].x - gpu_landmark[i].x) +
                       std::abs(cpu_landmark[i].y - gpu_landmark[i].y);
            max_diff = std::max(max_diff, diff);
        }
        std::cout << "Max landmark diff: " << max_diff << " pixels" << std::endl;
        if (max_diff > 2) {
            match = false;
        }
    }
    
    if (match) {
        std::cout << "\n✓ Results match!" << std::endl;
    } else {
        std::cout << "\n✗ Results differ!" << std::endl;
    }
    
    std::cout << "\nSpeedup: " << cpu_time / gpu_time << "x" << std::endl;
    
    std::cout << "\n=== Test Done ===" << std::endl;
    return 0;
}
