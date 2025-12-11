#include <iostream>
#include <chrono>
#include "../src/gpu_decoder.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return 1;
    }
    
    std::string video_path = argv[1];
    int num_threads = 4;
    
    std::cout << "=== GPU Decoder Test ===" << std::endl;
    std::cout << "Video: " << video_path << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    
    // 创建解码器
    GPUDecoder decoder;
    
    // 打开视频
    auto t0 = std::chrono::high_resolution_clock::now();
    
    if (!decoder.open(video_path, num_threads, 25)) {
        std::cerr << "Failed to open video!" << std::endl;
        return 1;
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    auto open_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    
    std::cout << "\nVideo Info:" << std::endl;
    std::cout << "  Resolution: " << decoder.getWidth() << "x" << decoder.getHeight() << std::endl;
    std::cout << "  FPS: " << decoder.getFPS() << std::endl;
    std::cout << "  Frame Count: " << decoder.getFrameCount() << std::endl;
    std::cout << "  Bitrate: " << decoder.getBitrate() << " kbps" << std::endl;
    std::cout << "  Open Time: " << open_time << " ms" << std::endl;
    
    // 测试解码性能
    int test_frames = std::min(100, decoder.getFrameCount());
    std::cout << "\nDecoding " << test_frames << " frames..." << std::endl;
    
    t0 = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < test_frames; i++) {
        int thread_id = i % num_threads;
        cv::cuda::GpuMat& gpu_frame = decoder.decodeFrame(i, thread_id);
        
        if (gpu_frame.empty()) {
            std::cerr << "Failed to decode frame " << i << std::endl;
            continue;
        }
        
        // 每10帧打印一次进度
        if (i % 10 == 0) {
            std::cout << "  Frame " << i << ": " << gpu_frame.cols << "x" << gpu_frame.rows 
                      << " (GPU memory)" << std::endl;
        }
    }
    
    t1 = std::chrono::high_resolution_clock::now();
    auto decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    
    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Total Time: " << decode_time << " ms" << std::endl;
    std::cout << "  Avg per Frame: " << (double)decode_time / test_frames << " ms" << std::endl;
    std::cout << "  FPS: " << (double)test_frames * 1000 / decode_time << std::endl;
    
    // 测试下载到CPU（验证数据正确性）
    std::cout << "\nTesting GPU->CPU download..." << std::endl;
    
    cv::cuda::GpuMat& gpu_frame = decoder.decodeFrame(0, 0);
    cv::Mat cpu_frame;
    gpu_frame.download(cpu_frame);
    
    std::cout << "  CPU Frame: " << cpu_frame.cols << "x" << cpu_frame.rows 
              << " channels=" << cpu_frame.channels() << std::endl;
    
    // 保存一帧用于验证
    cv::imwrite("test_gpu_decode_output.jpg", cpu_frame);
    std::cout << "  Saved to: test_gpu_decode_output.jpg" << std::endl;
    
    // 关闭
    decoder.close();
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    
    return 0;
}
