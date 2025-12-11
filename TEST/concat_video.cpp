/**
 * 视频循环拼接到5分钟（无重编码，极速）
 * 
 * 使用 -stream_loop 循环视频，-c copy 不重编码
 * 不处理音频
 */

#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <array>
#include <memory>
#include <cmath>

// 目标时长：5分钟 = 300秒
const double TARGET_DURATION = 300.0;

// 执行命令并获取输出
std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        return "";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

// 获取媒体时长（秒）
double getDuration(const std::string& path) {
    std::string cmd = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"" + path + "\"";
    std::string result = exec(cmd.c_str());
    if (result.empty()) {
        return 0.0;
    }
    return std::stod(result);
}

// 循环拼接视频到5分钟
bool loopVideoTo5Min(const std::string& video_path, 
                     const std::string& output_path) {
    
    // 1. 获取视频时长
    double video_duration = getDuration(video_path);
    
    if (video_duration <= 0) {
        std::cerr << "Error: Cannot get video duration" << std::endl;
        return false;
    }
    
    std::cout << "Video duration: " << video_duration << "s" << std::endl;
    std::cout << "Target duration: " << TARGET_DURATION << "s (5 minutes)" << std::endl;
    
    // 2. 如果视频已经比5分钟长，直接裁剪
    if (video_duration >= TARGET_DURATION) {
        std::cout << "Video is already longer than 5 minutes, trimming..." << std::endl;
        std::string cmd = "ffmpeg -y -i \"" + video_path + "\" -t " + std::to_string(TARGET_DURATION) + " -c copy -an \"" + output_path + "\"";
        std::cout << "Command: " << cmd << std::endl;
        return system(cmd.c_str()) == 0;
    }
    
    // 3. 计算需要循环多少次
    int loop_count = static_cast<int>(std::ceil(TARGET_DURATION / video_duration));
    std::cout << "Need " << loop_count << " loops to reach 5 minutes" << std::endl;
    
    // 4. 使用 stream_loop 循环拼接（无重编码，极速）
    std::cout << "Looping video (no re-encoding)..." << std::endl;
    
    // -stream_loop N 表示循环N次（总共播放N+1次）
    // -an 表示不要音频
    std::string cmd = "ffmpeg -y -stream_loop " + std::to_string(loop_count - 1) + 
                      " -i \"" + video_path + "\" -t " + std::to_string(TARGET_DURATION) + 
                      " -c copy -an \"" + output_path + "\"";
    
    std::cout << "Command: " << cmd << std::endl;
    
    if (system(cmd.c_str()) != 0) {
        std::cerr << "Error: Failed to loop video" << std::endl;
        return false;
    }
    
    // 5. 验证输出
    double output_duration = getDuration(output_path);
    std::cout << "\nOutput video duration: " << output_duration << "s (" 
              << output_duration / 60.0 << " minutes)" << std::endl;
    
    return true;
}

int main(int argc, char* argv[]) {
    std::string video_path = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/input/bb0087c4ff364deb97cdea8d5e4aaf3b.mp4";
    std::string output_path = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/output/out.mp4";
    
    if (argc >= 3) {
        video_path = argv[1];
        output_path = argv[2];
    } else if (argc == 2) {
        video_path = argv[1];
    }
    
    std::cout << "=== Video Loop to 5 Minutes (No Re-encoding) ===" << std::endl;
    std::cout << "Input:  " << video_path << std::endl;
    std::cout << "Output: " << output_path << std::endl;
    std::cout << std::endl;
    
    if (loopVideoTo5Min(video_path, output_path)) {
        std::cout << "\nDone!" << std::endl;
        return 0;
    } else {
        std::cerr << "\nFailed!" << std::endl;
        return 1;
    }
}
