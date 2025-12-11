/**
 * 视频循环拼接Demo（无重编码，极速）
 * 
 * 使用 -stream_loop 循环视频，-c copy 不重编码
 */

#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <array>
#include <memory>
#include <cmath>

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

// 循环拼接视频
bool loopVideo(const std::string& video_path, 
               const std::string& audio_path, 
               const std::string& output_path) {
    
    // 1. 获取视频和音频时长
    double video_duration = getDuration(video_path);
    double audio_duration = getDuration(audio_path);
    
    if (video_duration <= 0 || audio_duration <= 0) {
        std::cerr << "Error: Cannot get duration" << std::endl;
        return false;
    }
    
    std::cout << "Video duration: " << video_duration << "s" << std::endl;
    std::cout << "Audio duration: " << audio_duration << "s" << std::endl;
    
    // 2. 如果视频已经比音频长，直接复制
    if (video_duration >= audio_duration) {
        std::cout << "Video is already longer than audio, copying..." << std::endl;
        std::string cmd = "ffmpeg -y -i \"" + video_path + "\" -t " + std::to_string(audio_duration) + " -c copy -an \"" + output_path + "\"";
        return system(cmd.c_str()) == 0;
    }
    
    // 3. 计算需要循环多少次
    int loop_count = static_cast<int>(std::ceil(audio_duration / video_duration));
    std::cout << "Need " << loop_count << " loops to cover audio duration" << std::endl;
    
    // 4. 使用 stream_loop 循环拼接（无重编码，极速）
    std::cout << "Looping video (no re-encoding)..." << std::endl;
    
    // -stream_loop N 表示循环N次（总共播放N+1次）
    std::string cmd = "ffmpeg -y -stream_loop " + std::to_string(loop_count - 1) + 
                      " -i \"" + video_path + "\" -t " + std::to_string(audio_duration) + 
                      " -c copy -an \"" + output_path + "\"";
    
    std::cout << "Command: " << cmd << std::endl;
    
    if (system(cmd.c_str()) != 0) {
        std::cerr << "Error: Failed to loop video" << std::endl;
        return false;
    }
    
    // 5. 验证输出
    double output_duration = getDuration(output_path);
    std::cout << "\nOutput video duration: " << output_duration << "s" << std::endl;
    
    return true;
}

int main(int argc, char* argv[]) {
    std::string video_path = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/input/bb0087c4ff364deb97cdea8d5e4aaf3b.mp4";
    std::string audio_path = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/input/audio.wav";
    std::string output_path = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/output/out.mp4";
    
    std::cout << "=== Video Loop Demo (No Re-encoding) ===" << std::endl;
    std::cout << "Video: " << video_path << std::endl;
    std::cout << "Audio: " << audio_path << std::endl;
    std::cout << "Output: " << output_path << std::endl;
    std::cout << std::endl;
    
    if (loopVideo(video_path, audio_path, output_path)) {
        std::cout << "\nDone!" << std::endl;
        return 0;
    } else {
        std::cerr << "\nFailed!" << std::endl;
        return 1;
    }
}
