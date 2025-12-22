#include "video_palindrome.h"
#include "gpu_decoder.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <memory>
#include <chrono>
#include <sys/stat.h>
#include <sys/wait.h>  // for WEXITSTATUS

// 日志宏定义，模仿 Java 版本的 [INFO] [WARN] 格式
#define PAL_INFO(fmt, ...) printf("[INFO] " fmt, ##__VA_ARGS__)
#define PAL_WARN(fmt, ...) printf("[WARN] " fmt, ##__VA_ARGS__)
#define PAL_ERR(fmt, ...)  printf("[ERROR] " fmt, ##__VA_ARGS__)

// 执行命令并获取输出
static std::string execCommand(const std::string& cmd) {
    std::array<char, 256> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        return "";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    // 去除尾部空白字符
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || 
           result.back() == ' ' || result.back() == '\t')) {
        result.pop_back();
    }
    return result;
}

// 执行命令并只输出关键信息，返回是否成功（基于文件存在性检查）
// output_path: 可选，如果提供则检查输出文件是否存在来判断成功
static bool execCommandWithLog(const std::string& cmd, const std::string& prefix, const std::string& output_path = "") {
    std::string full_cmd = cmd + " 2>&1";
    std::array<char, 512> buffer;
    FILE* pipe_raw = popen(full_cmd.c_str(), "r");
    if (!pipe_raw) {
        PAL_ERR("popen 失败: %s\n", cmd.c_str());
        return false;
    }
    
    std::string last_error_line;  // 保存最后一条错误信息
    while (fgets(buffer.data(), buffer.size(), pipe_raw) != nullptr) {
        std::string line(buffer.data());
        // 去除尾部换行
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        
        // 只显示关键信息：Lsize=, error, Error, failed, Failed, cannot, Cannot
        if (line.find("Lsize=") != std::string::npos || 
            line.find("error") != std::string::npos || 
            line.find("Error") != std::string::npos ||
            line.find("failed") != std::string::npos ||
            line.find("Failed") != std::string::npos ||
            line.find("cannot") != std::string::npos ||
            line.find("Cannot") != std::string::npos ||
            line.find("Unknown") != std::string::npos ||
            line.find("not found") != std::string::npos ||
            line.find("No such") != std::string::npos) {
            printf("[%s] %s\n", prefix.c_str(), line.c_str());
            // 记录错误行
            if (line.find("error") != std::string::npos || 
                line.find("Error") != std::string::npos ||
                line.find("failed") != std::string::npos ||
                line.find("Failed") != std::string::npos) {
                last_error_line = line;
            }
        }
    }
    
    // pclose 返回子进程的退出状态
    int status = pclose(pipe_raw);
    int exit_code = -1;
    if (status != -1) {
        exit_code = WEXITSTATUS(status);
    }
    
    // 如果提供了输出路径，检查文件是否存在且大小大于0
    if (!output_path.empty()) {
        struct stat file_stat;
        if (stat(output_path.c_str(), &file_stat) == 0 && file_stat.st_size > 0) {
            // 文件存在且大小大于0，认为成功
            return true;
        } else {
            // 文件不存在或大小为0，输出更多调试信息
            if (!last_error_line.empty()) {
                PAL_ERR("命令失败，错误信息: %s\n", last_error_line.c_str());
            }
            PAL_ERR("命令退出码: %d, 输出文件不存在或为空\n", exit_code);
            return false;
        }
    }
    
    // 没有提供输出路径，根据退出码判断
    return (exit_code == 0);
}

// 检查文件是否存在
static bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// 删除文件
static void deleteFile(const std::string& path) {
    if (fileExists(path)) {
        remove(path.c_str());
    }
}

// 获取文件目录
static std::string getParentDir(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    return path.substr(0, pos);
}

// 获取文件名（不含路径）
static std::string getFileName(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

// 获取不含扩展名的文件名
static std::string getBaseName(const std::string& filename) {
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos || pos == 0) {
        return filename;
    }
    return filename.substr(0, pos);
}

// 获取 FFmpeg/FFprobe 命令路径（根据环境变量 LIPSYNC_FFMPEG_DIR）
static std::string getFFmpegCmd(const std::string& tool = "ffmpeg") {
    const char* ffmpeg_dir = std::getenv("LIPSYNC_FFMPEG_DIR");
    if (ffmpeg_dir != nullptr && ffmpeg_dir[0] != '\0') {
        return std::string(ffmpeg_dir) + "/bin/" + tool;
    }
    return tool;  // 使用系统 PATH 中的版本
}

// 解析帧率分数格式（如 "30000/1001"）
static double parseFpsFraction(const std::string& fraction) {
    if (fraction.empty()) {
        return -1;
    }
    size_t pos = fraction.find('/');
    if (pos != std::string::npos) {
        try {
            double num = std::stod(fraction.substr(0, pos));
            double den = std::stod(fraction.substr(pos + 1));
            if (den == 0) {
                return -1;
            }
            return num / den;
        } catch (...) {
            return -1;
        }
    } else {
        try {
            return std::stod(fraction);
        } catch (...) {
            return -1;
        }
    }
}

// 尝试获取媒体时长
static double tryProbeDuration(const std::string& media_path, const std::string& stream_specifier) {
    std::string cmd = getFFmpegCmd("ffprobe") + " -v error";
    if (!stream_specifier.empty()) {
        cmd += " -select_streams " + stream_specifier;
        cmd += " -show_entries stream=duration";
    } else {
        cmd += " -show_entries format=duration";
    }
    cmd += " -of default=noprint_wrappers=1:nokey=1 \"" + media_path + "\"";
    
    std::string result = execCommand(cmd);
    if (result.empty()) {
        return -1;
    }
    try {
        return std::stod(result);
    } catch (...) {
        return -1;
    }
}

double probeMediaDuration(const std::string& media_path, const std::string& stream_specifier) {
    double duration = -1;
    if (!stream_specifier.empty()) {
        duration = tryProbeDuration(media_path, stream_specifier);
    }
    if (duration <= 0) {
        duration = tryProbeDuration(media_path, "");
    }
    return duration;
}

double probeVideoFps(const std::string& video_path) {
    std::string cmd = getFFmpegCmd("ffprobe") + " -v error -select_streams v:0 "
                      "-show_entries stream=r_frame_rate "
                      "-of default=noprint_wrappers=1:nokey=1 \"" + video_path + "\"";
    std::string result = execCommand(cmd);
    if (result.empty()) {
        return -1;
    }
    return parseFpsFraction(result);
}

std::string ensurePalindromeVideo(const std::string& src_video_path, 
                                   const std::string& audio_path,
                                   std::string output_video_path,
                                   long* palindrome_cost_ms) {
    auto start_time = std::chrono::steady_clock::now();
    
    // 检查源视频是否存在
    if (!fileExists(src_video_path)) {
        PAL_ERR("源视频不存在: %s\n", src_video_path.c_str());
        return src_video_path;
    }
    
    // 检查音频文件是否有效
    if (audio_path.empty() || !fileExists(audio_path)) {
        PAL_WARN("音频路径无效，跳过正反拼接，直接返回原视频。\n");
        return src_video_path;
    }
    
    // 获取音视频时长
    double video_duration = probeMediaDuration(src_video_path, "v:0");
    double audio_duration = probeMediaDuration(audio_path, "a:0");
    
    if (video_duration <= 0 || audio_duration <= 0) {
        PAL_WARN("无法获取音视频时长，跳过正反拼接。\n");
        return src_video_path;
    }
    
    PAL_INFO("视频时长 %.3fs 音频时长 %.3fs \n", video_duration, audio_duration);
    
    // 判断是否需要正反拼接
    if (video_duration >= audio_duration - 0.2) {
        PAL_INFO("视频时长 %.3fs 大于音频时长 %.3fs - 0.2s，直接使用原视频。\n", 
                video_duration, audio_duration);
        return src_video_path;
    }
    
    // 准备输出路径
    std::string parent_dir = getParentDir(src_video_path);
    std::string filename = getFileName(src_video_path);
    std::string base_name = getBaseName(filename);
    
    if (output_video_path.empty()) {
        output_video_path = parent_dir + "/" + base_name + "_palindrome.mp4";
    }
    
    PAL_INFO("开始生成正反拼接视频: %s\n", output_video_path.c_str());
    
    // ========== 步骤1：统一正向片段编码 ==========
    std::string forward_path = parent_dir + "/" + base_name + "_forward_pal.mp4";
    PAL_INFO("预先转码原视频以统一编码参数: %s\n", forward_path.c_str());
    
    // 从环境变量 LIPSYNC_FFMPEG_DIR 获取 FFmpeg 路径
    std::string ffmpeg_cmd = getFFmpegCmd("ffmpeg");
    PAL_INFO("使用 FFmpeg: %s\n", ffmpeg_cmd.c_str());
    
    std::string forward_cmd = ffmpeg_cmd + " -y -i \"" + src_video_path + "\" "
                              "-c:v hevc_nvenc -preset fast -pix_fmt yuv420p -an \"" + forward_path + "\"";
    
    bool forward_success = execCommandWithLog(forward_cmd, "forward", forward_path);
    if (!forward_success) {
        deleteFile(forward_path);
        PAL_ERR("GPU编码失败，请检查 FFmpeg 路径和 NVENC 支持\n");
        return src_video_path;
    }
    
    // ========== 步骤2：分段反转处理 ==========
    int segment_duration = 30;  // 每段30秒
    int segment_count = (int)std::ceil(video_duration / segment_duration);
    
    PAL_INFO("视频时长 %.2fs, 分为 %d 段处理 (每段%ds)\n", 
             video_duration, segment_count, segment_duration);
    
    std::vector<std::string> reversed_segments;
    
    for (int i = 0; i < segment_count; i++) {
        double start_time_seg = i * segment_duration;
        double duration = std::min((double)segment_duration, video_duration - start_time_seg);
        
        if (duration <= 1e-3) {
            PAL_INFO("第 %d 段时长 %.4fs, 跳过生成无效片段。\n", i + 1, duration);
            continue;
        }
        
        std::string segment_path = parent_dir + "/" + base_name + "_seg" + std::to_string(i) + "_rev.mp4";
        
        PAL_INFO("处理第 %d/%d 段 (%.1fs-%.1fs)...\n", 
                 i + 1, segment_count, start_time_seg, start_time_seg + duration);
        
        char start_str[32], duration_str[32];
        snprintf(start_str, sizeof(start_str), "%.3f", start_time_seg);
        snprintf(duration_str, sizeof(duration_str), "%.3f", duration);
        
        std::string seg_cmd = ffmpeg_cmd + " -y -ss " + std::string(start_str) + 
                              " -t " + std::string(duration_str) +
                              " -i \"" + src_video_path + "\" "
                              "-vf reverse "
                              "-c:v hevc_nvenc -preset fast -pix_fmt yuv420p -an \"" + segment_path + "\"";
        
        bool seg_success = execCommandWithLog(seg_cmd, "seg" + std::to_string(i), segment_path);
        
        if (!seg_success) {
            // 清理已生成的片段
            for (const auto& seg : reversed_segments) {
                deleteFile(seg);
            }
            deleteFile(forward_path);
            PAL_ERR("分段 %d GPU编码失败\n", i);
            return src_video_path;
        }
        
        reversed_segments.push_back(segment_path);
    }
    
    PAL_INFO("所有分段反转完成, 开始拼接...\n");
    
    // ========== 步骤3：创建拼接列表 ==========
    std::string concat_list_path = parent_dir + "/" + base_name + "_concat_list.txt";
    
    {
        std::ofstream concat_file(concat_list_path);
        if (!concat_file.is_open()) {
            for (const auto& seg : reversed_segments) {
                deleteFile(seg);
            }
            deleteFile(forward_path);
            PAL_ERR("无法创建拼接列表文件\n");
            return src_video_path;
        }
        
        // 先加入原视频(正序)
        concat_file << "file '" << forward_path << "'" << std::endl;
        
        // 再加入反向分段(倒序拼接,形成完整反向)
        for (int i = (int)reversed_segments.size() - 1; i >= 0; i--) {
            concat_file << "file '" << reversed_segments[i] << "'" << std::endl;
        }
    }
    
    // ========== 步骤4：执行拼接 ==========
    std::string concat_cmd = ffmpeg_cmd + " -y -f concat -safe 0 -i \"" + concat_list_path + "\" "
                             "-c:v hevc_nvenc -preset fast -pix_fmt yuv420p -an \"" + output_video_path + "\"";
    
    bool concat_success = execCommandWithLog(concat_cmd, "concat", output_video_path);
    
    // ========== 步骤5：清理临时文件 ==========
    deleteFile(concat_list_path);
    for (const auto& seg_path : reversed_segments) {
        deleteFile(seg_path);
    }
    deleteFile(forward_path);
    
    if (!concat_success) {
        PAL_ERR("ffmpeg 拼接失败\n");
        return src_video_path;
    }
    
    PAL_INFO("正反拼接视频生成完成。\n");
    
    // 校验帧率
    double original_fps = probeVideoFps(src_video_path);
    double palindrome_fps = probeVideoFps(output_video_path);
    if (original_fps > 0 && palindrome_fps > 0) {
        bool same = std::abs(original_fps - palindrome_fps) < 1e-3;
        PAL_INFO("原视频 FPS: %.3f, 拼接视频 FPS: %.3f, 是否一致: %s\n",
                original_fps, palindrome_fps, same ? "是" : "否");
    } else {
        PAL_WARN("无法获取视频帧率信息。\n");
    }
    
    // 计算耗时
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    if (palindrome_cost_ms != nullptr) {
        *palindrome_cost_ms = duration_ms;
    }
    
    PAL_INFO("正反拼接耗时: %lld 毫秒 (%.2f 秒)\n", (long long)duration_ms, duration_ms / 1000.0);
    
    return output_video_path;
}

// ============================================================================
// AsyncVideoReverser 类实现
// ============================================================================

AsyncVideoReverser::AsyncVideoReverser() {
}

AsyncVideoReverser::~AsyncVideoReverser() {
    join();
    cleanup();
}

void AsyncVideoReverser::startAsync(const std::string& src_video_path,
                                     const std::string& reversed_video_path,
                                     double video_duration,
                                     double audio_duration) {
    src_video_path_ = src_video_path;
    reversed_video_path_ = reversed_video_path;
    video_duration_ = video_duration;
    audio_duration_ = audio_duration;
    
    // 判断是否需要反转
    if (video_duration >= audio_duration - 0.2) {
        PAL_INFO("[AsyncReverser] 视频时长 %.3fs >= 音频时长 %.3fs - 0.2s，无需反转\n",
                 video_duration, audio_duration);
        need_reverse_ = false;
        is_ready_.store(true);
        return;
    }
    
    need_reverse_ = true;
    is_ready_.store(false);
    is_failed_.store(false);
    
    PAL_INFO("[AsyncReverser] 视频时长 %.3fs < 音频时长 %.3fs，启动后台反转线程\n",
             video_duration, audio_duration);
    
    // 启动后台反转线程
    thread_started_ = true;
    reverse_thread_ = std::thread(&AsyncVideoReverser::doReverse, this);
}

void AsyncVideoReverser::doReverse() {
    PAL_INFO("[AsyncReverser] 后台线程开始反转视频...\n");
    auto start_time = std::chrono::steady_clock::now();
    
    bool success = generateReversedVideo(src_video_path_, reversed_video_path_);
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // 更新状态并通知等待的线程
    {
        std::lock_guard<std::mutex> lock(mutex_);
        is_failed_.store(!success);
        is_ready_.store(true);
    }
    cv_.notify_all();
    
    if (success) {
        PAL_INFO("[AsyncReverser] 后台反转完成，耗时 %lld ms\n", (long long)duration_ms);
    } else {
        PAL_ERR("[AsyncReverser] 后台反转失败，耗时 %lld ms\n", (long long)duration_ms);
    }
}

GPUDecoder* AsyncVideoReverser::waitAndGetDecoder(int num_threads, int target_fps) {
    // ========== 阶段1: 等待反转视频生成完成 ==========
    {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (!is_ready_.load()) {
            PAL_INFO("[AsyncReverser] 等待反转视频生成完成...\n");
        }
        
        // 条件等待：直到 is_ready_ 变为 true
        cv_.wait(lock, [this]() {
            return is_ready_.load();
        });
        
        // 检查是否失败
        if (is_failed_.load()) {
            PAL_WARN("[AsyncReverser] 反转视频生成失败，返回nullptr\n");
            return nullptr;
        }
    }
    
    // ========== 阶段2: 初始化解码器（只做一次，线程安全） ==========
    {
        std::lock_guard<std::mutex> lock(decoder_mutex_);
        
        // Double-check: 可能其他线程已经初始化了
        if (decoder_initialized_) {
            return reversed_decoder_;
        }
        
        // 第一个到达的线程负责初始化
        PAL_INFO("[AsyncReverser] 初始化反转视频解码器: %s\n", reversed_video_path_.c_str());
        
        reversed_decoder_ = new GPUDecoder();
        if (!reversed_decoder_->open(reversed_video_path_, num_threads, target_fps)) {
            PAL_ERR("[AsyncReverser] 打开反转视频解码器失败\n");
            delete reversed_decoder_;
            reversed_decoder_ = nullptr;
            decoder_initialized_ = true;  // 标记为已尝试（虽然失败了）
            return nullptr;
        }
        
        decoder_initialized_ = true;
        PAL_INFO("[AsyncReverser] 反转视频解码器初始化成功，帧数: %d\n", 
                 reversed_decoder_->getFrameCount());
    }
    
    return reversed_decoder_;
}

void AsyncVideoReverser::cleanup() {
    // 关闭并删除解码器
    {
        std::lock_guard<std::mutex> lock(decoder_mutex_);
        if (reversed_decoder_ != nullptr) {
            reversed_decoder_->close();
            delete reversed_decoder_;
            reversed_decoder_ = nullptr;
        }
        decoder_initialized_ = false;
    }
    
    // 删除临时反转视频文件
    if (!reversed_video_path_.empty() && fileExists(reversed_video_path_)) {
        PAL_INFO("[AsyncReverser] 清理临时反转视频: %s\n", reversed_video_path_.c_str());
        deleteFile(reversed_video_path_);
    }
}

void AsyncVideoReverser::join() {
    if (thread_started_ && reverse_thread_.joinable()) {
        reverse_thread_.join();
        thread_started_ = false;
    }
}

// ============================================================================
// 辅助函数实现
// ============================================================================

std::string generateReversedVideoPath(const std::string& src_video_path) {
    std::string parent_dir = getParentDir(src_video_path);
    std::string filename = getFileName(src_video_path);
    std::string base_name = getBaseName(filename);
    return parent_dir + "/" + base_name + "_reversed.mp4";
}

bool generateReversedVideo(const std::string& src_video_path,
                           const std::string& output_video_path) {
    PAL_INFO("[generateReversedVideo] 开始生成反转视频: %s -> %s\n",
             src_video_path.c_str(), output_video_path.c_str());
    
    // 检查源视频是否存在
    if (!fileExists(src_video_path)) {
        PAL_ERR("[generateReversedVideo] 源视频不存在: %s\n", src_video_path.c_str());
        return false;
    }
    
    // 获取视频时长
    double video_duration = probeMediaDuration(src_video_path, "v:0");
    if (video_duration <= 0) {
        PAL_ERR("[generateReversedVideo] 无法获取视频时长\n");
        return false;
    }
    
    std::string ffmpeg_cmd = getFFmpegCmd("ffmpeg");
    std::string parent_dir = getParentDir(src_video_path);
    std::string filename = getFileName(src_video_path);
    std::string base_name = getBaseName(filename);
    
    // ========== 分段反转处理 ==========
    int segment_duration = 30;  // 每段20秒
    int segment_count = (int)std::ceil(video_duration / segment_duration);
    
    PAL_INFO("[generateReversedVideo] 视频时长 %.2fs, 分为 %d 段处理 (每段%ds)\n",
             video_duration, segment_count, segment_duration);
    
    std::vector<std::string> reversed_segments;
    
    for (int i = 0; i < segment_count; i++) {
        double start_time_seg = i * segment_duration;
        double duration = std::min((double)segment_duration, video_duration - start_time_seg);
        
        if (duration <= 1e-3) {
            PAL_INFO("[generateReversedVideo] 第 %d 段时长 %.4fs, 跳过\n", i + 1, duration);
            continue;
        }
        
        std::string segment_path = parent_dir + "/" + base_name + "_revseg" + std::to_string(i) + ".mp4";
        
        PAL_INFO("[generateReversedVideo] 处理第 %d/%d 段 (%.1fs-%.1fs)...\n",
                 i + 1, segment_count, start_time_seg, start_time_seg + duration);
        
        char start_str[32], duration_str[32];
        snprintf(start_str, sizeof(start_str), "%.3f", start_time_seg);
        snprintf(duration_str, sizeof(duration_str), "%.3f", duration);
        
        // GPU 编码：使用 h264_nvenc (H.264)，比 hevc_nvenc 编码更快
        std::string seg_cmd = ffmpeg_cmd + " -y"
                              " -ss " + std::string(start_str) +
                              " -t " + std::string(duration_str) +
                              " -i \"" + src_video_path + "\""
                              " -vf reverse"
                              " -c:v h264_nvenc -preset fast -pix_fmt yuv420p"
                              " -an \"" + segment_path + "\"";
        
        PAL_INFO("[generateReversedVideo] 执行命令: %s\n", seg_cmd.c_str());
        
        bool seg_success = execCommandWithLog(seg_cmd, "revseg" + std::to_string(i), segment_path);
        
        if (!seg_success) {
            // 清理已生成的片段
            for (const auto& seg : reversed_segments) {
                deleteFile(seg);
            }
            PAL_ERR("[generateReversedVideo] 分段 %d 反转失败\n", i);
            return false;
        }
        
        reversed_segments.push_back(segment_path);
    }
    
    PAL_INFO("[generateReversedVideo] 所有分段反转完成, 开始拼接...\n");
    
    // ========== 创建拼接列表（倒序拼接形成完整反转视频） ==========
    std::string concat_list_path = parent_dir + "/" + base_name + "_rev_concat_list.txt";
    
    {
        std::ofstream concat_file(concat_list_path);
        if (!concat_file.is_open()) {
            for (const auto& seg : reversed_segments) {
                deleteFile(seg);
            }
            PAL_ERR("[generateReversedVideo] 无法创建拼接列表文件\n");
            return false;
        }
        
        // 倒序拼接，形成完整的反转视频
        // 原视频: [seg0][seg1][seg2]
        // 反转后: [seg2_rev][seg1_rev][seg0_rev]
        for (int i = (int)reversed_segments.size() - 1; i >= 0; i--) {
            concat_file << "file '" << reversed_segments[i] << "'" << std::endl;
        }
    }
    
    // ========== 执行拼接 ==========
    // 由于分段已经是 H.264 格式，拼接时可以直接 copy，速度更快
    std::string concat_cmd = ffmpeg_cmd + " -y -f concat -safe 0"
                             " -i \"" + concat_list_path + "\""
                             " -c:v copy -an \"" + output_video_path + "\"";
    
    PAL_INFO("[generateReversedVideo] 拼接命令: %s\n", concat_cmd.c_str());
    
    bool concat_success = execCommandWithLog(concat_cmd, "rev_concat", output_video_path);
    
    // ========== 清理临时文件 ==========
    deleteFile(concat_list_path);
    for (const auto& seg_path : reversed_segments) {
        deleteFile(seg_path);
    }
    
    if (!concat_success) {
        PAL_ERR("[generateReversedVideo] 拼接失败\n");
        return false;
    }
    
    PAL_INFO("[generateReversedVideo] 反转视频生成完成: %s\n", output_video_path.c_str());
    return true;
}