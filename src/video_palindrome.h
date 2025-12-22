#ifndef VIDEO_PALINDROME_H
#define VIDEO_PALINDROME_H

#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

// 前向声明
class GPUDecoder;

/**
 * 异步视频反转器
 * 
 * 在后台线程中生成反转视频，渲染线程可以在需要时等待完成。
 * 用于优化：渲染正向帧的同时，后台生成反转视频。
 */
class AsyncVideoReverser {
public:
    AsyncVideoReverser();
    ~AsyncVideoReverser();

    /**
     * 启动异步反转（非阻塞，立即返回）
     * 
     * @param src_video_path 源视频路径
     * @param reversed_video_path 反转视频输出路径
     * @param video_duration 视频时长（秒）
     * @param audio_duration 音频时长（秒）
     */
    void startAsync(const std::string& src_video_path,
                    const std::string& reversed_video_path,
                    double video_duration,
                    double audio_duration);

    /**
     * 等待反转完成并获取解码器（线程安全）
     * 多个线程可以同时调用，只有第一个线程会初始化解码器
     * 
     * @param num_threads GPU帧缓存池大小
     * @param target_fps 目标帧率
     * @return 反转视频的解码器指针，失败返回nullptr
     */
    GPUDecoder* waitAndGetDecoder(int num_threads, int target_fps);

    /**
     * 检查是否需要反转（视频时长 < 音频时长）
     */
    bool needReverse() const { return need_reverse_; }

    /**
     * 检查反转是否已完成
     */
    bool isReady() const { return is_ready_.load(); }

    /**
     * 检查反转是否失败
     */
    bool isFailed() const { return is_failed_.load(); }

    /**
     * 获取源视频路径
     */
    const std::string& getSourceVideoPath() const { return src_video_path_; }

    /**
     * 获取反转视频路径
     */
    const std::string& getReversedVideoPath() const { return reversed_video_path_; }

    /**
     * 清理资源（删除临时反转视频文件）
     */
    void cleanup();

    /**
     * 等待后台线程结束
     */
    void join();

private:
    // 执行实际的反转操作
    void doReverse();

private:
    std::string src_video_path_;
    std::string reversed_video_path_;
    double video_duration_ = 0;
    double audio_duration_ = 0;

    std::thread reverse_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;

    std::atomic<bool> is_ready_{false};
    std::atomic<bool> is_failed_{false};
    bool need_reverse_ = false;
    bool thread_started_ = false;

    // 反转视频解码器
    GPUDecoder* reversed_decoder_ = nullptr;
    bool decoder_initialized_ = false;
    std::mutex decoder_mutex_;
};

/**
 * 视频正反拼接处理（回文视频）- 旧版同步接口，保留兼容
 * 
 * 当源视频时长比音频时长短时，通过"正反拼接"的方式延长视频，
 * 使视频能够覆盖整个音频时长。
 * 
 * @param src_video_path 源视频路径
 * @param audio_path 音频路径
 * @param output_video_path 输出视频路径（如果为空，则自动生成）
 * @param palindrome_cost_ms 输出参数：正反拼接耗时（毫秒）
 * @return 处理后的视频路径（如果无需处理则返回原路径）
 */
std::string ensurePalindromeVideo(const std::string& src_video_path, 
                                   const std::string& audio_path,
                                   std::string output_video_path = "",
                                   long* palindrome_cost_ms = nullptr);

/**
 * 使用 ffprobe 获取媒体时长
 * 
 * @param media_path 媒体文件路径
 * @param stream_specifier 流选择器（如 "v:0" 或 "a:0"，为空则获取格式级别时长）
 * @return 时长（秒），失败返回 -1
 */
double probeMediaDuration(const std::string& media_path, const std::string& stream_specifier = "");

/**
 * 使用 ffprobe 获取视频帧率
 * 
 * @param video_path 视频文件路径
 * @return 帧率，失败返回 -1
 */
double probeVideoFps(const std::string& video_path);

/**
 * 生成反转视频（供 AsyncVideoReverser 内部调用）
 * 
 * @param src_video_path 源视频路径
 * @param output_video_path 输出反转视频路径
 * @return 是否成功
 */
bool generateReversedVideo(const std::string& src_video_path,
                           const std::string& output_video_path);

/**
 * 根据源视频路径生成反转视频的路径
 */
std::string generateReversedVideoPath(const std::string& src_video_path);

#endif // VIDEO_PALINDROME_H
