#ifndef FFMPEG_CONFIG_H
#define FFMPEG_CONFIG_H

#include <string>
#include <cstdlib>

class FFmpegConfig {
public:
    bool use_cuda;              // CUDA总开关
    int cuda_device;            // CUDA设备ID
    bool nvenc_available;       // NVENC编码器可用性
    bool nvdec_available;       // NVDEC解码器可用性

    FFmpegConfig();

    // 自动检测GPU编解码能力
    void detectCudaCapabilities();

    // 获取解码器前置参数(hwaccel相关)
    // decode_threads: 解码线程数，<1时不指定-threads参数，让ffmpeg自行决定
    //                 建议CUDA解码时使用1以避免解码表面数超过32个限制
    std::string getDecoderArgs(int decode_threads = 1) const;

    // 获取视频编码器及参数(替代libx264)
    // bitrate: 目标码率(kbps), 0表示不限制
    // maxrate: 最大码率(kbps)
    // cbr_mode: true=恒定码率CBR, false=可变码率VBR
    // force_cpu: true=强制使用CPU编码器, false=根据配置自动选择
    std::string getEncoderArgs(int bitrate, int maxrate, bool cbr_mode, bool force_cpu = false) const;

    // 获取缩放滤镜参数
    // width, height: 目标分辨率
    // force_cpu: true=强制使用CPU滤镜, false=根据配置自动选择
    std::string getScaleFilter(int width, int height, bool force_cpu = false) const;

    // 检查是否实际启用CUDA(总开关开启且硬件可用)
    bool isCudaEnabled() const;

private:
    // 执行shell命令并返回退出码
    int execCommand(const std::string& cmd) const;
};

#endif // FFMPEG_CONFIG_H
