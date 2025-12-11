#include "ffmpeg_config.h"
#include <cstdio>
#include <array>
#include <memory>
#include <sstream>

FFmpegConfig::FFmpegConfig()
    : use_cuda(false), cuda_device(0), nvenc_available(false), nvdec_available(false)
{
    // 读取环境变量 LIPSYNC_FFMPEG_CUDA
    const char* env_cuda = std::getenv("LIPSYNC_FFMPEG_CUDA");
    if (env_cuda != nullptr) {
        use_cuda = (std::string(env_cuda) == "1");
    }

    // 读取环境变量 LIPSYNC_FFMPEG_CUDA_DEVICE
    const char* env_device = std::getenv("LIPSYNC_FFMPEG_CUDA_DEVICE");
    if (env_device != nullptr) {
        try {
            cuda_device = std::stoi(env_device);
            if (cuda_device < 0) {
                cuda_device = 0;
            }
        } catch (...) {
            cuda_device = 0;
        }
    }

    // 如果开启了CUDA,自动检测硬件能力
    if (use_cuda) {
        detectCudaCapabilities();

        // 如果硬件不支持,自动禁用CUDA
        if (!nvenc_available || !nvdec_available) {
            use_cuda = false;
        }
    }
}

int FFmpegConfig::execCommand(const std::string& cmd) const
{
    std::array<char, 128> buffer;
    std::string result;

    // 使用popen执行命令
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return -1;
    }

    // 读取输出(虽然不使用,但需要清空管道)
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }

    // 获取退出码
    int exit_code = pclose(pipe);
    return WEXITSTATUS(exit_code);
}

void FFmpegConfig::detectCudaCapabilities()
{
    // 检测NVENC编码器是否可用
    // 使用grep查找h264_nvenc编码器
    std::string nvenc_check = "ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc";
    int nvenc_ret = execCommand(nvenc_check);
    nvenc_available = (nvenc_ret == 0);

    // 检测NVDEC解码器是否可用
    // 使用grep查找h264_cuvid解码器
    std::string nvdec_check = "ffmpeg -hide_banner -decoders 2>/dev/null | grep -q h264_cuvid";
    int nvdec_ret = execCommand(nvdec_check);
    nvdec_available = (nvdec_ret == 0);
}

std::string FFmpegConfig::getDecoderArgs(int decode_threads) const
{
    if (!isCudaEnabled()) {
        return "";
    }

    std::ostringstream oss;
    oss << "-hwaccel cuda -hwaccel_device " << cuda_device;

    // 添加解码线程数参数（如果指定）
    // CUDA解码器会创建 (threads + 1) * 2 + 1 个解码表面
    // 例如: threads=1 -> surfaces=5, threads=2 -> surfaces=7
    // 不限制时ffmpeg可能默认使用16个线程,导致surfaces=35 > 32
    if (decode_threads >= 1) {
        oss << " -threads " << decode_threads;
    }

    return oss.str();
}

std::string FFmpegConfig::getEncoderArgs(int bitrate, int maxrate, bool cbr_mode, bool force_cpu) const
{
    if (force_cpu || !isCudaEnabled()) {
        // CPU模式: 使用libx264
        std::ostringstream oss;

        if (bitrate > 0) {
            // 有码率限制
            oss << "-c:v libx264 -b:v " << bitrate << "k";
            if (maxrate > 0) {
                oss << " -maxrate " << maxrate << "k -bufsize " << maxrate << "k";
            }
            if (cbr_mode) {
                oss << " -x264-params \"nal-hrd=cbr\"";
            }
        } else {
            // 无码率限制,使用CRF质量控制
            oss << "-c:v libx264 -preset veryfast -crf 23";
        }

        return oss.str();
    }

    // CUDA模式: 使用h264_nvenc
    std::ostringstream oss;
    oss << "-c:v h264_nvenc -preset p4 -tune hq";

    if (cbr_mode && bitrate > 0) {
        // CBR恒定码率模式
        oss << " -rc cbr -b:v " << bitrate << "k";
        if (maxrate > 0) {
            oss << " -maxrate " << maxrate << "k";
        }
        oss << " -bufsize " << (bitrate > 0 ? bitrate : maxrate) << "k";
    } else if (bitrate > 0) {
        // VBR可变码率模式
        oss << " -rc vbr -b:v " << bitrate << "k";
        if (maxrate > 0) {
            oss << " -maxrate " << maxrate << "k";
        }
        oss << " -cq 23";
    } else {
        // 质量优先模式(无码率限制)
        oss << " -rc vbr -cq 23 -b:v 0";
    }

    // 添加高质量优化参数
    oss << " -spatial_aq 1 -temporal_aq 1";

    return oss.str();
}

std::string FFmpegConfig::getScaleFilter(int width, int height, bool force_cpu) const
{
    std::ostringstream oss;

    if (force_cpu || !isCudaEnabled()) {
        // CPU模式: 使用scale滤镜
        oss << "scale=" << width << ":" << height;
    } else {
        // CUDA模式: 使用scale_cuda滤镜
        // 注意: scale_cuda需要配合hwaccel使用,且输出需要hwdownload转回系统内存
        oss << "scale_cuda=" << width << ":" << height << ":interp_algo=lanczos";
    }

    return oss.str();
}

bool FFmpegConfig::isCudaEnabled() const
{
    return use_cuda && nvenc_available && nvdec_available;
}
