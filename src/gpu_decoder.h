#ifndef GPU_DECODER_H
#define GPU_DECODER_H

#include <string>
#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include "debug/util_debug.h"

/**
 * GPU视频解码器类
 * 
 * 使用FFmpeg NVDEC硬件解码，将视频帧直接解码到GPU显存(GpuMat)
 * 
 * 用法:
 *   GPUDecoder decoder;
 *   decoder.open("video.mp4", 4);  // 打开视频，分配4个GPU帧缓存
 *   
 *   cv::cuda::GpuMat& frame = decoder.decodeFrame(frame_idx, thread_id);
 *   // 直接使用GPU帧进行后续处理...
 */
class GPUDecoder {
public:
    GPUDecoder();
    ~GPUDecoder();

    /**
     * 打开视频文件并初始化解码器
     * @param video_path 视频文件路径
     * @param num_threads 渲染线程数，决定GPU帧缓存池大小
     * @param target_fps 目标帧率，0表示保持原始帧率
     * @return 是否成功
     */
    bool open(const std::string& video_path, int num_threads, int target_fps = 25);

    /**
     * 关闭解码器并释放资源
     */
    void close();

    /**
     * 解码指定帧到GPU，返回对应线程的GpuMat引用
     * @param frame_idx 底板帧索引 (会根据循环逻辑计算实际帧号)
     * @param thread_id 线程ID，用于选择GPU帧缓存槽位
     * @return GPU帧的引用
     */
    cv::cuda::GpuMat& decodeFrame(int frame_idx, int thread_id);

    /**
     * 获取指定线程的GPU帧缓存引用（用于外部直接访问）
     */
    cv::cuda::GpuMat& getGpuFrame(int thread_id);

    /**
     * 获取视频信息
     */
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    int getFPS() const { return fps_; }
    int getFrameCount() const { return frame_count_; }
    long getBitrate() const { return bitrate_; }
    
    /**
     * 获取原始视频fps
     */
    int getOriginalFPS() const { return original_fps_; }
    
    /**
     * 获取目标fps
     */
    int getTargetFPS() const { return target_fps_; }
    
    /**
     * 获取目标帧率下的帧数（重采样后的帧数）
     * 计算公式: 视频时长 * 目标fps
     */
    int getTargetFrameCount() const {
        if (original_fps_ <= 0) return frame_count_;
        double duration = (double)frame_count_ / (double)original_fps_;
        return (int)(duration * target_fps_);
    }
    
    /**
     * 获取fps比例（用于帧索引映射）
     * 返回 original_fps / target_fps
     */
    float getFPSRatio() const {
        if (target_fps_ <= 0 || target_fps_ == original_fps_) 
            return 1.0f;
        return (float)original_fps_ / (float)target_fps_;
    }

    /**
     * 检查是否已打开
     */
    bool isOpened() const { return is_opened_; }

private:
    // 初始化CUDA硬件加速上下文
    bool initHWContext();
    
    // 解码单帧到GPU
    bool decodeOneFrame(int target_frame);
    
    // NV12转BGR并存入GpuMat
    void convertNV12ToBGR(AVFrame* hw_frame, cv::cuda::GpuMat& output);

    // seek到指定帧
    bool seekToFrame(int frame_idx);

private:
    // FFmpeg相关
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;
    AVFrame* hw_frame_ = nullptr;      // 硬件帧(GPU)
    AVFrame* sw_frame_ = nullptr;      // 软件帧(用于转换)
    AVPacket* packet_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    
    // sws_ctx_ 缓存的源参数（用于检测是否需要重新创建）
    AVPixelFormat last_sws_src_format_ = AV_PIX_FMT_NONE;
    int last_sws_src_width_ = 0;
    int last_sws_src_height_ = 0;
    
    int video_stream_idx_ = -1;
    
    // 视频信息
    int width_ = 0;
    int height_ = 0;
    int fps_ = 25;
    int original_fps_ = 25;   // 原始视频fps
    int target_fps_ = 25;     // 目标fps
    int frame_count_ = 0;     // 原始帧数
    long bitrate_ = 0;
    
    // GPU帧缓存池 (每个线程一个槽位)
    std::vector<cv::cuda::GpuMat> gpu_frame_pool_;
    
    // 当前解码位置
    int current_frame_ = -1;
    
    // 状态
    bool is_opened_ = false;
    bool is_draining_ = false;  // 是否已进入drain模式（发送了flush包）
    
    // 线程安全
    std::mutex decode_mutex_;
    
    // CUDA流
    cv::cuda::Stream cuda_stream_;
};

#endif // GPU_DECODER_H
