#include "gpu_decoder.h"
#include <cuda_runtime.h>

GPUDecoder::GPUDecoder() {
    // 初始化FFmpeg（新版本不需要av_register_all）
}

GPUDecoder::~GPUDecoder() {
    close();
}

bool GPUDecoder::open(const std::string& video_path, int num_threads, int target_fps) {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    
    if (is_opened_) {
        close();
    }
    
    // 1. 打开视频文件
    fmt_ctx_ = avformat_alloc_context();
    int ret = avformat_open_input(&fmt_ctx_, video_path.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        DBG_LOGE("GPUDecoder: Cannot open video file: %s, error: %s (code: %d)\n", 
                 video_path.c_str(), errbuf, ret);
        return false;
    }
    
    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        DBG_LOGE("GPUDecoder: Cannot find stream info\n");
        close();
        return false;
    }
    
    // 2. 找到视频流
    video_stream_idx_ = -1;
    for (unsigned int i = 0; i < fmt_ctx_->nb_streams; i++) {
        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx_ = i;
            break;
        }
    }
    
    if (video_stream_idx_ < 0) {
        DBG_LOGE("GPUDecoder: No video stream found\n");
        close();
        return false;
    }
    
    AVStream* video_stream = fmt_ctx_->streams[video_stream_idx_];
    AVCodecParameters* codecpar = video_stream->codecpar;
    
    // 3. 获取视频信息
    width_ = codecpar->width;
    height_ = codecpar->height;
    bitrate_ = fmt_ctx_->bit_rate / 1024; // 转为kbps
    
    // 计算帧率
    if (video_stream->avg_frame_rate.den > 0) {
        fps_ = video_stream->avg_frame_rate.num / video_stream->avg_frame_rate.den;
    } else if (video_stream->r_frame_rate.den > 0) {
        fps_ = video_stream->r_frame_rate.num / video_stream->r_frame_rate.den;
    }
    if (target_fps > 0) {
        fps_ = target_fps;
    }
    
    // 计算总帧数
    if (video_stream->nb_frames > 0) {
        frame_count_ = video_stream->nb_frames;
    } else if (video_stream->duration > 0) {
        double duration_sec = video_stream->duration * av_q2d(video_stream->time_base);
        frame_count_ = static_cast<int>(duration_sec * fps_);
    } else if (fmt_ctx_->duration > 0) {
        double duration_sec = fmt_ctx_->duration / (double)AV_TIME_BASE;
        frame_count_ = static_cast<int>(duration_sec * fps_);
    }
    
    DBG_LOGI("GPUDecoder: Video info - %dx%d, %d fps, %d frames, %ld kbps\n", 
             width_, height_, fps_, frame_count_, bitrate_);
    
    // 4. 查找NVDEC解码器
    const AVCodec* decoder = nullptr;
    
    // 根据编码格式选择对应的CUDA解码器
    switch (codecpar->codec_id) {
        case AV_CODEC_ID_H264:
            decoder = avcodec_find_decoder_by_name("h264_cuvid");
            break;
        case AV_CODEC_ID_HEVC:
            decoder = avcodec_find_decoder_by_name("hevc_cuvid");
            break;
        case AV_CODEC_ID_VP9:
            decoder = avcodec_find_decoder_by_name("vp9_cuvid");
            break;
        case AV_CODEC_ID_AV1:
            decoder = avcodec_find_decoder_by_name("av1_cuvid");
            break;
        default:
            break;
    }
    
    // 如果没有对应的CUDA解码器，回退到软解码
    if (!decoder) {
        DBG_LOGW("GPUDecoder: NVDEC decoder not found for codec, falling back to software decoder\n");
        decoder = avcodec_find_decoder(codecpar->codec_id);
    }
    
    if (!decoder) {
        DBG_LOGE("GPUDecoder: Cannot find decoder\n");
        close();
        return false;
    }
    
    DBG_LOGI("GPUDecoder: Using decoder: %s\n", decoder->name);
    
    // 5. 创建解码器上下文
    codec_ctx_ = avcodec_alloc_context3(decoder);
    if (!codec_ctx_) {
        DBG_LOGE("GPUDecoder: Cannot allocate codec context\n");
        close();
        return false;
    }
    
    if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
        DBG_LOGE("GPUDecoder: Cannot copy codec parameters\n");
        close();
        return false;
    }
    
    // 6. 初始化CUDA硬件上下文
    if (!initHWContext()) {
        DBG_LOGW("GPUDecoder: Failed to init HW context, using software decoding\n");
        // 继续使用软解码
    }
    
    // 7. 打开解码器
    if (avcodec_open2(codec_ctx_, decoder, nullptr) < 0) {
        DBG_LOGE("GPUDecoder: Cannot open codec\n");
        close();
        return false;
    }
    
    // 8. 分配帧和包
    hw_frame_ = av_frame_alloc();
    sw_frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    
    if (!hw_frame_ || !sw_frame_ || !packet_) {
        DBG_LOGE("GPUDecoder: Cannot allocate frame/packet\n");
        close();
        return false;
    }
    
    // 9. 初始化GPU帧缓存池
    gpu_frame_pool_.resize(num_threads);
    for (int i = 0; i < num_threads; i++) {
        gpu_frame_pool_[i].create(height_, width_, CV_8UC3);
    }
    
    DBG_LOGI("GPUDecoder: Initialized %d GPU frame buffers\n", num_threads);
    
    is_opened_ = true;
    current_frame_ = -1;
    
    return true;
}

bool GPUDecoder::initHWContext() {
    // 创建CUDA硬件设备上下文
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (ret < 0) {
        DBG_LOGW("GPUDecoder: Failed to create CUDA device context\n");
        return false;
    }
    
    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    
    DBG_LOGI("GPUDecoder: CUDA hardware context initialized\n");
    return true;
}

void GPUDecoder::close() {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    
    if (packet_) {
        av_packet_free(&packet_);
        packet_ = nullptr;
    }
    
    if (hw_frame_) {
        av_frame_free(&hw_frame_);
        hw_frame_ = nullptr;
    }
    
    if (sw_frame_) {
        av_frame_free(&sw_frame_);
        sw_frame_ = nullptr;
    }
    
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
    }
    
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
    }
    
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    
    // 释放GPU帧缓存
    for (auto& gpu_frame : gpu_frame_pool_) {
        gpu_frame.release();
    }
    gpu_frame_pool_.clear();
    
    is_opened_ = false;
    current_frame_ = -1;
    video_stream_idx_ = -1;
}

bool GPUDecoder::seekToFrame(int frame_idx) {
    if (!fmt_ctx_ || video_stream_idx_ < 0) {
        return false;
    }
    
    AVStream* video_stream = fmt_ctx_->streams[video_stream_idx_];
    
    // 计算目标时间戳
    int64_t target_ts = av_rescale_q(frame_idx, 
                                      AVRational{1, fps_}, 
                                      video_stream->time_base);
    
    // seek到目标位置（向前seek到关键帧）
    int ret = av_seek_frame(fmt_ctx_, video_stream_idx_, target_ts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        DBG_LOGW("GPUDecoder: Seek failed, will decode from current position\n");
        return false;
    }
    
    // 清空解码器缓冲
    avcodec_flush_buffers(codec_ctx_);
    
    return true;
}

bool GPUDecoder::decodeOneFrame(int target_frame) {
    if (!is_opened_) {
        return false;
    }
    
    // 如果需要seek
    if (target_frame < current_frame_ || target_frame > current_frame_ + 30) {
        DBG_LOGI("GPUDecoder: SEEK triggered! target=%d current=%d\n", target_frame, current_frame_);
        seekToFrame(target_frame);
        current_frame_ = -1;
    }
    
    int frames_decoded = 0;
    // 解码直到目标帧
    while (true) {
        int ret = av_read_frame(fmt_ctx_, packet_);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                // 到达文件末尾，重新seek到开头
                av_seek_frame(fmt_ctx_, video_stream_idx_, 0, AVSEEK_FLAG_BACKWARD);
                avcodec_flush_buffers(codec_ctx_);
                current_frame_ = -1;
                continue;
            }
            return false;
        }
        
        if (packet_->stream_index != video_stream_idx_) {
            av_packet_unref(packet_);
            continue;
        }
        
        ret = avcodec_send_packet(codec_ctx_, packet_);
        av_packet_unref(packet_);
        
        if (ret < 0) {
            continue;
        }
        
        ret = avcodec_receive_frame(codec_ctx_, hw_frame_);
        if (ret == AVERROR(EAGAIN)) {
            continue;
        } else if (ret < 0) {
            return false;
        }
        
        current_frame_++;
        frames_decoded++;
        
        if (current_frame_ >= target_frame) {
            if (frames_decoded > 1) {
                DBG_LOGI("GPUDecoder: decoded %d frames to reach target %d\n", frames_decoded, target_frame);
            }
            return true;
        }
        
        av_frame_unref(hw_frame_);
    }
    
    return false;
}

void GPUDecoder::convertNV12ToBGR(AVFrame* frame, cv::cuda::GpuMat& output) {
    // NVDEC输出的是CUDA格式的NV12帧
    // 需要先下载到CPU，再转换格式，最后上传到GPU
    // 这是因为直接操作CUDA帧的内存布局比较复杂
    
    if (frame->format == AV_PIX_FMT_CUDA) {
        // 硬件帧：先传输到CPU
        av_frame_unref(sw_frame_);
        sw_frame_->format = AV_PIX_FMT_NV12;
        
        int ret = av_hwframe_transfer_data(sw_frame_, frame, 0);
        if (ret < 0) {
            DBG_LOGE("GPUDecoder: Failed to transfer HW frame to CPU\n");
            return;
        }
        
        // 现在sw_frame_包含NV12格式的CPU数据
        // 使用swscale转换为BGR
        if (!sws_ctx_) {
            sws_ctx_ = sws_getContext(sw_frame_->width, sw_frame_->height, 
                                       AV_PIX_FMT_NV12,
                                       sw_frame_->width, sw_frame_->height, 
                                       AV_PIX_FMT_BGR24,
                                       SWS_BILINEAR, nullptr, nullptr, nullptr);
        }
        
        // 分配BGR帧
        AVFrame* bgr_frame = av_frame_alloc();
        bgr_frame->format = AV_PIX_FMT_BGR24;
        bgr_frame->width = sw_frame_->width;
        bgr_frame->height = sw_frame_->height;
        av_frame_get_buffer(bgr_frame, 32);
        
        // NV12 -> BGR
        sws_scale(sws_ctx_, sw_frame_->data, sw_frame_->linesize, 0, sw_frame_->height,
                  bgr_frame->data, bgr_frame->linesize);
        
        // 上传到GPU
        cv::Mat cpu_frame(bgr_frame->height, bgr_frame->width, CV_8UC3, 
                          bgr_frame->data[0], bgr_frame->linesize[0]);
        output.upload(cpu_frame, cuda_stream_);
        
        av_frame_free(&bgr_frame);
        
    } else {
        // 软解码的情况，帧在CPU内存中
        if (!sws_ctx_) {
            sws_ctx_ = sws_getContext(frame->width, frame->height, 
                                       (AVPixelFormat)frame->format,
                                       frame->width, frame->height, 
                                       AV_PIX_FMT_BGR24,
                                       SWS_BILINEAR, nullptr, nullptr, nullptr);
        }
        
        // 分配BGR帧
        av_frame_unref(sw_frame_);
        sw_frame_->format = AV_PIX_FMT_BGR24;
        sw_frame_->width = frame->width;
        sw_frame_->height = frame->height;
        av_frame_get_buffer(sw_frame_, 32);
        
        // 转换
        sws_scale(sws_ctx_, frame->data, frame->linesize, 0, frame->height,
                  sw_frame_->data, sw_frame_->linesize);
        
        // 上传到GPU
        cv::Mat cpu_frame(frame->height, frame->width, CV_8UC3, 
                          sw_frame_->data[0], sw_frame_->linesize[0]);
        output.upload(cpu_frame, cuda_stream_);
    }
    
    cuda_stream_.waitForCompletion();
}

cv::cuda::GpuMat& GPUDecoder::decodeFrame(int frame_idx, int thread_id) {
    double t_start = (double)cv::getTickCount();
    
    std::lock_guard<std::mutex> lock(decode_mutex_);
    double t_lock = (double)cv::getTickCount();
    
    if (!is_opened_ || thread_id < 0 || thread_id >= (int)gpu_frame_pool_.size()) {
        DBG_LOGE("GPUDecoder: Invalid state or thread_id\n");
        static cv::cuda::GpuMat empty;
        return empty;
    }
    
    // 计算实际帧索引（处理循环播放）
    int actual_frame = frame_idx % frame_count_;
    
    // 解码帧
    double t_decode_start = (double)cv::getTickCount();
    if (!decodeOneFrame(actual_frame)) {
        DBG_LOGE("GPUDecoder: Failed to decode frame %d\n", actual_frame);
        return gpu_frame_pool_[thread_id];
    }
    double t_decode_end = (double)cv::getTickCount();
    
    // 转换并存入对应线程的GPU帧缓存
    double t_convert_start = (double)cv::getTickCount();
    convertNV12ToBGR(hw_frame_, gpu_frame_pool_[thread_id]);
    double t_convert_end = (double)cv::getTickCount();
    
    av_frame_unref(hw_frame_);
    
    double t_end = (double)cv::getTickCount();
    double freq = cv::getTickFrequency();
    
    DBG_LOGI("GPUDecoder::decodeFrame[%d] frame=%d actual=%d | wait_lock=%.1fms decode=%.1fms convert=%.1fms total=%.1fms\n",
             thread_id, frame_idx, actual_frame,
             (t_lock - t_start) * 1000 / freq,
             (t_decode_end - t_decode_start) * 1000 / freq,
             (t_convert_end - t_convert_start) * 1000 / freq,
             (t_end - t_start) * 1000 / freq);
    
    return gpu_frame_pool_[thread_id];
}

cv::cuda::GpuMat& GPUDecoder::getGpuFrame(int thread_id) {
    if (thread_id < 0 || thread_id >= (int)gpu_frame_pool_.size()) {
        static cv::cuda::GpuMat empty;
        return empty;
    }
    return gpu_frame_pool_[thread_id];
}
