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
    
    // 计算原始帧率
    if (video_stream->avg_frame_rate.den > 0) {
        original_fps_ = video_stream->avg_frame_rate.num / video_stream->avg_frame_rate.den;
    } else if (video_stream->r_frame_rate.den > 0) {
        original_fps_ = video_stream->r_frame_rate.num / video_stream->r_frame_rate.den;
    }
    
    // 设置目标帧率
    if (target_fps > 0) {
        target_fps_ = target_fps;
    } else {
        target_fps_ = original_fps_;  // 默认保持原fps
    }
    
    // fps_用于输出和seek计算，设为目标fps
    fps_ = target_fps_;
    
    // 计算原始总帧数（注意：必须用original_fps_，因为frame_count_是原始帧数）
    if (video_stream->nb_frames > 0) {
        frame_count_ = video_stream->nb_frames;
    } else if (video_stream->duration > 0) {
        double duration_sec = video_stream->duration * av_q2d(video_stream->time_base);
        frame_count_ = static_cast<int>(duration_sec * original_fps_);
    } else if (fmt_ctx_->duration > 0) {
        double duration_sec = fmt_ctx_->duration / (double)AV_TIME_BASE;
        frame_count_ = static_cast<int>(duration_sec * original_fps_);
    }
    
    DBG_LOGI("GPUDecoder: Video info - %dx%d, original_fps=%d, target_fps=%d, frames=%d, target_frames=%d, %ld kbps\n", 
             width_, height_, original_fps_, target_fps_, frame_count_, getTargetFrameCount(), bitrate_);
    
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
    
    // 重置sws缓存参数
    last_sws_src_format_ = AV_PIX_FMT_NONE;
    last_sws_src_width_ = 0;
    last_sws_src_height_ = 0;
    
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
    is_draining_ = false;
    current_frame_ = -1;
    last_decoded_thread_id_ = -1;
    last_decoded_frame_idx_ = -1;
    video_stream_idx_ = -1;
}

bool GPUDecoder::seekToFrame(int frame_idx) {
    if (!fmt_ctx_ || video_stream_idx_ < 0) {
        return false;
    }
    
    AVStream* video_stream = fmt_ctx_->streams[video_stream_idx_];
    
    // 计算目标时间戳（使用原始fps，因为frame_idx是原始帧索引）
    int64_t target_ts = av_rescale_q(frame_idx, 
                                      AVRational{1, original_fps_}, 
                                      video_stream->time_base);
    
    // seek到目标位置（向前seek到关键帧）
    int ret = av_seek_frame(fmt_ctx_, video_stream_idx_, target_ts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        DBG_LOGW("GPUDecoder: Seek failed, will decode from current position\n");
        return false;
    }
    
    // 清空解码器缓冲，重置drain状态
    avcodec_flush_buffers(codec_ctx_);
    is_draining_ = false;
    
    return true;
}

bool GPUDecoder::decodeOneFrame(int target_frame) {
    if (!is_opened_) {
        return false;
    }
    
    // 保护：目标帧不能超过有效范围
    if (target_frame >= frame_count_ ) {
        DBG_LOGW("GPUDecoder: target_frame %d too close to end (frame_count=%d), clamping\n", 
                 target_frame, frame_count_);
        target_frame = frame_count_ - 1;
        if (target_frame < 0) target_frame = 0;
    }
    
    // 判断是否需要跳过或SEEK
    if (current_frame_ >= 0 && target_frame <= current_frame_) {
        int gap = current_frame_ - target_frame;
        // fps_ratio < 1 时，相邻帧映射到同一个 actual_decode_idx，gap 最多为1-2
        // 如果 gap > 2，很可能是视频循环（即使视频只有几帧）
        if (gap <= 5) {
            // fps_ratio < 1 导致的重复请求，跳过解码
            // DBG_LOGI("GPUDecoder: skip decode, target=%d <= current=%d (gap=%d)\n", target_frame, current_frame_, gap);
            return false;  // 跳过，复用已有帧
        } else {
            // 视频循环：需要 SEEK 回去
            DBG_LOGI("GPUDecoder: SEEK backward (loop)! target=%d current=%d\n", target_frame, current_frame_);
            seekToFrame(target_frame);
            current_frame_ = -1;
        }
    } else if (target_frame > current_frame_ + 30) {
        // 向前跳跃太大，需要 SEEK
        DBG_LOGI("GPUDecoder: SEEK forward! target=%d current=%d\n", target_frame, current_frame_);
        seekToFrame(target_frame);
        current_frame_ = -1;
    }
    
    int frames_decoded = 0;
    
    // 如果已经在drain模式，直接从缓冲区获取帧
    if (is_draining_) {
        while (true) {
            int ret = avcodec_receive_frame(codec_ctx_, hw_frame_);
            if (ret == AVERROR_EOF || ret < 0) {
                // 缓冲区也空了，真正没有更多帧了
                DBG_LOGW("GPUDecoder: EOF (drain exhausted) at frame %d (target=%d), will reuse last frame\n", 
                         current_frame_, target_frame);
                return false;
            }
            
            current_frame_++;
            frames_decoded++;
            
            if (current_frame_ >= target_frame) {
                return true;
            }
            av_frame_unref(hw_frame_);
        }
    }
    
    // 正常解码模式：解码直到目标帧
    while (true) {
        int ret = av_read_frame(fmt_ctx_, packet_);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                // 文件读完了，但解码器缓冲区里可能还有帧（B帧延迟）
                // 发送空包来刷新解码器，进入drain模式
                avcodec_send_packet(codec_ctx_, nullptr);
                is_draining_ = true;
                
                // 尝试从缓冲区获取剩余帧
                while (true) {
                    ret = avcodec_receive_frame(codec_ctx_, hw_frame_);
                    if (ret == AVERROR_EOF || ret < 0) {
                        // 缓冲区也空了，真正没有更多帧了
                        DBG_LOGW("GPUDecoder: EOF reached at frame %d (target=%d), will reuse last frame\n", 
                                 current_frame_, target_frame);
                        return false;
                    }
                    
                    current_frame_++;
                    frames_decoded++;
                    
                    if (current_frame_ >= target_frame) {
                        return true;
                    }
                    av_frame_unref(hw_frame_);
                }
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
            // if (frames_decoded > 1) {
            //     DBG_LOGI("GPUDecoder: decoded %d frames to reach target %d\n", frames_decoded, target_frame);
            // }
            return true;
        }
        
        av_frame_unref(hw_frame_);
    }
    
    return false;
}

void GPUDecoder::convertNV12ToBGR(AVFrame* frame, cv::cuda::GpuMat& output) {
    // NVDEC输出的是CUDA格式的NV12帧
    // 需要先下载到CPU，再转换格式，最后上传到GPU
    
    AVPixelFormat src_format;
    int src_width, src_height;
    AVFrame* src_frame = nullptr;
    
    if (frame->format == AV_PIX_FMT_CUDA) {
        // 硬件帧：先传输到CPU
        av_frame_unref(sw_frame_);
        // 不要预设format，让av_hwframe_transfer_data自动设置
        
        int ret = av_hwframe_transfer_data(sw_frame_, frame, 0);
        if (ret < 0) {
            DBG_LOGE("GPUDecoder: Failed to transfer HW frame to CPU\n");
            return;
        }
        
        // 获取实际的像素格式（可能是NV12或其他格式）
        src_format = (AVPixelFormat)sw_frame_->format;
        src_width = sw_frame_->width;
        src_height = sw_frame_->height;
        src_frame = sw_frame_;
        
        // 调试信息：打印实际格式和linesize
        static bool format_logged = false;
        if (!format_logged) {
            DBG_LOGI("GPUDecoder: HW frame transferred, format=%d (%s), size=%dx%d, linesize=[%d,%d,%d,%d]\n",
                     src_format, av_get_pix_fmt_name(src_format),
                     src_width, src_height,
                     sw_frame_->linesize[0], sw_frame_->linesize[1], 
                     sw_frame_->linesize[2], sw_frame_->linesize[3]);
            format_logged = true;
        }
    } else {
        // 软解码的情况，帧在CPU内存中
        src_format = (AVPixelFormat)frame->format;
        src_width = frame->width;
        src_height = frame->height;
        src_frame = frame;
        
        // 调试信息
        static bool sw_format_logged = false;
        if (!sw_format_logged) {
            DBG_LOGI("GPUDecoder: SW frame, format=%d (%s), size=%dx%d, linesize=[%d,%d,%d,%d]\n",
                     src_format, av_get_pix_fmt_name(src_format),
                     src_width, src_height,
                     frame->linesize[0], frame->linesize[1], 
                     frame->linesize[2], frame->linesize[3]);
            sw_format_logged = true;
        }
    }
    
    // 检查是否需要重新创建 sws_ctx_（源格式或尺寸变化时）
    if (sws_ctx_ == nullptr || 
        last_sws_src_format_ != src_format || 
        last_sws_src_width_ != src_width || 
        last_sws_src_height_ != src_height) {
        
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        
        // 使用SWS_ACCURATE_RND和SWS_FULL_CHR_H_INT以获得更准确的颜色转换
        sws_ctx_ = sws_getContext(src_width, src_height, src_format,
                                   src_width, src_height, AV_PIX_FMT_BGR24,
                                   SWS_BILINEAR | SWS_ACCURATE_RND, 
                                   nullptr, nullptr, nullptr);
        
        if (!sws_ctx_) {
            DBG_LOGE("GPUDecoder: Failed to create sws context for format %d (%s)\n", 
                     src_format, av_get_pix_fmt_name(src_format));
            return;
        }
        
        last_sws_src_format_ = src_format;
        last_sws_src_width_ = src_width;
        last_sws_src_height_ = src_height;
        
        DBG_LOGI("GPUDecoder: Created sws context for %dx%d format=%d (%s)\n", 
                 src_width, src_height, src_format, av_get_pix_fmt_name(src_format));
    }
    
    // 使用av_image_alloc创建对齐的缓冲区（消除swscaler警告，提高性能）
    uint8_t* dst_data[4] = { nullptr, nullptr, nullptr, nullptr };
    int dst_linesize[4] = { 0, 0, 0, 0 };
    
    int ret = av_image_alloc(dst_data, dst_linesize, 
                              src_width, src_height, AV_PIX_FMT_BGR24, 32);
    if (ret < 0) {
        DBG_LOGE("GPUDecoder: av_image_alloc failed\n");
        return;
    }
    
    // 执行颜色空间转换
    ret = sws_scale(sws_ctx_, 
                    src_frame->data, src_frame->linesize, 
                    0, src_height,
                    dst_data, dst_linesize);
    
    if (ret != src_height) {
        DBG_LOGE("GPUDecoder: sws_scale failed, ret=%d expected=%d\n", ret, src_height);
        av_freep(&dst_data[0]);
        return;
    }
    
    // 创建cv::Mat包装对齐的数据，然后拷贝到连续内存
    cv::Mat aligned_frame(src_height, src_width, CV_8UC3, dst_data[0], dst_linesize[0]);
    cv::Mat cpu_frame;
    aligned_frame.copyTo(cpu_frame);  // 拷贝到连续内存的Mat
    
    // 释放对齐缓冲区
    av_freep(&dst_data[0]);
    
    // 上传到GPU
    output.upload(cpu_frame, cuda_stream_);
    cuda_stream_.waitForCompletion();
}

cv::cuda::GpuMat& GPUDecoder::decodeFrame(int frame_idx, int thread_id) {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    
    if (!is_opened_ || thread_id < 0 || thread_id >= (int)gpu_frame_pool_.size()) {
        DBG_LOGE("GPUDecoder: Invalid state or thread_id\n");
        static cv::cuda::GpuMat empty;
        return empty;
    }
    
    // 计算实际帧索引（处理循环播放）
    int actual_frame = frame_idx % frame_count_;
    
    // 解码帧
    bool decoded = decodeOneFrame(actual_frame);
    
    if (!decoded) {
        // 跳过解码时，需要从最后解码的线程复制帧到当前线程
        if (last_decoded_thread_id_ >= 0 && 
            last_decoded_thread_id_ != thread_id &&
            last_decoded_thread_id_ < (int)gpu_frame_pool_.size() &&
            !gpu_frame_pool_[last_decoded_thread_id_].empty()) {
            gpu_frame_pool_[last_decoded_thread_id_].copyTo(gpu_frame_pool_[thread_id]);
        }
        return gpu_frame_pool_[thread_id];
    }
    
    // 检查 hw_frame_ 是否有效
    if (!hw_frame_ || hw_frame_->width <= 0 || hw_frame_->height <= 0) {
        DBG_LOGE("GPUDecoder: hw_frame_ is invalid after decode\n");
        return gpu_frame_pool_[thread_id];
    }
    
    // 转换并存入对应线程的GPU帧缓存
    convertNV12ToBGR(hw_frame_, gpu_frame_pool_[thread_id]);
    
    // 记录最后解码的线程ID
    last_decoded_thread_id_ = thread_id;
    
    av_frame_unref(hw_frame_);
    
    return gpu_frame_pool_[thread_id];
}

cv::cuda::GpuMat& GPUDecoder::getGpuFrame(int thread_id) {
    if (thread_id < 0 || thread_id >= (int)gpu_frame_pool_.size()) {
        static cv::cuda::GpuMat empty;
        return empty;
    }
    return gpu_frame_pool_[thread_id];
}
