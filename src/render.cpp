#include "talkingface.h"


Status TalkingFace::render(const char *src_video_path, 
                           const char *audio_path, 
                           const char *json_save_path, 
                           const char *render_video_save_path, 
                           const char *set_params,
                           const char *vocal_audio_path,
                           const char *id_params)
{
    Status status;
    infos.reset();

    // 创建tmp路径
    std::string dir_command = "mkdir -p " + (std::string)tmp_frame_dir;
    system(dir_command.c_str());

    // 获取传参
    this->readVideoParam(set_params);

    // 获取音频时长
    this->readAudioInfo(audio_path);

    // 读取底板视频 - 使用GPU解码器
    DBG_LOGI("read video start (GPU decoder).\n");
    double t0 = (double)cv::getTickCount();
    
    // 先获取视频基本信息
    int frame_width, frame_height, fps;
    long bitrate;
    Status video_info_status = this->readVideoInfo(src_video_path, frame_width, frame_height, fps, bitrate);
    if (!video_info_status.IsOk())
        return video_info_status;
    
    // 计算目标分辨率
    infos.video_width = video_params.width == 0 ? frame_width : video_params.width;
    infos.video_height = video_params.height == 0 ? frame_height : video_params.height;
    int max_side = infos.video_width >= infos.video_height ? infos.video_width : infos.video_height;
    if (video_params.video_max_side > 0 && video_params.video_max_side < max_side)
    {
        float scale = static_cast<float>(max_side) / static_cast<float>(video_params.video_max_side);
        if (infos.video_height >= infos.video_width)
        {
            infos.video_height = video_params.video_max_side;
            infos.video_width = static_cast<int>(static_cast<float>(infos.video_width) / scale);
        }
        else
        {
            infos.video_height = static_cast<int>(static_cast<float>(infos.video_height / scale));
            infos.video_width = video_params.video_max_side;
        }
    }
    if (infos.video_height % 2 == 1) infos.video_height++;
    if (infos.video_width % 2 == 1) infos.video_width++;
    
    // fps设置
    infos.fps = (video_params.keep_fps == 0) ? 25 : fps;
    if (infos.fps > 60) infos.fps = 60;
    infos.bitrate = bitrate;
    
    // 初始化GPU解码器
    gpu_decoder_ = new GPUDecoder();
    if (!gpu_decoder_->open(src_video_path, n_threads, infos.fps)) {
        DBG_LOGE("GPU decoder open failed.\n");
        delete gpu_decoder_;
        gpu_decoder_ = nullptr;
        return Status(Status::Code::VIDEO_READ_FAIL, "GPU decoder open failed.");
    }
    
    // 减10帧避免访问到末尾可能的空帧（视频编码器通常会在末尾添加几个空帧）
    infos.frame_nums = gpu_decoder_->getFrameCount() - 10;
    if (infos.frame_nums < 1) {
        infos.frame_nums = 1;
        DBG_LOGW("Video too short, frame_nums clamped to 1\n");
    }
    
    double t1 = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    DBG_LOGI("read video finish (GPU decoder) cost time: %.2fs, frames: %d.\n", t1, infos.frame_nums);

    // 单人特例：读取遮挡判断和说话人判断的json文件
    bool json_ret = this->readJsonFile(json_save_path);
    if (json_ret)
    {
        // id
        infos.ids.emplace_back(0);

        // roi
        cv::Rect2i roi(0, 0, infos.video_width, infos.video_height);
        infos.id_rois.emplace_back(roi);

        // audio
        if (access(vocal_audio_path, F_OK) == -1)
        {
            Status audio_status = this->extractAudioFeat(audio_path);
            if (!audio_status.IsOk())
                return audio_status;
        }
        else
        {
            Status audio_status = this->extractAudioFeat(vocal_audio_path);
            if (!audio_status.IsOk())
                return audio_status;
        }
        infos.min_audio_cnt = infos.audio_cnts[0];
    }
    
    else
    {
        // 多人处理，如果为单人，也统一为多人格式
        Status id_status;
        if (id_params[0] == '\0')
        {
            // 把单人处理成多人传参
            std::string id_params_str;
            id_params_str += "[{\"id\":0,\"box\":[";
            id_params_str += std::to_string(video_params.roi_rect.x) + "," + std::to_string(video_params.roi_rect.y) + ",";
            id_params_str += std::to_string(video_params.roi_rect.width) + "," + std::to_string(video_params.roi_rect.height) + "],";
            if (access(vocal_audio_path, F_OK) == -1)
                id_params_str += ("\"audio\":\"" + std::string(audio_path) + "\"");
            else
                id_params_str += ("\"audio\":\"" + std::string(vocal_audio_path) + "\"");
            id_params_str += "}]";
            id_status = this->ProcessIDParam(id_params_str.c_str());
        }
        else
            id_status = this->ProcessIDParam(id_params);
        if (!id_status.IsOk())
            return id_status;
    }

    // 校验音频特征长度
    if (infos.min_audio_cnt < 1)
        return Status(Status::Code::AUDIO_FEAT_EXTRACT_FAIL, "audio feat length error.");

    // 开启渲染线程
    std::vector<std::thread> render_threads;
    for (int i = 0; i < n_threads; i++)
        render_threads.emplace_back(std::thread(&TalkingFace::renderProducer, this, i));

    // 开启写入线程
    infos.video_writer.open(tmp_video_path, infos.render_codec, infos.fps, cv::Size2d(infos.video_width, infos.video_height));
    std::thread writer_thread(&TalkingFace::writeConsumer, this);

    for (int i = 0; i < n_threads; i++)
        render_threads[i].join();
    writer_thread.join();

    // 校验渲染视频
    try
    {
        cv::VideoCapture cap;
        cap.open(tmp_video_path);
        if (!cap.isOpened())
        {
            DBG_LOGE("tmp video check fail.\n");
            return Status(Status::Code::AUDIO_DUBBING_FAIL, "tmp video check fail!");
        }
        cap.release();
    }
    catch(...)
    {
        DBG_LOGE("tmp video check fail.\n");
        return Status(Status::Code::AUDIO_DUBBING_FAIL, "tmp video check fail!");
    }

    // 配音
    Status dubbing_status = this->audioDubbing(audio_path, render_video_save_path);
    
    // 清理GPU解码器
    if (gpu_decoder_) {
        gpu_decoder_->close();
        delete gpu_decoder_;
        gpu_decoder_ = nullptr;
    }
    
    if (!dubbing_status.IsOk())
        return dubbing_status;

    if (!status.IsOk())
        return status;
    return Status(Status::Code::SUCCESS, "success");
}
