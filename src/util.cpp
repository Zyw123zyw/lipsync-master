#include "talkingface.h"

void TalkingFace::traverseFiles(std::string path, std::vector<std::string> &filenames, const std::string& extension)
{
    DIR *pDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str())))
    {
        return;
    }
    filenames.clear();
    while ((ptr = readdir(pDir)) != 0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            std::string file_path = path + "/" + ptr->d_name;
            std::string file_ext = file_path.substr(file_path.find_last_of(".")+1);
            if (file_ext == extension)
                filenames.push_back(file_path);
        }
    }
    closedir(pDir);
    std::sort(filenames.begin(), filenames.end());
}

bool TalkingFace::fileExists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}

void TalkingFace::drawBox(cv::Mat &img, cv::Rect2i &rect)
{
    cv::Scalar color(0, 255, 0);
    cv::rectangle(img, rect, color, 2);
}

void TalkingFace::drawLandmarks(cv::Mat &img, std::vector<cv::Point2i> &landmark_pts)
{
    cv::Scalar color(0, 0, 255);
    int num_pts = landmark_pts.size();
    for (int i = 0; i < num_pts; i++)
    {
        cv::Point2i pt;
        pt.x = (int)landmark_pts[i].x;
        pt.y = (int)landmark_pts[i].y;
        cv::circle(img, pt, 2, color, -1);
    }
}

// 读取预处理（说话人判断，遮挡判断）得到的人脸信息
bool TalkingFace::readJsonFile(const char *json_save_path)
{
    try
    {
        std::ifstream jsonFile(json_save_path);
        if (!jsonFile.is_open())
        {
            DBG_LOGI("json file read fail.\n");
            return false;
        }

        Json::Reader info_reader;
        Json::Value info_root;

        // 判断能否打开
        if (!info_reader.parse(jsonFile, info_root))
        {
            DBG_LOGI("json file parse fail.\n");
            return false;
        }
        
        // 校验视频信息
        if (!info_root.isMember("video_info") || !info_root.isMember("face_bbox_info") || !info_root.isMember("face_landmark_info"))
        {
            DBG_LOGI("json file read fail.\n");
            return false;
        }
        int info_width = info_root["video_info"]["video_frame_width"].asInt();
        int info_height = info_root["video_info"]["video_frame_height"].asInt();
        int info_frame_nums = info_root["video_info"]["video_frame_nums"].asInt();
        if (infos.video_height != info_height || infos.video_width != info_width || infos.frame_nums != info_frame_nums)
        {
            DBG_LOGI("json file read fail.\n");
            return false;
        }
        if (info_root["face_bbox_info"].size() != info_root["face_landmark_info"].size())
        {
            DBG_LOGI("json file read fail.\n");
            return false;
        }

        // 读取json内容
        std::vector<cv::Rect2i> bboxes;
        std::vector<std::vector<cv::Point>> landmarks;
        for (size_t i = 0; i < info_root["face_bbox_info"].size(); i++)
        {
            char buffer[10];
            std::sprintf(buffer, "%06d", i);
            std::string indexStr = std::string(buffer);

            // box
            int x = info_root["face_bbox_info"][indexStr][0].asInt();
            int y = info_root["face_bbox_info"][indexStr][1].asInt();
            int w = info_root["face_bbox_info"][indexStr][2].asInt();
            int h = info_root["face_bbox_info"][indexStr][3].asInt();
            cv::Rect2i box(x, y, w, h);
            bboxes.emplace_back(box);

            // landmark
            std::vector<cv::Point> pts;
            for (unsigned int j = 0; j < 68; j++)
            {
                int x_ = info_root["face_landmark_info"][indexStr][2*j].asInt();
                int y_ = info_root["face_landmark_info"][indexStr][2*j+1].asInt();
                cv::Point pt(x_, y_);
                pts.emplace_back(pt);
            }
            landmarks.emplace_back(pts);
        }
        infos.face_bboxes.emplace_back(bboxes);
        infos.face_landmarks.emplace_back(landmarks);
    }
    catch(...)
    {
        DBG_LOGI("json file read fail.\n");
        return false;
    }
    return true;
}

Status TalkingFace::ProcessIDParam(const char *id_params)
{
    Status status;
    try
    {
        Json::Reader reader;
        Json::Value root;

        if (reader.parse((std::string)id_params, root))
        {
            // id image信息
            for (unsigned int i = 0; i < root.size(); i++)
            {
                // id idx
                infos.ids.emplace_back(i);

                // box
                cv::Rect2i box;
                box.x = root[i]["box"][0].asInt();
                box.y = root[i]["box"][1].asInt();
                box.width = root[i]["box"][2].asInt();
                box.height = root[i]["box"][3].asInt();
                this->checkROI(box);    // 检查box的有效性
                infos.id_rois.emplace_back(box);
                DBG_LOGI("id %d, box : [%d, %d, %d, %d]\n", i, box.x, box.y, box.width, box.height);

                // face
                // std::string face_path = root[i]["face"].asString();
                // cv::Mat id_image = cv::imread(face_path);

                // audio
                std::string audio_path = root[i]["audio"].asString();
                DBG_LOGI("id %d, audio %s\n", i, audio_path.c_str());
                Status ret_status = this->extractAudioFeat(audio_path.c_str());
                if (!ret_status.IsOk())
                    return ret_status;
            }

            int min_length = infos.audio_cnts[0];
            for (int id: infos.ids)
            {
                int length = infos.audio_cnts[id];
                if (length < min_length)
                    min_length = length;
            }
            infos.min_audio_cnt = min_length;
        }
    }
    catch(...)
    {
        DBG_LOGI("read id param fail..\n");
        return Status(Status::Code::READ_ID_PARAM_FAIL, "read id param fail.");
    }
    return status;
}

void TalkingFace::readVideoParam(const char *set_params)
{
    video_params.reset();
    try
    {
        Json::Reader reader;
        Json::Value root;

        if (reader.parse((std::string)set_params, root))
        {
            /*输入判断参数*/
            // 长边限制，避免内存打爆
            if (root.isMember("video_max_side"))
            {
                if (root["video_max_side"].asInt() > 0 && root["video_max_side"].asInt() < video_params.video_max_side)
                    video_params.video_max_side = root["video_max_side"].asInt();
            }

            // 音频时长限制
            if (root.isMember("audio_max_time"))
            {
                if (root["audio_max_time"].asInt() > 0)
                    video_params.audio_max_time = root["audio_max_time"].asInt();
            }

            // roi
            if (root.isMember("face_box"))
            {
                if (root["face_box"][2].asInt() > 0 && root["face_box"][3].asInt() > 0)
                {
                    video_params.roi_rect.x = root["face_box"][0].asInt();
                    video_params.roi_rect.y = root["face_box"][1].asInt();
                    video_params.roi_rect.width = root["face_box"][2].asInt();
                    video_params.roi_rect.height = root["face_box"][3].asInt();
                }
            }

            // 侧脸检测
            if (root.isMember("filter_head_pose"))
            {
                if (root["filter_head_pose"].asInt() == 1)
                {
                    video_params.filter_head_pose = true;
                }
            }

            // 人脸检测阈值
            if (root.isMember("face_det_threshold"))
            {
                if (root["face_det_threshold"].asFloat() > 0 && root["face_det_threshold"].asFloat() < 1)
                    video_params.face_det_threshold = root["face_det_threshold"].asFloat();
            }

            // 说话张嘴幅度系数
            if (root.isMember("amplifier"))
            {
                if (root["amplifier"].asFloat() > 0)
                    video_params.amplifier = root["amplifier"].asFloat();
            }

            /*输出视频设置参数*/
            // 合成视频的码率
            if (root.isMember("video_bitrate"))
            {
                video_params.bitrate = root["video_bitrate"].asInt();
            }

            if (root.isMember("video_max_bitrate"))
            {
                video_params.max_bitrate = root["video_max_bitrate"].asInt();
            }

            // 合成视频的分辨率
            if (root.isMember("video_width"))
            {
                if (root["video_width"].asInt() > 0)
                    video_params.width = root["video_width"].asInt();
            }

            if (root.isMember("video_height"))
            {
                if (root["video_height"].asInt() > 0)
                    video_params.height = root["video_height"].asInt();
            }

            // 视频超分/增强
            if (root.isMember("video_enhance"))
            {
                video_params.video_enhance = root["video_enhance"].asInt();
            }

            // 首帧静音驱动
            if (root.isMember("shutup_first"))
            {
                video_params.shutup_first = root["shutup_first"].asInt();
            }

            // 保持视频帧率
            if (root.isMember("keep_fps"))
            {
                if (root["keep_fps"].asInt() == 1)
                    video_params.keep_fps = 1;
            }

            // 保持视频码率
            if (root.isMember("keep_bitrate"))
            {
                if (root["keep_bitrate"].asInt() == 1)
                    video_params.keep_bitrate = 1;
            }
        }
    }
    catch(...)
    {
        DBG_LOGE("video params failed, please check video parmas.\n");
    }

    DBG_LOGI("param face box : [%d, %d, %d, %d]\n", video_params.roi_rect.x, video_params.roi_rect.y, video_params.roi_rect.width, video_params.roi_rect.height);
    DBG_LOGI("param filter_head_pose : %d\n", video_params.filter_head_pose);
    DBG_LOGI("param face det threshold : %.2f\n", video_params.face_det_threshold);
    DBG_LOGI("param amplifier : %.1f\n", video_params.amplifier);
    
    DBG_LOGI("param bitrate : %dk\n", video_params.bitrate);
    DBG_LOGI("param max bitrate : %dk\n", video_params.max_bitrate);
    DBG_LOGI("param width x height : [%d, %d]\n", video_params.width, video_params.height);
    DBG_LOGI("param video_enhance : %d\n", video_params.video_enhance);
    DBG_LOGI("param shutup_first : %d\n", video_params.shutup_first);

    DBG_LOGI("param keep_fps : %d\n", video_params.keep_fps);
    DBG_LOGI("param keep_bitrate : %d\n", video_params.keep_bitrate);

    DBG_LOGI("param video_max_side : %d\n", video_params.video_max_side);
    DBG_LOGI("param audio_max_time : %d\n", video_params.audio_max_time);
}

std::string TalkingFace::exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    // 使用 popen 执行命令，并通过管道读取输出
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() 调用失败！");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

Status TalkingFace::readVideoInfo(const char *src_video_path, int&  frame_width, int& frame_height, int& fps, long& bitrate)
{
    try
    {
        cv::VideoCapture vcap;
        vcap.open(src_video_path);
        if (!vcap.isOpened())
        {
            DBG_LOGE("video open failed.\n");
            return Status(Status::Code::VIDEO_READ_FAIL, "video open failed.");
        }
        frame_width = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_WIDTH));
        frame_height = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps = static_cast<int>(vcap.get(5) + 0.5);
        vcap.release();
    }
    catch(...)
    {
        DBG_LOGE("video open failed.\n");
        return Status(Status::Code::VIDEO_READ_FAIL, "video open failed.");
    }

    // 获取视频码率
    try
    {
        // 首先尝试从视频流读取码率
        std::string vb_command = "ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 " + std::string(src_video_path);
        std::string bitrate_str = exec(vb_command.c_str());
        bitrate_str.erase(bitrate_str.find_last_not_of(" \n\r\t") + 1);

        // 如果视频流码率为N/A，则从format级别读取总码率
        if (bitrate_str == "N/A" || bitrate_str.empty())
        {
            DBG_LOGI("stream bit_rate is N/A, try to get from format\n");
            std::string fb_command = "ffprobe -v error -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1 " + std::string(src_video_path);
            bitrate_str = exec(fb_command.c_str());
            bitrate_str.erase(bitrate_str.find_last_not_of(" \n\r\t") + 1);
        }

        bitrate = std::stol(bitrate_str);
        bitrate = bitrate / 1024;  // 转换为 kbps
    }
    catch(...)
    {
        DBG_LOGE("video info read failed, get bitrate fail\n");
        return Status(Status::Code::VIDEO_READ_FAIL, "video info read failed, get bitrate fail");
    }

    return Status(Status::Code::SUCCESS, "success");
}

void TalkingFace::readAudioInfo(const char *audio_path)
{
    try
    {
        std::string at_command = "ffprobe -v error -show_entries stream=duration -of csv=p=0 " + std::string(audio_path);
        std::string at_str = exec(at_command.c_str());
        int audio_duration = std::stoi(at_str);
        if (video_params.audio_max_time > 0)
            video_params.video_ffmpeg_duration = std::min(video_params.audio_max_time, audio_duration);
        else
            video_params.video_ffmpeg_duration = audio_duration;
    }
    catch(...)
    {
        DBG_LOGE("audio info read failed\n");
    }
}


Status TalkingFace::readVideo(const char *src_video_path)
{
    try
    {
        // 判断视频是否受损
        int frame_width;
        int frame_height;
        int fps;
        long bitrate;

        bool is_convert = false;

        Status video_status = readVideoInfo(src_video_path, frame_width, frame_height, fps, bitrate);
        if (!video_status.IsOk())
        {
            // 视频转为标准兼容格式 - 同时构建CUDA和CPU两个版本的命令
            std::string convert_command_cuda = "ffmpeg -loglevel quiet -y ";
            std::string convert_command_cpu = "ffmpeg -loglevel quiet -y ";

            // CUDA版本：添加CUDA解码参数
            std::string decoder_args = ffmpeg_config.getDecoderArgs();
            if (!decoder_args.empty()) {
                convert_command_cuda += (decoder_args + " ");
            }

            // 两个版本都添加输入文件
            convert_command_cuda += ("-i " + std::string(src_video_path) + " ");
            convert_command_cpu += ("-i " + std::string(src_video_path) + " ");

            // CUDA版本：添加CUDA编码参数
            std::string encoder_args_cuda = ffmpeg_config.getEncoderArgs(0, 0, false, false);
            convert_command_cuda += encoder_args_cuda;

            // CPU版本：使用CPU编码
            std::string encoder_args_cpu = ffmpeg_config.getEncoderArgs(0, 0, false, true);
            convert_command_cpu += encoder_args_cpu;

            // 两个版本都添加音频拷贝
            convert_command_cuda += " -c:a copy";
            convert_command_cpu += " -c:a copy";

            // 添加线程参数
            if (ffmpeg_threads > 0) {
                convert_command_cuda += (" -threads " + (std::to_string)(ffmpeg_threads));
                convert_command_cpu += (" -threads " + (std::to_string)(ffmpeg_threads));
            }

            // 添加输出路径
            convert_command_cuda += (" " + std::string(tmp_convert_video_path));
            convert_command_cpu += (" " + std::string(tmp_convert_video_path));

            try
            {
                int convert_status_code = executeFFmpegWithFallback(convert_command_cuda, convert_command_cpu, "video convert");
                if (convert_status_code != 0)
                {
                    DBG_LOGE("ffmpeg convert video fail\n");
                    return Status(Status::Code::VIDEO_READ_FAIL, "ffmpeg convert video fail");
                }
            }
            catch(...)
            {
                DBG_LOGE("ffmpeg convert video fail\n");
                return Status(Status::Code::VIDEO_READ_FAIL, "ffmpeg convert video fail");
            }

            // 再次校验
            video_status = readVideoInfo(tmp_convert_video_path, frame_width, frame_height, fps, bitrate);
            if (!video_status.IsOk())
                return video_status;

            is_convert = true;
        }

        infos.fps = fps;
        infos.bitrate = bitrate;

        // 视频转为图像帧序列, fps设为25 - 同时构建CUDA和CPU两个版本的命令
        std::string frame_command_cuda = "ffmpeg -loglevel quiet -y ";
        std::string frame_command_cpu = "ffmpeg -loglevel quiet -y ";

        // 先确定有效输出分辨率
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

        if (infos.video_height % 2 == 1)
            infos.video_height++;
        if (infos.video_width % 2 == 1)
            infos.video_width++;

        // 判断是否需要缩放
        bool need_scale = (infos.video_width != frame_width || infos.video_height != frame_height);

        // CUDA版本：添加CUDA解码参数
        // 只在需要缩放且启用CUDA时,添加 -hwaccel_output_format cuda 参数
        std::string decoder_args = ffmpeg_config.getDecoderArgs();
        if (!decoder_args.empty()) {
            frame_command_cuda += (decoder_args + " ");
            // 如果需要缩放且使用CUDA,添加hwaccel_output_format参数确保解码器输出CUDA格式帧
            if (need_scale && ffmpeg_config.isCudaEnabled()) {
                frame_command_cuda += "-hwaccel_output_format cuda ";
            }
        }

        // CPU版本：添加解码线程数限制
        if (ffmpeg_threads > 0)
            frame_command_cpu += (" -threads " + (std::to_string)(ffmpeg_threads)) + " ";

        // 两个版本都添加输入文件
        std::string input_path;
        if (is_convert)
        {
            input_path = "-i " + (std::string)tmp_convert_video_path;
        }
        else
        {
            input_path = "-i " + (std::string)src_video_path;
        }
        frame_command_cuda += input_path;
        frame_command_cpu += input_path;

        // 两个版本都添加时长限制
        if (video_params.video_ffmpeg_duration > 0)
        {
            std::string duration_arg = " -t " + std::to_string(video_params.video_ffmpeg_duration + 2);
            frame_command_cuda += duration_arg;
            frame_command_cpu += duration_arg;
        }

        // 添加缩放滤镜(CUDA或CPU)
        if (need_scale)
        {
            // CUDA版本：使用scale_cuda滤镜
            std::string scale_filter_cuda = ffmpeg_config.getScaleFilter(infos.video_width, infos.video_height, false);
            frame_command_cuda += (" -vf \"" + scale_filter_cuda + ",hwdownload,format=nv12,format=yuv420p\"");

            // CPU版本：使用CPU scale滤镜
            std::string scale_filter_cpu = ffmpeg_config.getScaleFilter(infos.video_width, infos.video_height, true);
            frame_command_cpu += (" -vf " + scale_filter_cpu);
        }

        // fps设置
        if (video_params.keep_fps == 0)
            infos.fps = 25;
        if (infos.fps > 60)
            infos.fps = 60;

        // 两个版本都添加输出参数
        std::string output_args = " -start_number 0 -qmin 1 -q:v 1 -r " + std::to_string(infos.fps);
        if (ffmpeg_threads > 0)
            output_args += (" -threads " + (std::to_string)(ffmpeg_threads));
        output_args += (" " + (std::string)tmp_frame_dir + "/%06d.jpg");

        frame_command_cuda += output_args;
        frame_command_cpu += output_args;

        try
        {
            int frame_status_code = executeFFmpegWithFallback(frame_command_cuda, frame_command_cpu, "video to frames");
            if (frame_status_code != 0)
            {
                DBG_LOGE("ffmpeg convert video to local file fail.\n");
                return Status(Status::Code::VIDEO_READ_FAIL, "ffmpeg convert video to local file fail.");
            }
        }
        catch(...)
        {
            DBG_LOGE("ffmpeg convert video to local file fail.\n");
            return Status(Status::Code::VIDEO_READ_FAIL, "ffmpeg convert video to local file fail.");
        }

        // 获取图像帧序列的路径
        try
        {
            this->traverseFiles((std::string)tmp_frame_dir, infos.frame_paths, "jpg");
        }
        catch(...)
        {
            DBG_LOGE("local frame empty.\n");
            return Status(Status::Code::VIDEO_READ_FAIL, "local frame empty.");
        }
        if (infos.frame_paths.size() == 0)
        {
            DBG_LOGE("local frame empty.\n");
            return Status(Status::Code::VIDEO_READ_FAIL, "local frame empty.");
        }
        infos.frame_nums = infos.frame_paths.size();
    }

    catch(...)
    {
        DBG_LOGE("video read fail.\n");
        return Status(Status::Code::VIDEO_READ_FAIL, "video read fail.");
    }

    return Status(Status::Code::SUCCESS, "success");
}

Status TalkingFace::audioDubbing(const char *audio_path, const char *render_video_save_path)
{
    DBG_LOGI("audio dubbing start.\n");
    double t0 = (double)cv::getTickCount();

    // 音视频合成 - 同时构建CUDA和CPU两个版本的命令
    std::string ffmpeg_command_cuda = "ffmpeg -loglevel quiet -y ";
    std::string ffmpeg_command_cpu = "ffmpeg -loglevel quiet -y ";

    // CUDA版本：添加CUDA解码参数
    std::string decoder_args = ffmpeg_config.getDecoderArgs();
    if (!decoder_args.empty()) {
        ffmpeg_command_cuda += (decoder_args + " ");
    }

    // 两个版本都添加输入视频
    ffmpeg_command_cuda += ("-i " + (std::string)tmp_video_path);
    ffmpeg_command_cpu += ("-i " + (std::string)tmp_video_path);

    // 叠加音频，如果存在音频时长限制则截断（音频处理不需要CUDA）
    if (video_params.audio_max_time > 0)
    {
        std::string ffmpeg_crop_audio = "ffmpeg -loglevel quiet -y -i " + (std::string)audio_path + " -t " + (std::to_string)(video_params.audio_max_time) + " -f wav ";

        if (ffmpeg_threads > 0)
            ffmpeg_crop_audio += (" -threads " + (std::to_string)(ffmpeg_threads));
        ffmpeg_crop_audio += (" " + (std::string)(tmp_crop_audio_path));

        DBG_LOGI("ffmpeg dubbing crop audio command:  %s\n", ffmpeg_crop_audio.c_str());
        try
        {
            if (system(ffmpeg_crop_audio.c_str()) != 0)
            {
                DBG_LOGE("ffmpeg dubbing crop audio fail.\n");
                return Status(Status::Code::AUDIO_DUBBING_FAIL, "ffmpeg dubbing crop audio fail.");
            }
        }
        catch(...)
        {
            DBG_LOGE("ffmpeg dubbing crop audio fail.\n");
            return Status(Status::Code::AUDIO_DUBBING_FAIL, "ffmpeg dubbing crop audio fail.");
        }
        std::string audio_args = " -i " + (std::string)tmp_crop_audio_path + " -map 0:v:0 -map 1:a:0";
        ffmpeg_command_cuda += audio_args;
        ffmpeg_command_cpu += audio_args;
    }
    else
    {
        std::string audio_args = " -i " + (std::string)audio_path + " -map 0:v:0 -map 1:a:0";
        ffmpeg_command_cuda += audio_args;
        ffmpeg_command_cpu += audio_args;
    }

    // 视频码率设置
    if (video_params.bitrate > 0)
    {
        if (video_params.max_bitrate > 0 && video_params.bitrate > video_params.max_bitrate)
            video_params.bitrate = video_params.max_bitrate;
        if (video_params.bitrate > 50000)       // 兜底，限制在50M码率之内
            video_params.bitrate = 50000;

        // CUDA版本：使用CUDA编码器(CBR模式)
        std::string encoder_args_cuda = ffmpeg_config.getEncoderArgs(
            video_params.bitrate,
            video_params.bitrate,
            true,  // CBR模式
            false  // 不强制CPU
        );
        ffmpeg_command_cuda += (" " + encoder_args_cuda);

        // CPU版本：使用CPU编码器(CBR模式)
        std::string encoder_args_cpu = ffmpeg_config.getEncoderArgs(
            video_params.bitrate,
            video_params.bitrate,
            true,  // CBR模式
            true   // 强制CPU
        );
        ffmpeg_command_cpu += (" " + encoder_args_cpu);
    }
    else if (video_params.keep_bitrate == 1 && infos.bitrate > 0)
    {
        if (video_params.max_bitrate > 0 && infos.bitrate > video_params.max_bitrate)
            infos.bitrate = video_params.max_bitrate;
        if (infos.bitrate > 50000)       // 兜底，限制在50M码率之内
            infos.bitrate = 50000;

        // CUDA版本：使用CUDA编码器(CBR模式)
        std::string encoder_args_cuda = ffmpeg_config.getEncoderArgs(
            infos.bitrate,
            infos.bitrate,
            true,  // CBR模式
            false  // 不强制CPU
        );
        ffmpeg_command_cuda += (" " + encoder_args_cuda);

        // CPU版本：使用CPU编码器(CBR模式)
        std::string encoder_args_cpu = ffmpeg_config.getEncoderArgs(
            infos.bitrate,
            infos.bitrate,
            true,  // CBR模式
            true   // 强制CPU
        );
        ffmpeg_command_cpu += (" " + encoder_args_cpu);
    }
    else
    {
        ffmpeg_command_cuda += (" -c:v copy");
        ffmpeg_command_cpu += (" -c:v copy");
    }

    // 两个版本都添加音频编码和输出参数
    ffmpeg_command_cuda += (" -c:a aac");
    ffmpeg_command_cpu += (" -c:a aac");

    if (ffmpeg_threads > 0) {
        ffmpeg_command_cuda += (" -threads " + (std::to_string)(ffmpeg_threads));
        ffmpeg_command_cpu += (" -threads " + (std::to_string)(ffmpeg_threads));
    }

    ffmpeg_command_cuda += (" " + (std::string)render_video_save_path);
    ffmpeg_command_cpu += (" " + (std::string)render_video_save_path);

    try
    {
        int dubbing_status_code = executeFFmpegWithFallback(ffmpeg_command_cuda, ffmpeg_command_cpu, "audio dubbing");
        if (dubbing_status_code != 0)
        {
            DBG_LOGE("ffmpeg dubbing fail.\n");
            return Status(Status::Code::AUDIO_DUBBING_FAIL, "ffmpeg dubbing fail.");
        }
    }
    catch(...)
    {
        DBG_LOGE("ffmpeg dubbing fail.\n");
        return Status(Status::Code::AUDIO_DUBBING_FAIL, "ffmpeg dubbing fail.");
    }

    try
    {
        cv::VideoCapture cap;
        cap.open(render_video_save_path);
        if (!cap.isOpened())
        {
            DBG_LOGE("render out video verify failed.\n");
            return Status(Status::Code::AUDIO_DUBBING_FAIL, "render out video verify failed!");
        }
        cap.release();
    }
    catch(...)
    {
        DBG_LOGE("render out video verify failed.\n");
        return Status(Status::Code::AUDIO_DUBBING_FAIL, "render out video verify failed!");
    }
    
    double t1 = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    DBG_LOGI("audio dubbing finish cost time: %.2fs.\n", t1);
    
    return Status(Status::Code::SUCCESS, "success");
}

bool TalkingFace::DetectSideHeadPose(const DetectionBox &box, std::vector<cv::Point> &landmark)
{
    // 侧脸返回true，否则返回false

    // // 左右眼间距判断法
    // cv::Point le = landmark[36];    // 左眼角
    // cv::Point re = landmark[45];    // 右眼角
    // cv::Point ce = landmark[27];
    // double dl = std::sqrt(std::pow(le.x - ce.x, 2) + std::pow(le.y - ce.y, 2));
    // double dr = std::sqrt(std::pow(re.x - ce.x, 2) + std::pow(re.y - ce.y, 2));
    // if (dr > dl * 1.8 || dl > dr * 1.8)
    //     return true;
    // return false;

    // 左右脸宽判断法
    int dr = std::max(static_cast<int>(landmark[30].x - box.x1), 0);
    int dl = std::max(static_cast<int>(box.x1 + box.w - landmark[30].x), 0);
    if (dr > dl * 6 || dl > dr * 6)
        return true;
    return false;
}

void TalkingFace::checkROI(cv::Rect2i &roi_rect)
{
    // 校验roi，限制在图像size内
    if (roi_rect.width == 0 || roi_rect.height== 0)
    {
        roi_rect.x = 0;
        roi_rect.y = 0;
        roi_rect.width= infos.video_width - 1;
        roi_rect.height = infos.video_height - 1;
        return;
    }

    if (roi_rect.x <= 0)
        roi_rect.x = 0;

    if (roi_rect.y <= 0)
        roi_rect.y = 0;

    if (roi_rect.x + roi_rect.width > infos.video_width - 1)
        roi_rect.width = infos.video_width - 1 - roi_rect.x;

    if (roi_rect.y + roi_rect.height > infos.video_height - 1)
        roi_rect.height = infos.video_height - 1 - roi_rect.y;

    return;
}

void TalkingFace::expand_box(
    const cv::Size s, const cv::Rect2i &box,
    cv::Rect_<int> &out_rect, float increase_area, float increase_margin[4])
{
    /*
    float increase_area: 超参数，控制外扩的比例
    */
    // 向上扩大人脸区域，配合后续扩边逻辑，能够囊括头部和帽子区域
    float box_x1 = box.x;
    float box_y1 = box.y;
    float box_h = box.height;
    float box_w = box.width;
    if (box.height / box.width < 1.6)
    {
        float height_ori = box.height;
        box_h = box.width * 1.6;
        box_y1 = box_y1 - (box_h - height_ori);
    }

    int width = (int)(box_w);
    int height = (int)(box_h);
    // 扩边逻辑源自 https://github.com/AliaksandrSiarohin/first-order-model/blob/master/crop-video.py
    float width_increase = std::max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width));
    float height_increase = std::max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height));
    int x1 = std::max(0, (int)(box_x1 - width_increase * width * increase_margin[0]));
    int y1 = std::max(0, (int)(box_y1 - height_increase * height * increase_margin[1]));
    int x2 = std::min(s.width, (int)(box_x1 + width +  width_increase * width * increase_margin[2]));
    int y2 = std::min(s.height, (int)(box_y1 + height +  height_increase * height * increase_margin[3]));
    out_rect.x = x1;
    out_rect.y = y1;
    out_rect.width = x2 - x1;
    out_rect.height = y2 - y1;
}

// CUDA兜底执行：先尝试CUDA命令，失败则自动降级到CPU命令重试
int TalkingFace::executeFFmpegWithFallback(const std::string& cuda_cmd, const std::string& cpu_cmd, const char* operation_name)
{
    // 如果CUDA未启用，直接执行CPU命令
    if (!ffmpeg_config.isCudaEnabled())
    {
        DBG_LOGI("%s command (CPU): %s\n", operation_name, cpu_cmd.c_str());
        return system(cpu_cmd.c_str());
    }

    // 尝试执行CUDA命令
    DBG_LOGI("%s command (CUDA): %s\n", operation_name, cuda_cmd.c_str());
    int cuda_ret = system(cuda_cmd.c_str());

    if (cuda_ret == 0)
    {
        return 0;  // CUDA执行成功
    }

    // CUDA失败，降级到CPU重试
    DBG_LOGI("%s CUDA execution failed (code: %d), fallback to CPU\n", operation_name, cuda_ret);
    DBG_LOGI("%s command (CPU fallback): %s\n", operation_name, cpu_cmd.c_str());
    return system(cpu_cmd.c_str());
}