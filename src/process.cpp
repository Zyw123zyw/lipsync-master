#include "talkingface.h"


Status TalkingFace::process(const char *src_video_path, const char *info_save_path, const char *set_params)
{
    try
    {
        infos.reset();

        // 创建tmp路径
        std::string dir_command = "mkdir -p " + (std::string)tmp_frame_dir;
        system(dir_command.c_str());

        // 获取传参
        this->readVideoParam(set_params);

        // 读取底板视频
        Status video_status = this->readVideo(src_video_path);
        if (!video_status.IsOk())
            return video_status;

        // 初始化infos的face_bboxes和face_landmarks
        std::vector<cv::Rect2i> bboxes(infos.frame_paths.size(), cv::Rect2i(0, 0, 0, 0));
        infos.face_bboxes.emplace_back(bboxes);

        std::vector<cv::Point2i> landmark(68, cv::Point2i(0, 0));
        std::vector<std::vector<cv::Point2i>> landmarks(infos.frame_paths.size(), landmark);
        infos.face_landmarks.emplace_back(landmarks);

        // 人脸关键点检测
        std::vector<std::thread> detect_threads;
        for (int i = 0; i < n_threads; i++)
            detect_threads.emplace_back(std::thread(&TalkingFace::getVideoLandmark, this, i));

        for (int i = 0; i < n_threads; i++)
            detect_threads[i].join();

        // 保存为json文件
        Json::Value root;
        Json::Value video_info;
        video_info["video_path"] = src_video_path;
        video_info["video_frame_nums"] = infos.face_bboxes[0].size();
        video_info["video_FPS"] = infos.fps;
        video_info["video_frame_width"] = infos.video_width;
        video_info["video_frame_height"] = infos.video_height;
        root["video_info"] = video_info;

        // face-bbox-info
        Json::Value face_bbox_info;
        for (size_t i = 0; i < infos.face_bboxes[0].size(); i++)
        {
            Json::Value jsonArray(Json::arrayValue);

            jsonArray.append(infos.face_bboxes[0].at(i).x);
            jsonArray.append(infos.face_bboxes[0].at(i).y);
            jsonArray.append(infos.face_bboxes[0].at(i).width);
            jsonArray.append(infos.face_bboxes[0].at(i).height);

            char buffer[10];
            std::sprintf(buffer, "%06d", i);
            face_bbox_info[std::string(buffer)] = jsonArray;
        }
        root["face_bbox_info"] = face_bbox_info;

        // face-landmark-info
        Json::Value face_landmark_info;
        for (size_t i = 0; i < infos.face_landmarks[0].size(); i++)
        {
            Json::Value jsonArray(Json::arrayValue);

            for (const auto &num : infos.face_landmarks[0].at(i))
            {
                jsonArray.append(static_cast<int>(num.x));
                jsonArray.append(static_cast<int>(num.y));
            }

            char buffer[10];
            std::sprintf(buffer, "%06d", i);
            face_landmark_info[std::string(buffer)] = jsonArray;
        }
        root["face_landmark_info"] = face_landmark_info;

        // save2local-path
        std::ofstream outputFile(info_save_path);
        if (outputFile.is_open())
        {
            Json::StreamWriterBuilder writerBuilder;
            std::unique_ptr<Json::StreamWriter> writer(writerBuilder.newStreamWriter());
            writer->write(root, &outputFile);
            outputFile.close();
            DBG_LOGI("JSON data has been written to %s.\n", info_save_path);
        }
        else
        {
            DBG_LOGE("json save fail.\n");
            return Status(Status::Code::PROCESS_PREDICT_FAIL, "json save fail.");
        }
    }
    catch(...)
    {
        DBG_LOGE("process fail.\n");
        return Status(Status::Code::PROCESS_PREDICT_FAIL, "process fail.");
    }

    return Status(Status::Code::SUCCESS, "success");
}
