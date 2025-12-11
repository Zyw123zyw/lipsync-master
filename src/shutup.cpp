#include "talkingface.h"


std::vector<cv::Rect2i> TalkingFace::parseShutupIdParams(const char *id_params, const int frame_width, const int frame_height)
{
    std::vector<cv::Rect2i> id_rois;
    if (id_params[0] == '\0')
    {
        cv::Rect2i id_roi(0, 0, frame_width-1, frame_height-1);
        DBG_LOGI("id %d, box : [%d, %d, %d, %d]\n", 0, id_roi.x, id_roi.y, id_roi.width, id_roi.height);
        id_rois.push_back(id_roi);
    }
    else
    {
        try
        {
            Json::Reader reader;
            Json::Value root;

            if (reader.parse((std::string)id_params, root))
            {
                for (unsigned int i = 0; i < root.size(); i++)
                {
                    cv::Rect2i id_roi;
                    id_roi.x = root[i]["box"][0].asInt();
                    id_roi.y = root[i]["box"][1].asInt();
                    id_roi.width = root[i]["box"][2].asInt();
                    id_roi.height = root[i]["box"][3].asInt();

                    if (id_roi.x < 0)
                        id_roi.x = 0;
                    if (id_roi.y < 0)
                        id_roi.y = 0;
                    if (id_roi.x + id_roi.width > frame_width - 1)
                        id_roi.width = frame_width - 1 - id_roi.x;
                    if (id_roi.y + id_roi.height > frame_height - 1)
                        id_roi.height = frame_height - 1 - id_roi.y;

                    id_rois.emplace_back(id_roi);
                    DBG_LOGI("id %d, box : [%d, %d, %d, %d]\n", i, id_roi.x, id_roi.y, id_roi.width, id_roi.height);
                }
            }
        }
        catch(...)
        {
            cv::Rect2i id_roi(0, 0, frame_width-1, frame_height-1);
            DBG_LOGI("id %d, box : [%d, %d, %d, %d]\n", 0, id_roi.x, id_roi.y, id_roi.width, id_roi.height);
            id_rois.push_back(id_roi);
        }
    }
    return id_rois;
}

Status TalkingFace::shutup(const char *image_path, 
                           const char *save_path, 
                           const char *set_params,
                           const char *id_params)
{
    try
    {
        infos.reset();
        static const int work_idx = 0;

        // 获取传参
        this->readVideoParam(set_params);

        if (!this->fileExists(std::string(image_path)))
        {
            DBG_LOGE("image not exists.\n");
            return Status(Status::Code::SHUT_UP_FAIL, "image not exists.");
        }

        // 读取图片
        cv::Mat frame = cv::imread(image_path);
        if (frame.empty())
        {
            DBG_LOGE("image read failed.\n");
            return Status(Status::Code::SHUT_UP_FAIL, "image read failed.");
        }
        if (frame.channels() == 1)
        {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        // 分辨率限制
        int frame_width = frame.cols;
        int frame_height = frame.rows;

        int save_width = video_params.width == 0 ? frame_width : video_params.width;
        int save_height = video_params.height == 0 ? frame_height : video_params.height;
        int max_side = save_width >= save_height ? save_width : save_height;
        if (video_params.video_max_side > 0 && video_params.video_max_side < max_side)
        {
            float scale = static_cast<float>(max_side) / static_cast<float>(video_params.video_max_side);
            if (save_height >= save_width)
            {
                save_height = video_params.video_max_side;
                save_width = static_cast<int>(static_cast<float>(save_width) / scale);
            }
            else
            {
                save_height = static_cast<int>(static_cast<float>(save_height / scale));
                save_width = video_params.video_max_side;
            }
        }

        if (frame_height != save_height || frame_width != save_width)
            cv::resize(frame, frame, cv::Size(save_width, save_height));

        // 使用新函数解析id_params
        std::vector<cv::Rect2i> id_rois = this->parseShutupIdParams(id_params, save_width, save_height);

        for (cv::Rect2i id_roi: id_rois)
        {
            // 人脸检测
            cv::Rect2i box;
            std::vector<cv::Point2i> landmark;
            this->detectLandmark(work_idx, frame, id_roi, box, landmark);

            if (box.width != 0 && box.height != 0)
            {
                cv::Mat crop, src, ref, output, mask;
                cv::Rect2i rect;
                m_generators[work_idx]->make_mask_img(frame, landmark, crop, ref, src, rect, mask);

                m_generators[work_idx]->predict(src, ref, audio_extractor->silence_feat.data(), output, video_params.amplifier);

                cv::resize(output, output, crop.size());
                m_generators[work_idx]->pasterBack(crop, output);
                output.copyTo(frame(rect));

                // 人脸增强
                if (video_params.video_enhance == 1)
                {
                    // 增强区域
                    cv::Rect2i enhance_rect;
                    float increase_margin[4] = {1.0, 1.0, 1.0, 1.0};
                    this->expand_box(frame.size(), box, enhance_rect, 0.25, increase_margin);
                    cv::Mat enhance_img = frame(enhance_rect).clone();
                    cv::Mat enhance_out;
                    m_enhancers[work_idx]->predict(enhance_img, enhance_out);

                    // 回贴方案1: 关键点
                    std::vector<cv::Point2i> contour_pts(landmark.begin(), landmark.begin() + 17);
                    std::vector<cv::Point2i> contour_pts_eye(landmark.begin() + 17, landmark.begin() + 27);
                    std::reverse(contour_pts_eye.begin(), contour_pts_eye.end());
                    contour_pts.insert(contour_pts.end(), contour_pts_eye.begin(), contour_pts_eye.end());

                    std::vector<std::vector<cv::Point>> contours;
                    contours.emplace_back(contour_pts);
                    cv::Mat enhance_frame_mask(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
                    cv::drawContours(enhance_frame_mask, contours, 0, cv::Scalar(255), cv::FILLED);
                    cv::cvtColor(enhance_frame_mask, enhance_frame_mask, cv::COLOR_GRAY2BGR);
                    cv::GaussianBlur(enhance_frame_mask, enhance_frame_mask, cv::Size(11, 11), 0);

                    cv::Mat enhance_mask = enhance_frame_mask(enhance_rect);
                    m_enhancers[work_idx]->pasterBack(enhance_img, enhance_out, enhance_mask);

                    // // 回帖方案2: 直接回帖
                    // m_enhancers[work_idx]->pasterBack(enhance_img, enhance_out);

                    enhance_out.copyTo(frame(enhance_rect));
                }
            }
        }
        cv::imwrite(save_path, frame);
    }
    catch(...)
    {
        DBG_LOGE("render image fail.\n");
        return Status(Status::Code::SHUT_UP_FAIL, "render image fail.");
    }
    
    return Status(Status::Code::SUCCESS, "success");
}