#include "talkingface.h"

void TalkingFace::detectLandmark(int work_idx,
                                 const cv::Mat &frame,
                                 const cv::Rect2i &roi_rect,
                                 cv::Rect2i &face_bbox,
                                 std::vector<cv::Point2i> &face_landmark)
{
    try
    {
        cv::Mat roi_img = frame(roi_rect).clone();

        // face box detect
        std::vector<DetectionBox> faceboxes;
        m_face_detectors[work_idx]->detect(roi_img, faceboxes, this->video_params.face_det_threshold);
        if (!faceboxes.empty())
        {
            DetectionBox detect_box = faceboxes[0];
            detect_box.x1 += roi_rect.x;
            detect_box.y1 += roi_rect.y;

            // face landmark detect
            cv::Rect2i lmsdet_rect;
            m_face_detectors[work_idx]->expand_box_for_pipnet(frame.size(), detect_box, lmsdet_rect, 1.2);
            cv::Mat kp_img = frame(lmsdet_rect).clone();
            std::vector<cv::Point2i> landmark;
            m_face_landmarkers[work_idx]->predict(kp_img, lmsdet_rect, landmark);

            // side face detect
            if (video_params.filter_head_pose == true)
            {
                if (this->DetectSideHeadPose(detect_box, landmark) == false)
                {
                    face_bbox.x = detect_box.x1;
                    face_bbox.y = detect_box.y1;
                    face_bbox.width = detect_box.w;
                    face_bbox.height = detect_box.h;
                    face_landmark = landmark;
                }
            }
            else{
                face_bbox.x = detect_box.x1;
                face_bbox.y = detect_box.y1;
                face_bbox.width = detect_box.w;
                face_bbox.height = detect_box.h;
                face_landmark = landmark;
            }
        }
    }
    catch(...)
    {
        DBG_LOGE("face landmark detect error.\n");
    }
}


void TalkingFace::getVideoLandmark(int work_idx)
{
    DBG_LOGI("detect landmark %d thread start.\n", work_idx);

    std::string framepath;
    char buffer[256];
    int img_idx;
    cv::Mat frame;
    cv::Rect2i roi = video_params.roi_rect;
    this->checkROI(roi);

    int cnt = 0;
    while (1)
    {
        img_idx = cnt * n_threads + work_idx;
        if (img_idx >= infos.frame_paths.size())
            break;

        std::sprintf(buffer, "/%06d.jpg", img_idx);
        framepath = (std::string)tmp_frame_dir + std::string(buffer);
        frame = cv::imread(framepath);

        cv::Rect2i box(0, 0, 0, 0);
        std::vector<cv::Point2i> landmark(68, cv::Point2i(0, 0));
        this->detectLandmark(work_idx, frame, roi, box, landmark);
        infos.face_bboxes[0][img_idx] = box;
        infos.face_landmarks[0][img_idx] = landmark;
        cnt++;
    }
    DBG_LOGI("detect landmark %d thread finish.\n", work_idx);
}