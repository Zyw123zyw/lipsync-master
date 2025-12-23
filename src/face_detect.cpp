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


void TalkingFace::detectLandmarkGPU(int work_idx,
                                    const cv::cuda::GpuMat &gpu_frame,
                                    const cv::Rect2i &roi_rect,
                                    cv::Rect2i &face_bbox,
                                    std::vector<cv::Point2i> &face_landmark)
{
    try
    {
        // 1. GPU上裁剪ROI（零拷贝，只是指针偏移）
        cv::cuda::GpuMat gpu_roi = gpu_frame(roi_rect);

        // 2. GPU检测（无H2D传输）
        std::vector<DetectionBox> faceboxes;
        m_face_detectors[work_idx]->detectGPU(gpu_roi, faceboxes, this->video_params.face_det_threshold);
        
        if (!faceboxes.empty())
        {
            DetectionBox detect_box = faceboxes[0];
            // 坐标还原到原图
            detect_box.x1 += roi_rect.x;
            detect_box.y1 += roi_rect.y;

            // 3. PIPNet关键点检测（GPU版本，无需下载）
            cv::Rect2i lmsdet_rect;
            m_face_detectors[work_idx]->expand_box_for_pipnet(
                cv::Size(gpu_frame.cols, gpu_frame.rows), detect_box, lmsdet_rect, 1.2);
            
            // GPU上裁剪人脸区域，直接调用predictGPU
            cv::cuda::GpuMat gpu_face = gpu_frame(lmsdet_rect);
            
            std::vector<cv::Point2i> landmark;
            m_face_landmarkers[work_idx]->predictGPU(gpu_face, lmsdet_rect, landmark);

            // 4. 侧脸检测
            if (video_params.filter_head_pose == true)
            {
                cv::Mat filter_img;
                gpu_frame(lmsdet_rect).download(filter_img);
                EulerAngles angles;

                m_face_posefilters[work_idx]->detect_pose(filter_img, angles);

                if ((angles.yaw < 65 && angles.yaw > -65) && 
                    (angles.pitch < 40 && angles.pitch > -40) &&
                    (angles.roll < 30 && angles.roll > -30))
                {
                    face_bbox.x = detect_box.x1;
                    face_bbox.y = detect_box.y1;
                    face_bbox.width = detect_box.w;
                    face_bbox.height = detect_box.h;
                    face_landmark = landmark;
                }
            }
            else
            {
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
        DBG_LOGE("face landmark detect (GPU) error.\n");
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