#include "talkingface.h"
#include <opencv2/cudawarping.hpp>

std::mutex mtx;
std::condition_variable m_cond_write;

void TalkingFace::renderProducer(int work_idx)
{
    DBG_LOGI("render producer %d thread start.\n", work_idx);
    double t0 = (double)cv::getTickCount();

    int divisor, remainder, diban_idx, render_idx;

    int cnt = 0;
    while (1)
    {
        render_idx = cnt * n_threads + work_idx;       // 渲染帧的索引
        if (render_idx >= infos.min_audio_cnt)
            break;

        // 简单循环播放（不再乒乓来回）
        diban_idx = render_idx % infos.frame_nums;

        // 使用GPU解码器获取帧
        double t_gpu_start = (double)cv::getTickCount();
        cv::cuda::GpuMat& gpu_frame = gpu_decoder_->decodeFrame(diban_idx, work_idx);
        double t_gpu_decode = (double)cv::getTickCount();
        
        // GPU缩放（如果需要）
        cv::cuda::GpuMat gpu_frame_resized;
        if (gpu_frame.cols != infos.video_width || gpu_frame.rows != infos.video_height) {
            cv::cuda::resize(gpu_frame, gpu_frame_resized, 
                           cv::Size(infos.video_width, infos.video_height));
        } else {
            gpu_frame_resized = gpu_frame;
        }
        double t_gpu_resize = (double)cv::getTickCount();
        
        // 下载到CPU（Wav2Lip等后续处理需要）
        cv::Mat frame;
        gpu_frame_resized.download(frame);
        double t_download = (double)cv::getTickCount();
        double freq = cv::getTickFrequency();
        
        DBG_LOGI("Producer[%d] render_idx=%d diban_idx=%d | gpu_decode=%.1fms gpu_resize=%.1fms download=%.1fms\n",
                 work_idx, render_idx, diban_idx,
                 (t_gpu_decode - t_gpu_start) * 1000 / freq,
                 (t_gpu_resize - t_gpu_decode) * 1000 / freq,
                 (t_download - t_gpu_resize) * 1000 / freq);

        // cv::Mat frame_src = frame.clone();

        std::vector<cv::Rect2i> bboxes; // [ids, 4]
        std::vector<std::vector<cv::Point2i>> landmarks;    // [ids, 68, 2]

        for (int id: infos.ids)
        {
            // face detect
            cv::Rect2i box;
            std::vector<cv::Point2i> landmark;

            // 判断该底板帧是否已检测过了
            if (infos.face_bboxes.size() == 0)
            {
                // 使用GPU版本的人脸检测（消除H2D传输）
                this->detectLandmarkGPU(work_idx, gpu_frame_resized, infos.id_rois[id], box, landmark);
            }
            else if (diban_idx >= infos.face_bboxes[id].size())
            {
                // 使用GPU版本的人脸检测（消除H2D传输）
                this->detectLandmarkGPU(work_idx, gpu_frame_resized, infos.id_rois[id], box, landmark);
            }
            else
            {
                box = infos.face_bboxes[id][diban_idx];
                landmark = infos.face_landmarks[id][diban_idx];
            }
            bboxes.emplace_back(box);
            landmarks.emplace_back(landmark);

            // render
            if (box.width != 0 && box.height != 0)
            {
                cv::Mat crop, src, ref, output, mask;
                cv::Rect2i rect;
                m_generators[work_idx]->make_mask_img(frame, landmark, crop, ref, src, rect, mask);

                if (video_params.shutup_first == 1 && render_idx == 0)
                {
                    m_generators[work_idx]->predict(src, ref, audio_extractor->silence_feat.data(), output, video_params.amplifier);
                }
                else
                {
                    // std::vector<float> audio = infos.audio_feats[id][render_idx];
                    std::vector<float> audio = audio_extractor->get_audio_feat(infos.audio_feats[id], render_idx, infos.audio_intervals[id]);
                    m_generators[work_idx]->predict(src, ref, audio.data(), output, video_params.amplifier);
                }

                cv::resize(output, output, crop.size());
                m_generators[work_idx]->pasterBack(crop, output);
                output.copyTo(frame(rect));

                // cv::rectangle(frame, rect, cv::Scalar(0,255,0), 2);

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
        
        std::unique_lock<std::mutex> lock(mtx);
        while (infos.last_idx != (render_idx - 1) || infos.rendered_frames.size() > 1)
            m_cond_write.wait(lock);

        infos.last_idx = render_idx;
        infos.rendered_frames.push(frame);

        // cv::Mat combined_frame;
        // cv::hconcat(frame_src, frame, combined_frame);
        // infos.rendered_frames.push(combined_frame);

        m_cond_write.notify_all();

        // 缓存底板帧的detect结果
        if (infos.face_bboxes.size() == 0)
        {
            for (int id: infos.ids)
            {
                std::vector<cv::Rect2i> bboxes_0(1, bboxes[id]);
                std::vector<std::vector<cv::Point2i>> landmarks_0(1, landmarks[id]);
                infos.face_bboxes.emplace_back(bboxes_0);
                infos.face_landmarks.emplace_back(landmarks_0);
            }
        }
        if (diban_idx >= infos.face_bboxes[0].size())
        {
            for(int id: infos.ids)
            {
                cv::Rect2i box = bboxes[id];
                std::vector<cv::Point2i> landmark = landmarks[id];
                infos.face_bboxes[id].emplace_back(box);
                infos.face_landmarks[id].emplace_back(landmark);
            }
        }

        cnt++;
    }

    double t1 = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    DBG_LOGI("render %d thread finish cost time: %.2fs.\n", work_idx, t1);
}

void TalkingFace::writeConsumer()
{
    DBG_LOGI("write consumer thread start.\n");
    double t0 = (double)cv::getTickCount();

    cv::Mat frame;
    int write_idx = 0;
    while (1)
    {
        std::unique_lock<std::mutex> lock(mtx);
        while (infos.rendered_frames.empty())
            m_cond_write.wait(lock);

        frame = infos.rendered_frames.front();
        infos.rendered_frames.pop();
        
        m_cond_write.notify_all();

        infos.video_writer.write(frame);

        write_idx++;
        if (write_idx >= infos.min_audio_cnt)
            break;
    }

    if (infos.video_writer.isOpened())
        infos.video_writer.release();
    
    double t1 = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    DBG_LOGI("write video thread finish cost time: %.2fs.\n", t1);
}
