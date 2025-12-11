#pragma once
#ifndef WAV2LIP_H
#define WAV2LIP_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "../core/trt_handler.h"
#include "../core/trt_utils.h"

namespace Function
{

class Wav2Lip : public BasicTRTHandler
{
private:
    const int audio_feat_channel = 1024;
    const int audio_feat_duration = 20;

    cv::Mat fusion_mask;

public:
    const int target_size = 256;

public:
    explicit Wav2Lip(const std::string &_engine_path);
    ~Wav2Lip() override = default;

    void predict(const cv::Mat &src, const cv::Mat &ref, float *audio_feat, cv::Mat &output, float amplifier);

    void predict(const cv::Mat &src, const cv::Mat &ref, float *audio_feat, cv::Mat &output, float amplifier, bool enable_sr);

    void warmup();

    void make_mask_img(const cv::Mat &img, const std::vector<cv::Point> &landmark, cv::Mat &crop, cv::Mat &ref, cv::Mat &src, cv::Rect2i &rect, cv::Mat &mask);

    void pasterBack(cv::Mat &ori_img, cv::Mat &pred_img);
    void pasterBack(cv::Mat &ori_img, cv::Mat &pred_img, cv::Mat &mask);

    void generate_fusion_mask();
};

}

#endif