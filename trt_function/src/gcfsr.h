#pragma once
#ifndef GCFSR_H
#define GCFSR_H

#include <opencv2/imgproc/imgproc.hpp>
#include "../core/trt_handler.h"
#include "../core/trt_utils.h"

namespace Function
{

class GCFSR : public BasicTRTHandler // hair face hat segmentation network
{
private:
    const float mean_vals[3] = {128.f, 128.f, 128.f};
    const float norm_vals[3] = {1.f/128.f, 1.f/128.f, 1.f/128.f};
    int size = 512;
    int output_size = 3 * 512 * 512;

    const float mean_vals2[3] = {1.f, 1.f, 1.f};
    const float norm_vals2[3] = {127.5, 127.5, 127.5};

    cv::Mat fusion_mask;

private:
    std::vector<float> prepare(const cv::Mat& img);

public:
    explicit GCFSR(const std::string &_engine_path);
    ~GCFSR() override = default;

    void predict(const cv::Mat &mat, cv::Mat &output);
    void warmup();
    void pasterBack(cv::Mat &ori_img, cv::Mat &pred_img);
    void pasterBack(cv::Mat &ori_img, cv::Mat &pred_img, cv::Mat &mask);
    void generate_fusion_mask();
};

}

#endif