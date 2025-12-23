#pragma once 
#ifndef SIXDREP_H
#define SIXDREP_H

#include "../core/trt_handler.h"
#include "../core/trt_utils.h"
#include "opencv2/imgproc/imgproc.hpp"

namespace Function
{
    struct EulerAngles {
        float pitch;  // 俯仰角 (绕Y轴)
        float yaw;    // 偏航角 (绕Z轴)
        float roll;   // 滚转角 (绕X轴)
    };

    class SixDRep : public BasicTRTHandler
    {
    public:
        explicit SixDRep(const std::string &_engine_path);
        ~SixDRep() override = default;

    private:
        const float mean_vals[3] = {0.485f, 0.456f, 0.406f}; // RGB order
        const float norm_vals[3] = {1.f/0.229f/255.f, 1.f/0.224f/255.f, 1.f/0.225f/255.f};
        const int output_size = 9;
        const int size = 224;

    public:
        std::vector<float> prepare(const cv::Mat& src);
        void detect_pose(const cv::Mat &mat, EulerAngles &angles);
        void warmup();

        
    };
}



#endif