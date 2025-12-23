#include "sixdrep.h"

using Function::SixDRep;
using Function::EulerAngles;

SixDRep::SixDRep(const std::string &_engine_path) : BasicTRTHandler(_engine_path) {}

std::vector<float> SixDRep::prepare(const cv::Mat& mat)
{
    cv::Mat canva = mat.clone();
	cv::resize(canva, canva, cv::Size(size, size));
    normalize_inplace(canva, mean_vals, norm_vals, false);
    std::vector<float> result(size * size * 3);
    trans2chw(canva, result);
    return result;
}

void SixDRep::warmup()
{
    DBG_LOGI("SixDRep warm up start\n");
    cv::Mat mat = cv::Mat(size, size, CV_8UC3, cv::Scalar(0, 0, 0));

    std::vector<float> cur_input = this->prepare(mat);

    CHECK(cudaMemcpy(buffers[0], cur_input.data(), buffer_size[0], cudaMemcpyHostToDevice));

    context->executeV2(buffers);

    float *output = new float[output_size];
    CHECK(cudaMemcpy(output, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost));

    delete output;

    DBG_LOGI("SixDRep warm up done\n");
}

void SixDRep::detect_pose(const cv::Mat &mat, EulerAngles &angles)
{
    if (mat.empty()) return;
    // 创建输入
    std::vector<float> cur_input = this->prepare(mat);

    // 将输入传递到GPU
    CHECK(cudaMemcpy(buffers[0], cur_input.data(), buffer_size[0], cudaMemcpyHostToDevice));

    // 异步执行
    context->executeV2(buffers);

    // 将输出传递到CPU
    float *output = new float[output_size];
    CHECK(cudaMemcpy(output, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost));

    float sy = std::sqrt(output[0]*output[0] + output[3]*output[3]);
    bool singular = sy < 1e-6;
    if (!singular) {
        // 非奇异情况
        float x = std::atan2(output[7], output[8]);
        float y = std::atan2(-output[6], sy);
        float z = std::atan2(output[3], output[0]);
        
        // 转换为角度
        angles.pitch = x * 180.0 / M_PI;
        angles.yaw = y * 180.0 / M_PI;
        angles.roll = z * 180.0 / M_PI;
    } else {
        // 奇异情况
        float xs = std::atan2(-output[5], output[4]);
        float ys = std::atan2(-output[6], sy);
        float zs = 0.0;
        
        // 转换为角度
        angles.pitch = xs * 180.0 / M_PI;
        angles.yaw = ys * 180.0 / M_PI;
        angles.roll = zs * 180.0 / M_PI;
    }

}