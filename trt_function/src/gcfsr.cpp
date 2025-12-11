#include "gcfsr.h"


using Function::GCFSR;


GCFSR::GCFSR(const std::string &_engine_path) : BasicTRTHandler(_engine_path) {
    this->generate_fusion_mask();
}

void GCFSR::generate_fusion_mask()
{
    // 制作用于羽化回帖的mask
    fusion_mask.create(size, size, CV_8UC3);
    fusion_mask.setTo(cv::Scalar(255, 255, 255));

    int k = size / 16;
    if (k % 2 == 0)
        k = k + 1;
    cv::rectangle(fusion_mask, cv::Point(0, 0), cv::Point(size-1, size-1), cv::Scalar(0, 0, 0), k);
    cv::GaussianBlur(fusion_mask, fusion_mask, cv::Size(k, k), 0);
}

std::vector<float> GCFSR::prepare(const cv::Mat& mat)
{
    cv::Mat canva = mat.clone();
    cv::resize(canva, canva, cv::Size(size, size));
    normalize_inplace(canva, mean_vals, norm_vals, false);
    std::vector<float> result(size * size * 3);
    trans2chw(canva, result);
    return result;
}

void GCFSR::warmup()
{
    DBG_LOGI("GCFSR warm up start\n");
    cv::Mat mat = cv::Mat(size, size, CV_8UC3, cv::Scalar(0, 0, 0));

    std::vector<float> cur_input = this->prepare(mat);

    CHECK(cudaMemcpy(buffers[0], cur_input.data(), buffer_size[0], cudaMemcpyHostToDevice));

    context->executeV2(buffers);

    float *output = new float[output_size];
    CHECK(cudaMemcpy(output, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost));

    delete output;

    DBG_LOGI("GCFSR warm up done\n");
}

void GCFSR::predict(const cv::Mat &mat, cv::Mat &result)
{
    /*
    output: CV_8CU3, 0<=val<=255
    */
    int im_h = mat.rows;
    int im_w = mat.cols;
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

    cv::Mat out(size, size, CV_32FC3, output);
    out.convertTo(out, CV_8UC3);
//     cv::cvtColor(out, result, cv::COLOR_RGB2BGR);
    cv::resize(out, result, cv::Size(im_w, im_h));

    delete output;
}

void GCFSR::pasterBack(cv::Mat &ori_img, cv::Mat &pred_img)
{
    // cv::Mat mask(ori_img.rows, ori_img.cols, CV_8UC3, cv::Scalar(255,255,255));
    // int mk = std::min(ori_img.cols, ori_img.rows) / 16;
    // if (mk % 2 == 0)
    //     mk = mk + 1;
    // cv::rectangle(mask, cv::Point(0, 0), cv::Point(ori_img.cols-1, ori_img.rows-1), cv::Scalar(0, 0, 0), mk);
    // cv::GaussianBlur(mask, mask, cv::Size(mk, mk), 0);

    cv::Mat mask;
    cv::resize(fusion_mask, mask, ori_img.size());
    cv::Mat inverse_mask;
    cv::bitwise_not(mask, inverse_mask);

    mask.convertTo(mask, CV_32FC3, 1.0 / 255.0);
    inverse_mask.convertTo(inverse_mask, CV_32FC3, 1.0 / 255.0);
    ori_img.convertTo(ori_img, CV_32FC3);
    pred_img.convertTo(pred_img, CV_32FC3);

    cv::multiply(pred_img, mask, pred_img);  // img1 * mask
    cv::multiply(ori_img, inverse_mask, ori_img);  // img2 * (1 - mask)
    cv::add(pred_img, ori_img, pred_img);  // img1 * mask + img2 * (1 - mask)

    pred_img.convertTo(pred_img, CV_8UC3);
}

void GCFSR::pasterBack(cv::Mat &ori_img, cv::Mat &pred_img, cv::Mat &mask)
{
    cv::Mat inverse_mask;
    cv::bitwise_not(mask, inverse_mask);

    mask.convertTo(mask, CV_32FC3, 1.0 / 255.0);
    inverse_mask.convertTo(inverse_mask, CV_32FC3, 1.0 / 255.0);
    ori_img.convertTo(ori_img, CV_32FC3);
    pred_img.convertTo(pred_img, CV_32FC3);

    cv::multiply(pred_img, mask, pred_img);  // img1 * mask
    cv::multiply(ori_img, inverse_mask, ori_img);  // img2 * (1 - mask)
    cv::add(pred_img, ori_img, pred_img);  // img1 * mask + img2 * (1 - mask)

    pred_img.convertTo(pred_img, CV_8UC3);
}