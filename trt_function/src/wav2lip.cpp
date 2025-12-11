#include "wav2lip.h"


using Function::Wav2Lip;


Wav2Lip::Wav2Lip(const std::string &_engine_path) : BasicTRTHandler(_engine_path) {
    this->generate_fusion_mask();
}

void Wav2Lip::generate_fusion_mask()
{
    // 制作用于羽化回帖的mask
    fusion_mask.create(target_size, target_size, CV_8UC3);
    fusion_mask.setTo(cv::Scalar(255, 255, 255));

    int k = target_size / 16;
    if (k % 2 == 0)
        k = k + 1;
    cv::rectangle(fusion_mask, cv::Point(0, 0), cv::Point(target_size-1, target_size-1), cv::Scalar(0, 0, 0), k);
    cv::GaussianBlur(fusion_mask, fusion_mask, cv::Size(k, k), 0);
}

void Wav2Lip::warmup()
{
    DBG_LOGI("Wav2Lip warm up start\n");

    // src
    cv::Mat src_img = cv::Mat(target_size, target_size, CV_8UC3, cv::Scalar(0, 0, 0));
    src_img.convertTo(src_img, CV_32FC3, 1.0 / 255.0);
    auto srcTensorSize = src_img.cols * src_img.rows * src_img.channels();
    std::vector<float> src_input(srcTensorSize);
    trans2chw(src_img, src_input);

    // ref
    cv::Mat ref_img = cv::Mat::zeros(target_size, target_size, CV_8UC3);
    ref_img.convertTo(ref_img, CV_32FC3, 1.0 / 255.0);
    auto refTensorSize = ref_img.cols * ref_img.rows * ref_img.channels();
    std::vector<float> ref_input(refTensorSize);
    trans2chw(ref_img, ref_input);

    // audio
    auto audioTensorSize = 1 * audio_feat_channel * audio_feat_duration;
    std::vector<float> audio_input(audioTensorSize, 0);

    // amplifier
    float amplifier_ = 1.8;

    CHECK(cudaMemcpy(buffers[0], src_input.data(), buffer_size[0], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(buffers[1], ref_input.data(), buffer_size[1], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(buffers[2], audio_input.data(), buffer_size[2], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(buffers[3], &amplifier_, buffer_size[3], cudaMemcpyHostToDevice));

    context->executeV2(buffers);

    float *output = new float[srcTensorSize];
    CHECK(cudaMemcpy(output, buffers[4], buffer_size[4], cudaMemcpyDeviceToHost));
    delete output;

    // float *output_sr = new float[srcTensorSize];
    // CHECK(cudaMemcpy(output_sr, buffers[5], buffer_size[5], cudaMemcpyDeviceToHost));
    // delete output_sr;

    DBG_LOGI("Wav2Lip warm up done\n");
}

void Wav2Lip::predict(const cv::Mat &src, const cv::Mat &ref, float *audio, cv::Mat &output, float amplifier)
{
    // src
    cv::Mat src_img = src.clone();
    src_img.convertTo(src_img, CV_32FC3, 1.0 / 255.0);
    auto srcTensorSize = src_img.cols * src_img.rows * src_img.channels();
    std::vector<float> src_input(srcTensorSize);
    trans2chw(src_img, src_input);

    // ref
    cv::Mat ref_img = ref.clone();
    ref_img.convertTo(ref_img, CV_32FC3, 1.0 / 255.0);
    auto refTensorSize = ref_img.cols * ref_img.rows * ref_img.channels();
    std::vector<float> ref_input(refTensorSize);
    trans2chw(ref_img, ref_input);
    
    // 将输入传递到GPU
    CHECK(cudaMemcpy(buffers[0], src_input.data(), buffer_size[0], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(buffers[1], ref_input.data(), buffer_size[1], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(buffers[2], audio, buffer_size[2], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(buffers[3], &amplifier, buffer_size[3], cudaMemcpyHostToDevice));

    // 异步执行
    context->executeV2(buffers);

    // 将输出传递到CPU
    float *out_ = new float[srcTensorSize];
    CHECK(cudaMemcpy(out_, buffers[4], buffer_size[4], cudaMemcpyDeviceToHost));

    cv::Mat out(target_size, target_size, CV_32FC3, out_);
    out.convertTo(output, CV_8UC3, 255.f);

    delete out_;
}

// 伪超分
void Wav2Lip::predict(const cv::Mat &src, const cv::Mat &ref, float *audio, cv::Mat &output, float amplifier, bool enable_sr)
{
    // src
    cv::Mat src_img = src.clone();
    src_img.convertTo(src_img, CV_32FC3, 1.0 / 255.0);
    auto srcTensorSize = src_img.cols * src_img.rows * src_img.channels();
    std::vector<float> src_input(srcTensorSize);
    trans2chw(src_img, src_input);

    // ref
    cv::Mat ref_img = ref.clone();
    ref_img.convertTo(ref_img, CV_32FC3, 1.0 / 255.0);
    auto refTensorSize = ref_img.cols * ref_img.rows * ref_img.channels();
    std::vector<float> ref_input(refTensorSize);
    trans2chw(ref_img, ref_input);
    
    // 将输入传递到GPU
    CHECK(cudaMemcpy(buffers[0], src_input.data(), buffer_size[0], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(buffers[1], ref_input.data(), buffer_size[1], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(buffers[2], audio, buffer_size[2], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(buffers[3], &amplifier, buffer_size[3], cudaMemcpyHostToDevice));

    // 异步执行
    context->executeV2(buffers);

    // 将输出传递到CPU
    float *out_ = new float[srcTensorSize];
    if (enable_sr == false)
        CHECK(cudaMemcpy(out_, buffers[4], buffer_size[4], cudaMemcpyDeviceToHost));
    else
        CHECK(cudaMemcpy(out_, buffers[5], buffer_size[5], cudaMemcpyDeviceToHost));

    cv::Mat out(target_size, target_size, CV_32FC3, out_);
    out.convertTo(output, CV_8UC3, 255.f);

    delete out_;
}


// void Wav2Lip::make_mask_img(const cv::Mat &img, const std::vector<cv::Point> &landmark, cv::Mat &crop, cv::Mat &ref, cv::Mat &src, cv::Rect2i &rect)
// {
//     int img_w = img.cols;
//     int img_h = img.rows;

//     int x1 = landmark[0].x;
//     int x2 = landmark[0].x;
//     int y1 = landmark[0].y;
//     int y2 = landmark[0].y;
//     for (cv::Point pt : landmark)
//     {
//         x1 = std::min(x1, pt.x);
//         x2 = std::max(x2, pt.x);
//         y1 = std::min(y1, pt.y);
//         y2 = std::max(y2, pt.y);
//     }
//     int face_w = x2 - x1;
//     int face_h = y2 - y1;

//     // expand
//     int expand_x1 = static_cast<int>(x1 - face_w * 0.20);
//     int expand_x2 = static_cast<int>(x2 + face_w * 0.20);
//     int expand_y1 = static_cast<int>(y1 - face_h * 0.10);
//     int expand_y2 = static_cast<int>(y2 + face_h * 0.25);

//     expand_x1 = std::min(std::max(expand_x1, 0), img_w-1);
//     expand_x2 = std::min(std::max(expand_x2, 0), img_w-1);
//     expand_y1 = std::min(std::max(expand_y1, 0), img_h-1);
//     expand_y2 = std::min(std::max(expand_y2, 0), img_h-1);

//     rect.x = expand_x1;
//     rect.y = expand_y1;
//     rect.width = expand_x2 - expand_x1;
//     rect.height = expand_y2 - expand_y1;
    
//     // crop
//     crop = img.clone();
//     crop = crop(rect);

//     // ref
//     cv::resize(crop, ref, cv::Size(target_size, target_size));

//     // src mask
//     std::vector<cv::Point> resize_landmark;
//     for (size_t i = 0; i < landmark.size(); i++)
//     {
//         int x = static_cast<float>(landmark[i].x - expand_x1) / rect.width * target_size;
//         int y = static_cast<float>(landmark[i].y - expand_y1) / rect.height * target_size;
//         x = std::min(std::max(x, 0), target_size - 1);
//         y = std::min(std::max(y, 0), target_size - 1);
//         resize_landmark.emplace_back(cv::Point2i(x, y));
//     }

//     // mask
//     int offset_x = 0.05 * face_w;
//     int offset_y = 0.12 * face_h;

//     int lx = resize_landmark[0].x;
//     int rx = resize_landmark[16].x;
//     for (size_t i = 0; i < 17; ++i)
//     {
//         // 更新最小值lx
//         if (resize_landmark[i].x < lx)
//             lx = resize_landmark[i].x;
//         // 更新最大值rx
//         if (resize_landmark[i].x > rx)
//             rx = resize_landmark[i].x;
//     }
//     lx = lx - offset_x;
//     rx = rx + offset_x;

//     int limit_minx = target_size / 16;
//     int limit_maxx = target_size - limit_minx;
//     int limit_maxy = limit_maxx;
    
//     if (lx < limit_minx)
//         lx = limit_minx;
//     if (rx > limit_maxx)
//         rx = limit_maxx;

//     std::vector<cv::Point> mask_pts;
//     mask_pts.emplace_back(cv::Point2i(lx, std::min(std::max(resize_landmark[0].y + offset_y, 0), target_size - 1)));
//     mask_pts.emplace_back(cv::Point2i(limit_minx, std::min(std::max(resize_landmark[0].y + offset_y, 0), target_size - 1)));
//     mask_pts.emplace_back(cv::Point2i(limit_minx, limit_maxy));
//     mask_pts.emplace_back(cv::Point2i(limit_maxx, limit_maxy));
//     mask_pts.emplace_back(cv::Point2i(limit_maxx, std::min(std::max(resize_landmark[16].y + offset_y, 0), target_size - 1)));
//     mask_pts.emplace_back(cv::Point2i(rx, std::min(std::max(resize_landmark[16].y + offset_y, 0), target_size - 1)));
//     mask_pts.emplace_back(cv::Point2i(std::min(std::max(resize_landmark[35].x, 0), target_size - 1), std::min(std::max(resize_landmark[29].y, 0), target_size - 1)));
//     mask_pts.emplace_back(cv::Point2i(std::min(std::max(resize_landmark[31].x, 0), target_size - 1), std::min(std::max(resize_landmark[29].y, 0), target_size - 1)));

//     cv::Mat mask = cv::Mat::zeros(target_size, target_size, CV_8UC1);
//     cv::fillPoly(mask, mask_pts, cv::Scalar(255));

//     src = ref.clone();
//     src.setTo(cv::Scalar(0,0,0), mask==255);
// }


void Wav2Lip::make_mask_img(const cv::Mat &img, const std::vector<cv::Point> &landmark, cv::Mat &crop, cv::Mat &ref, cv::Mat &src, cv::Rect2i &rect, cv::Mat &mask)
{
    int img_w = img.cols;
    int img_h = img.rows;

    int x1 = landmark[0].x;
    int x2 = landmark[0].x;
    int y1 = landmark[0].y;
    int y2 = landmark[0].y;
    for (cv::Point pt : landmark)
    {
        x1 = std::min(x1, pt.x);
        x2 = std::max(x2, pt.x);
        y1 = std::min(y1, pt.y);
        y2 = std::max(y2, pt.y);
    }
    int face_w = x2 - x1;
    int face_h = y2 - y1;

    // expand
    int expand_x1 = static_cast<int>(x1 - face_w * 0.20);
    int expand_x2 = static_cast<int>(x2 + face_w * 0.20);
    int expand_y1 = static_cast<int>(y1 - face_h * 0.10);
    int expand_y2 = static_cast<int>(y2 + face_h * 0.25);

    expand_x1 = std::min(std::max(expand_x1, 0), img_w-1);
    expand_x2 = std::min(std::max(expand_x2, 0), img_w-1);
    expand_y1 = std::min(std::max(expand_y1, 0), img_h-1);
    expand_y2 = std::min(std::max(expand_y2, 0), img_h-1);

    rect.x = expand_x1;
    rect.y = expand_y1;
    rect.width = expand_x2 - expand_x1;
    rect.height = expand_y2 - expand_y1;
    
    // crop
    crop = img.clone();
    crop = crop(rect);

    // ref
    cv::resize(crop, ref, cv::Size(target_size, target_size));

    // src mask
    std::vector<cv::Point> resize_landmark;
    int resize_min_x = target_size;
    int resize_max_x = 0;
    int resize_min_y = target_size;
    int resize_max_y = 0;
    for (size_t i = 0; i < landmark.size(); i++)
    {
        int x = static_cast<float>(landmark[i].x - expand_x1) / rect.width * target_size;
        int y = static_cast<float>(landmark[i].y - expand_y1) / rect.height * target_size;
        x = std::min(std::max(x, 0), target_size - 1);
        y = std::min(std::max(y, 0), target_size - 1);
        resize_landmark.emplace_back(cv::Point2i(x, y));

        resize_min_x = std::min(resize_min_x, x);
        resize_max_x = std::max(resize_max_x, x);
        resize_min_y = std::min(resize_min_y, y);
        resize_max_y = std::max(resize_max_y, y);
    }
    int resize_face_w = resize_max_x - resize_min_x;
    int resize_face_h = resize_max_y - resize_min_y;

    // mask
    std::vector<cv::Point> mask_pts;
    int offset_x = 0.03 * resize_face_w;
    int offset_y = 0.15 * resize_face_h;
    for (size_t i = 0; i < 17; ++i)
    {
        int x = resize_landmark[i].x;
        int y = resize_landmark[i].y;
        y = y + offset_y;
        if (i < 8)
            x = x - offset_x;
        if (i > 8)
            x = x + offset_x;
        x = std::min(std::max(x, 0), target_size-1);
        y = std::min(std::max(y, 0), target_size-1);
        mask_pts.emplace_back(cv::Point2i(x, y));
    }
    mask_pts.emplace_back(cv::Point2i(std::min(std::max(resize_landmark[35].x, 0), target_size-1), std::min(std::max(resize_landmark[29].y, 0), target_size-1)));
    mask_pts.emplace_back(cv::Point2i(std::min(std::max(resize_landmark[31].x, 0), target_size-1), std::min(std::max(resize_landmark[29].y, 0), target_size-1)));

    mask = cv::Mat::zeros(target_size, target_size, CV_8UC1);
    cv::fillPoly(mask, mask_pts, cv::Scalar(255));
    cv::rectangle(mask, cv::Point(0, 0), cv::Point(target_size-1, target_size-1), cv::Scalar(0), 3);

    src = ref.clone();
    src.setTo(cv::Scalar(0,0,0), mask==255);

    // cv::resize(mask, mask, crop.size());
    // cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
}

void Wav2Lip::pasterBack(cv::Mat &ori_img, cv::Mat &pred_img)
{
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

void Wav2Lip::pasterBack(cv::Mat &ori_img, cv::Mat &pred_img, cv::Mat &mask)
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