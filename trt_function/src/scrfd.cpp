#include "scrfd.h"

using Function::DetectionBox;
using Function::SCRFD;


SCRFD::SCRFD(const std::string &_engine_path) : BasicTRTHandler(_engine_path) {
    this->generate_points(size, size);
}

void SCRFD::generate_points(const int target_height, const int target_width)
{
    if (center_points_is_update) return;
    // 8, 16, 32
    for (auto stride : feat_stride_fpn)
    {
        unsigned int num_grid_w = target_width / stride;
        unsigned int num_grid_h = target_height / stride;
        // y
        for (unsigned int i = 0; i < num_grid_h; ++i)
        {
        // x
        for (unsigned int j = 0; j < num_grid_w; ++j)
        {
            // num_anchors, col major
            for (unsigned int k = 0; k < num_anchors; ++k)
            {
            SCRFDPoint point;
            point.cx = (float) j;
            point.cy = (float) i;
            point.stride = (float) stride;
            center_points[stride].push_back(point);
            }

        }
        }
    }

    center_points_is_update = true;
}

void SCRFD::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                           int target_height, int target_width,
                           SCRFDScaleParams &scale_params)
{
    if (mat.empty()) return;
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                    cv::Scalar(0, 0, 0));
    // scale ratio (new / old) new_shape(h,w)
    float w_r = (float) target_width / (float) img_width;
    float h_r = (float) target_height / (float) img_height;
    float r = std::min(w_r, h_r);
    // compute padding
    int new_unpad_w = static_cast<int>((float) img_width * r); // floor
    int new_unpad_h = static_cast<int>((float) img_height * r); // floor
    int pad_w = target_width - new_unpad_w; // >=0
    int pad_h = target_height - new_unpad_h; // >=0

    int dw = pad_w / 2;
    int dh = pad_h / 2;

    // resize with unscaling
    cv::Mat new_unpad_mat;
    // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
    cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // record scale params.
    scale_params.ratio = r;
    scale_params.dw = dw;
    scale_params.dh = dh;
    scale_params.flag = true;
}

std::vector<float> SCRFD::prepare(const cv::Mat& mat)
{
    cv::Mat canva = mat.clone();
    normalize_inplace(canva, mean_vals, scale_vals, false);
    std::vector<float> result(size * size * 3);
    trans2chw(canva, result);
    return result;
}

void SCRFD::warmup()
{
    DBG_LOGI("SCRFD warm up start\n");
    cv::Mat mat = cv::Mat(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<float> cur_input = this->prepare(mat);

    CHECK(cudaMemcpy(buffers[0], cur_input.data(), buffer_size[0], cudaMemcpyHostToDevice));
    
    context->executeV2(buffers);

    float *score_8 = new float[s8];
    float *bbox_8 = new float[b8];
    float *score_16 = new float[s16];
    float *bbox_16 = new float[b16];
    float *score_32 = new float[s32];
    float *bbox_32 = new float[b32];

    CHECK(cudaMemcpy(score_8, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bbox_8, buffers[2], buffer_size[2], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(score_16, buffers[3], buffer_size[3], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bbox_16, buffers[4], buffer_size[4], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(score_32, buffers[5], buffer_size[5], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bbox_32, buffers[6], buffer_size[6], cudaMemcpyDeviceToHost));

    delete score_8;
    delete bbox_8;
    delete score_16;
    delete bbox_16;
    delete score_32;
    delete bbox_32;

    DBG_LOGI("SCRFD warm up done\n");
}

void SCRFD::detect(const cv::Mat &mat, std::vector<DetectionBox> &detected_boxes_kps,
                   float score_threshold, float iou_threshold, unsigned int topk)
{
    if (mat.empty()) return;
    auto img_height = static_cast<float>(mat.rows);
    auto img_width = static_cast<float>(mat.cols);

    // resize & unscale
    cv::Mat mat_rs;
    SCRFDScaleParams scale_params;
    this->resize_unscale(mat, mat_rs, size, size, scale_params);

    // 创建输入
    std::vector<float> cur_input = this->prepare(mat_rs);

    // 将输入传递到GPU
    CHECK(cudaMemcpy(buffers[0], cur_input.data(), buffer_size[0], cudaMemcpyHostToDevice));

    // 异步执行
    context->executeV2(buffers);

    //输出传回给CPU
    float *score_8 = new float[s8];
    float *bbox_8 = new float[b8];
    float *score_16 = new float[s16];
    float *bbox_16 = new float[b16];
    float *score_32 = new float[s32];
    float *bbox_32 = new float[b32];

    CHECK(cudaMemcpy(score_8, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bbox_8, buffers[2], buffer_size[2], cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(score_16, buffers[3], buffer_size[3], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bbox_16, buffers[4], buffer_size[4], cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(score_32, buffers[5], buffer_size[5], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bbox_32, buffers[6], buffer_size[6], cudaMemcpyDeviceToHost));

    // 3. rescale & exclude.
    std::vector<DetectionBox> bbox_kps_collection;
    this->generate_bboxes_kps(scale_params, bbox_kps_collection,
                              score_8, bbox_8,
                              score_16, bbox_16,
                              score_32, bbox_32,
                              score_threshold, img_height, img_width);
    // 4. hard nms with topk.
    // this->nms_bboxes_kps(bbox_kps_collection, detected_boxes_kps, iou_threshold, topk);
    this->nms_bboxes_kps_combined(bbox_kps_collection, detected_boxes_kps, iou_threshold, topk);

    delete score_8;
    delete bbox_8;
    delete score_16;
    delete bbox_16;
    delete score_32;
    delete bbox_32;
}

void SCRFD::generate_bboxes_kps(const SCRFDScaleParams &scale_params,
                                std::vector<DetectionBox> &bbox_kps_collection,
                                const float *score_8, const float *bbox_8,
                                const float *score_16, const float *bbox_16,
                                const float *score_32, const float *bbox_32,
                                float score_threshold,
                                float img_height,
                                float img_width)
{
    // generate center points.
    // this->generate_points(size, size);

    // level 8 & 16 & 32
    this->generate_bboxes_single_stride(scale_params, score_8, bbox_8, s8, 8, score_threshold,
                                        img_height, img_width, bbox_kps_collection);
    this->generate_bboxes_single_stride(scale_params, score_16, bbox_16, s16, 16, score_threshold,
                                        img_height, img_width, bbox_kps_collection);
    this->generate_bboxes_single_stride(scale_params, score_32, bbox_32, s32, 32, score_threshold,
                                        img_height, img_width, bbox_kps_collection);
}

void SCRFD::generate_bboxes_single_stride(
    const SCRFDScaleParams &scale_params, 
    const float *score_ptr, 
    const float *bbox_ptr,
    unsigned int num_points,
    unsigned int stride, 
    float score_threshold, 
    float img_height, 
    float img_width,
    std::vector<DetectionBox> &bbox_kps_collection)
{
    unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
    nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

    float ratio = scale_params.ratio;
    int dw = scale_params.dw;
    int dh = scale_params.dh;

    unsigned int count = 0;
    auto &stride_points = center_points[stride];

    for (unsigned int i = 0; i < num_points; ++i)
    {
        // std::cout << i << std::endl;
        const float cls_conf = score_ptr[i];
        if (cls_conf < score_threshold) continue; // filter
        auto &point = stride_points.at(i);
        const float cx = point.cx; // cx
        const float cy = point.cy; // cy
        const float s = point.stride; // stride

        // bbox
        const float *offsets = bbox_ptr + i * 4;
        float l = offsets[0]; // left
        float t = offsets[1]; // top
        float r = offsets[2]; // right
        float b = offsets[3]; // bottom

        DetectionBox box_kps;
        float x1 = ((cx - l) * s - (float) dw) / ratio;  // cx - l x1
        float y1 = ((cy - t) * s - (float) dh) / ratio;  // cy - t y1
        float x2 = ((cx + r) * s - (float) dw) / ratio;  // cx + r x2
        float y2 = ((cy + b) * s - (float) dh) / ratio;  // cy + b y2
        box_kps.x1 = std::max(0.f, x1);
        box_kps.y1 = std::max(0.f, y1);
        box_kps.w = std::min(img_width - 1.f, x2 - x1);
        box_kps.h = std::min(img_height - 1.f, y2 - y1);
        box_kps.score = cls_conf;

        bbox_kps_collection.push_back(box_kps);

        count += 1; // limit boxes for nms.
        if (count > max_nms)
            break;
    }

    if (bbox_kps_collection.size() > nms_pre_)
    {
        std::sort(
            bbox_kps_collection.begin(), bbox_kps_collection.end(),
            [](const DetectionBox &a, const DetectionBox &b)
            { return a.score > b.score; }
        ); // sort inplace
        // trunc
        bbox_kps_collection.resize(nms_pre_);
    }

}


inline float SCRFD::intersection_area(const DetectionBox& a, const DetectionBox& b)
{
    if (a.x1 > b.x1 + b.w || a.x1 + a.w < b.x1 || a.y1 > b.y1 + b.h || a.y1 + a.h < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x1 + a.w, b.x1 + b.w) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y1 + a.h, b.y1 + b.h) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

void SCRFD::nms_bboxes_kps(std::vector<DetectionBox> &input,
                           std::vector<DetectionBox> &output,
                           float iou_threshold, unsigned int topk)
{
    if (input.empty()) return;
    std::sort(
        input.begin(), input.end(),
        [](const DetectionBox &a, const DetectionBox &b)
        { return a.score > b.score; }
    );
    const unsigned int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    std::vector<float> areas(box_num);
    for (int i = 0; i < box_num; i++)
    {
        areas[i] = input[i].w * input[i].h;
    }

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i)
    {
        if (merged[i]) continue;
        std::vector<DetectionBox> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        for (unsigned int j = i + 1; j < box_num; ++j)
        {
            if (merged[j]) continue;

            float inter_area = intersection_area(input[i], input[j]);
            float union_area = areas[i] + areas[j] - inter_area;

            if (inter_area / union_area > iou_threshold)
            {
                merged[j] = 1;
                buf.push_back(input[j]);
            }

        }
        output.push_back(buf[0]);

        // keep top k
        count += 1;
        if (count >= topk)
        break;
    }
}

void SCRFD::nms_bboxes_kps_combined(std::vector<DetectionBox> &input,
                           std::vector<DetectionBox> &output,
                           float iou_threshold, unsigned int topk)
{
    if (input.empty()) return;

    // 按照置信度从高到低排序检测框 (保留这一步是为了保持NMS的稳定性)
    std::sort(
        input.begin(), input.end(),
        [](const DetectionBox &a, const DetectionBox &b)
        { return a.score > b.score; }
    );

    const unsigned int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    std::vector<float> areas(box_num);
    for (int i = 0; i < box_num; i++)
    {
        areas[i] = input[i].w * input[i].h;
    }

    // 计算图像中心点
    float img_center_x = size / 2.0f;
    float img_center_y = size / 2.0f;

    // 计算最大面积和最大距离用于归一化
    float max_area = 0.0f;
    float max_dist = 0.0f;
    for (const auto& box : input) {
        max_area = std::max(max_area, box.w * box.h);
        float dist = std::sqrt(std::pow(box.x1 + box.w/2.0f - img_center_x, 2) +
                             std::pow(box.y1 + box.h/2.0f - img_center_y, 2));
        max_dist = std::max(max_dist, dist);
    }

    // 定义综合评分计算函数
    auto compute_combined_score = [img_center_x, img_center_y, max_area, max_dist]
                                (const DetectionBox &box) {
        // 计算归一化的人脸大小 (0~1)
        float norm_size = (box.w * box.h) / max_area;

        // 计算归一化的中心距离 (0~1)
        float dist = std::sqrt(std::pow(box.x1 + box.w/2.0f - img_center_x, 2) +
                             std::pow(box.y1 + box.h/2.0f - img_center_y, 2));
        float norm_dist = dist / max_dist;

        // 综合评分计算
        // 权重可以根据实际需求调整
        const float w_conf = 0.4f;    // 置信度权重
        const float w_size = 0.3f;    // 尺寸权重
        const float w_dist = 0.3f;    // 距离权重（负向影响）

        return w_conf * box.score + w_size * norm_size - w_dist * norm_dist;
    };

    std::vector<DetectionBox> combined_boxes;
    for (unsigned int i = 0; i < box_num; ++i)
    {
        if (merged[i]) continue;
        std::vector<DetectionBox> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        // NMS处理
        for (unsigned int j = i + 1; j < box_num; ++j)
        {
            if (merged[j]) continue;

            float inter_area = intersection_area(input[i], input[j]);
            float union_area = areas[i] + areas[j] - inter_area;

            if (inter_area / union_area > iou_threshold)
            {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }

        // 对于每组重叠的框，选择综合评分最高的一个
        auto max_score_box = std::max_element(buf.begin(), buf.end(),
            [&compute_combined_score](const DetectionBox &a, const DetectionBox &b) {
                return compute_combined_score(a) < compute_combined_score(b);
            });

        combined_boxes.push_back(*max_score_box);
    }

    // 按综合评分对所有保留的框进行排序
    std::sort(
        combined_boxes.begin(), combined_boxes.end(),
        [&compute_combined_score](const DetectionBox &a, const DetectionBox &b) {
            return compute_combined_score(a) > compute_combined_score(b);
        }
    );

    // 选择前topk个框
    for (unsigned int i = 0; i < std::min(topk, static_cast<unsigned int>(combined_boxes.size())); ++i)
    {
        output.push_back(combined_boxes[i]);
    }
}

void SCRFD::expand_box(
    const cv::Size s, const DetectionBox &box,
    cv::Rect_<int> &out_rect, float increase_area, float increase_margin[4])
{
    /*
    float increase_area: 超参数，控制外扩的比例
    */
    // 向上扩大人脸区域，配合后续扩边逻辑，能够囊括头部和帽子区域
    float box_x1 = box.x1;
    float box_y1 = box.y1;
    float box_h = box.h;
    float box_w = box.w;
    if (box.h / box.w < 1.6)
    {
        float height_ori = box.h;
        box_h = box.w * 1.6;
        box_y1 = box_y1 - (box_h - height_ori);
    }

    int width = (int)(box_w);
    int height = (int)(box_h);
    // 扩边逻辑源自 https://github.com/AliaksandrSiarohin/first-order-model/blob/master/crop-video.py
    float width_increase = std::max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width));
    float height_increase = std::max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height));
    int x1 = std::max(0, (int)(box_x1 - width_increase * width * increase_margin[0]));
    int y1 = std::max(0, (int)(box_y1 - height_increase * height * increase_margin[1]));
    int x2 = std::min(s.width, (int)(box_x1 + width +  width_increase * width * increase_margin[2]));
    int y2 = std::min(s.height, (int)(box_y1 + height +  height_increase * height * increase_margin[3]));
    out_rect.x = x1;
    out_rect.y = y1;
    out_rect.width = x2 - x1;
    out_rect.height = y2 - y1;
}

void SCRFD::expand_box(
    const cv::Size s, const cv::Rect2i &box,
    cv::Rect_<int> &out_rect, float increase_area, float increase_margin[4])
{
    /*
    float increase_area: 超参数，控制外扩的比例
    */
    // 向上扩大人脸区域，配合后续扩边逻辑，能够囊括头部和帽子区域
    float box_x1 = box.x;
    float box_y1 = box.y;
    float box_h = box.height;
    float box_w = box.width;
    if (box.height / box.width < 1.6)
    {
        float height_ori = box.height;
        box_h = box.width * 1.6;
        box_y1 = box_y1 - (box_h - height_ori);
    }

    int width = (int)(box_w);
    int height = (int)(box_h);
    // 扩边逻辑源自 https://github.com/AliaksandrSiarohin/first-order-model/blob/master/crop-video.py
    float width_increase = std::max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width));
    float height_increase = std::max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height));
    int x1 = std::max(0, (int)(box_x1 - width_increase * width * increase_margin[0]));
    int y1 = std::max(0, (int)(box_y1 - height_increase * height * increase_margin[1]));
    int x2 = std::min(s.width, (int)(box_x1 + width +  width_increase * width * increase_margin[2]));
    int y2 = std::min(s.height, (int)(box_y1 + height +  height_increase * height * increase_margin[3]));
    out_rect.x = x1;
    out_rect.y = y1;
    out_rect.width = x2 - x1;
    out_rect.height = y2 - y1;
}

void SCRFD::expand_box_for_pipnet(const cv::Size s, const DetectionBox &box,
                                  cv::Rect_<int> &out_rect, float box_scale)
{
    /*
    float increase_area: 超参数，控制外扩的比例
    */
    // 向上扩大人脸区域，配合后续扩边逻辑，能够囊括头部和帽子区域
    float x1 = box.x1;
    float y1 = box.y1;
    float x2 = x1 + box.w;
    float y2 = y1 + box.h;

    int x1_expand = std::max(int(x1 - (box.w * (box_scale - 1) / 2)), 0);
    int y1_expand = std::max(int(y1 + (box.h * (box_scale - 1) / 2)), 0);
    int x2_expand = std::min(int(x2 + (box.w * (box_scale - 1) / 2)), s.width - 1);
    int y2_expand = std::min(int(y2 + (box.h * (box_scale - 1)) / 2), s.height - 1);

    out_rect.x = x1_expand;
    out_rect.y = y1_expand;
    out_rect.width = x2_expand - x1_expand;
    out_rect.height = y2_expand - y1_expand;
}

void SCRFD::paintRect(cv::Mat &image, const std::vector<DetectionBox> detection)
{
    cv::Scalar col_red{0, 0, 255};
    cv::Scalar col_blue{255, 0, 0};

    for (size_t i = 0; i < detection.size(); i++)
    {
        cv::Rect det_rect(detection[i].x1, detection[i].y1, detection[i].w, detection[i].h);
        cv::rectangle(image, det_rect, col_red);
        for (int ks=0; ks < detection[i].fiveLand.size(); ++ks) {
            cv::Point2f p(detection[i].fiveLand[ks]);
            cv::circle(image, p, 2, col_blue, -1);
        }
    }
}

void SCRFD::paintRect(cv::Mat &image, const std::vector<cv::Rect_<int>> out_rect)
{
    cv::Scalar col_red{0, 0, 255};

    for (size_t i = 0; i < out_rect.size(); i++)
    {
        cv::rectangle(image, out_rect[i], col_red);
    }
}

void SCRFD::detectGPU(const cv::cuda::GpuMat &gpu_mat, std::vector<DetectionBox> &detected_boxes_kps,
                      float score_threshold, float iou_threshold, unsigned int topk)
{
    if (gpu_mat.empty()) return;
    auto img_height = static_cast<float>(gpu_mat.rows);
    auto img_width = static_cast<float>(gpu_mat.cols);

    // 1. GPU上进行resize_unscale（保持宽高比 + padding）
    if (gpu_resized_.rows != size || gpu_resized_.cols != size) {
        gpu_resized_.create(size, size, CV_8UC3);
    }
    
    gpuResizeUnscale(gpu_mat.ptr<unsigned char>(), gpu_resized_.ptr<unsigned char>(),
                     gpu_mat.cols, gpu_mat.rows, size, size,
                     3, gpu_mat.step, gpu_resized_.step,
                     gpu_scale_params_, nullptr);

    // 2. GPU预处理（normalize + HWC→CHW），结果直接写入TensorRT的输入buffer
    //    无需H2D传输！数据已经在GPU上了
    gpu_preprocessor_.process(gpu_resized_, (float*)buffers[0], size, mean_vals, scale_vals);

    // 3. TensorRT推理
    context->executeV2(buffers);

    // 4. D2H传输（保留，和CPU版本一样）
    float *score_8 = new float[s8];
    float *bbox_8 = new float[b8];
    float *score_16 = new float[s16];
    float *bbox_16 = new float[b16];
    float *score_32 = new float[s32];
    float *bbox_32 = new float[b32];

    CHECK(cudaMemcpy(score_8, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bbox_8, buffers[2], buffer_size[2], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(score_16, buffers[3], buffer_size[3], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bbox_16, buffers[4], buffer_size[4], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(score_32, buffers[5], buffer_size[5], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bbox_32, buffers[6], buffer_size[6], cudaMemcpyDeviceToHost));

    // 5. CPU后处理（保留，和CPU版本一样）
    // 需要把GPU的scale_params转换为CPU版本的格式
    SCRFDScaleParams scale_params;
    scale_params.ratio = gpu_scale_params_.ratio;
    scale_params.dw = gpu_scale_params_.dw;
    scale_params.dh = gpu_scale_params_.dh;
    scale_params.flag = true;

    std::vector<DetectionBox> bbox_kps_collection;
    this->generate_bboxes_kps(scale_params, bbox_kps_collection,
                              score_8, bbox_8,
                              score_16, bbox_16,
                              score_32, bbox_32,
                              score_threshold, img_height, img_width);

    // 6. NMS
    this->nms_bboxes_kps_combined(bbox_kps_collection, detected_boxes_kps, iou_threshold, topk);

    delete[] score_8;
    delete[] bbox_8;
    delete[] score_16;
    delete[] bbox_16;
    delete[] score_32;
    delete[] bbox_32;
}