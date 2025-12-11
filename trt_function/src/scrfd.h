#ifndef SCRFD_H
#define SCRFD_H

#include <fstream>
#include "../core/trt_handler.h"
#include "../core/trt_utils.h"

namespace Function
{

struct DetectionBox {
    float x1, y1, w, h;
    float score;
    std::vector<cv::Point2f> fiveLand;
};

class SCRFD : public BasicTRTHandler
{
public:
    // explicit SCRFD(const std::string &_onnx_path, unsigned int _num_threads=1, int gpu_id=-1);
    // ~SCRFD() override = default;
    explicit SCRFD(const std::string &_engine_path);
    ~SCRFD() override = default;

private:
    // nested classes
    typedef struct
    {
      float cx;
      float cy;
      float stride;
    } SCRFDPoint;
    typedef struct
    {
      float ratio;
      int dw;
      int dh;
      bool flag;
    } SCRFDScaleParams;

private:
    // Ort::Value transform(const cv::Mat &mat) override;
    std::vector<float> prepare(const cv::Mat& src);
    
    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        SCRFDScaleParams &scale_params);

    // generate once.
    void generate_points(const int target_height, const int target_width);

    void generate_bboxes_single_stride(const SCRFDScaleParams &scale_params,
                                       const float *score_pred,
                                       const float *bbox_pred,
                                       unsigned int num_points,
                                       unsigned int stride,
                                       float score_threshold,
                                       float img_height,
                                       float img_width,
                                       std::vector<DetectionBox> &objects);

    void generate_bboxes_kps(const SCRFDScaleParams &scale_params,
                             std::vector<DetectionBox> &objects,
                             const float *score_8, const float *bbox_8,
                             const float *score_16, const float *bbox_16,
                             const float *score_32, const float *bbox_32,
                             float score_threshold, float img_height,
                             float img_width); // rescale & exclude
    
    inline float intersection_area(const DetectionBox& a, const DetectionBox& b);

    void nms_bboxes_kps(std::vector<DetectionBox> &input,
                        std::vector<DetectionBox> &output,
                        float iou_threshold, unsigned int topk);

    void nms_bboxes_kps_combined(std::vector<DetectionBox> &input,
                        std::vector<DetectionBox> &output,
                        float iou_threshold, unsigned int topk);

private:
    // blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f}; // RGB
    const float scale_vals[3] = {1.f / 128.f, 1.f / 128.f, 1.f / 128.f};
    unsigned int fmc = 3; // feature map count
    bool use_kps = false;
    unsigned int num_anchors = 2;
    std::vector<int> feat_stride_fpn = {8, 16, 32}; // steps, may [8, 16, 32, 64, 128]
    // if num_anchors>1, then stack points in col major -> (height*num_anchor*width,2)
    // anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
    std::unordered_map<int, std::vector<SCRFDPoint>> center_points;
    bool center_points_is_update = false;
    static constexpr const unsigned int nms_pre = 1000;
    static constexpr const unsigned int max_nms = 30000;

    const int size = 640;

    const int s8 = size * size / 8 / 4;
    const int b8 = s8 * 4;

    const int s16 = s8 / 4;
    const int b16 = s16 * 4;

    const int s32 = s8 / 4 / 4;
    const int b32 = s32 * 4;


public:
    void detect(const cv::Mat &mat, std::vector<DetectionBox> &objects,
                float score_threshold = 0.5f, float iou_threshold = 0.45f,
                unsigned int topk = 1);

    void expand_box(
        const cv::Size s, const DetectionBox &box,
        cv::Rect_<int> &out_rect, float increase_area, float increase_margin[4]);

    void expand_box(
        const cv::Size s, const cv::Rect2i &box,
        cv::Rect_<int> &out_rect, float increase_area, float increase_margin[4]);

    void expand_box_for_pipnet(const cv::Size s, const DetectionBox &box,
                               cv::Rect_<int> &out_rect, float box_scale);

    void paintRect(cv::Mat &img, const std::vector<DetectionBox> detection);
    void paintRect(cv::Mat &img, const std::vector<cv::Rect_<int>> out_rect);

    void warmup();

  };

};

#endif
