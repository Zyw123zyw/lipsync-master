#ifndef TRT_UTILS_H
#define TRT_UTILS_H

#include <opencv2/opencv.hpp>
#include "debug/util_debug.h"
#include <sys/stat.h>


void normalize_inplace(cv::Mat &mat_inplace, 
                       const float mean[3], 
                       const float norm[3], 
                       bool swapRB=false);
void trans2chw(const cv::Mat &mat, std::vector<float> &input_tensor);

cv::Mat meanAxis0(const cv::Mat &src);

cv::Mat elementwiseMinus(const cv::Mat &A, const cv::Mat &B);

cv::Mat varAxis0(const cv::Mat &src);

int MatrixRank(cv::Mat M);

cv::Mat similarTransform(cv::Mat src, cv::Mat dst);

cv::Mat invComplex(const cv::Mat& m);

void normalize_kp_inline(std::vector<cv::Point2f> &kp);
void mean_point(std::vector<cv::Point2f> &landmarks);

float cal_norm(std::vector<cv::Point2f> kp_source, std::vector<cv::Point2f> kp_driving);

int devide(std::string video_path, std::string& voise_path);
int getvoise(std::string dvideo_path, std::string dvoise_path);
int merge(std::string video_path, std::string voise_path, std::string output_path);
std::string rand_str(const int len);
static std::vector<int> innermouth_index = {61, 62, 63, 64, 65, 66, 67, 68};
static std::vector<int> lefteye_index = {37, 38, 39, 40, 41, 42};
static std::vector<int> righteye_index = {43, 44, 45, 46, 47, 48};

void pick_rect(cv::Rect_<int> &g_box, std::vector<cv::Rect_<int>> &t_boxes, 
               cv::Rect_<int> &p_box, float threshold);

void merge_mask(cv::Mat &source_mask, cv::Mat &drived_mask, cv::Mat &out_mask);
void mask2dist(cv::Mat &mask, cv::Mat &dist, int gap, int pad=5);

#endif
