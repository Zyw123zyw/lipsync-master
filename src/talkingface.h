#ifndef TALKINGFACE_H
#define TALKINGFACE_H

#include <string>
#include <unistd.h>
#include <cstdio>
#include <mutex>
#include <opencv2/core/core.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <dirent.h>
#include <condition_variable>
#include <queue>
#include <thread>
#include <omp.h>

#include "debug/util_debug.h"
#include "json/json.h"
#include "cuda_runtime_api.h"

#include "../trt_function/utils.h"
#include "../trt_function/models.h"
#include "../src/audio/wavdata.h"
#include "../src/audio/wavhelper.h"
#include "ffmpeg_config.h"
#include "gpu_decoder.h"

using namespace Function;
using namespace live;


struct VideoParam {
    int bitrate = 0;
    int max_bitrate = 50000;
    int width = 0;
    int height = 0;
    int video_enhance = 0;

    float amplifier = 1.0;
    float face_det_threshold = 0.5;
    bool filter_head_pose = false;
    cv::Rect2i roi_rect = cv::Rect2i(0, 0, 0, 0);
    int video_max_side = 5120;    // 视频长边限制，兜底5120
    int audio_max_time = 0;    // 音频时长限制
    int video_ffmpeg_duration = 0;  // 视频转frame的-t参数设置
    int shutup_first = 0;     // 首帧静音驱动
    int keep_fps = 0;   // 0为固定25fps， 1为保持原fps
    int keep_bitrate = 0;   // 0为不设置码率，1为保持码率

    void reset()
    {
        *this = VideoParam();
    }
};

struct Infos
{
    int video_width;    // 输出视频宽度
    int video_height;   // 输出视频高度
    int frame_nums;     // 底板视频帧数
    int fps;            // 底板视频帧率
    long bitrate;       // 视频码率
    std::vector<std::string> frame_paths;   // 底板视频帧路径集

    std::vector<std::vector<cv::Rect2i>> face_bboxes;   // 底板视频的bboxes, [id, len, 4]
    std::vector<std::vector<std::vector<cv::Point2i>>> face_landmarks;    // 底板视频的landmarks, [id, len, 68, 2]

    // render
    int last_idx = -1;       // 最新渲染帧的索引
    std::vector<int> ids;   // [id,]
    std::vector<cv::Rect2i> id_rois;    // [id,]
    // std::vector<std::vector<std::vector<float>>> audio_feats;   // hubert特征 [id, len, 20*1024]
    std::vector<std::vector<std::vector<float>>> audio_feats;   // hubert特征 [id, len, 1024]
    std::vector<int> audio_cnts;   // 音频特征数量
    std::vector<float> audio_intervals;   // 音频特征间隔
    int min_audio_cnt = -1;     // 多人did音频特征中最短的长度

    std::queue<cv::Mat> rendered_frames;        // 渲染帧缓存
    cv::VideoWriter video_writer;

    // // // multi person，兼容单人情况
    // std::vector<std::vector<cv::Rect2i>> stamp_rois;    // 提前算好每一帧中的所有roi，兼容多人和单人情况

    int render_codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    // int render_codec = cv::VideoWriter::fourcc('F', 'F', 'V', '1');

    void reset()
    {
        *this = Infos();
    }
};

class TalkingFace
{
private:
    int n_threads;      // 渲染线程数
    int ffmpeg_threads = 0; // ffmpeg线程数

    VideoParam video_params;   // 传参
    Infos infos;
    FFmpegConfig ffmpeg_config; // FFmpeg CUDA配置

    HuBERT *audio_extractor = nullptr;
    std::vector<SCRFD*> m_face_detectors;
    std::vector<PIPNet*> m_face_landmarkers;
    std::vector<Wav2Lip*> m_generators;
    std::vector<GCFSR*> m_enhancers;

    std::vector<float> silence_hubert_feat;

    bool is_running;
    
    // GPU视频解码器
    GPUDecoder* gpu_decoder_ = nullptr;

private:

    void traverseFiles(std::string path, std::vector<std::string> &filenames, const std::string& extension = "jpg");

    bool fileExists(const std::string &filename);

    void drawBox(cv::Mat &img, cv::Rect2i &rect);

    void drawLandmarks(cv::Mat &img, std::vector<cv::Point2i> &landmark_pts);

    void readVideoParam(const char *video_params);

    bool readJsonFile(const char *json_save_path);

    void detectLandmark(int work_idx,
                        const cv::Mat &frame,
                        const cv::Rect2i &roi_rect,
                        cv::Rect2i &face_bbox,
                        std::vector<cv::Point2i> &face_landmark);

    void getVideoLandmark(int work_idx);

    Status extractAudioFeat(const char *audio_path);
    
    Status readVideo(const char *src_video_path);

    Status readVideoInfo(const char *src_video_path, int&  frame_width, int& frame_height, int& fps, long& bitrate);

    void readAudioInfo(const char *audio_path);

    Status audioDubbing(const char *audio_path, const char *render_video_save_path);

    bool DetectSideHeadPose(const DetectionBox &box,
                            std::vector<cv::Point> &landmark);

    void renderProducer(int work_idx);
    void writeConsumer();

    void checkROI(cv::Rect2i &roi_rect);

    void expand_box(
        const cv::Size s, const cv::Rect2i &box,
        cv::Rect_<int> &out_rect, float increase_area, float increase_margin[4]);

    Status ProcessIDParam(const char *id_params);

    std::vector<cv::Rect2i> parseShutupIdParams(const char *id_params, const int frame_width, const int frame_height);

    std::string exec(const char* cmd);

    // CUDA兜底执行：先尝试CUDA命令，失败则自动降级到CPU命令重试
    int executeFFmpegWithFallback(const std::string& cuda_cmd, const std::string& cpu_cmd, const char* operation_name);

public:
    const char *tmp_dir = "./tmp";
    const char *tmp_audio_path = "./tmp/audio.wav";
    const char *tmp_video_path = "./tmp/video.mp4";
    const char *tmp_convert_video_path = "./tmp/convert_video.mp4";
    const char *tmp_frame_dir = "./tmp/frames";
    const char *tmp_crop_audio_path = "./tmp/crop_audio.wav";

public:
    TalkingFace(/* args */);

    ~TalkingFace() = default;

    Status init(const std::string model_dir, int n=1, int fn=0);

    Status process(const char *src_video_path, const char *json_save_path, const char *set_params);

    Status render(const char *src_video_path, 
                  const char *audio_path, 
                  const char *json_save_path, 
                  const char *render_video_save_path, 
                  const char *set_params,
                  const char *vocal_audio_path,
                  const char *id_params);

    Status shutup(const char *input_image_path,
                  const char *save_image_path,
                  const char *set_params,
                  const char *id_params);

    Status stop();

    // Ort::Env ort_env;
};

#endif // TALKINGFACE_H
