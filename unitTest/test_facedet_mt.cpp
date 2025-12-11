#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <condition_variable>
#include <queue>
#include <thread>

#include "debug/util_debug.h"
#include "cuda_runtime_api.h"
#include "../trt_function/models.h"
#include "../trt_function/utils.h"

using namespace Function;


void get_file_names(std::string path, std::vector<std::string> &filenames, const std::string& extension = "*")
{
    DIR *pDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str())))
    {
        std::cout << "Folder doesn't Exist!" << std::endl;
        return;
    }
    while ((ptr = readdir(pDir)) != 0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            std::string file_path = path + "/" + ptr->d_name;
            std::string file_ext = file_path.substr(file_path.find_last_of(".")+1);
            if (extension == "*")
                filenames.push_back(file_path);
            else if (file_ext == extension)
                filenames.push_back(file_path);
            // filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
    std::sort(filenames.begin(), filenames.end());
}

struct PredictInfos
{
    int n_threads;       // 线程数
    int video_width;
    int video_height;
    int last_idx;       // 最新渲染帧的索引
    int frame_num;      // 底板视频帧数
    int render_num;     // 渲染帧数，根据音频特征长度而定
    
    std::string frame_dir;      // 底板视频帧路径
    std::string tmp_video_path = "./tmp_video.mp4";
    std::queue<cv::Mat> rendered_frames;        // 渲染完成的帧
    cv::VideoWriter video_writer;
};

class MultiThead
{
public:
    MultiThead();
    ~MultiThead();

    bool init(const std::string model_dir, const int n_threads);
    bool stop();
    void reset(const int render_num, const int frame_num, const std::string frame_dir, const int video_width, const int video_height);
    void predict(const int render_num, const int frame_num, const std::string frame_dir, const int video_width, const int video_height);
    void renderProducer(int work_idx);      // 合成渲染帧, 并push至队列中
    void writeConsumer();                   // 写入视频文件

private:
    std::vector<SCRFD*> m_face_detectors;
    std::vector<PIPNet*> m_face_landmarkers;
    PredictInfos m_predictInfos;

    std::mutex mtx;
    std::condition_variable m_cond_save;
    std::condition_variable m_cond_write;
};

MultiThead::MultiThead(){}

MultiThead::~MultiThead()=default;

bool MultiThead::init(const std::string model_dir, int n_threads) {
    m_predictInfos.n_threads = n_threads;

    std::string scrfd_path = model_dir + "/scrfd_2.5g_shape640x640.engine";
    DBG_LOGI("SCRFD Model Loading from %s\n", scrfd_path.c_str());

    std::string pipnet_path = model_dir + "/pipnet.engine";
    DBG_LOGI("PIPNet Loading from %s\n", pipnet_path.c_str());
    
    for(int i = 0; i < n_threads; i++){
        SCRFD *face_detector = new SCRFD(scrfd_path);
        face_detector->initialize_handler();
        face_detector->warmup();
        m_face_detectors.push_back(face_detector);
        DBG_LOGI("SCRFD %d Model is initialized.\n", i);

        PIPNet *face_landmarker = new PIPNet(pipnet_path);
        face_landmarker->initialize_handler();
        face_landmarker->warmup();
        m_face_landmarkers.push_back(face_landmarker);
        DBG_LOGI("PIPNet %d Model is initialized.\n", i);
    }
    DBG_LOGE("Model is initialized! \n");
    return true;
}

bool MultiThead::stop() {
    for (int i = 0; i < m_face_detectors.size(); i++)
    {
        m_face_detectors[i] = nullptr;
        m_face_landmarkers[i] = nullptr;
    }
    return true;
}

void MultiThead::reset(const int render_num, const int frame_num, const std::string frame_dir, const int video_width, const int video_height)
{
    m_predictInfos.render_num = render_num;
    m_predictInfos.frame_num = frame_num;
    m_predictInfos.frame_dir = frame_dir;
    m_predictInfos.video_width = video_width;
    m_predictInfos.video_height = video_height;

    m_predictInfos.last_idx = -1;
    m_predictInfos.rendered_frames = std::queue<cv::Mat>();

    const int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    const double fps = 25.0;
    cv::Size2d video_size = cv::Size2d(m_predictInfos.video_width, m_predictInfos.video_height);
    m_predictInfos.video_writer.open(m_predictInfos.tmp_video_path, codec, fps, video_size);
}

void MultiThead::renderProducer(int work_idx)
{
    DBG_LOGI("render producer %d thread start.\n", work_idx);
    cv::Mat frame;
    std::string framepath;
    char buffer[256];
    int divisor, remainder, diban_idx, img_idx;
    int cnt = 0;
    while (1)
    {
        img_idx = cnt * m_predictInfos.n_threads + work_idx;     // 渲染帧的索引
        if (img_idx >= m_predictInfos.render_num)
            break;

        divisor = img_idx / m_predictInfos.frame_num;
        remainder = img_idx % m_predictInfos.frame_num;
        if (divisor % 2 == 0)
            diban_idx = remainder;
        else
            diban_idx = m_predictInfos.frame_num - 1 - remainder;
        
        std::sprintf(buffer, "/%06d.jpg", diban_idx);
        framepath = m_predictInfos.frame_dir + std::string(buffer);
        frame = cv::imread(framepath);

        std::vector<DetectionBox> faceboxes;
        m_face_detectors[work_idx]->detect(frame, faceboxes, 0.1, 0.45, 1);

        for (DetectionBox box : faceboxes)
        {
            // face landmark detect
            cv::Rect2i lmsdet_rect;
            m_face_detectors[work_idx]->expand_box_for_pipnet(frame.size(), faceboxes[0], lmsdet_rect, 1.2);
            cv::Mat kp_img = frame(lmsdet_rect).clone();
            std::vector<cv::Point2i> landmark_pts;
            m_face_landmarkers[work_idx]->predict(kp_img, lmsdet_rect, landmark_pts);

            // draw result
            for (int i = 0; i < landmark_pts.size(); ++i)
            {
                cv::Point2i pt;
                pt.x = (int)landmark_pts[i].x;
                pt.y = (int)landmark_pts[i].y;
                cv::circle(frame, pt, 2, cv::Scalar(255,0,0), -1);
            }
            cv::Rect2i rect(box.x1, box.y1, box.w, box.h);
            cv::rectangle(frame, rect, cv::Scalar(0,0,255), 4);
            cv::putText(frame, std::to_string(box.score), cv::Point2i(box.x1, box.y1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 2);
        }

        std::unique_lock<std::mutex> save_lock(mtx);
        while (m_predictInfos.last_idx != (img_idx - 1) || m_predictInfos.rendered_frames.size() > 2*m_predictInfos.n_threads)
            m_cond_save.wait(save_lock);

        m_predictInfos.last_idx = img_idx;
        m_predictInfos.rendered_frames.push(frame);
        cnt++;
        
        save_lock.unlock();
        m_cond_save.notify_all();
        m_cond_write.notify_one();
    }
    DBG_LOGI("render producer %d thread finish.\n", work_idx);
}

void MultiThead::writeConsumer()
{
    DBG_LOGI("write consumer thread start.\n");
    cv::Mat frame;
    int write_idx = 0;
    while (1)
    {
        std::unique_lock<std::mutex> write_lock(mtx);
        while (m_predictInfos.rendered_frames.empty())
            m_cond_write.wait(write_lock);

        frame = m_predictInfos.rendered_frames.front();
        m_predictInfos.rendered_frames.pop();
        
        write_lock.unlock();
        m_cond_save.notify_all();

        m_predictInfos.video_writer.write(frame);

        write_idx++;
        if (write_idx >= m_predictInfos.render_num)
            break;
    }

    if (m_predictInfos.video_writer.isOpened())
        m_predictInfos.video_writer.release();
    DBG_LOGI("write consumer thread finish.\n");
}

void MultiThead::predict(const int render_num, const int frame_num, const std::string frame_dir, 
                         const int video_width, const int video_height)
{
    MultiThead::reset(render_num, frame_num, frame_dir, video_width, video_height);

    // 开启渲染线程
    std::vector<std::thread> render_threads;
    for (int i = 0; i < m_predictInfos.n_threads; i++)
        render_threads.push_back(std::thread(&MultiThead::renderProducer, this, i));
    
    // 开启写入线程
    std::thread write_thread(&MultiThead::writeConsumer, this);

    for (int i = 0; i < m_predictInfos.n_threads; i++)
        render_threads[i].join();

    write_thread.join();
}


int main(int argc, char* argv[])
{
    // const char* imagedir = argv[1];
    const char* imagedir = "/workspace/project/talkingface/test_asserts/images/720p";
    // const int n = std::stoi(argv[1]);
    const int n = 2;

    std::vector<std::string> image_files;
    get_file_names(imagedir, image_files);
    std::cout << "images nums: " << image_files.size() << std::endl;

    MultiThead *mt = new MultiThead();
    bool flag = mt->init("./models", n);
    DBG_LOGI("MultiThead Model Init Down !\n");

    for (unsigned int i = 0; i < 1; ++i)
    {
        DBG_LOGI("model start\n");
        double start_time = (double)cv::getTickCount();
        mt->predict(image_files.size(), image_files.size(), imagedir, 720, 1280);
        double end_time = ((double)cv::getTickCount() - start_time) / cv::getTickFrequency();
        DBG_LOGI("model inference time: %f s\n\n", end_time);
    }

    mt->stop();
    delete mt;
    mt = nullptr;
    
    return 1;
}
