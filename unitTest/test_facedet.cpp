#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

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


class ModelFacecrop{
public:
    ModelFacecrop();
    ~ModelFacecrop();

    bool init(const std::string model_dir);
    bool stop();

    void predict(const std::vector<std::string> image_files);

    const int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    const double fps = 25.0;
    cv::Size2d video_size = cv::Size2d(720, 1280);
    cv::VideoWriter video_writer;

    std::string tmp_video_path = "./tmp_video.mp4";

private:
    SCRFD *face_detector = nullptr;
    PIPNet *face_landmarker = nullptr;
};


ModelFacecrop::ModelFacecrop(){}

ModelFacecrop::~ModelFacecrop()=default;


bool ModelFacecrop::init(const std::string model_dir) {

    // 人脸检测
    std::string scrfd_path_str = model_dir + "/scrfd_2.5g_shape640x640.engine";
    const char *scrfd_path = scrfd_path_str.c_str();
    DBG_LOGI("SCRFD Model Loading from %s\n", scrfd_path);

    face_detector = new SCRFD(scrfd_path_str);
    face_detector->initialize_handler();
    face_detector->warmup();
    DBG_LOGI("SCRFD Model is initialized.\n");

    // 关键点检测
    std::string pipnet_path_str = model_dir + "/pipnet.engine";
    const char *pipnet_path = pipnet_path_str.c_str();
    DBG_LOGI("PIPNet Loading from %s\n", pipnet_path);

    face_landmarker = new PIPNet(pipnet_path_str);
    face_landmarker->initialize_handler();
    face_landmarker->warmup();
    DBG_LOGI("PIPNet Model is initialized.\n");

    return true;
}

bool ModelFacecrop::stop() {
    if (face_detector)
        delete face_detector;
    if (face_landmarker)
        delete face_landmarker;

    face_detector = nullptr;
    face_landmarker = nullptr;

    return true;
}


void ModelFacecrop::predict(const std::vector<std::string> image_files)
{
    video_writer.open(tmp_video_path, codec, fps, video_size);

    for (std::string image_path : image_files)
    {
        // 打开图像
        std::cout << "path:" << image_path << "  size:" << image_files.size() << std::endl;
        cv::Mat frame = cv::imread(image_path);

        // 输入图像人脸检测
        std::vector<DetectionBox> faceboxes;
        face_detector->detect(frame, faceboxes, 0.5, 0.45, 1);
        if (faceboxes.empty()) {
            continue;
        }

        for (DetectionBox box : faceboxes)
        {
            // face landmark detect
            cv::Rect2i lmsdet_rect;
            face_detector->expand_box_for_pipnet(frame.size(), faceboxes[0], lmsdet_rect, 1.2);
            // cv::Mat kp_img = frame(lmsdet_rect).clone();
            // std::vector<cv::Point2i> landmark_pts;
            // face_landmarker->predict(kp_img, lmsdet_rect, landmark_pts);

            // // draw result
            // for (int i = 0; i < landmark_pts.size(); ++i)
            // {
            //     cv::Point2i pt;
            //     pt.x = (int)landmark_pts[i].x;
            //     pt.y = (int)landmark_pts[i].y;
            //     cv::circle(frame, pt, 2, cv::Scalar(255,0,0), -1);
            // }
            cv::Rect2i rect(box.x1, box.y1, box.w, box.h);
            cv::rectangle(frame, rect, cv::Scalar(0,0,255), 4);
            cv::putText(frame, std::to_string(box.score), cv::Point2i(box.x1, box.y1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 2);
        }

        video_writer.write(frame);
    }
    if (video_writer.isOpened())
        video_writer.release();
    
    return;
}

int main(int argc, char* argv[])
{
    // const char* imagedir = argv[1];
    const char* imagedir = "/workspace/project/talkingface/test_asserts/det/images";

    std::vector<std::string> image_files;
    get_file_names(imagedir, image_files);
    std::cout << "images nums: " << image_files.size() << std::endl;

    ModelFacecrop *facecroper = new ModelFacecrop();
    bool flag = facecroper->init("./models");
    DBG_LOGI("ModelFacecrop Model Init Down !\n");

    for (unsigned int i = 0; i < 1; ++i)
    {
        DBG_LOGI("model start\n");
        double modelStart = (double)cv::getTickCount();
        facecroper->predict(image_files);
        double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
        DBG_LOGI("model inference time: %f s\n\n", modelTime);
    }

    facecroper->stop();
    delete facecroper;
    facecroper = nullptr;
    
    return 1;
}