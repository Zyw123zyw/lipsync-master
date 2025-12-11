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

    void read_images(const std::vector<std::string> image_files);
    void read_video(const char *video_path);

    const int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    const double fps = 25.0;
    cv::Size2d video_size = cv::Size2d(1080, 1920);
    cv::VideoWriter video_writer;

    std::string tmp_video_path = "./tmp_video.mp4";

private:
    SCRFD *face_detector = nullptr;
    PIPNet *face_landmarker = nullptr;
};


ModelFacecrop::ModelFacecrop(){}

ModelFacecrop::~ModelFacecrop()=default;


void ModelFacecrop::read_images(const std::vector<std::string> image_files)
{
    for (std::string image_path : image_files)
    {
        double modelStart = (double)cv::getTickCount();
        cv::Mat frame = cv::imread(image_path);
        double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
        DBG_LOGI("read image time: %f s\n", modelTime);
    }
    return;
}

void ModelFacecrop::read_video(const char *video_path)
{
    cv::VideoCapture vcap;
    vcap.open(video_path);
    
    int frame_nums = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_COUNT));
    
    cv::Mat frame;
    int cnt = 0;
    while (true)
    {
        double modelStart = (double)cv::getTickCount();
        vcap >> frame;
        if (frame.empty())
            break;

        cnt++;
        if (cnt == frame_nums)
        {
            cnt = 0;
            vcap.set(cv::CAP_PROP_POS_FRAMES, 0);
        }

        double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
        DBG_LOGI("read video time: %fs, %d\n", modelTime, cnt);
    }
    vcap.release();
    return;
}


int main(int argc, char* argv[])
{
    // const char* imagedir = argv[1];
    const char* imagedir = "/workspace/project/talkingface/test_asserts/images/720p";
    const char* videopath = "/workspace/project/talkingface/test_asserts/videos/720p.mp4";

    std::vector<std::string> image_files;
    get_file_names(imagedir, image_files);
    std::cout << "images nums: " << image_files.size() << std::endl;

    ModelFacecrop *facecroper = new ModelFacecrop();

    for (unsigned int i = 0; i < 1; ++i)
    {
        DBG_LOGI("model start\n");
        double modelStart = (double)cv::getTickCount();
        facecroper->read_images(image_files);   // 4k:47ms  720p:5.5ms
        facecroper->read_video(videopath);  // 4k:16ms  720p:2ms
        double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
        DBG_LOGI("model inference time: %f s\n\n", modelTime);
    }

    delete facecroper;
    facecroper = nullptr;

    return 1;
}
