#include "../src/talkingface.h"

int main(int argc, char* argv[])
{

    /*onnx model path*/
    const char *model_dir = "/workspace/project/talkingface/models";

    const char *src_video_path = argv[1];
    const char *save_json_path = argv[2];
    const char *src_audio_path = argv[3];
    const char *save_render_video_path = argv[4];
    const char *vocal_audio_path = argv[5];

    const int n = std::stoi(argv[6]);

    const char *video_params;
    // video_params = "{\"face_box_x\":300,\"face_box_y\":250,\"face_box_w\":400,\"face_box_h\":500}";
   
    TalkingFace *tf = new TalkingFace();
    tf->init(model_dir, n);

    for (int i=0; i<1; i++) {
        double modelStart = (double)cv::getTickCount();
        tf->process(src_video_path, save_json_path, video_params);
        double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
        DBG_LOGI("process inference time: %f s\n", modelTime);
    }

    std::string delete_command = "rm -rf " + (std::string)tf->tmp_dir;
    system(delete_command.c_str());

    tf->stop();

    return 0;
}
