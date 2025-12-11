#include "../src/talkingface.h"

int main(int argc, char* argv[])
{

    /*onnx model path*/
    const char *model_dir = "/workspace/project/talkingface/models";

    const char *image_path = argv[1];
    const char *save_path = argv[2];
    
    const char *video_params = "";
    video_params = "{\"video_enhance\":0}";

    const char *id_params = "";

    id_params = R"([
        {
            "id": 0,
            "box": [70, 150, 2000, 2000]
        },
        {
            "id": 1,
            "box": [220, 220, 2000, 2000]
        }
    ])";

    TalkingFace *tf = new TalkingFace();
    const int n = 2;
    tf->init(model_dir, n);

    for (int i=0; i<1; i++) {
        double modelStart = (double)cv::getTickCount();
        tf->shutup(image_path, save_path, video_params, id_params);
        double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
        DBG_LOGI("render inference time: %f s\n", modelTime);
    }

    tf->stop();

    return 0;
}
