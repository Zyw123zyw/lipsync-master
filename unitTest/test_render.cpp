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
    
    const char *video_params = "";
    // video_params = "{\"filter_head_pose\":1}";
    // video_params = "{\"video_bitrate\":0,\"video_width\":0,\"video_height\":0,\"video_enhance\":0,\"start_index_ratio\":0.5}";
    // video_params = "{\"video_bitrate\":0,\"video_width\":0,\"video_height\":0,\"video_enhance\":0,\"start_index_ratio\":0.5,\"face_box_x\":300,\"face_box_y\":250,\"face_box_w\":400,\"face_box_h\":500}";  //xlh
    // video_params = "{\"face_box_x\":400,\"face_box_y\":0,\"face_box_w\":0,\"face_box_h\":0}"; //wjq
    // video_params = "{\"video_enhance\":0}";
    // video_params = "{\"video_enhance\":0, \"video_bitrate\":0,  \"amplifier\": 1.0, \"video_max_side\": 1920, \"audio_max_time\": 1800}";
    // video_params = "{\"keep_fps\":1}";
    // video_params = "{\"face_box\":[0,0,500,500]}";
    // video_params = R"({
    //     "face_box":[0,0,500,500]
    // })";
    int KKK=1;
    const char *id_params = "";
    // id_params = R"([
    //     {
    //         "id": 0,
    //         "box": [150, 150, 300, 600],
    //         "face": "path0.jpg",
    //         "audio": "/workspace/project/talkingface/test_asserts/multi_person/audio2_id0.MP3"
    //     },
    //     {
    //         "id": 1,
    //         "box": [430, 150, 300, 600],
    //         "face": "path1.jpg",
    //         "audio": "/workspace/project/talkingface/test_asserts/multi_person/audio2_id1.MP3"
    //     },
    //     {
    //         "id": 2,
    //         "box": [700, 150, 300, 600],
    //         "face": "path2.jpg",
    //         "audio": "/workspace/project/talkingface/test_asserts/multi_person/audio2_id2.MP3"
    //     },
    //     {
    //         "id": 3,
    //         "box": [980, 150, 300, 600],
    //         "face": "path3.jpg",
    //         "audio": "/workspace/project/talkingface/test_asserts/multi_person/audio2_id3.MP3"
    //     }
    // ])";

    // id_params = R"([
    //     {
    //         "id": 0,
    //         "box": [0, 0, 1080, 960],
    //         "face": "path1.jpg",
    //         "audio": "/workspace/project/talkingface/test_asserts/multi_person/audio3_id0.wav"
    //     },
    //     {
    //         "id": 1,
    //         "box": [0, 960, 1080, 960],
    //         "face": "path2.jpg",
    //         "audio": "/workspace/project/talkingface/test_asserts/multi_person/audio3_id1.wav"
    //     }
    // ])";

    TalkingFace *tf = new TalkingFace();
    tf->init(model_dir, n);

    for (int i=0; i<1; i++) {
        double modelStart = (double)cv::getTickCount();
        tf->render(src_video_path, src_audio_path, save_json_path, save_render_video_path, video_params, vocal_audio_path, id_params);
        double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
        DBG_LOGI("render inference time: %f s\n", modelTime);
    }

    std::string delete_command = "rm -rf " + (std::string)tf->tmp_dir;
    system(delete_command.c_str());

    tf->stop();

    return 0;
}
