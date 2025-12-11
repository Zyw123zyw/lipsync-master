#include "com_hexin_ai_jni_TalkingFace.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "util_debug.h"
#include "talkingface.h"
#include "error_code.h"


static TalkingFace *talkingface = 0;
static bool intFlag = false;

JNIEXPORT void JNICALL Java_com_hexin_ai_jni_TalkingFace_sayHello(JNIEnv *env, jobject obj)
{
    DBG_LOGI("Helloworld.\n");
    return;
}

JNIEXPORT jboolean JNICALL Java_com_hexin_ai_jni_TalkingFace_init(JNIEnv *env, jobject obj, jint gpu_id, jint numWorkers, jint ffmpegThreads, jstring modelDir)
{
    if (intFlag == true)
    {
        DBG_LOGI("Already initialized\n");
        return JNI_TRUE;
    }

    //**************************************************************//
    DBG_LOGI("TalkingFace Model Init Begin\n");
    const char *model_dir = env->GetStringUTFChars(modelDir, 0);
    std::string model_dir_str = model_dir;
    Status status;
    if (!talkingface)
    {
        talkingface = new TalkingFace();
        status = talkingface->init(model_dir_str, numWorkers, ffmpegThreads);
    }
    else
    {
        delete talkingface;
        talkingface = 0;
        talkingface = new TalkingFace();
        status = talkingface->init(model_dir_str, numWorkers, ffmpegThreads);
    }
    DBG_LOGI("TalkingFace Model Init Done\n");
    //**************************************************************//

    env->ReleaseStringUTFChars(modelDir, model_dir);

    if (status.IsOk())
    {
        intFlag = true;
        return JNI_TRUE;
    }
    else
    {
        return JNI_FALSE;
    }
}

// JNIEXPORT jstring JNICALL Java_com_hexin_ai_jni_TalkingFace_process(JNIEnv *env, jobject obj,
//                                                                           jstring src_video_path,
//                                                                           jstring json_save_path,
//                                                                           jstring video_params)
// {
//     std::string msg;
//     Status status;

//     DBG_LOGE("process interface is deprecated.\n");
//     status = Status(
//         Status::Code::PROCESS_PREDICT_FAIL, "process interface is deprecated");

//     msg = status.AsString();
//     const char *msg_char = msg.c_str();
//     jstring jstrmsg = env->NewStringUTF(msg_char);
//     return jstrmsg;
// }

JNIEXPORT jstring JNICALL Java_com_hexin_ai_jni_TalkingFace_process(JNIEnv *env, jobject obj,
                                                                          jstring src_video_path,
                                                                          jstring json_save_path,
                                                                          jstring video_params)
{

    std::string msg;
    Status status;

    const char *srcVideoPath = env->GetStringUTFChars(src_video_path, 0);
    const char *jsonSavePath = env->GetStringUTFChars(json_save_path, 0);
    const char *videoParams = env->GetStringUTFChars(video_params, 0);

    if (talkingface)
    {
        double modelStart = (double)cv::getTickCount();
        try
        {
            status = talkingface->process(srcVideoPath, jsonSavePath, videoParams);
            DBG_LOGI("Get process done and has been feedback.\n");
        }
        catch(...)
        {
            DBG_LOGE("process predict fail.\n");
            status = Status(
                Status::Code::PROCESS_PREDICT_FAIL, "process predict fail");
        }

        // delete tmp dir
        std::string delete_command = "rm -rf " + (std::string)talkingface->tmp_dir;
        try
        {
            int dir_status_code = system(delete_command.c_str());
            if (dir_status_code != 0)
            {
                DBG_LOGE("delete tmp file fail.\n");
                status = Status(
                    Status::Code::FILE_TMP_DELETE_FAIL, "delete tmp file fail");
            }
        }
        catch(...)
        {
            DBG_LOGE("delete tmp file fail.\n");
            status = Status(
                Status::Code::FILE_TMP_DELETE_FAIL, "delete tmp file fail");
        }

        double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
        DBG_LOGI("process time: %f s\n", modelTime);
    }
    else
    {
        DBG_LOGE("extractor does not init when it predict.\n");
        status = Status(
            Status::Code::MODEL_INIT_FAIL,
            "extractor call predict but extractor is not loaded");
    }

    msg = status.AsString();
    const char *msg_char = msg.c_str();
    jstring jstrmsg = env->NewStringUTF(msg_char);

    env->ReleaseStringUTFChars(src_video_path, srcVideoPath);
    env->ReleaseStringUTFChars(json_save_path, jsonSavePath);
    env->ReleaseStringUTFChars(video_params, videoParams);

    return jstrmsg;
}

JNIEXPORT jstring JNICALL Java_com_hexin_ai_jni_TalkingFace_render(JNIEnv *env, jobject obj,
                                                                         jstring src_video_path,
                                                                         jstring audio_path,
                                                                         jstring json_save_path,
                                                                         jstring render_video_save_path,
                                                                         jstring video_params,
                                                                         jstring vocal_audio_path,
                                                                         jstring id_params)
{
    std::string msg;
    Status status;

    const char *srcVideoPath = env->GetStringUTFChars(src_video_path, 0);
    const char *audioPath = env->GetStringUTFChars(audio_path, 0);
    const char *jsonSavePath = env->GetStringUTFChars(json_save_path, 0);
    const char *renderVideoSavePath = env->GetStringUTFChars(render_video_save_path, 0);
    const char *videoParams = env->GetStringUTFChars(video_params, 0);
    const char *vocalAudioPath = env->GetStringUTFChars(vocal_audio_path, 0);
    const char *idParams = env->GetStringUTFChars(id_params, 0);

    if (talkingface)
    {
        double modelStart = (double)cv::getTickCount();
        try
        {
            status = talkingface->render(srcVideoPath, audioPath, jsonSavePath, renderVideoSavePath, videoParams, vocalAudioPath, idParams);
            DBG_LOGI("Get render done and has been feedback.\n");
        }
        catch(...)
        {
            DBG_LOGE("render predict fail.\n");
            status = Status(Status::Code::RENDER_PREDICT_FAIL, "render predict fail");
        }

        // delete tmp dir
        std::string delete_command = "rm -rf " + (std::string)talkingface->tmp_dir;
        try
        {
            int dir_status_code = system(delete_command.c_str());
            if (dir_status_code != 0)
            {
                DBG_LOGE("delete tmp file fail.\n");
                status = Status(Status::Code::FILE_TMP_DELETE_FAIL, "delete tmp file fail");
            }
        }
        catch(...)
        {
            DBG_LOGE("delete tmp file fail.\n");
            status = Status(Status::Code::FILE_TMP_DELETE_FAIL, "delete tmp file fail");
        }

        double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
        DBG_LOGI("render time: %f s\n", modelTime);
    }
    else
    {
        DBG_LOGE("model not init.\n");
        status = Status(Status::Code::MODEL_INIT_FAIL, "model not init");
    }

    msg = status.AsString();
    const char *msg_char = msg.c_str();
    jstring jstrmsg = env->NewStringUTF(msg_char);

    env->ReleaseStringUTFChars(src_video_path, srcVideoPath);
    env->ReleaseStringUTFChars(audio_path, audioPath);
    env->ReleaseStringUTFChars(json_save_path, jsonSavePath);
    env->ReleaseStringUTFChars(render_video_save_path, renderVideoSavePath);
    env->ReleaseStringUTFChars(video_params, videoParams);
    env->ReleaseStringUTFChars(vocal_audio_path, vocalAudioPath);
    env->ReleaseStringUTFChars(id_params, idParams);

    return jstrmsg;
}


JNIEXPORT jstring JNICALL Java_com_hexin_ai_jni_TalkingFace_shutup(JNIEnv *env, jobject obj,
                                                                         jstring image_path,
                                                                         jstring save_path,
                                                                         jstring video_params,
                                                                         jstring id_params)
{
    std::string msg;
    Status status;

    const char *imagePath = env->GetStringUTFChars(image_path, 0);
    const char *savePath = env->GetStringUTFChars(save_path, 0);
    const char *videoParams = env->GetStringUTFChars(video_params, 0);
    const char *idParams = env->GetStringUTFChars(id_params, 0);

    if (talkingface)
    {
        double modelStart = (double)cv::getTickCount();
        try
        {
            status = talkingface->shutup(imagePath, savePath, videoParams, idParams);
            DBG_LOGI("Get shutup done and has been feedback.\n");
        }
        catch(...)
        {
            DBG_LOGE("shutup predict fail.\n");
            status = Status(Status::Code::SHUT_UP_FAIL, "shutup predict fail");
        }

        double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
        DBG_LOGI("shutup time: %f s\n", modelTime);
    }
    else
    {
        DBG_LOGE("model not init.\n");
        status = Status(Status::Code::MODEL_INIT_FAIL, "model not init");
    }

    msg = status.AsString();
    const char *msg_char = msg.c_str();
    jstring jstrmsg = env->NewStringUTF(msg_char);

    env->ReleaseStringUTFChars(image_path, imagePath);
    env->ReleaseStringUTFChars(save_path, savePath);
    env->ReleaseStringUTFChars(video_params, videoParams);
    env->ReleaseStringUTFChars(id_params, idParams);

    return jstrmsg;
}

JNIEXPORT jboolean JNICALL Java_com_hexin_ai_jni_TalkingFace_stop(JNIEnv *env, jobject obj)
{
    //**************************************************************//

    if (talkingface)
    {
        delete talkingface;
        talkingface = 0;
        free(talkingface);
    }
    //**************************************************************//

    return JNI_TRUE;
}
