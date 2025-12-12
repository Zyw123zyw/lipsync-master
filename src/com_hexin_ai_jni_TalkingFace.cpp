#include "com_hexin_ai_jni_TalkingFace.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "util_debug.h"
#include "talkingface.h"
#include "error_code.h"

// 多实例模式：通过handle管理实例

JNIEXPORT void JNICALL Java_com_hexin_ai_jni_TalkingFace_sayHello(JNIEnv *env, jobject obj)
{
    DBG_LOGI("Helloworld.\n");
    return;
}

// 创建实例
JNIEXPORT jlong JNICALL Java_com_hexin_ai_jni_TalkingFace_nativeCreate(JNIEnv *env, jobject obj)
{
    TalkingFace* instance = new TalkingFace();
    DBG_LOGI("TalkingFace instance created: %p (id=%d)\n", instance, instance->getInstanceId());
    return reinterpret_cast<jlong>(instance);
}

// 销毁实例
JNIEXPORT void JNICALL Java_com_hexin_ai_jni_TalkingFace_nativeDestroy(JNIEnv *env, jobject obj, jlong handle)
{
    TalkingFace* instance = reinterpret_cast<TalkingFace*>(handle);
    if (instance) {
        DBG_LOGI("TalkingFace instance destroying: %p (id=%d)\n", instance, instance->getInstanceId());
        instance->stop();
        delete instance;
    }
}

JNIEXPORT jboolean JNICALL Java_com_hexin_ai_jni_TalkingFace_init(JNIEnv *env, jobject obj, jlong handle, jint gpu_id, jint numWorkers, jint ffmpegThreads, jstring modelDir)
{
    TalkingFace* instance = reinterpret_cast<TalkingFace*>(handle);
    if (!instance) {
        DBG_LOGE("Invalid handle\n");
        return JNI_FALSE;
    }

    DBG_LOGI("TalkingFace instance %d init begin\n", instance->getInstanceId());
    const char *model_dir = env->GetStringUTFChars(modelDir, 0);
    std::string model_dir_str = model_dir;
    
    Status status = instance->init(model_dir_str, numWorkers, ffmpegThreads);
    
    DBG_LOGI("TalkingFace instance %d init done\n", instance->getInstanceId());
    env->ReleaseStringUTFChars(modelDir, model_dir);

    return status.IsOk() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL Java_com_hexin_ai_jni_TalkingFace_process(JNIEnv *env, jobject obj, jlong handle,
                                                                          jstring src_video_path,
                                                                          jstring json_save_path,
                                                                          jstring video_params)
{
    std::string msg;
    Status status;

    TalkingFace* instance = reinterpret_cast<TalkingFace*>(handle);
    if (!instance) {
        status = Status(Status::Code::MODEL_INIT_FAIL, "Invalid handle");
        msg = status.AsString();
        return env->NewStringUTF(msg.c_str());
    }

    const char *srcVideoPath = env->GetStringUTFChars(src_video_path, 0);
    const char *jsonSavePath = env->GetStringUTFChars(json_save_path, 0);
    const char *videoParams = env->GetStringUTFChars(video_params, 0);

    double modelStart = (double)cv::getTickCount();
    try
    {
        status = instance->process(srcVideoPath, jsonSavePath, videoParams);
        DBG_LOGI("Instance %d process done.\n", instance->getInstanceId());
    }
    catch(...)
    {
        DBG_LOGE("Instance %d process fail.\n", instance->getInstanceId());
        status = Status(Status::Code::PROCESS_PREDICT_FAIL, "process predict fail");
    }

    // delete tmp dir
    std::string delete_command = "rm -rf " + std::string(instance->tmp_dir());
    try
    {
        system(delete_command.c_str());
    }
    catch(...) {}

    double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
    DBG_LOGI("Instance %d process time: %f s\n", instance->getInstanceId(), modelTime);

    msg = status.AsString();
    env->ReleaseStringUTFChars(src_video_path, srcVideoPath);
    env->ReleaseStringUTFChars(json_save_path, jsonSavePath);
    env->ReleaseStringUTFChars(video_params, videoParams);

    return env->NewStringUTF(msg.c_str());
}

JNIEXPORT jstring JNICALL Java_com_hexin_ai_jni_TalkingFace_render(JNIEnv *env, jobject obj, jlong handle,
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

    TalkingFace* instance = reinterpret_cast<TalkingFace*>(handle);
    if (!instance) {
        status = Status(Status::Code::MODEL_INIT_FAIL, "Invalid handle");
        msg = status.AsString();
        return env->NewStringUTF(msg.c_str());
    }

    const char *srcVideoPath = env->GetStringUTFChars(src_video_path, 0);
    const char *audioPath = env->GetStringUTFChars(audio_path, 0);
    const char *jsonSavePath = env->GetStringUTFChars(json_save_path, 0);
    const char *renderVideoSavePath = env->GetStringUTFChars(render_video_save_path, 0);
    const char *videoParams = env->GetStringUTFChars(video_params, 0);
    const char *vocalAudioPath = env->GetStringUTFChars(vocal_audio_path, 0);
    const char *idParams = env->GetStringUTFChars(id_params, 0);

    double modelStart = (double)cv::getTickCount();
    try
    {
        status = instance->render(srcVideoPath, audioPath, jsonSavePath, renderVideoSavePath, videoParams, vocalAudioPath, idParams);
        DBG_LOGI("Instance %d render done.\n", instance->getInstanceId());
    }
    catch(...)
    {
        DBG_LOGE("Instance %d render fail.\n", instance->getInstanceId());
        status = Status(Status::Code::RENDER_PREDICT_FAIL, "render predict fail");
    }

    // delete tmp dir
    std::string delete_command = "rm -rf " + std::string(instance->tmp_dir());
    try
    {
        system(delete_command.c_str());
    }
    catch(...) {}

    double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
    DBG_LOGI("Instance %d render time: %f s\n", instance->getInstanceId(), modelTime);

    msg = status.AsString();
    env->ReleaseStringUTFChars(src_video_path, srcVideoPath);
    env->ReleaseStringUTFChars(audio_path, audioPath);
    env->ReleaseStringUTFChars(json_save_path, jsonSavePath);
    env->ReleaseStringUTFChars(render_video_save_path, renderVideoSavePath);
    env->ReleaseStringUTFChars(video_params, videoParams);
    env->ReleaseStringUTFChars(vocal_audio_path, vocalAudioPath);
    env->ReleaseStringUTFChars(id_params, idParams);

    return env->NewStringUTF(msg.c_str());
}

JNIEXPORT jstring JNICALL Java_com_hexin_ai_jni_TalkingFace_shutup(JNIEnv *env, jobject obj, jlong handle,
                                                                         jstring image_path,
                                                                         jstring save_path,
                                                                         jstring video_params,
                                                                         jstring id_params)
{
    std::string msg;
    Status status;

    TalkingFace* instance = reinterpret_cast<TalkingFace*>(handle);
    if (!instance) {
        status = Status(Status::Code::MODEL_INIT_FAIL, "Invalid handle");
        msg = status.AsString();
        return env->NewStringUTF(msg.c_str());
    }

    const char *imagePath = env->GetStringUTFChars(image_path, 0);
    const char *savePath = env->GetStringUTFChars(save_path, 0);
    const char *videoParams = env->GetStringUTFChars(video_params, 0);
    const char *idParams = env->GetStringUTFChars(id_params, 0);

    double modelStart = (double)cv::getTickCount();
    try
    {
        status = instance->shutup(imagePath, savePath, videoParams, idParams);
        DBG_LOGI("Instance %d shutup done.\n", instance->getInstanceId());
    }
    catch(...)
    {
        DBG_LOGE("Instance %d shutup fail.\n", instance->getInstanceId());
        status = Status(Status::Code::SHUT_UP_FAIL, "shutup predict fail");
    }

    double modelTime = ((double)cv::getTickCount() - modelStart) / cv::getTickFrequency();
    DBG_LOGI("Instance %d shutup time: %f s\n", instance->getInstanceId(), modelTime);

    msg = status.AsString();
    env->ReleaseStringUTFChars(image_path, imagePath);
    env->ReleaseStringUTFChars(save_path, savePath);
    env->ReleaseStringUTFChars(video_params, videoParams);
    env->ReleaseStringUTFChars(id_params, idParams);

    return env->NewStringUTF(msg.c_str());
}

JNIEXPORT jboolean JNICALL Java_com_hexin_ai_jni_TalkingFace_stop(JNIEnv *env, jobject obj, jlong handle)
{
    TalkingFace* instance = reinterpret_cast<TalkingFace*>(handle);
    if (instance) {
        instance->stop();
    }
    return JNI_TRUE;
}

// 性能测试接口（静态方法，不需要handle）
JNIEXPORT void JNICALL Java_com_hexin_ai_jni_TalkingFace_startPerfTest(JNIEnv *env, jclass cls, jint durationMinutes)
{
    TalkingFace::startPerfTest(durationMinutes);
}

JNIEXPORT jlong JNICALL Java_com_hexin_ai_jni_TalkingFace_getPerfFrameCount(JNIEnv *env, jclass cls)
{
    return TalkingFace::getPerfFrameCount();
}

JNIEXPORT void JNICALL Java_com_hexin_ai_jni_TalkingFace_resetPerfCounter(JNIEnv *env, jclass cls)
{
    TalkingFace::resetPerfCounter();
}
