#include "talkingface.h"

TalkingFace::TalkingFace()
{
    is_running = false;
}

Status TalkingFace::init(const std::string model_dir, int n, int fn)
{
    Status status;

    DBG_LOGI("----------------------TalkingFace init start-----------------------------.\n");

    if (n < 1)
        n = 1;
    n_threads = n;

    if (fn < 1)
        fn = 0;
    ffmpeg_threads = fn;

    // 读取环境变量控制线程数
    const char* env_threads = std::getenv("LIPSYNC_NUM_THREADS");
    int max_threads = n_threads;
    if (env_threads != nullptr) {
        try {
            max_threads = std::stoi(env_threads);
            if (max_threads < 1) {
                max_threads = 1;
            }
            DBG_LOGI("Using LIPSYNC_NUM_THREADS from environment: %d\n", max_threads);
        } catch (...) {
            DBG_LOGE("Invalid LIPSYNC_NUM_THREADS value, using default: %d\n", n_threads);
            max_threads = n_threads;
        }
    }

    // 设置OpenMP最大线程数
    omp_set_num_threads(max_threads);
    DBG_LOGI("OpenMP threads set to: %d\n", max_threads);

    // 设置OpenCV线程数
    cv::setNumThreads(max_threads);
    DBG_LOGI("OpenCV threads set to: %d\n", max_threads);

    // 可选:通过环境变量覆盖FFmpeg线程数
    const char* env_ffmpeg_threads = std::getenv("LIPSYNC_FFMPEG_THREADS");
    if (env_ffmpeg_threads != nullptr) {
        try {
            ffmpeg_threads = std::stoi(env_ffmpeg_threads);
            DBG_LOGI("Using LIPSYNC_FFMPEG_THREADS from environment: %d\n", ffmpeg_threads);
        } catch (...) {
            DBG_LOGE("Invalid LIPSYNC_FFMPEG_THREADS value, using default: %d\n", fn);
        }
    }

    DBG_LOGI("init n_threads : %d\n", n_threads);
    DBG_LOGI("init ffmpeg_threads : %d\n", ffmpeg_threads);

    // 初始化FFmpeg CUDA配置
    // ffmpeg_config已在构造时自动初始化,这里可以记录配置状态
    if (ffmpeg_config.isCudaEnabled()) {
        DBG_LOGI("FFmpeg CUDA acceleration enabled (device: %d)\n", ffmpeg_config.cuda_device);
        DBG_LOGI("  NVENC available: %s\n", ffmpeg_config.nvenc_available ? "yes" : "no");
        DBG_LOGI("  NVDEC available: %s\n", ffmpeg_config.nvdec_available ? "yes" : "no");
    } else {
        DBG_LOGI("FFmpeg CUDA acceleration disabled (using CPU)\n");
    }

    // HuBERT
    std::string hubert_path_str = model_dir + "/hubert-large-ll60k.engine";
    DBG_LOGI ("HuBERT Model Loading from %s\n", hubert_path_str.c_str());
    if (access(hubert_path_str.c_str(), F_OK) == -1)
    {
        DBG_LOGE("%s not exists.\n", hubert_path_str.c_str());
        return Status(Status::Code::MODEL_INIT_FAIL, hubert_path_str + " not exists.");
    }
    audio_extractor = new HuBERT(hubert_path_str);
    audio_extractor->initialize_handler();
    audio_extractor->warmup();
    DBG_LOGI ("HuBERT Model is initialized.\n");

    for (int i = 0; i < n_threads; i++)
    {
        // face detect
        std::string scrfd_path_str = model_dir + "/scrfd_2.5g_shape640x640.engine";
        DBG_LOGI("SCRFD Model Loading from %s\n", scrfd_path_str.c_str());
        if (access(scrfd_path_str.c_str(), F_OK) == -1)
        {
            DBG_LOGE("%s not exists.\n", scrfd_path_str.c_str());
            return Status(Status::Code::MODEL_INIT_FAIL, scrfd_path_str + " not exists.");
        }
        SCRFD *face_detector = new SCRFD(scrfd_path_str);
        face_detector->initialize_handler();
        face_detector->warmup();
        m_face_detectors.emplace_back(face_detector);
        DBG_LOGI("SCRFD %d Model is initialized.\n", i);

        // face landmark detect
        std::string pipnet_path_str = model_dir + "/pipnet.engine";
        DBG_LOGI("PIPNet Loading from %s\n", pipnet_path_str.c_str());
        if (access(pipnet_path_str.c_str(), F_OK) == -1)
        {
            DBG_LOGE("%s not exists.\n", pipnet_path_str.c_str());
            return Status(Status::Code::MODEL_INIT_FAIL, pipnet_path_str + " not exists.");
        }
        PIPNet *face_landmarker = new PIPNet(pipnet_path_str);
        face_landmarker->initialize_handler();
        face_landmarker->warmup();
        m_face_landmarkers.emplace_back(face_landmarker);
        DBG_LOGI("PIPNet %d Model is initialized.\n", i);

        // wav2lip
        std::string wav2lip_path_str = model_dir + "/wav2lip.engine";
        DBG_LOGI ("Wav2Lip Model Loading from %s\n", wav2lip_path_str.c_str());
        if (access(wav2lip_path_str.c_str(), F_OK) == -1)
        {
            DBG_LOGE("%s not exists.\n", wav2lip_path_str.c_str());
            return Status(Status::Code::MODEL_INIT_FAIL, wav2lip_path_str + " not exists.");
        }
        Wav2Lip *generator = new Wav2Lip(wav2lip_path_str);
        generator->initialize_handler();
        generator->warmup();
        m_generators.emplace_back(generator);
        DBG_LOGI ("Wav2Lip %d Model is initialized.\n", i);

        // gcfsr
        std::string gcfsr_path_str = model_dir + "/gcfsr_blind_512.engine";
        DBG_LOGI ("GCFSR Model Loading from %s\n", gcfsr_path_str.c_str());
        if (access(gcfsr_path_str.c_str(), F_OK) == -1)
        {
            DBG_LOGE("%s dose not exists.\n", gcfsr_path_str.c_str());
            return Status(Status::Code::MODEL_INIT_FAIL, gcfsr_path_str + " not exists.");
        }
        GCFSR *face_enhancer = new GCFSR(gcfsr_path_str);
        face_enhancer->initialize_handler();
        face_enhancer->warmup();
        m_enhancers.emplace_back(face_enhancer);
        DBG_LOGI ("GCFSR %d Model is initialized.\n", i);
    }

    is_running = true;
    DBG_LOGI("----------------------TalkingFace init end-----------------------------.\n");
    return Status::Success;
}

Status TalkingFace::stop()
{
    DBG_LOGI("----------------------TalkingFace stop start-----------------------------.\n");
    if (audio_extractor)
        delete audio_extractor;
    
    audio_extractor = nullptr;
    for (int i = 0; i < n_threads; i++)
    {
        m_face_detectors[i] = nullptr;
        m_face_landmarkers[i] = nullptr;
        m_generators[i] = nullptr;
        m_enhancers[i] = nullptr;
    }
    DBG_LOGI("----------------------TalkingFace stop end-----------------------------.\n");
    return Status::Success;
}
