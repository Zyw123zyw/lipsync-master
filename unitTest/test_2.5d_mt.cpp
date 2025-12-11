#include "../src/speech2face.h"
#include <math.h>
#include "util_debug.h"
#include "../src/util_live.h"
#include "omp.h"
#include "../src/talkingface.h"


class MultiThead
{
private:
    Function::HuBERT* m_hubert_extractor = nullptr;
    std::vector<Function::Wav2Lip*> m_wav2lip_renderers;

    std::mutex m_mtx_frame;     // 互斥锁：用于保护共享资源的同步原语。它确保在同一时刻，只有一个线程能够访问共享资源，从而避免数据竞争
    std::mutex m_mtx_result;
    std::condition_variable m_cond_consume;     // 条件变量：是一个同步原语，用于阻塞一个或多个线程，直到某个条件为真时通知线程继续执行
    std::condition_variable m_cond_read;
    std::condition_variable m_cond_save_frame;
    std::condition_variable m_cond_write_frame;

    // std::thread* m_detect_thread;  // 检测线程
    // std::thread* m_extract_thread; // 音频处理线程

    PredictInfos m_predictInfos;  // 渲染生成需要的信息

public:
    MultiThead();
    ~MultiThead();

    bool init(std::string model_dir, int gpu_id, int m_render_worker_nb);
    bool stop();

    int audioExtract();
    int read_img(int totframes, std::string data_path);
    int render(const std::vector<std::vector<float>>& audio_feats, int work_idx);
    
    // 生成结果写入视频函数
    int open_intermediate_save_path(std::string save_path);
    int write_intermediate_video_thread();

    int predict(int frames_number, std::string data_path, int m_render_worker_nb, std::string save_path);
};

MultiThead::MultiThead(){}

MultiThead::~MultiThead()=default;

bool MultiThead::init(std::string model_dir, int gpu_id, int m_render_worker_nb)
{
    std::string hubert_path = model_dir + "/hubert-large-ll60k.engine";
    std::string wav2lip_path = model_dir + "/wav2lip.engine";

    if(!(access(hubert_path.c_str(), 0) == 0)){
        DBG_LOGE ("HuBERT model path is failed load from %s \n", hubert_path.c_str());
        return false;
    }

    if(!(access(wav2lip_path.c_str(), 0) == 0)){
        DBG_LOGE ("Wav2Lip model path is failed load from %s \n", wav2lip_path.c_str());
        return false;
    }

    DBG_LOGI("START init! \n");
    try {
        m_hubert_extractor = new Function::HuBERT(hubert_path);
        DBG_LOGI("START initialize_handler! \n");
        m_hubert_extractor->initialize_handler();
        DBG_LOGI("START warmup! \n");
        m_hubert_extractor->warmup();
    } catch (...)
    {
        DBG_LOGE("HuBERT init fail! \n");
        return false;
    }

    DBG_LOGI("HuBERT Model is initialized.\n");

    DBG_LOGI("Wav2Lip Model Loading from %s\n", wav2lip_path.c_str());
    for(int i = 0; i < m_render_worker_nb; i++){
        try {
            Function::Wav2Lip *w2l_renderer = new Function::Wav2Lip(wav2lip_path);
            w2l_renderer->initialize_handler();
            w2l_renderer->warmup();
            m_wav2lip_renderers.push_back(w2l_renderer);
        }
        catch (...)
        {
            DBG_LOGE("Wav2Lip init fail! \n");
            return false;
        }
    }
    DBG_LOGI("Wav2Lip Model is initialized.\n");
    return true;
}

bool MultiThead::stop()
{
    if (m_hubert_extractor) {
        delete m_hubert_extractor;
    }

    m_hubert_extractor = nullptr;
    for (int i = 0; i < m_wav2lip_renderers.size(); i++) {
        m_wav2lip_renderers[i] = nullptr;
    }
    return true;
}

int MultiThead::audioExtract()
{
    DBG_LOGI("START audioExtract! \n");
    while (1)
    {
        int pad_len = 16000 * 5;
        std::vector<float> wav2vec_buffer(pad_len, 0);
        std::vector<std::vector<float>> hb_feats;
        auto t1 = std::chrono::system_clock::now();
        m_hubert_extractor->predict_live(wav2vec_buffer, wav2vec_buffer.size(), hb_feats, 0);
        auto t2 = std::chrono::system_clock::now();
        printf("audioExtract time %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
        usleep(1000 * 150);
    }

    return 0;
}

int MultiThead::read_img(int totframes, std::string data_path)
{
    int startFrame = 0;  // 底板素材起始帧
    int endFrame = 500;  // 底板素材结束帧
    int cnt = 0; // 读取视频帧计数
    int frameid = 0; // 底板帧id
    int reverseSeq;

    m_predictInfos.totframes = totframes; // 需要渲染的素材总帧数
    m_predictInfos.finish_read = 0;
    
    while(1)
    {
        RenderFrameInfos frame_info;
        char fn[512];
        frame_info.cnt = cnt;
        frame_info.frame_id = frameid;
        sprintf(fn, "%s/%06d_src.png", data_path.c_str(), frameid);
        frame_info.img_path = fn;
        sprintf(fn, "%s/%06d_ref.png", data_path.c_str(), frameid);
        frame_info.mask_path = fn;

        // 读取输入人脸图
        frame_info.img = cv::imread(frame_info.img_path.c_str(), 1);
        frame_info.mask = cv::imread(frame_info.mask_path.c_str(), 1);
        
        if (frameid >= endFrame - 1){
                reverseSeq = -1;
        }
        if (frameid <= startFrame) {
            reverseSeq = 1;
        }

        std::unique_lock<std::mutex> locker(m_mtx_frame);
        while(!m_predictInfos.all_render_frame_infos.empty()){  // 读取图片消费完线程等待
            m_cond_read.wait(locker);
        }

        m_predictInfos.all_render_frame_infos.push_back(frame_info);
        locker.unlock();

        m_cond_consume.notify_all();  // 通知所有消费线程

        cnt ++;
        frameid += reverseSeq;

        DBG_LOGI("cnt: %d \n", cnt);

        if(cnt >= totframes){
            break;
        }
    }

    // 通知所有等待的线程
    // 拿一下锁再通知，防止卡住的问题
    // 先拿锁
    std::unique_lock<std::mutex> locker_frame(m_mtx_frame);
    m_predictInfos.finish_read = 1;
    locker_frame.unlock();
    // 通知所有等待的线程
    m_cond_consume.notify_all();

    DBG_LOGI("finsih read img thread \n");
    
    return 0;
}

int MultiThead::render(const std::vector<std::vector<float>>& audio_feats, int work_idx)
{
    DBG_LOGI("worker idx: %d\n", work_idx);

    int totframes = m_predictInfos.totframes;
    RenderFrameInfos frame_info;

    bool inferenced = false;
    cv::Mat w2l_out;
    int cnt;

    while (1)
    {
        inferenced = false;
        std::unique_lock<std::mutex> locker(m_mtx_frame);
        while(m_predictInfos.all_render_frame_infos.empty() && (m_predictInfos.finish_read == 0)){
            m_cond_consume.wait(locker); 
        }

        if(!m_predictInfos.all_render_frame_infos.empty()){
            assert(m_predictInfos.all_render_frame_infos.size() == 1);
            inferenced = true;
            frame_info = m_predictInfos.all_render_frame_infos[0];
            m_predictInfos.all_render_frame_infos.pop_back();
        }
        locker.unlock();
        m_cond_read.notify_one();
        
        if (inferenced) {
            cv::Mat src = frame_info.img;
            cv::Mat ref = frame_info.mask;
            
            cnt = frame_info.cnt;
            DBG_LOGI("render cnt: %d \n", cnt);
            std::vector<float> audio = audio_feats[cnt];
            
            auto t1 = std::chrono::system_clock::now();
            m_wav2lip_renderers[work_idx]->predict(src, ref, audio.data(), w2l_out);
            auto t2 = std::chrono::system_clock::now();
            printf("render time %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

            // 限制m_predictInfos.rendered_frames的数量，防止占用内存太大
            std::unique_lock<std::mutex> locker_res(m_mtx_result);
            while(m_predictInfos.last_outputed_cnt != (cnt - 1) || m_predictInfos.rendered_frames.size() > 25){
                m_cond_save_frame.wait(locker_res); 
            }

            m_predictInfos.last_outputed_cnt = cnt;
            // cv::imwrite("/data/text2face/text2face-sdk/output/nerfblendshape/1017/" + to_string(cnt) + ".png", merged_frame);
            m_predictInfos.rendered_frames.push(w2l_out.clone());

            locker_res.unlock();
            m_cond_save_frame.notify_all();

            m_cond_write_frame.notify_one();

        } else {
            break;
        }
        
    }
    return 0;
}

int MultiThead::open_intermediate_save_path(std::string save_path)
{
    const int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    const double fps = 25.0;
    cv::Size2d video_size = cv::Size2d(256, 256);
    m_predictInfos.videoOut.open(save_path, codec, fps, video_size);

    usleep(100);  // 等待100微秒
    if(! m_predictInfos.videoOut.isOpened())
    {
        return 1;
    }

    return 0;
}

int MultiThead::write_intermediate_video_thread()
{
    DBG_LOGI("write intermediate video thread started\n");

    cv::Mat frame;
    bool write = false;
    int write_num = 0;
    while(1){
        write = false;
        std::unique_lock<std::mutex> locker(m_mtx_result);
        while(m_predictInfos.rendered_frames.empty() && m_predictInfos.finish_render == 0){
            m_cond_write_frame.wait(locker);
        }

        if(m_predictInfos.rendered_frames.size() > 0){
            frame = m_predictInfos.rendered_frames.front();
            m_predictInfos.rendered_frames.pop();
            write = true;
        }

        locker.unlock();

        if(write){
            m_cond_save_frame.notify_all();
            m_predictInfos.videoOut.write(frame);
            write_num ++;
            DBG_LOGI("write number: %d\n", write_num);
        }
        else{
            break;
        }
    }

    DBG_LOGI("write intermediate video thread finished\n");

    return 0;
}

int MultiThead::predict(int frames_number, std::string data_path, int m_render_worker_nb, std::string save_path)
{
    std::vector<std::vector<float>> audio_feats;
    std::vector<float> hbFeat(5120, 0);

    m_predictInfos.init();

    for (int i = 0; i < frames_number; i++) {
        audio_feats.push_back(hbFeat);
    }
    
    thread read_thread(&MultiThead::read_img, this, frames_number, data_path);
    
    // 等到读取完第一帧
    std::unique_lock<std::mutex> locker(m_mtx_frame);
    while(m_predictInfos.all_render_frame_infos.empty())
    {
        DBG_LOGI("cond consume wait \n");
        m_cond_consume.wait(locker);  // Unlock mu and wait to be notified
    }
    locker.unlock();
    DBG_LOGI("finish cond consume wait \n");

    MultiThead::open_intermediate_save_path(save_path);

    vector<thread> render_threads;
    int startFrame = 0;
    m_predictInfos.last_outputed_cnt = startFrame - 1;  // 预先设置初始值，方便后续线程判断

    for(int i = 0; i < m_render_worker_nb; i++){
        render_threads.push_back(thread(&MultiThead::render, this, audio_feats, i));
    }

    // 开启opencv写入中间结果视频的线程
    thread write_thread(&MultiThead::write_intermediate_video_thread, this);

    read_thread.join();
    printf("read finish \n");
    for(int i = 0; i < render_threads.size(); i++){
        render_threads[i].join();
    }

    DBG_LOGI("all render thread finished \n");
    std::unique_lock<std::mutex> locker1(m_mtx_result);
    m_predictInfos.finish_render = 1;
    locker1.unlock();
    m_cond_write_frame.notify_one();

    write_thread.join();

    if (m_predictInfos.videoOut.isOpened()){
        m_predictInfos.videoOut.release();
    }
    return 0;
}

int main(int argc, char* argv[]) {
    std::string model_dir = "/workspace/project/audio-talkingface/models/7.5";

    std::string data_path = "/workspace/project/audio-talkingface/test_asserts/images";
    
    int frames_number = 125;
    int m_render_worker_nb = 3;

    std::string save_path = "/workspace/project/audio-talkingface/test_asserts/out.mp4";

    MultiThead *Mt = new MultiThead();
    Mt->init(model_dir, 0, m_render_worker_nb);
    DBG_LOGI("START predict \n");

    Mt->predict(frames_number, data_path, m_render_worker_nb, save_path);

    Mt->stop();
    return 0;
}