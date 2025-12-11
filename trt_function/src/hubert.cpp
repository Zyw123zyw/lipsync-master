#include "hubert.h"

using Function::HuBERT;


HuBERT::HuBERT(const std::string &_engine_path) : BasicTRTHandler(_engine_path) {}

void HuBERT::preprocess(const short *constData, long wav_length, long sample_rate,
                        std::vector<std::vector<float>>& input_wav_chunk,
                        std::vector<std::vector<int>>& input_valid_index)
{
    double mean = 0.0;
    double variance = 0.0;
    long overlap = sample_rate * overlap_duration;
    long chunk_size = sample_rate * chunk_duration;

    std::vector<float> input_wav(wav_length + overlap * 2, 0);

    for (size_t i = 0; i < wav_length; i++){
        input_wav[i + overlap] = constData[i] / (3.2768e4);
        mean += input_wav[i + overlap];
    }

    mean /= input_wav.size();
    for (size_t i = 0; i < input_wav.size(); i++){
        variance += (input_wav[i] - mean) * (input_wav[i] - mean);
    }

    variance /= input_wav.size();
    double standard_deviation = std::sqrt(variance + 1e-7);
    for (size_t i = 0; i < input_wav.size(); i++){
        input_wav[i] = (input_wav[i] - mean) / standard_deviation;
    }

    // chunk
    long quotient = wav_length / chunk_size;    // 商
    long remainder = wav_length % chunk_size;   // 余
    if (quotient > 0)
    {
        for (int i = 0; i < quotient; i++)
        {
            int start = i * chunk_size;
            int end = (i + 1) * chunk_size + 2 * overlap;
            std::vector<float> chunk(input_wav.begin() + start, input_wav.begin() + end);
            std::vector<int> index = {overlap_duration*fps, static_cast<int>((chunk.size() / 16000.0 - overlap_duration)*fps)};
            input_wav_chunk.emplace_back(chunk);
            input_valid_index.emplace_back(index);
        }
        if (remainder > 0)
        {
            int start = quotient * chunk_size;
            int end = (quotient + 1) * chunk_size + 2 * overlap;
            std::vector<float> chunk(input_wav.begin() + start, input_wav.end());
            std::vector<int> index = {overlap_duration*fps, static_cast<int>((chunk.size() / 16000.0 - overlap_duration)*fps)};
            // pad
            float pad_value = input_wav.back();
            int pad_num = chunk_size + 2 * overlap - chunk.size();
            chunk.insert(chunk.end(), pad_num, pad_value);
            input_wav_chunk.emplace_back(chunk);
            input_valid_index.emplace_back(index);
        }
    }
    else{
        std::vector<float> chunk(input_wav.begin(), input_wav.end());
        std::vector<int> index = {overlap_duration*fps, static_cast<int>((chunk.size() / 16000.0 - overlap_duration)*fps)};
        // pad
        float pad_value = input_wav.back();
        int pad_num = chunk_size + 2 * overlap - chunk.size();
        chunk.insert(chunk.end(), pad_num, pad_value);
        input_wav_chunk.emplace_back(chunk);
        input_valid_index.emplace_back(index);
    }
}

void HuBERT::warmup()
{
    DBG_LOGI("HuBERT warm up start\n");

    int wav_length = (chunk_duration + overlap_duration * 2) * 16000;
    std::vector<float> input_wav(wav_length, 0);

    CHECK(cudaMemcpy(buffers[0], input_wav.data(), buffer_size[0], cudaMemcpyHostToDevice));

    context->executeV2(buffers);

    float *output = new float[buffer_size[1] / sizeof(float)];

    CHECK(cudaMemcpy(output, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost));

    int feat_length = static_cast<int>(static_cast<float>(input_wav.size()) / 16000 * fps) - 1;   // 少一帧
    std::vector<std::vector<float>> feats_(feat_length, std::vector<float>(feat_channel, 0));

    for (int l = 0; l < feat_length; ++l)
    {
        for (int c = 0; c < feat_channel; ++c)
        {
            feats_[l][c] = output[l*feat_channel + c];
        }
    }

    silence_feat.resize(feat_channel * feat_duration, 0);
    for (int j = 0; j < feat_channel; ++j)
    {
        for (int k = 0; k < feat_duration; ++k)
        {
            silence_feat[j * feat_duration + k] = feats_[feat_length/2][j];
        }
    }

    delete output;

    DBG_LOGI("HuBERT warm up done\n");
}

void HuBERT::predict(const short *constData, long wav_length, long sample_rate, const int render_fps,
                    std::vector<std::vector<float>>& audio_feats, int& audio_cnt, float& audio_interval)
{
    DBG_LOGI("HuBERT predict start\n");
    std::vector<std::vector<float>> input_wav_chunk;
    std::vector<std::vector<int>> input_valid_index;
    this->preprocess(constData, wav_length, sample_rate, input_wav_chunk, input_valid_index);

    std::vector<std::vector<float>> hubert_feats;
    for ( int i = 0; i < input_wav_chunk.size(); ++i)
    {
        // 创建输入
        std::vector<float> input_wav = input_wav_chunk[i];

        // 将输入传递到GPU
        CHECK(cudaMemcpy(buffers[0], input_wav.data(), buffer_size[0], cudaMemcpyHostToDevice));

        // 异步执行
        context->executeV2(buffers);

        // 将输出传递到CPU
        float *output = new float[buffer_size[1] / sizeof(float)];
        CHECK(cudaMemcpy(output, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost));

        int feat_length = static_cast<int>(static_cast<float>(input_wav.size()) / sample_rate * fps) - 1;   // 少一帧
        std::vector<std::vector<float>> feats_(feat_length, std::vector<float>(feat_channel, 0));

        for (int l = 0; l < feat_length; ++l)
        {
            for (int c = 0; c < feat_channel; ++c)
            {
                feats_[l][c] = output[l*feat_channel + c];
            }
        }

        int start_valid_index = input_valid_index[i][0];
        int end_valid_index = input_valid_index[i][1];
        hubert_feats.insert(hubert_feats.end(), feats_.begin() + start_valid_index, feats_.begin() + end_valid_index);

        delete output;
    }
    DBG_LOGI("HuBERT predict done\n");

    // // postprocess: pad, duration_chunk and transpose
    // std::vector<std::vector<float>> hubert_feats_pad;
    // for (int i = 0; i < 10; ++i)
    //     hubert_feats_pad.emplace_back(hubert_feats.front());
    // hubert_feats_pad.insert(hubert_feats_pad.end(), hubert_feats.begin(), hubert_feats.end());
    // for (int i = 0; i < 20; ++i)
    //     hubert_feats_pad.emplace_back(hubert_feats.back());

    // int cnts = static_cast<int>(wav_length / 16000.0 * render_fps);
    // float interval = hubert_feats.size() / static_cast<float>(cnts);
    // feats.resize(cnts, std::vector<float>(feat_channel*feat_duration, 0));

    // int ii;
    // for (int i = 0; i < cnts; ++i)
    // {
    //     for (int j = 0; j < feat_channel; ++j)
    //     {
    //         for (int k = 0; k < feat_duration; ++k)
    //         {
    //             ii = static_cast<int>(i * interval);
    //             feats[i][j * feat_duration + k] = hubert_feats_pad[ii + k][j];
    //         }
    //     }
    // }

    // pad
    for (int i = 0; i < 10; ++i)
        audio_feats.emplace_back(hubert_feats.front());
    audio_feats.insert(audio_feats.end(), hubert_feats.begin(), hubert_feats.end());
    for (int i = 0; i < 20; ++i)
        audio_feats.emplace_back(hubert_feats.back());

    audio_cnt = static_cast<int>(wav_length / 16000.0 * render_fps);
    audio_interval = hubert_feats.size() / static_cast<float>(audio_cnt);
}

std::vector<float> HuBERT::get_audio_feat(std::vector<std::vector<float>>& audio_feats, int idx, float interval)
{
    std::vector<float> feat(feat_channel*feat_duration);

    int i = static_cast<int>(idx * interval);
    for (int j = 0; j < feat_channel; ++j)
    {
        for (int k = 0; k < feat_duration; ++k)
        {
            feat[j * feat_duration + k] = audio_feats[i + k][j];
        }
    }
    return feat;
}
