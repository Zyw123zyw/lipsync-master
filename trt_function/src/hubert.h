#pragma once
#ifndef HUBERT_H
#define HUBERT_H

#include "../core/trt_handler.h"
#include "../core/trt_utils.h"

namespace Function
{
    class HuBERT : public BasicTRTHandler // extract hubert feature from audio data
    {
        private:
            const int feat_channel = 1024;
            const int feat_duration = 20;
            const int fps = 50;

            const int chunk_duration = 13; // 15s
            const int overlap_duration = 1; // 1s

        private:
            /* function */
            void preprocess(const short *constData, long wav_length, long sample_rate,
                            std::vector<std::vector<float>>& input_wav_chunk,
                            std::vector<std::vector<int>>& input_valid_index);

        public:
            explicit HuBERT(const std::string &_engine_path);
            ~HuBERT() override = default;

            void predict(const short *constData, long wav_length, long sample_rate, const int render_fps,
                    std::vector<std::vector<float>>& audio_feats, int& audio_cnt, float& audio_interval);

            void warmup();

            std::vector<float> get_audio_feat(std::vector<std::vector<float>>& audio_feats, int idx, float interval);

        public:
            std::vector<float> silence_feat;
    };
}

#endif
