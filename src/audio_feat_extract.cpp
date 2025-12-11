#include "talkingface.h"


// 提取渲染音频的hubert特征
Status TalkingFace::extractAudioFeat(const char *audio_path)
{
    Status status;

    try
    {
        // 统一转换为16khz、单通道音频
        if (access(audio_path, F_OK) == -1)
        {
            DBG_LOGE("extract audio feat fail, audio file not exists.\n");
            return Status(Status::Code::FFMPEG_AUDIO_CONVERT_FAIL, "extract audio feat fail, audio file not exists.");
        }

        std::string audio_command = "ffmpeg -loglevel quiet -y -i " + (std::string)audio_path;

        if (video_params.audio_max_time > 0)
            audio_command += (" -t " + (std::to_string)(video_params.audio_max_time));

        audio_command += (" -f wav -ac 1 -ar 16000");

        if (ffmpeg_threads > 0)
            audio_command += (" -threads " + (std::to_string)(ffmpeg_threads));
        audio_command += (" " + (std::string)tmp_audio_path);

        DBG_LOGI("ffmpeg audio command:  %s\n", audio_command.c_str());
        try
        {
            int audio_status_code = system(audio_command.c_str());
            if (audio_status_code != 0)
            {
                DBG_LOGE("extract audio feat fail, audio ffmpeg convert fail.\n");
                return Status(Status::Code::FFMPEG_AUDIO_CONVERT_FAIL, "extract audio feat fail, audio ffmpeg convert fail.");
            }
        }
        catch(...)
        {
            DBG_LOGE("extract audio feat fail, audio ffmpeg convert fail.\n");
            return Status(Status::Code::FFMPEG_AUDIO_CONVERT_FAIL, "extract audio feat fail, audio ffmpeg convert fail.");
        }

        // 判断转换后的音频是否存在，避免后面WavHelper直接崩溃
        std::ifstream audioFile(tmp_audio_path);
        if (!audioFile.is_open())
        {
            DBG_LOGE("extract audio feat fail, convert audio check fail.\n");
            return Status(Status::Code::FFMPEG_AUDIO_CONVERT_FAIL, "extract audio feat fail, convert audio check fail.");
        }

        // 读取音频数据
        WavData data;
        try
        {
            WavHelper::readWav(data, tmp_audio_path);
        }
        catch(...)
        {
            DBG_LOGE("audio data read fail.\n");
            return Status(Status::Code::AUDIO_READ_FAIL, "audio data read fail.");
        }
        if (data.wavLength() <= (17 + 640))
        {
            DBG_LOGE("audio data read empty.\n");
            return Status(Status::Code::AUDIO_READ_FAIL, "audio data read empty.");
        }

        // 提取hubert特征
        std::vector<std::vector<float>> audio_feats;
        int audio_cnt = 0;
        float audio_interval = 0;
        try
        {
            const short *audioData = data.data() + 17;
            long wav_length = data.wavLength() - 17;
            long sample_rate = data.sampleRate();
            audio_extractor->predict(audioData, wav_length, sample_rate, infos.fps,
                audio_feats, audio_cnt, audio_interval);
        }
        catch(...)
        {
            DBG_LOGE("audio feat extract fail.\n");
            return Status(Status::Code::AUDIO_FEAT_EXTRACT_FAIL, "audio feat extract fail.");
        }
        if (audio_feats.size() <= 0)
        {
            DBG_LOGE("audio feat extract empty.\n");
            return Status(Status::Code::AUDIO_FEAT_EXTRACT_FAIL, "audio feat extract empty.");
        }
        infos.audio_feats.emplace_back(audio_feats);
        infos.audio_cnts.emplace_back(audio_cnt);
        infos.audio_intervals.emplace_back(audio_interval);
    }

    catch(...)
    {
        DBG_LOGE("audio feat extract fail.\n");
        return Status(Status::Code::AUDIO_FEAT_EXTRACT_FAIL, "audio feat extract fail.");
    }

    return status;
}
