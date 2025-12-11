#ifndef ERROR_CODE_H
#define ERROR_CODE_H

#include <string>
#include <iostream>

class Status
{
public:
  // The status codes
  enum class Code
  {
    SUCCESS,
    //****************************************************************************/
    /* 0、初始化 */
    //****************************************************************************/
    // 模型初始化失败
    MODEL_INIT_FAIL,

    //****************************************************************************/
    /* 1、预处理 */
    //****************************************************************************/
    // 底板视频解析错误
    VIDEO_READ_FAIL,
    //****************************************************************************/
    // // 人脸检测失败
    // FACE_DETECT_FAIL,
    // // 人脸关键点检测失败
    // FACE_LANDMARK_FAIL,
    //****************************************************************************/
    // // JSON保存错误
    // JSON_SAVE_FAIL,
    //****************************************************************************/
    // 预处理预测执行异常捕获
    PROCESS_PREDICT_FAIL,
    
    //****************************************************************************/
    /* 2、视频合成 */
    //****************************************************************************/
    // VIDEO PARAM读取失败
    VIDEO_PARAM_READ_FAIL,
    // 输入音频读取失败
    AUDIO_READ_FAIL,
    // FFMPEG音频转换失败
    FFMPEG_AUDIO_CONVERT_FAIL,
    // // 输入视频读取失败
    // VIDEO_READ_FAIL,
    // FFMPEG视频转换失败
    // FFMPEG_VIDEO_CONVERT_FAIL,
    // 人脸json读取失败
    JSON_READ_FAIL,
    //****************************************************************************/
    // 音频特征提取失败
    AUDIO_FEAT_EXTRACT_FAIL,
    // 合成帧失败
    FRAME_GENERATE_FAIL,
    //****************************************************************************/
    // FFMPEG配音
    AUDIO_DUBBING_FAIL,
    // // 输出视频校验失败
    // RENDER_OUT_VIDEO_VERIFY_FAIL,
    //****************************************************************************/
    // Render预测执行异常捕获
    RENDER_PREDICT_FAIL,
    //****************************************************************************/
    // 缓存文件删除失败
    FILE_TMP_DELETE_FAIL,
    //****************************************************************************/
    // ID参数读取失败
    READ_ID_PARAM_FAIL,
    //****************************************************************************/

    //****************************************************************************/
    /* 3、闭嘴合成 */
    //****************************************************************************/
    SHUT_UP_FAIL,


    /* 未知错误 */
    UNKNOWN,

  };

public:
  // Construct a status from a code with no message.
  explicit Status(Code code = Code::SUCCESS) : code_(code) {}

  // Construct a status from a code and message.
  explicit Status(Code code, const std::string &msg) : code_(code), msg_(msg) {}

  // Convenience "success" value. Can be used as Status::Success to
  // indicate no error.
  static const Status Success;

  // Return the code for this status.
  Code StatusCode() const { return code_; }

  // Return the message for this status.
  const std::string &Message() const { return msg_; }

  // Return true if this status indicates "ok"/"success", false if
  // status indicates some kind of failure.
  bool IsOk() const { return code_ == Code::SUCCESS; }

  // Return the status as a string.
  std::string AsString() const;

  // Return the constant string name for a code.
  static std::string CodeString(const Code code);

private:
  Code code_;
  std::string msg_;
};

#endif