#include "error_code.h"

const Status Status::Success(Status::Code::SUCCESS);

std::string
Status::AsString() const
{
  std::string message = "{'code': '" + CodeString(code_) +
                        "', 'msg': '" + msg_ + "'}";
  return message;
}

std::string
Status::CodeString(const Code code)
{
  switch (code)
  {
  case Status::Code::SUCCESS:
    return "300";

  //****************************************************************************/
  /* 0、初始化 */
  //****************************************************************************/
  case Status::Code::MODEL_INIT_FAIL:
    return "301";
  
  //****************************************************************************/
  /* 1、预处理 */
  //****************************************************************************/
  case Status::Code::VIDEO_READ_FAIL:
    return "331";
  //****************************************************************************/
  // case Status::Code::FACE_DETECT_FAIL:
  //   return "332";
  // case Status::Code::FACE_LANDMARK_FAIL:
  //   return "333";
  //****************************************************************************/
  // case Status::Code::JSON_SAVE_FAIL:
  //   return "334";
  //****************************************************************************/
  case Status::Code::PROCESS_PREDICT_FAIL:
    return "340";
  
  //****************************************************************************/
  /* 2、视频合成 */
  //****************************************************************************/
  case Status::Code::VIDEO_PARAM_READ_FAIL:
    return "351";
  case Status::Code::AUDIO_READ_FAIL:
    return "352";
  case Status::Code::FFMPEG_AUDIO_CONVERT_FAIL:
    return "353";
  // case Status::Code::VIDEO_READ_FAIL:
  //   return "354";
  // case Status::Code::FFMPEG_VIDEO_CONVERT_FAIL:
  //   return "355";
  case Status::Code::JSON_READ_FAIL:
    return "356";
  //****************************************************************************/
  case Status::Code::AUDIO_FEAT_EXTRACT_FAIL:
    return "357";
  case Status::Code::FRAME_GENERATE_FAIL:
    return "358";
  //****************************************************************************/
  case Status::Code::AUDIO_DUBBING_FAIL:
    return "359";
  // case Status::Code::RENDER_OUT_VIDEO_VERIFY_FAIL:
  //   return "360";

  //****************************************************************************/
  /* 3、闭嘴合成 */
  //****************************************************************************/
  case Status::Code::SHUT_UP_FAIL:
    return "360";

  //****************************************************************************/
  case Status::Code::RENDER_PREDICT_FAIL:
    return "370";
  //****************************************************************************/
  case Status::Code::FILE_TMP_DELETE_FAIL:
    return "371";
  //****************************************************************************/
  case Status::Code::READ_ID_PARAM_FAIL:
    return "372";
  //****************************************************************************/
  case Status::Code::UNKNOWN:
    return "399";

  default:
    break;
  }

  return "<invalid code>";
}