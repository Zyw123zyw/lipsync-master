#ifndef VIDEO_PALINDROME_H
#define VIDEO_PALINDROME_H

#include <string>

/**
 * 视频正反拼接处理（回文视频）
 * 
 * 当源视频时长比音频时长短时，通过"正反拼接"的方式延长视频，
 * 使视频能够覆盖整个音频时长。
 * 
 * @param src_video_path 源视频路径
 * @param audio_path 音频路径
 * @param output_video_path 输出视频路径（如果为空，则自动生成）
 * @param palindrome_cost_ms 输出参数：正反拼接耗时（毫秒）
 * @return 处理后的视频路径（如果无需处理则返回原路径）
 */
std::string ensurePalindromeVideo(const std::string& src_video_path, 
                                   const std::string& audio_path,
                                   std::string output_video_path = "",
                                   long* palindrome_cost_ms = nullptr);

/**
 * 使用 ffprobe 获取媒体时长
 * 
 * @param media_path 媒体文件路径
 * @param stream_specifier 流选择器（如 "v:0" 或 "a:0"，为空则获取格式级别时长）
 * @return 时长（秒），失败返回 -1
 */
double probeMediaDuration(const std::string& media_path, const std::string& stream_specifier = "");

/**
 * 使用 ffprobe 获取视频帧率
 * 
 * @param video_path 视频文件路径
 * @return 帧率，失败返回 -1
 */
double probeVideoFps(const std::string& video_path);

#endif // VIDEO_PALINDROME_H
