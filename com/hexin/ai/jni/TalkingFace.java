package com.hexin.ai.jni;

/**
 * TalkingFace JNI 接口类
 * 
 * 注意：视频正反拼接功能已移至 C++ 层（render 函数入口处自动处理），
 * Java 端无需再调用 ensurePalindromeVideo 方法。
 */
public class TalkingFace {

    public native void sayHello();

    public native boolean init(int gpuId, int numWorkers, int ffmpegThreads, String modelDir);

    public native String process(String videoPath, String faceJsonSavePath, String videoParams);

    public native String render(String videoPath, String audioPath, String faceJsonPath, String saveVideoPath, String videoParams, String vocalAudioPath, String idParams);

    public native String shutup(String imagePath, String savePath, String videoParams, String idParams);

    public native boolean stop();

    static {
        System.load("/workspace/project/talkingface/lipsync-master/build/libtalkingface.so"); // 加载动态链接库 
    }

    public static void main(String[] args) {

        // 记录程序开始时间
        long startTime = System.currentTimeMillis();

        TalkingFace tf = new TalkingFace();

        tf.sayHello();

        tf.init(0, 2, 4, "/workspace/models_8.9/");
        String msg;
        
        String src_video_path = "/workspace/project/talkingface/in.mp4";
        String audioPath = "/workspace/project/talkingface/audio.wav";
        String faceJsonSavePath = "";
        String saveVideoPath = "/workspace/project/talkingface/out.mp4";
        String videoParams = "";
        String vocalAudioPath = "";
        String idParams = "";

        // 视频正反拼接已在 C++ render 函数入口处自动处理，无需在 Java 端调用
        for (int i = 1; i <= 1; i++) { // for循环为了测试用的循环，实际使用时应该去掉
            msg = tf.render(src_video_path, audioPath, faceJsonSavePath, saveVideoPath, videoParams, vocalAudioPath, idParams);
            System.out.println(msg);
        }
    
        tf.stop();
        
        // 计算并输出总耗时
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        long seconds = totalTime / 1000;
        long minutes = seconds / 60;
        long hours = minutes / 60;
        
        System.out.println("========================================");
        System.out.println("程序执行完成！");
        System.out.println("总耗时: " + totalTime + " 毫秒");
        System.out.println("总耗时: " + seconds + " 秒 (" + (minutes % 60) + " 分 " + (seconds % 60) + " 秒)");
        if (hours > 0) {
            System.out.println("总耗时: " + hours + " 小时 " + (minutes % 60) + " 分 " + (seconds % 60) + " 秒");
        }
        System.out.println("========================================");
    }
}



