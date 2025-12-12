package com.hexin.ai.jni;

import java.io.IOException;

public class TalkingFace {

    private long nativeHandle;  // C++实例指针

    // 构造函数 - 创建C++实例
    public TalkingFace() {
        nativeHandle = nativeCreate();
    }

    // 销毁实例
    public void destroy() {
        if (nativeHandle != 0) {
            nativeDestroy(nativeHandle);
            nativeHandle = 0;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        destroy();
        super.finalize();
    }

    // Native方法 - 实例管理
    private native long nativeCreate();
    private native void nativeDestroy(long handle);

    // Native方法 - 带handle参数
    public native void sayHello();
    private native boolean init(long handle, int gpuId, int numWorkers, int ffmpegThreads, String modelDir);
    private native String process(long handle, String videoPath, String faceJsonSavePath, String videoParams);
    private native String render(long handle, String videoPath, String audioPath, String faceJsonPath, String saveVideoPath, String videoParams, String vocalAudioPath, String idParams);
    private native String shutup(long handle, String imagePath, String savePath, String videoParams, String idParams);
    private native boolean stop(long handle);

    // 性能测试接口（静态方法）
    public static native void startPerfTest(int durationMinutes);
    public static native long getPerfFrameCount();
    public static native void resetPerfCounter();

    // 包装方法 - 对外接口保持不变
    public boolean init(int gpuId, int numWorkers, int ffmpegThreads, String modelDir) {
        return init(nativeHandle, gpuId, numWorkers, ffmpegThreads, modelDir);
    }

    public String process(String videoPath, String faceJsonSavePath, String videoParams) {
        return process(nativeHandle, videoPath, faceJsonSavePath, videoParams);
    }

    public String render(String videoPath, String audioPath, String faceJsonPath, String saveVideoPath, String videoParams, String vocalAudioPath, String idParams) {
        return render(nativeHandle, videoPath, audioPath, faceJsonPath, saveVideoPath, videoParams, vocalAudioPath, idParams);
    }

    public String shutup(String imagePath, String savePath, String videoParams, String idParams) {
        return shutup(nativeHandle, imagePath, savePath, videoParams, idParams);
    }

    public boolean stop() {
        return stop(nativeHandle);
    }

    static {
        System.load("/mnt/data/vision-devel/zhangyiwei/lipsync-master/build/libtalkingface.so");
    }

    // 测试入口 - 单实例测试
    public static void main(String[] args) throws IOException {
        TalkingFace tf = new TalkingFace();
        tf.sayHello();
        tf.init(0, 2, 4, "/root/models_8.9/");

        String videoParams = "{\"video_enhance\":0}";
        String src_video_path = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/input/vedio.mp4";
        String audioPath = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/input/audio.wav";
        String saveVideoPath = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/output/out.mp4";

        String msg = tf.render(src_video_path, audioPath, "", saveVideoPath, videoParams, "", "");
        System.out.println(msg);

        tf.stop();
        tf.destroy();
    }
}
