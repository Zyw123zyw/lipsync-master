package com.hexin.ai.jni;
import java.io.File;
import java.util.Arrays;
import java.util.Comparator;

import java.io.IOException;

public class TalkingFace {

    public native void sayHello();

    public native boolean init(int gpuId, int numWorkers, int ffmpegThreads, String modelDir);

    public native String process(String videoPath, String faceJsonSavePath, String videoParams);

    public native String render(String videoPath, String audioPath, String faceJsonPath, String saveVideoPath, String videoParams, String vocalAudioPath, String idParams);

    public native String shutup(String imagePath, String savePath, String videoParams, String idParams);

    public native boolean stop();

    static {
        System.load("/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/build/libtalkingface.so"); // 加载动态链接库 
    }

    public static void main(String[] args) throws IOException {

        TalkingFace tf = new TalkingFace();

        tf.sayHello();

        tf.init(0, 2, 4, "/root/models_8.9/");

        String msg;

        String src_video_path = "";
        String audioPath = "";
        String faceJsonSavePath = "";
        String saveVideoPath = "";
        String videoParams = "{\"video_enhance\":0}";
        String vocalAudioPath = "";
        String idParams = "";

        // videoParams = "{\"video_bitrate\":0,\"video_width\":256,\"video_height\":511,\"video_enhance\":1,\"filter_head_pose\":1,\"face_det_threshold\":0.5}";
        // videoParams = "{\"video_enhance\":1, \"face_det_threshold\":0.5, \"face_box\":[800,0,960,1080]}";
        // videoParams = "{\"video_bitrate\":0,\"video_enhance\":1,\"video_height\":1920,\"video_width\":1080}}";
        // videoParams = "{\"video_enhance\":1, \"amplifier\": 2.0, \"filter_head_pose\": 0, \"face_det_threshold\":0.5, \"face_box\":[0,0,0,0]}";
        // idParams = "[{\"id\":0,\"box\":[0,0,600,1000],\"face\":\"path1.jpg\",\"audio\":\"/workspace/project/talkingface/test_asserts/multi_person/audio3_id0.wav\"},{\"id\":1,\"box\":[700,0,600,1000],\"face\":\"path2.jpg\",\"audio\":\"/workspace/project/talkingface/test_asserts/multi_person/audio3_id1.wav\"}]";
        // idParams = "[{\"id\":0,\"box\":[250,200,650,700],\"face\":\"path1.jpg\",\"audio\":\"/workspace/project/talkingface/test_asserts/multi_person/audio2_id0.MP3\"},{\"id\":1,\"box\":[600,200,650,700],\"face\":\"path2.jpg\",\"audio\":\"/workspace/project/talkingface/test_asserts/multi_person/audio2_id1.MP3\"\"},{\"id\": 2,\"box\": [930, 200, 650, 700],\"face\": \"path2.jpg\",\"audio\": \"/workspace/project/talkingface/test_asserts/multi_person/audio2_id2.MP3\"\"},{\"id\":3,\"box\":[1300,200,650,700],\"face\":\"path2.jpg\",\"audio\":\"/workspace/project/talkingface/test_asserts/multi_person/audio2_id3.MP3\"\"}]";
        // idParams = "[{\"id\":0,\"box\":[150,150,300,550],\"face\":\"path0.jpg\",\"audio\":\"/workspace/project/talkingface/test_asserts/multi_person/audio2_id0.MP3\"},{\"id\":1,\"box\":[430,150,300,550],\"face\":\"path2.jpg\",\"audio\":\"/workspace/project/talkingface/test_asserts/multi_person/audio2_id1.MP3\"},{\"id\": 2,\"box\": [700,150,300,550],\"face\": \"path2.jpg\",\"audio\": \"/workspace/project/talkingface/test_asserts/multi_person/audio2_id2.MP3\"},{\"id\":3,\"box\":[980,150,300,550],\"face\":\"path2.jpg\",\"audio\":\"/workspace/project/talkingface/test_asserts/multi_person/audio2_id3.MP3\"}]";
        // videoParams = "{\"video_enhance\":1, \"video_bitrate\":0,  \"amplifier\": 1.0, \"video_max_side\": 1440, \"audio_max_time\": 1800}";
        // videoParams = "{\"video_enhance\":0, \"video_bitrate\":0,  \"amplifier\": 1.0, \"video_max_side\": 0, \"audio_max_time\": 0}";
        // videoParams = "{\"video_max_side\":-1, \"video_width\":2000, \"video_height\":4000}";
        // videoParams = "{\"audio_max_time\":20, \"video_enhance\":1}";
        // videoParams = "{\"video_bitrate\":2000, \"video_enhance\":1}";
        // videoParams = "{\"video_enhance\":1, \"audio_max_time\":0, \"video_max_side\": 1080}";
        // videoParams = "{\"video_enhance\":1, \"amplifier\": 1.5, \"shutup_first\": 0}";

        // // test-for-process
        // src_video_path = "/workspace/project/talkingface/117d07764ae94ffabbfe16aeaf2a3d30.mp4";
        // faceJsonSavePath = "/workspace/project/talkingface/117d07764ae94ffabbfe16aeaf2a3d30.json";
        // msg = tf.process(src_video_path, faceJsonSavePath, videoParams);
        // System.out.println(msg);

        // // // test-for-shutup
        // String imagePath = "/workspace/project/talkingface/assets/1/image3.png";
        // String savePath = "/workspace/project/talkingface/out.png";
        // // idParams = "[{\"box\":[707,43,307,399]}]";
        // // idParams = "[{\"box\":[70,150,200,200]},{\"box\":[200,200,200,200]}]";
        // // idParams = "[{\"box\":[60,60,150,150]},{\"box\":[100,60,150,150]},{\"box\":[200,60,150,150]}]";
        // msg = tf.shutup(imagePath, savePath, videoParams, idParams);
        // System.out.println(msg);

        // // // test-for-render
        // videoParams = "{\"video_enhance\":0, \"video_bitrate\":0,  \"amplifier\": 2.0, \"video_max_side\": 0, \"audio_max_time\": 0}";
        // src_video_path = "/workspace/project/talkingface/assets/多人/双人视频-左1右2.mp4";
        // audioPath = "/workspace/project/talkingface/assets/多人/说话人.wav";
        // saveVideoPath = "/workspace/project/talkingface/assets/多人/out.mp4";
        // idParams = "[{\"id\":0,\"box\":[400,0,600,600],\"audio\":\"/workspace/project/talkingface/assets/多人/说话人1.WAV\"},{\"id\":1,\"box\":[1000,0,600,600],\"audio\":\"/workspace/project/talkingface/assets/多人/说话人2.WAV\"}]";
        // msg = tf.render(src_video_path, audioPath, faceJsonSavePath, saveVideoPath, videoParams, vocalAudioPath, idParams);
        // System.out.println(msg);
        

        // videoParams = "{\"video_enhance\": 0, \"keep_fps\": 1, \"keep_bitrate\": 1}";
        // src_video_path = "/workspace/project/talkingface/assets/0620/4k_25fps.mp4";
        // audioPath = "/workspace/project/talkingface/assets/0620/audio_30m.wav";
        // saveVideoPath = "/workspace/project/talkingface/out.mp4";
        // msg = tf.render(src_video_path, audioPath, faceJsonSavePath, saveVideoPath, videoParams, vocalAudioPath, idParams);
        // System.out.println(msg);

        src_video_path = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/input/bb0087c4ff364deb97cdea8d5e4aaf3b.mp4";
        // src_video_path = "/workspace/project/talkingface/assets/0620/4k_25fps.mp4";
        audioPath = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/input/audio.wav";
        saveVideoPath = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/output/out.mp4";
        msg = tf.render(src_video_path, audioPath, faceJsonSavePath, saveVideoPath, videoParams, vocalAudioPath, idParams);
        System.out.println(msg);


        // // 批量测试
        // String path = "/workspace/project/talkingface/assets/batch_test";
        // File file = new File(path);
        // File[] fs = file.listFiles();
        // // 按文件名进行排序（不区分大小写）
        // Arrays.sort(fs, new Comparator<File>() {
        //     @Override
        //     public int compare(File file1, File file2) {
        //         return file1.getName().compareToIgnoreCase(file2.getName());
        //     }
        // });
        // // 遍历合成
        // for(File f:fs){
        //     if(!f.isDirectory() && f.getName().toLowerCase().endsWith(".mp4"))
        //     {
        //         src_video_path = f.getAbsolutePath();
        //         // audioPath = src_video_path.replace(".mp4", ".wav").replace("/videos/", "/audios/");
        //         // saveVideoPath = src_video_path.replace("/videos/", "/outs/");
        //         audioPath = src_video_path.replace(".mp4", ".wav").replace("diban", "audio");
        //         saveVideoPath = src_video_path.replace("/diban", "/out_");
        //         System.out.println(src_video_path);
        //         System.out.println(audioPath);
        //         System.out.println(saveVideoPath);
        //         // System.out.println();

        //         msg = tf.render(src_video_path, audioPath, faceJsonSavePath, saveVideoPath, videoParams, vocalAudioPath, idParams);
        //         System.out.println(msg);
        //     }
        // }


        // // 批量测试
        // String path = "/workspace/project/talkingface/assets/像不像问题/api素材/data";
        // File file = new File(path);
        // File[] fs = file.listFiles();
        // // 按文件名进行排序（不区分大小写）
        // Arrays.sort(fs, new Comparator<File>() {
        //     @Override
        //     public int compare(File file1, File file2) {
        //         return file1.getName().compareToIgnoreCase(file2.getName());
        //     }
        // });
        // // 遍历合成
        // for(File f:fs){
        //     if(!f.isDirectory() && f.getName().toLowerCase().endsWith(".wav"))
        //     {
        //         // src_video_path = f.getAbsolutePath();
        //         audioPath = f.getAbsolutePath();
        //         src_video_path = audioPath.replace(".wav", ".mp4");
        //         saveVideoPath = src_video_path.replace(".mp4", "_wav2lip_new.mp4");
        //         System.out.println(src_video_path);
        //         System.out.println(audioPath);
        //         System.out.println(saveVideoPath);
        //         // System.out.println();

        //         msg = tf.render(src_video_path, audioPath, faceJsonSavePath, saveVideoPath, videoParams, vocalAudioPath, idParams);
        //         System.out.println(msg);
        //     }
        // }

        tf.stop();
    }
}

