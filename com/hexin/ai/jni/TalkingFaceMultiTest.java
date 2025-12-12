package com.hexin.ai.jni;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.ArrayList;
import java.util.List;

/**
 * 多路并发性能测试 - 多实例模式
 */
public class TalkingFaceMultiTest {

    // ============ 配置区 ============
    private static final int CONCURRENT_NUM = 5;
    private static final int RUN_MINUTES = 20;
    private static final String MODEL_DIR = "/root/models_8.9/";
    private static final String INPUT_VIDEO = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/input/vedio.mp4";
    private static final String INPUT_AUDIO = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/input/audio.wav";
    private static final String OUTPUT_DIR = "/mnt/data/vision-devel/zhangyiwei/lipsync-sdk-master/output/";
    // ================================

    private static volatile boolean running = true;
    private static AtomicInteger successTasks = new AtomicInteger(0);
    private static AtomicInteger failTasks = new AtomicInteger(0);

    public static void main(String[] args) throws Exception {
        int concurrentNum = args.length >= 1 ? Integer.parseInt(args[0]) : CONCURRENT_NUM;
        int runMinutes = args.length >= 2 ? Integer.parseInt(args[1]) : RUN_MINUTES;
        final long runMillis = runMinutes * 60 * 1000L;

        System.out.println("========================================");
        System.out.println("多路并发性能测试 (多实例模式)");
        System.out.println("并发路数: " + concurrentNum);
        System.out.println("运行时长: " + runMinutes + " 分钟");
        System.out.println("========================================\n");

        // 创建多个实例
        System.out.println("创建 " + concurrentNum + " 个TalkingFace实例...");
        TalkingFace[] instances = new TalkingFace[concurrentNum];
        long initStart = System.currentTimeMillis();
        
        for (int i = 0; i < concurrentNum; i++) {
            instances[i] = new TalkingFace();
            instances[i].init(0, 2, 4, MODEL_DIR);
            System.out.println("实例 " + i + " 初始化完成");
        }
        
        long initEnd = System.currentTimeMillis();
        System.out.println("所有实例初始化完成，总耗时: " + (initEnd - initStart) + "ms\n");

        // 开始性能测试计时
        TalkingFace.startPerfTest(runMinutes);
        
        ExecutorService executor = Executors.newFixedThreadPool(concurrentNum);
        long testStart = System.currentTimeMillis();

        // 定时停止
        Thread stopThread = new Thread(() -> {
            try {
                Thread.sleep(runMillis);
                running = false;
                System.out.println("\n[时间到] 停止接收新任务，等待进行中的任务完成...");
            } catch (InterruptedException e) {}
        });
        stopThread.start();

        // 进度打印
        Thread progressThread = new Thread(() -> {
            while (running) {
                try {
                    Thread.sleep(30000);
                    if (running) {
                        long elapsed = System.currentTimeMillis() - testStart;
                        long frames = TalkingFace.getPerfFrameCount();
                        System.out.println(String.format(
                            "[进度] 已运行: %.1f分钟, 完成任务: %d, 渲染帧数: %d, 实时FPS: %.2f",
                            elapsed / 60000.0, successTasks.get(), frames, 
                            frames * 1000.0 / elapsed));
                    }
                } catch (InterruptedException e) { break; }
            }
        });
        progressThread.setDaemon(true);
        progressThread.start();

        // 提交任务 - 每路使用独立实例
        List<Future<?>> futures = new ArrayList<>();
        for (int i = 0; i < concurrentNum; i++) {
            final int threadId = i;
            final TalkingFace instance = instances[i];
            futures.add(executor.submit(() -> runRenderLoop(threadId, instance)));
        }

        // 等待完成
        for (Future<?> future : futures) {
            try { future.get(); } catch (Exception e) { e.printStackTrace(); }
        }

        long testEnd = System.currentTimeMillis();
        long totalFrames = TalkingFace.getPerfFrameCount();

        // 输出结果
        printStatistics(testStart, testEnd, concurrentNum, runMinutes, totalFrames);

        // 清理
        executor.shutdown();
        System.out.println("\n清理实例...");
        for (int i = 0; i < concurrentNum; i++) {
            instances[i].stop();
            instances[i].destroy();
        }
        System.out.println("完成");
    }

    private static void runRenderLoop(int threadId, TalkingFace instance) {
        String videoParams = "{\"video_enhance\":0}";
        String saveVideoPath = OUTPUT_DIR + "out_thread" + threadId + ".mp4";
        int iteration = 0;

        while (true) {
            if (!running) break;
            
            long start = System.currentTimeMillis();
            try {
                String msg = instance.render(INPUT_VIDEO, INPUT_AUDIO, "", 
                                             saveVideoPath, videoParams, "", "");
                long cost = System.currentTimeMillis() - start;
                
                boolean success = msg != null && msg.contains("success");
                String overtime = !running ? " [超时后完成]" : "";
                
                if (success) {
                    successTasks.incrementAndGet();
                    System.out.println("[Thread-" + threadId + "] 任务" + iteration + 
                                     " 成功, 耗时: " + cost + "ms" + overtime);
                } else {
                    failTasks.incrementAndGet();
                    System.out.println("[Thread-" + threadId + "] 任务" + iteration + " 失败: " + msg);
                }
            } catch (Exception e) {
                failTasks.incrementAndGet();
                System.out.println("[Thread-" + threadId + "] 任务" + iteration + " 异常: " + e.getMessage());
            }
            iteration++;
        }
        System.out.println("[Thread-" + threadId + "] 退出, 完成 " + iteration + " 个任务");
    }

    private static void printStatistics(long start, long end, int concurrentNum, 
                                         int targetMinutes, long totalFrames) {
        double totalSeconds = (end - start) / 1000.0;
        double totalMinutes = totalSeconds / 60.0;
        int totalTasks = successTasks.get() + failTasks.get();
        
        System.out.println("\n========================================");
        System.out.println("性能测试结果");
        System.out.println("========================================");
        System.out.println("并发路数: " + concurrentNum);
        System.out.println("目标时长: " + targetMinutes + " 分钟");
        System.out.println("实际时长: " + String.format("%.2f", totalMinutes) + " 分钟");
        System.out.println("----------------------------------------");
        System.out.println("总任务数: " + totalTasks);
        System.out.println("成功: " + successTasks.get() + ", 失败: " + failTasks.get());
        if (totalTasks > 0) {
            System.out.println("成功率: " + String.format("%.2f", successTasks.get() * 100.0 / totalTasks) + "%");
        }
        System.out.println("----------------------------------------");
        System.out.println("【核心指标 - C++层精确统计】");
        System.out.println("总渲染帧数: " + totalFrames + " 帧");
        System.out.println("平均帧率 (FPS): " + String.format("%.2f", totalFrames / totalSeconds) + " 帧/秒");
        System.out.println("每分钟渲染帧数: " + String.format("%.0f", totalFrames / totalMinutes) + " 帧/分钟");
        System.out.println("----------------------------------------");
        System.out.println("任务吞吐量: " + String.format("%.2f", successTasks.get() / totalMinutes) + " 任务/分钟");
        if (successTasks.get() > 0) {
            System.out.println("平均单任务耗时: " + String.format("%.2f", totalSeconds / successTasks.get() * concurrentNum) + " 秒");
        }
        System.out.println("========================================");
    }
}
