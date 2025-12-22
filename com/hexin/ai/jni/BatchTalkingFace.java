package com.hexin.ai.jni;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Small utility to run TalkingFace.render across every "in*" directory that
 * carries an audio.wav and vedio.mp4 input pair.
 */
public class BatchTalkingFace {

    private enum JobResult {
        SUCCESS,
        SKIPPED,
        FAILED
    }

    private static final class Config {
        final Path rootDir;
        final String modelDir;
        final int gpuId;
        final int numWorkers;
        final int ffmpegThreads;
        final String audioFileName;
        final String videoFileName;
        final String outputFileName;
        final String videoParams;
        final String inputParamsPrefix;
        final boolean mergeInputParams;
        final String vocalAudioPath;
        final String idParams;
        final String faceJsonPath;
        final boolean forceOverwrite;
        final String dirPrefix;

        private Config(Path rootDir,
                       String modelDir,
                       int gpuId,
                       int numWorkers,
                       int ffmpegThreads,
                       String audioFileName,
                       String videoFileName,
                       String outputFileName,
                       String videoParams,
                       String inputParamsPrefix,
                       boolean mergeInputParams,
                       String vocalAudioPath,
                       String idParams,
                       String faceJsonPath,
                       boolean forceOverwrite,
                       String dirPrefix) {
            this.rootDir = rootDir;
            this.modelDir = modelDir;
            this.gpuId = gpuId;
            this.numWorkers = numWorkers;
            this.ffmpegThreads = ffmpegThreads;
            this.audioFileName = audioFileName;
            this.videoFileName = videoFileName;
            this.outputFileName = outputFileName;
            this.videoParams = videoParams;
            this.inputParamsPrefix = inputParamsPrefix;
            this.mergeInputParams = mergeInputParams;
            this.vocalAudioPath = vocalAudioPath;
            this.idParams = idParams;
            this.faceJsonPath = faceJsonPath;
            this.forceOverwrite = forceOverwrite;
            this.dirPrefix = dirPrefix;
        }

        static Config fromArgs(String[] args) {
            Map<String, String> argMap = parseArgs(args);
            Path root = Paths.get(argMap.getOrDefault("root", "/mnt/data/vision-devel/zhangyiwei/in/dreamface"));
            if (!Files.isDirectory(root)) {
                throw new IllegalArgumentException("无效的 root 目录: " + root.toAbsolutePath());
            }
            String modelDir = argMap.getOrDefault("model-dir", "/workspace/models_8.9/");
            int gpuId = Integer.parseInt(argMap.getOrDefault("gpu", "0"));
            int numWorkers = Integer.parseInt(argMap.getOrDefault("workers", "2"));
            int ffmpegThreads = Integer.parseInt(argMap.getOrDefault("ffmpeg-threads", "4"));
            String audioFileName = argMap.getOrDefault("audio-name", "audio.wav");
            String videoFileName = argMap.getOrDefault("video-name", "vedio.mp4");
            String outputFileName = argMap.getOrDefault("output-name", "vediogpu.mp4");
            // 默认不强制注入任何参数，完全以用户传入/每目录 inputX 为准
            String videoParams = argMap.getOrDefault("video-params", "{}");
            String inputParamsPrefix = argMap.getOrDefault("input-params-prefix", "input");
            boolean mergeInputParams = argMap.containsKey("merge-input-params");
            String vocalAudioPath = argMap.getOrDefault("vocal-audio", "");
            String idParams = argMap.getOrDefault("id-params", "");
            String faceJsonPath = argMap.getOrDefault("face-json", "");
            boolean forceOverwrite = argMap.containsKey("force");
            String dirPrefix = argMap.getOrDefault("dir-prefix", "in");
            return new Config(root, modelDir, gpuId, numWorkers, ffmpegThreads, audioFileName,
                              videoFileName, outputFileName, videoParams, inputParamsPrefix, mergeInputParams, vocalAudioPath,
                              idParams, faceJsonPath, forceOverwrite, dirPrefix);
        }

        private static Map<String, String> parseArgs(String[] args) {
            Map<String, String> parsed = new HashMap<>();
            for (String arg : args) {
                if (!arg.startsWith("--")) {
                    continue;
                }
                String withoutPrefix = arg.substring(2);
                int eqIdx = withoutPrefix.indexOf('=');
                if (eqIdx < 0) {
                    parsed.put(withoutPrefix, "");
                } else {
                    String key = withoutPrefix.substring(0, eqIdx);
                    String value = withoutPrefix.substring(eqIdx + 1);
                    parsed.put(key, value);
                }
            }
            return parsed;
        }
    }

    public static void main(String[] args) throws Exception {
        Config cfg = Config.fromArgs(args);
        System.out.printf(Locale.CHINA,
                "[BATCH] Root=%s, 模型目录=%s, GPU=%d, workers=%d, ffmpegThreads=%d%n",
                cfg.rootDir.toAbsolutePath(), cfg.modelDir, cfg.gpuId, cfg.numWorkers, cfg.ffmpegThreads);

        TalkingFace tf = new TalkingFace();
        tf.sayHello();
        if (!tf.init(cfg.gpuId, cfg.numWorkers, cfg.ffmpegThreads, cfg.modelDir)) {
            throw new IllegalStateException("TalkingFace 初始化失败");
        }

        int success = 0;
        int skipped = 0;
        int failed = 0;

    List<Path> candidates;
    try (java.util.stream.Stream<Path> stream = Files.list(cfg.rootDir)) {
            candidates = stream
                    .filter(Files::isDirectory)
                    .filter(path -> cfg.dirPrefix == null || cfg.dirPrefix.isEmpty()
                            || path.getFileName().toString().startsWith(cfg.dirPrefix))
                    .sorted(Comparator.comparing(path -> path.getFileName().toString(), String.CASE_INSENSITIVE_ORDER))
                    .collect(Collectors.toCollection(ArrayList::new));
        }

        System.out.printf("[BATCH] 待处理目录总数: %d%n", candidates.size());

        for (Path dir : candidates) {
            JobResult result = processDirectory(tf, dir, cfg);
            switch (result) {
                case SUCCESS:
                    success++;
                    break;
                case SKIPPED:
                    skipped++;
                    break;
                case FAILED:
                    failed++;
                    break;
                default:
                    break;
            }
        }

        tf.stop();

        System.out.printf("[BATCH] 完成: success=%d, skipped=%d, failed=%d%n", success, skipped, failed);
        if (failed > 0) {
            throw new RuntimeException("仍有 " + failed + " 个目录处理失败");
        }
    }

    private static JobResult processDirectory(TalkingFace tf, Path dir, Config cfg) {
        Path audioPath = dir.resolve(cfg.audioFileName);
        Path videoPath = dir.resolve(cfg.videoFileName);
        Path outputPath = dir.resolve(cfg.outputFileName);

        if (!Files.isRegularFile(audioPath) || !Files.isRegularFile(videoPath)) {
            System.out.printf("[BATCH][SKIP] %s 缺少 %s 或 %s%n", dir.getFileName(), cfg.audioFileName, cfg.videoFileName);
            return JobResult.SKIPPED;
        }

        if (Files.exists(outputPath) && !cfg.forceOverwrite) {
            System.out.printf("[BATCH][SKIP] %s 已存在 %s，使用 --force 可覆盖%n", dir.getFileName(), cfg.outputFileName);
            return JobResult.SKIPPED;
        }

        long start = System.currentTimeMillis();
        try {
            String effectiveVideoParams = resolveVideoParamsForDir(dir, cfg);

            // 视频正反拼接已在 C++ render 函数入口处自动处理，无需在 Java 端调用
            String renderMsg = tf.render(
                    videoPath.toAbsolutePath().toString(),
                    audioPath.toAbsolutePath().toString(),
                    cfg.faceJsonPath,
                    outputPath.toAbsolutePath().toString(),
                    effectiveVideoParams,
                    cfg.vocalAudioPath,
                    cfg.idParams);
            long costMs = System.currentTimeMillis() - start;
            System.out.printf("[BATCH][OK] %s -> %s (%.2fs) %s%n",
                    dir.getFileName(), outputPath.getFileName(), costMs / 1000.0, renderMsg);
            return JobResult.SUCCESS;
        } catch (Exception ex) {
            System.err.printf("[BATCH][FAIL] %s 处理失败: %s%n", dir.getFileName(), ex.getMessage());
            return JobResult.FAILED;
        }
    }

    /**
     * 每个 inX 目录下允许放一个 inputX 文件（例如 in1/input1），内容为 JSON，作为 videoParams 传入 render。
     *
     * 默认行为：如果存在 inputX，则优先使用其内容；如果不存在，则用 cfg.videoParams。
     * 如果传入 --merge-input-params，则将 cfg.videoParams 与 inputX 做浅层合并（inputX 覆盖同名 key）。
     */
    private static String resolveVideoParamsForDir(Path dir, Config cfg) {
        String dirName = dir.getFileName().toString();
        String inputFileName = cfg.inputParamsPrefix + dirName.replaceFirst("^" + cfg.dirPrefix, "");
        Path inputPath = dir.resolve(inputFileName);

        if (!Files.isRegularFile(inputPath)) {
            return cfg.videoParams;
        }

        try {
            String fileJson = new String(Files.readAllBytes(inputPath)).trim();
            if (fileJson.isEmpty()) {
                System.out.printf("[BATCH][WARN] %s 参数文件为空: %s，回退到默认 videoParams%n",
                        dir.getFileName(), inputPath.getFileName());
                return cfg.videoParams;
            }
            if (!cfg.mergeInputParams) {
                return fileJson;
            }
            return mergeJsonShallow(cfg.videoParams, fileJson);
        } catch (Exception e) {
            System.out.printf("[BATCH][WARN] %s 读取参数文件失败: %s (%s)，回退到默认 videoParams%n",
                    dir.getFileName(), inputPath.getFileName(), e.getMessage());
            return cfg.videoParams;
        }
    }

    /**
     * 非严格 JSON 解析的“浅合并”实现：仅支持一层 object，value 为数字/字符串/boolean。
     * 目的：避免引入额外 JSON 依赖，同时满足 {"video_width":0,...} 这种场景。
     */
    private static String mergeJsonShallow(String baseJson, String overrideJson) {
        Map<String, String> a = parseFlatJsonObject(baseJson);
        Map<String, String> b = parseFlatJsonObject(overrideJson);
        a.putAll(b);
        return toFlatJsonObject(a);
    }

    private static Map<String, String> parseFlatJsonObject(String json) {
        Map<String, String> map = new HashMap<>();
        if (json == null) {
            return map;
        }
        String s = json.trim();
        if (s.startsWith("{") && s.endsWith("}")) {
            s = s.substring(1, s.length() - 1).trim();
        }
        if (s.isEmpty()) {
            return map;
        }

        // split by comma not inside quotes (simple)
        List<String> parts = new ArrayList<>();
        StringBuilder buf = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' && (i == 0 || s.charAt(i - 1) != '\\')) {
                inQuotes = !inQuotes;
            }
            if (c == ',' && !inQuotes) {
                parts.add(buf.toString());
                buf.setLength(0);
            } else {
                buf.append(c);
            }
        }
        parts.add(buf.toString());

        for (String part : parts) {
            String p = part.trim();
            if (p.isEmpty()) continue;
            int idx = p.indexOf(':');
            if (idx < 0) continue;
            String key = p.substring(0, idx).trim();
            String val = p.substring(idx + 1).trim();
            if (key.startsWith("\"") && key.endsWith("\"")) {
                key = key.substring(1, key.length() - 1);
            }
            if (!key.isEmpty()) {
                map.put(key, val);
            }
        }
        return map;
    }

    private static String toFlatJsonObject(Map<String, String> map) {
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, String> e : map.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append('"').append(escapeJsonString(e.getKey())).append('"').append(':').append(e.getValue());
        }
        sb.append('}');
        return sb.toString();
    }

    private static String escapeJsonString(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
