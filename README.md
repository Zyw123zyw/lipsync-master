# TalkingFace_Serve

## 编译&测试

### 编译

```
bash build.sh
```

> 注意：`CMakeLists.txt` 不再包含任何机器相关的绝对路径（例如 `/workspace/...`）。
> 编译前你需要通过 **CMake 变量** 或 **环境变量** 提供依赖安装目录：CUDA / TensorRT / OpenCV / FFmpeg。

#### 方式 A：使用环境变量（推荐）

```bash
source scripts/env.example.sh
```

#### 方式 B：直接传 CMake 参数

```bash
rm -rf build && mkdir build
cd build
cmake .. \
   -DCUDA_DIR=/usr/local/cuda-11.2 \
   -DTENSORRT_DIR=/workspace/TensorRT-8.6.1.6 \
   -DOPENCV_CUDA_DIR=/workspace/opencv-install-cuda \
   -DFFMPEG_DIR=/workspace/ffmpeg-install
make -j6
```

1. 根据编译对象不同，手动注释`CMakeLists.txt`的编译目标

### 测试

0. 起容器
nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=0 -v /xxx/lipsync-sdk:/workspace/project/talkingface 82351da86aa2 /bin/bash
或  
sudo docker run -it --cpus 6 --gpus 'device=0' -e NVIDIA_DRIVER_CAPABILITIES=all -v /xxx/lipsync-sdk:/workspace/project/talkingface 82351da86aa2 /bin/bash

1. 模型转换, onnx -> tensorrt fp16
   ```bash
   python3 onnx_convert.py
   ```

2. 算法编译：

   ```bash
   sh build.sh
   ```

   - 应在编译时选择编译出可执行文件

3. java调用测试

   ```bash
   sh test_java.sh
   ```

   - 应在编译时选择编译出共享库so文件

### 运行时环境变量（重要）

如果你遇到 `extract audio feat fail, audio ffmpeg convert fail.`，最常见原因是运行时找不到 `ffmpeg` 或找不到动态库。

运行前确保：

- `ffmpeg` 在 PATH 里：`${FFMPEG_DIR}/bin`
- 运行期动态库可被加载：`${FFMPEG_DIR}/lib`、`${OPENCV_CUDA_DIR}/lib`、`${TENSORRT_DIR}/lib`、`${CUDA_DIR}/lib64`

使用 `source scripts/env.example.sh` 可以一次性设置这些变量。


4. c++调用测试（可选）
   - 如果需要c++调试，请先安装gdb ```apt install gdb```
   - CMakeLists.txt 打开 Unit Test 注释，然后sh build.sh得到 testRender 可执行文件，
   - 然后在 /workspace/project/talkingface/.vscode/launch.json 下指定输入，
   - 最后按F5运行即可。
# lipsync-master 11111111111111
