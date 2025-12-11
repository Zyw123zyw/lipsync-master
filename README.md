# TalkingFace_Serve

## 编译&测试

### 编译

```
bash build.sh
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


4. c++调用测试（可选）
   - 如果需要c++调试，请先安装gdb ```apt install gdb```
   - CMakeLists.txt 打开 Unit Test 注释，然后sh build.sh得到 testRender 可执行文件，
   - 然后在 /workspace/project/talkingface/.vscode/launch.json 下指定输入，
   - 最后按F5运行即可。
