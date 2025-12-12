#!/bin/bash

# 编译 Java 文件
echo "Compiling Java files..."
javac com/hexin/ai/jni/TalkingFace.java
javac com/hexin/ai/jni/TalkingFaceMultiTest.java

# 运行多实例测试
# 参数1: 并发路数 (默认5)
# 参数2: 运行时长(分钟) (默认20)
CONCURRENT=${1:-1}
DURATION=${2:-1}

echo "Running multi-instance test: $CONCURRENT channels, $DURATION minutes"
java com/hexin/ai/jni/TalkingFaceMultiTest $CONCURRENT $DURATION
