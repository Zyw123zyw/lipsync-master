# 人脸裁剪 ./build/testFacecrop [image path] [video path] [save dir]
./build/testFacecrop \
    testMedia/0_images/test.jpg \
    testMedia/0_videos/test.mp4 \
    testMedia/0_caches

# 人头分割 ./build/testFaceseg [image path] [save dir]
./build/testFaceseg \
    testMedia/0_images/test.jpg \
    testMedia/0_caches

# 预处理 ./build/testPreProcess [video_path] [save_json_path]
./build/testPreProcess \
    test_asserts/test.mp4 \
    test_asserts/test.json

# 合成 ./build/testRender [video_path] [json_path] [audio_path] [save_video_path] [enhance]
./build/testRender \
    test_asserts/test.mp4 \
    test_asserts/test.json \
    test_asserts/test.wav \
    test_asserts/out.mp4 \
    1