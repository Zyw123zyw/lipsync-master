#ifndef TRT_HANDLER_H
#define TRT_HANDLER_H

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <stdio.h>
// #include "trt-common/buffers.h"
// #include "trt-common/argsParser.h"
// #include "trt-common/common.h"
// #include "trt-common/logger.h"
#include "buffers.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>


class BasicTRTHandler {
    public:
        void initialize_handler();
    
    protected:
        explicit BasicTRTHandler(const std::string &_engine_path);
        virtual ~BasicTRTHandler();


        const std::string engine_file;
        nvinfer1::ICudaEngine *engine;
        nvinfer1::IRuntime *runtime;
        nvinfer1::IExecutionContext *context;
        // cudaStream_t stream;

        void *buffers[10]; // 待创建的空间
        std::vector<int64_t> buffer_size; // 待创建空间大小

        std::unique_ptr<char[]> readEngineFile(int &length);
        int64_t volume(const nvinfer1::Dims &d);
        unsigned int getElementSize(nvinfer1::DataType t);

    protected: // 禁止拷贝
        BasicTRTHandler(const BasicTRTHandler &) = delete;
        BasicTRTHandler(BasicTRTHandler &&) = delete;
        BasicTRTHandler &operator = (const BasicTRTHandler &) = delete;
        BasicTRTHandler &operator = (BasicTRTHandler &&) = delete;
};

#endif
