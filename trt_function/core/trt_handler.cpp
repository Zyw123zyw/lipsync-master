#include "trt_handler.h"


#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

using ILogger = nvinfer1::ILogger;
using Severity = nvinfer1::ILogger::Severity;

class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};


BasicTRTHandler::BasicTRTHandler(const std::string &_engine_path) : engine_file(_engine_path) {}


void BasicTRTHandler::initialize_handler() {
    int length = 0; // 记录data的长度
    std::unique_ptr<char[]> data = readEngineFile(length);
    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    
    runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
    engine = runtime->deserializeCudaEngine(data.get(), length);
    if (!engine)
    {
        std::cout << "Failed to create engine" << std::endl;
    }
    context = engine->createExecutionContext();
    assert(context != nullptr);
    int nbBindings = engine->getNbBindings();

    // 为输入和输出创建空间
    buffer_size.resize(nbBindings);
    for (int i = 0; i < nbBindings; i++)
    {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);    // (3, 224, 224)  (1000)
        nvinfer1::DataType dtype = engine->getBindingDataType(i); // 0, 0 也就是两个都是kFloat类型
        int64_t total_size = volume(dims) * 1 * getElementSize(dtype);
        buffer_size[i] = total_size;
        // std::cout << "total_size: " << total_size << " " << volume(dims) << " " << getElementSize(dtype) << std::endl;
        CHECK(cudaMalloc(&buffers[i], total_size));
    }
    std::cout << "Load engine successfully" << std::endl;
}


// 从plan文件读取数据
std::unique_ptr<char[]> BasicTRTHandler::readEngineFile(int &length)
{
    std::ifstream file;
    file.open(engine_file, std::ios::in | std::ios::binary);
    // 获得文件流的长度
    file.seekg(0, std::ios::end); // 把指针移到末尾
    length = file.tellg();        // 返回当前指针位置
    // 指针移到开始
    file.seekg(0, std::ios::beg);
    // 定义缓存
    std::unique_ptr<char[]> data(new char[length]);
    // 读取文件到缓存区
    file.read(data.get(), length);
    file.close();
    return data;
}


// 累积乘法 对binding的维度累乘 (3,224,224) => 3*224*224
inline int64_t BasicTRTHandler::volume(const nvinfer1::Dims &d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}


inline unsigned int BasicTRTHandler::getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}


BasicTRTHandler::~BasicTRTHandler() {
    context->destroy();
    context = nullptr;
    engine->destroy();
    engine = nullptr;
    runtime->destroy();
    runtime = nullptr;
}
