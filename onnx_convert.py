import os, glob

ONNX_DIR = '/workspace/project/talkingface/models'
TENSORRT_DIR = '/workspace/TensorRT-8.6.1.6'


def onnx2trt_convert(type="fp16"):

    onnx_path_list = glob.glob(os.path.join(ONNX_DIR, "*.onnx"))
    for onnx_path in onnx_path_list:
        engine_path = onnx_path.replace('.onnx', '.engine')

        command = ' '.join([
            'CUDA_VISIBLE_DEVICES=0 {}/bin/trtexec'.format(TENSORRT_DIR),
            '--onnx={}'.format(onnx_path),
            '--saveEngine={}'.format(engine_path)
        ])

        # if os.path.basename(onnx_path) == 'hubert-base-ls960.onnx':
        #     command += ' --minShapes=input:1x320 --optShapes=input:1x240000 --maxShapes=input:1x240000'

        if type == "fp16":
            command += ' --fp16'
        elif type == "int8":
            command += ' --int8'
        print("------------------------------------------------")
        print(command)
        print("------------------------------------------------")
        os.system(command)


# onnx2trt_convert("fp32")
onnx2trt_convert("fp16")


# dynamic size hubert convert
# ./bin/trtexec --onnx=/mnt/disk0/projects/talkingface/sdk/talkingface_trt/models/hubert-base-ls960.onnx --saveEngine=/mnt/disk0/projects/talkingface/sdk/talkingface_trt/models/hubert-base-ls960.engine --minShapes=input:1x320 --optShapes=input:1x240000 --maxShapes=input:1x240000 #--fp16
# static size hubert convert
# /workspace/TensorRT-8.6.1.6/bin/trtexec --onnx=/workspace/project/talkingface/models/hubert-large-ll60k.onnx --saveEngine=/workspace/project/talkingface/models/hubert-large-ll60k.engine --fp16

# wav2lip onnx to tensorrt
# /workspace/TensorRT-8.6.1.6/bin/trtexec --onnx=/workspace/project/talkingface/models/wav2lip.onnx --saveEngine=/workspace/project/talkingface/models/wav2lip.engine --fp16
