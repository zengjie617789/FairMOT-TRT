import time
import torch
import tensorrt as trt
import numpy as np
import os
import time
# import pycuda.driver as cuda
# import pycuda.autoinit
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# from mmcv.tensorrt import (TRTWrapper, onnx2trt, save_trt_engine,is_tensorrt_plugin_loaded)
# assert is_tensorrt_plugin_loaded(), 'Requires to complie TensorRT plugins in mmcv'


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTModel:

    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):

        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()


    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):

        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)


            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream


    def __call__(self, x, batch_size=1):

        x = x.astype(self.dtype)
        if isinstance(x, np.ndarray):
            np.copyto(self.inputs[0].host, x.ravel())
        else:
            self.inputs[0].host = x.flatten()

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        # t1 = time.time()
        self.stream.synchronize()
        # print(f"np copytime cost: {time.time() - t1 }")

        return [out.host.reshape(batch_size, -1) for out in self.outputs]


class TRTModelTorchv2:

    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):

        self.engine_path = engine_path
        self.dtype = dtype
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings= self.allocate_buffers_v2()
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream()
        self.builder = trt.Builder(logger)

    @staticmethod
    def load_trt(trt_runtime, trt_path):
        pass


    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
            engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers_v2(self):
        inputs = []
        outputs = []
        bindings = []
        with torch.no_grad():
            for binding in self.engine:
                # shape = (self.engine.max_batch_size, ) + tuple(self.engine.get_binding_shape(binding))
                shape = tuple(self.engine.get_binding_shape(binding))
                dtype = torch.float32
                device = torch.device('cuda')
                # temp_tensor = torch.empty(size=shape, dtype=dtype, device=device)
                temp_tensor = torch.ones(size=shape, dtype=dtype, device=device)

                # Append the device buffer to device bindings.
                bindings.append(int(temp_tensor.data_ptr()))
                # Append to the appropriate list.
                if self.engine.binding_is_input(binding):
                    inputs.append(temp_tensor)
                else:
                    outputs.append(temp_tensor)
        return inputs, outputs, bindings




    def do_inference_v2(self):
        # Run inference.
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.cuda_stream)
        # self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.cuda_stream)
        # self.context.execute(bindings=self.bindings)
        # Synchronize the stream
        # self.stream.synchronize()
        # torch.cuda.synchronize(torch.cuda.current_device())
        self.stream.synchronize()

        return self.outputs

    def __call__(self, input, batch_size=1):

        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)

        self.inputs[0].copy_(input)

        with torch.no_grad():
        # Run inference
           output = self.do_inference_v2()
        return output


def cal_time(model):
    shape = model.engine.get_binding_shape(0)
    data = np.random.randint(0, 255, (batch_size, *shape[1:])) / 255
    t1 = time.time()
    result = model(data)
    print(f"model inference cost : {time.time() - t1}")

if __name__ == "__main__":
    batch_size = 1
    # trt_engine_path = os.path.join("..", "models", "main.trt")
    # trt_engine_path = '/home/awifi/zzzj/Projects/FairMOT/models/onnx_engine/fairmot_dla34.engine'
    trt_engine_path = '/home/zzzj/Projects/models/onnx_engine/test.engine'
    # trt_engine_path = '/home/zzzj/Projects/models/onnx_engine/fairmot_dla34_DCN_TRT_mmcv.trt'
    # trt_engine_path = '/home/zzzj/Projects/models/onnx_engine/fairmot_lite.engine'
    # trt_engine_path = '/home/zzzj/Projects/models/onnx_engine/fairmot_dla34.trt'

    model = TRTModel(trt_engine_path)
    cal_time(model)

    # model = TRTModelTorch(trt_engine_path, input_names={'0'}, output_names={'698', '701', '704', '707'})
    # model = TRTModelTorchv2(trt_engine_path)
    # cal_time(model)


