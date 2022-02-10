import time
import numpy as np
import sys
import os
lib_path = os.path.join( os.path.dirname(os.path.abspath(__file__)), '..')
print(lib_path)
sys.path.append(lib_path)
from tracker.engine_module import TRTModel


def run_inference(trt_file, num=1):
    # inputs = torch.rand(1, 3, 608, 1088).cuda()
    inputs = np.random.randn(1,3,608,1088)
    ## Run inference with TensorRT
    trt_model = TRTModel(trt_file)

    sum = []
    for i in range(num):
        # torch.cuda.synchronize()
        t1 = time.time()
        trt_outputs = trt_model(inputs)
        # torch.cuda.synchronize()
        # outputs = [trt_outputs['922'], trt_outputs['925'], trt_outputs['928'], trt_outputs['931']]
        # outputs = [trt_outputs['hm'], trt_outputs['reg'], trt_outputs['wh'], trt_outputs['id_feature']]
        time_cost = time.time() - t1
        sum.append(time_cost)
        print(f"{i} th inference cost: {time_cost}")
    print(f"average time: {np.mean(sum)}")

def main():
    trt_file = '/home/zzzj/Projects/models/onnx_engine/test.engine'
    run_inference(trt_file=trt_file)

if __name__ == '__main__':
    main()