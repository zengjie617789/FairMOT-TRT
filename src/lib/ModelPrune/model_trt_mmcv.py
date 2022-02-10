import time

import torch
import onnx
import os
import numpy as np

from mmcv.tensorrt import (TRTWrapper, onnx2trt, save_trt_engine,
                                   is_tensorrt_plugin_loaded)

assert is_tensorrt_plugin_loaded(), 'Requires to complie TensorRT plugins in mmcv'

def gen_trt(onnx_file='sample.onnx', trt_file='sample.trt'):
    onnx_model = onnx.load(onnx_file)

    ## Model input
    inputs = torch.rand(1, 3, 608, 1088).cuda()
    ## Model input shape info
    opt_shape_dict = {
        'input.1': [list(inputs.shape),
                    list(inputs.shape),
                    list(inputs.shape)]
    }

    ## Create TensorRT engine
    max_workspace_size = 1 << 30
    trt_engine = onnx2trt(
        onnx_model,
        opt_shape_dict,
        max_workspace_size=max_workspace_size)

    ## Save TensorRT engine
    save_trt_engine(trt_engine, trt_file)

def run_inference(trt_file, num=1):
    inputs = torch.rand(1, 3, 608, 1088).cuda()
    ## Run inference with TensorRT
    # trt_model = TRTWrapper(trt_file, ['input.1'], ['922', '925', '928', '931'])
    trt_model = TRTWrapper(trt_file, ['input.1'], ['hm', 'reg', 'wh', 'id_feature'])

    sum = []
    for i in range(num):
        torch.cuda.synchronize()
        t1 = time.time()
        trt_outputs = trt_model({'input.1': inputs})
        torch.cuda.synchronize()
        # outputs = [trt_outputs['922'], trt_outputs['925'], trt_outputs['928'], trt_outputs['931']]
        outputs = [trt_outputs['hm'], trt_outputs['reg'], trt_outputs['wh'], trt_outputs['id_feature']]
        time_cost = time.time() - t1
        sum.append(time_cost)
        print(f"{i} th inference cost: {time_cost}")
    print(f"average time: {np.mean(sum)}")

def main():
    # onnx_name = 'fairmot_dla34_mmcv.onnx'
    # onnx_name = 'fairmot_dla34_mmcv_opt.onnx'
    # onnx_name = 'fairmot_dla34_whole_mmcv.onnx'
    onnx_name = 'fairmot_dla34_whole_mmcv_opt13.onnx'
    save_name = onnx_name.replace('.onnx', '.trt')

    # onnx_name = 'fairmot_dla34_whole_mmcv.onnx'
    # save_name = 'fairmot_dla34_whole_mmcv.trt'

    # onnx_name = 'yolo_lite_mmcv.onnx'
    # save_name = 'yolo_lite_mmcv.trt'

    # onnx_file = os.path.join('/home/zzzj/Projects/models/onnx_engine/', onnx_name)
    # trt_file = os.path.join('/home/zzzj/Projects/models/onnx_engine/', save_name)
    # gen_trt(onnx_file, trt_file)


    trt_file = os.path.join('/home/zzzj/Projects/models', 'test.engine')
    run_inference(trt_file, 100)

if __name__ == '__main__':
    main()