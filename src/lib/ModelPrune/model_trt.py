import time

from models.model import create_model, load_model, save_model
from types import MethodType
import torch
import torch.onnx as onnx
from torch.onnx import OperatorExportTypes
from collections import OrderedDict
import os
import onnxruntime as ort
import numpy as np
from models.decode import mot_decode
from models.utils import _tranpose_and_gather_feat, _sigmoid
import torch.nn.functional as F
import glob
# import mmcv
# from mmcv.onnx import register_extra_symbolics
# opset_version = 11
# register_extra_symbolics(opset_version)

## onnx is not support dict return value
## for dla34
def pose_dla_forward(self, x):
    x = self.base(x)
    x = self.dla_up(x)
    y = []
    for i in range(self.last_level - self.first_level):
        y.append(x[i].clone())
    self.ida_up(y, 0, len(y))
    ret = []  ## change dict to list
    for head in self.heads:
        print(f"the head name is : {head}")
        ret.append(self.__getattr__(head)(y[-1]))
    return ret

# def pose_dla_forward(self, x):
#     x = self.base(x)
#     x = self.dla_up(x)
#     y = []
#     for i in range(self.last_level - self.first_level):
#         y.append(x[i].clone())
#     self.ida_up(y, 0, len(y))
#     results = []  ## change dict to list
#     ret = {}
#
#     for head in self.heads:
#         print(f"the head name is : {head}")
#         ret[head] = self.__getattr__(head)(y[-1])
#
#     hm, reg, id, wh = ret['hm'], ret['reg'], ret['id'], ret['wh']
#     hm = hm.sigmoid_()
#     # id_feature = F.normalize(id, dim=1)
#     eps_min = 1e-12
#     eps_max = 1
#     id_feature = id.norm(p=2, dim=1, keepdim=True).clamp(min=eps_min, max=eps_max).expand_as(id)
#
#     dets, inds = mot_decode(hm, wh, reg=reg, ltrb=True, K=1000)
#     id_feature = _tranpose_and_gather_feat(id_feature, inds)
#     id_feature = id_feature.squeeze(0)
#
#     results = [dets, id_feature]
#     return results


## for dla34v0
def dlav0_forward(self, x):
    x = self.base(x)
    x = self.dla_up(x[self.first_level:])
    # x = self.fc(x)
    # y = self.softmax(self.up(x))
    ret = []  ## change dict to list
    for head in self.heads:
        ret.append(self.__getattr__(head)(x))
    return ret
## for resdcn
def resnet_dcn_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.deconv_layers(x)
    ret = []  ## change dict to list
    for head in self.heads:
        ret.append(self.__getattr__(head)(x))
    return ret

# def yolo_dcn_forward(self, x):
#     x = self.backbone(x)
#     ret = []
#     for head in self.heads:
#         ret.append(self.__getattr__(head)(x))
#     return ret


def yolo_dcn_forward(self, x):
    x = self.backbone(x)

    ret = {}

    for head in self.heads:
        print(f"the head name is : {head}")
        ret[head] = self.__getattr__(head)(x)

    hm, reg, id, wh = ret['hm'], ret['reg'], ret['id'], ret['wh']
    hm = hm.sigmoid_()
    # id_feature = F.normalize(id, dim=1)
    eps_min = 1e-12
    eps_max = 1
    id_feature = id.norm(p=2, dim=1, keepdim=True).clamp(min=eps_min, max=eps_max).expand_as(id)

    dets, inds = mot_decode(hm, wh, reg=reg, ltrb=True, K=1000)
    id_feature = _tranpose_and_gather_feat(id_feature, inds)
    id_feature = id_feature.squeeze(0)

    results = [dets, id_feature]
    return results


def onnx_check(onnx_path):
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)



def onnx_load_model(arch, load_path):
    forward = {'dla':pose_dla_forward,'dlav0':dlav0_forward,'resdcn':resnet_dcn_forward, 'yolo': yolo_dcn_forward}
    # head_conv = 256 if 'dla' in arch else 64
    head_conv = 256
    if arch == 'dla_34':
        heads = OrderedDict([('hm', 1), ('reg', 2), ('wh', 4), ('id', 128)])

    else:
        heads = OrderedDict([('hm', 1), ('reg', 2), ('wh', 4), ('id', 64)])
    model = create_model(arch, heads, head_conv)
    model.forward = MethodType(forward[arch.split('_')[0]], model)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        print("model convert !!!")

    # state_dict = model.state_dict()
    # new_state_dict = {}
    # for k in state_dict:
    #     if 'conv_offset' in k:
    #         new_key = k.replace('conv_offset', 'conv_offset_mask')
    #     else:
    #         new_key = k
    #     new_state_dict[new_key] = state_dict[k]
    #
    #
    # model.load_state_dict(new_state_dict)
    # model.state_dict = new_state_dict
    #
    model = load_model(model, load_path)
    model.cuda()
    model.eval()

    return model

def export_onnx(input, load_model, export_onnx=True, onnx_name='test.onnx', arch='dla_34', parent_dir = '/home/zzzj/Projects/models/'):
    # arch = 'dla_34'
    model = onnx_load_model(arch, load_model)
    # input = torch.randn((1, 3, 608, 1088)).cuda()
    if export_onnx:
        save_path = os.path.join(parent_dir, 'onnx_engine', onnx_name)
        # res = model(input)
        # onnx.export(model, input, save_path, verbose=True, operator_export_type=OperatorExportTypes.ONNX, opset_version=11)
        # output = onnx.export(model, input, save_path, verbose=False, input_names=['input.1'], output_names=['dets', 'id_feature'],operator_export_type=OperatorExportTypes.ONNX, opset_version=12,custom_opsets={"custom_domain":13})
        onnx.export(model, input, save_path, verbose=False,  opset_version=11, input_names=['input.1'], output_names=['hm', 'reg', 'wh', 'id_feature'])
        # onnx.export(model, input, save_path, verbose=False,  opset_version=13, input_names=['input.1'], output_names=['dets', 'id_feature'])
        print("Completed onnx export ...")
    else:
        return model


def find_register_library():
    build_file = os.path.join(os.path.dirname(__file__), "..", '..', 'mmcv', '*', '_ext_ort.*.so')
    ort_custom_op_path = glob.glob(build_file)[0]
    assert os.path.exists(ort_custom_op_path)
    return ort_custom_op_path

def validate_onnx(onnx_file, input_shape):
    #运行onnx的示例,校验结果
    from mmcv.ops import get_onnxruntime_op_path

    # ort_custom_op_path = get_onnxruntime_op_path()
    ## exported ONNX model with custom operators
    ort_custom_op_path = find_register_library()
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(ort_custom_op_path)
    a, b, c, d = input_shape
    input_data = np.random.randn(a, b, c, d).astype(np.float32)
    sess = ort.InferenceSession(onnx_file, session_options)
    t1 = time.time()
    onnx_results = sess.run(None, {'input.1' : input_data})
    print(f"inference cost: {time.time() - t1 }")
    # print(onnx_results)

def edit_onnx_model(onnx_path, modified=True):
    import onnx
    if modified:
        onnx_model = onnx.load(onnx_path)
        graph = onnx_model.graph
        for node in graph.node:
            if 'Clip' in str(node):
                print(node.name)
                print(node.inputs)

def onnx_optimizer(onnx_file):
    import onnx
    import onnxoptimizer
    from onnxsim import simplify
    onnx_name, suffix = onnx_file.split('.')
    new_onnx_name = onnx_name + '_opt'
    new_onnx_file = new_onnx_name + '.' + suffix
    model = onnx.load(onnx_file)
    library_path = find_register_library()
    # model = onnxoptimizer.optimize(model, ['fuse_bn_into_conv'])
    model = onnxoptimizer.optimize(model)
    # new_model = onnxoptimizer.optimize(model)
    model_sim, check = simplify(model, custom_lib=library_path)
    assert check, "Simplified ONNX model could not be validated"
    with open(new_onnx_file, 'wb') as f:
        f.write(model_sim.SerializeToString())

def main():
    input = torch.rand((1, 3, 608, 1088)).cuda()
    onnx_name = 'fairmot_dla34_rename.onnx'
    parent_dir = '/home/zzzj/Projects/models/onnx_engine/'
    onnx_path = os.path.join(parent_dir, onnx_name)

    # onnx_name = 'fairmot_dla34_whole_mmcv_opt12.onnx'
    # onnx_name = 'fairmot_dla34_whole_opt12.onnx'
    model_path = '/home/zzzj/Projects/models/fairmot_dla34.pth'
    export_onnx(input, model_path, onnx_name=onnx_name)



    # onnx_name = 'yolo_lite_whole.onnx'
    # model_path = '/home/zzzj/Projects/models/fairmot_lite.pth'
    # export_onnx(input, model_path, onnx_name=onnx_name, arch='yolo_s')
    # onnx_path = os.path.join('/home/zzzj/Projects/models/onnx_engine/', onnx_name)
    # validate_onnx(onnx_path, input_shape=(1, 3, 608, 1088))

    # import onnx
    parent_dir = '/home/zzzj/Projects/models/'
    onnx_path = os.path.join(parent_dir, 'onnx_engine', onnx_name)
    # # onnx.checker.check_model(onnx_path)
    #
    # onnx_optimizer(onnx_path)
#

if __name__ == '__main__':
    main()
