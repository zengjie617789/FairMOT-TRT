from models.model import create_model, load_model, save_model
from types import MethodType
import torch.onnx as onnx
import torch
from torch.onnx import OperatorExportTypes
from collections import OrderedDict
import os

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



def onnx_check(onnx_path):
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)


forward = {'dla':pose_dla_forward,'dlav0':dlav0_forward,'resdcn':resnet_dcn_forward}

arch = 'dla_34'
heads = OrderedDict([('hm', 1), ('reg', 2), ('wh', 4), ('id', 128)])
head_conv = 256 if 'dla' in arch else 64
model = create_model(arch, heads, head_conv)
model.forward = MethodType(forward[arch.split('_')[0]], model)
parent_dir = '/home/awifi/zzzj/Projects/FairMOT/models/'
load_path = os.path.join(parent_dir, 'fairmot_dla34.pth')
save_path = os.path.join(parent_dir,  'onnx_engine', "fairmot_dla34.onnx")
print(f"load_path: {load_path} | save_path: {save_path}")


if isinstance(model, torch.nn.DataParallel):
    model = model.module
    print("model convert !!!")
model = load_model(model, load_path)

model.eval()
model.cuda()
# input = torch.zeros([1, 3, 512, 512]).cuda()
input = torch.randn((1, 3, 608, 1088)).cuda()

output = onnx.export(model, input, save_path, verbose=True, operator_export_type=OperatorExportTypes.ONNX)
# output = onnx.export(model, input, save_path, verbose=False, operator_export_type=OperatorExportTypes.ONNX, custom_opsets={"custom_domain": 12})
# 运行onnx的示例
# import onnxruntime as ort
# ort_session = ort.InferenceSession(save_path)
# onnx_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: x.cpu().numpy().astype(np.float32)})
#
# # 校验结果
# print(onnx_outputs)
