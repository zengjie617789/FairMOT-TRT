from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import _init_paths
import os
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, LevelPruner
from nni.compression.pytorch import ModelSpeedup
import time
import torch.nn as nn
from nni.common.graph_utils import build_module_graph
from models.yolo import PoseYOLOv5m, PoseYOLOv5s, initialize_weights
import datasets.dataset.jde_yolov5 as datasets
import argparse
import numpy as np
import torch.nn.functional as F
from models.decode import mot_decode
from utils.post_process import ctdet_post_process, ctdet_post_process_cuda
from models.utils import _tranpose_and_gather_feat
import cv2

class ModifiedModel(nn.Module):
    def __init__(self, model_path, base_model=None):
        super(ModifiedModel, self).__init__()
        self.model = get_model(model_path)
        self.base_model = base_model

    def forward(self, x):
        if self.base_model is not None:
            self.model.base = self.base_model
            # self.model.backbone = self.base_model
        output = self.model(x)[0]
        # output = outputs[0]
        return output['hm'], output['wh'], output['id'], output['reg']


class WrappedModel(nn.Module):
    def __init__(self, input_model):
        super(WrappedModel, self).__init__()
        self.input_model = input_model

    def forward(self, x):
        output = self.input_model(x)[0]
        return output
        # return output['hm'], output['wh'], output['id'], output['reg']


def get_model(model_path):
    print('Creating model...')

    arch = 'dla_34'
    head_conv = 256
    heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}

    model = create_model(arch, heads, head_conv)
    model = load_model(model, model_path)
    model.eval()
    return model

def prune_model():
    model_path = '../../../models/fairmot_dla34.pth'
    model = get_model(model_path)
    modified_model = ModifiedModel(model_path)
    model = modified_model.model
    [print('op_name: {}\nop_type: {}\n'.format(name, type(module))) for name, module in model.named_modules()]
    config_list = [{
        'sparsity': 0.5,
        'op_types': ['Conv2d'],
    }]

    print(f"before pruning size: {model.hm[0].weight.data.size()}")
    print(f"before pruning data: {model.hm[0].weight.data}")
    pruner = L1FilterPruner(modified_model, config_list)
    pruner.compress()
    # print(f"after pruning data: {model.hm[0].weight.data}")
    # print(f"pruning data mask: {model.hm[0].weight_mask}")
    # model(torch.randn(1,3,28,28))
    # print(f"after pruning data:{model.hm[0].module.weight.data}")
    # pruner.export_model(model_path='../models/pruned_fairmot_dla34.pth', mask_path='../models/mask_fairmot_dla34.pth')

def test_model():
    device = 'cuda'
    model_path = '../../../models/fairmot_dla34.pth'

    model = ModifiedModel(model_path).to(device)

    model = WrappedModel(get_yolo_model('yolov5s'))

    dummy_input = torch.randn((1, 3, 1088, 608)).to(device)
    torch_graph = build_module_graph(model, dummy_input)
    # print(f"name_to_node: \n{torch_graph.name_to_node}")
    # print(f"input_to_node: \n{torch_graph.input_to_node}")


    # torch_graph.unpack_manually()

    # graph_input = []
    # for name, nodeio in torch_graph.nodes_py.nodes_io.items():
    #     if nodeio.input_or_output == "input":
    #         graph_input.append((name, nodeio))
    #     print(f"name:{name}  nodeio:{nodeio}")
    # print(graph_input)

    # for node in torch_graph.nodes_py.nodes_op:
    #     print(node)

    # graph_input_trace = torch_graph.trace.graph
    # internal_result = {}
    # for graph_input in graph_input_trace.inputs():
    #     if graph_input.type().kind() == 'ClassType':
    #         internal_result[graph_input.debugName()] = model
    #         break
    # print(f"internal_result is :{internal_result.keys()}")

    out_degree = {}
    in_degree = {}
    zero_degree = {}
    for node in torch_graph.nodes_py.nodes_op:
        successors = torch_graph.find_successors(node.unique_name)
        out_degree[node.unique_name] = len(successors)
        predecessors = torch_graph.find_predecessors(node.unique_name)
        in_degree[node.unique_name] = len(predecessors)
        if in_degree[node.unique_name] == 0:
            zero_degree[node.unique_name] = 0
    # print(f"out_degree : {out_degree}")
    # print(f"in_degree : {in_degree}")
    print(f"zero degree : {zero_degree}")

    # for _input in name_to_node[unique_name].inputs:
    #     if not _input in self.output_to_node:
    #         _logger.debug("cannot find node with %s as its output", _input)
    #     else:
    #         node_py = self.output_to_node[_input]
    #         predecessors.append(node_py.unique_name)

    name_to_node = torch_graph.name_to_node
    output_to_node = torch_graph.output_to_node
    input_to_node = torch_graph.input_to_node
    print(f"output_to_node: \n{output_to_node.keys()}")
    print(f"intput_to_node: \n{input_to_node.keys()}")

    # key = 'model.dla_up.ida_0.proj_1.actf.0'
    # print(f"{key} input : {name_to_node[key].inputs}")
    # print(output_to_node['input.221'])
    # print(output_to_node['out.25'])
    # print(output_to_node['input.225'])


def test_load_model():
    pretrained = '../models/yolov5m.pt'

    ckpt = torch.load(pretrained)  # load checkpoint
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    print(ckpt.keys())


def get_yolo_model(name, device='cuda', trt_compress=False):
    # name = 'yolov5m'
    # # name = 'yolov5s'
    head_conv = 256
    heads = {'hm': 1, 'wh': 4, 'id': 64, 'reg': 2}
    if trt_compress:
        heads = {'hm': 1, 'wh': 4, 'id': 256, 'reg': 2}

    config_file = os.path.join(os.path.expanduser("~/zzzj/projects/FairMOT"),
                               f'src/lib/models/networks/config/{name}.yaml')
    pretrained = os.path.join(os.path.expanduser("~/zzzj/projects/FairMOT"), f'models/{name}.pt')
    model = PoseYOLOv5s(heads, config_file) if name == 'yolov5s' else PoseYOLOv5m(heads, config_file, trt_compress)
    if os.path.exists(pretrained):
        initialize_weights(model, pretrained)
    return model.to(device)

def speedup_model(model_path):
    # model_path = '../models/fairmot_dla34.pth'

    prun_model_path = '../models/pruned'
    device = 'cuda'
    # device = 'cpu'
    dummy_input = torch.randn((1, 3, 1088, 608)).to(device)
    # s1 = time.time()
    # # model(dummy_input)
    modified_model = ModifiedModel(model_path).to(device)
    # modified_model(dummy_input)
    # print(f"cost time: {time.time() - s1}")


    config_list = [{
        'sparsity': 0.5,
        'op_types': ['Conv2d'],
        # 'op_types': ['default'],
    }]

    arch = 'dla_34'
    input_model = modified_model.model.base
    # input_model = torch.nn.DataParallel(input_model).eval()

    pruner = L1FilterPruner(input_model, config_list)
    # # pruner = LevelPruner(modified_model, config_list)
    pruner.compress()
    pruner._unwrap_model()
    pruner.export_model(model_path=f'{prun_model_path}/model_{arch}_test.pth', mask_path=f'{prun_model_path}/mask_{arch}_test.pth')
    # print(f"model: {modified_model}")
    # masks_data = torch.load(masks_file)
    s1 = time.time()
    modified_model(dummy_input)
    print(f"cost time: {time.time() - s1}")
    masks_file = f"{prun_model_path}/mask_{arch}_test.pth"
    # model = get_model(model_path).to(device)


    input_model.eval()
    # with torch.no_grad():
    m_speedup = ModelSpeedup(input_model, (dummy_input,), masks_file)
    m_speedup.speedup_model()
    s1 = time.time()

    with torch.no_grad():
        modified_model = ModifiedModel(model_path, base_model=input_model).to(device)
        modified_model.eval()
        modified_model(dummy_input)

    print(f"cost time: {time.time() - s1}")

def speedup_yolo_model(arch, device='cuda'):

    prun_model_path = '../../models/pruned'


    model = get_yolo_model(arch, device)
    wrapper_model = WrappedModel(model)

    # dummy_input = torch.randn((1, 3, 1088, 608)).to(device)
    dummy_input = torch.randn((1, 3, 608, 1088)).to(device)
    masks_file = f"{prun_model_path}/mask_{arch}_test.pth"
    pruned_model_file = f'{prun_model_path}/model_{arch}_test.pth'

    config_list = [{
        'sparsity': 0.5,
        'op_types': ['Conv2d'],
        # 'op_types': ['default'],
    }]

    pruner = L1FilterPruner(wrapper_model, config_list)
    # # pruner = LevelPruner(modified_model, config_list)
    pruner.compress()
    pruner._unwrap_model()
    pruner.export_model(model_path=pruned_model_file, mask_path=masks_file)

    m_speedup = ModelSpeedup(wrapper_model, dummy_input, masks_file)
    m_speedup.speedup_model()


def export_onnx(model_path, device='cuda', save_path = '../models/pruned/'):
    model_name = os.path.split(model_path)[-1]
    save_path = os.path.join(save_path, model_name+'.onnx')
    dummpy_input = torch.randn((3, 608, 1088)).to(device)
    model = get_yolo_model('yolov5s')
    # model = torch.load(model_path)
    torch.onnx.export(model, (dummpy_input), save_path, opset_version=11)


def get_model_layers(mode):
    model = get_yolo_model('yolov5s')

    if mode == 'children':
        print(f"length: {len(list(model.children()))}")
        for i in model.children():
            print(i)

    elif mode == 'modules':
        print(f"length: {len(list(model.modules()))}")
        for i in model.modules():
            print(i)
    else:
        print("mode is wrong !!!")



def decode(output, im_blob, img0):
    num_classes = 1
    max_per_image = K = 500
    conf_thres = 0.5
    down_ratio = 4
    width = img0.shape[1]
    height = img0.shape[0]
    inp_height = im_blob.shape[2]
    inp_width = im_blob.shape[3]

    c = np.array([width / 2., height / 2.], dtype=np.float32)  # center point
    s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
    meta = {'c': c, 's': s,
            'out_height': inp_height // down_ratio,
            'out_width': inp_width // down_ratio}

    def post_process(dets, meta):
        dets = dets.detach().cpu().numpy()

        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], num_classes)
        for j in range(1, num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(detections):
        results = {}
        for j in range(1, num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, num_classes + 1)])
        if len(scores) > max_per_image:
            kth = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results





    with torch.no_grad():
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)

        reg = output['reg']
        dets, inds = mot_decode(hm, wh, reg=reg, ltrb=True, K=K)
        id_feature = _tranpose_and_gather_feat(id_feature, inds)
        id_feature = id_feature.squeeze(0)
        id_feature = id_feature.cpu().numpy()

    dets = post_process(dets, meta)  # 12fps
    dets = merge_outputs([dets])[1]

    remain_inds = dets[:, 4] > conf_thres
    dets = dets[remain_inds]
    id_feature = id_feature[remain_inds]
    return dets, id_feature
    # print(id_feature.shape)

def draw_picture(img, dets):
        img = img.copy()
        for det in dets:
            # x, y, w, h, score = det
            xyxy, score = det[:-1], det[-1]
            x0, y0, x1, y1 = map(int, xyxy)
            # x0, y0, x1, y1 = int(x)-int(w)/2, int(y)-int(h)/2, int(x)+int(w)/2, int(y)+int(h)/2
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
        return img

def get_model_outputs(model_path, target_path, device='cuda'):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_video', required=True, type=str)
    ap.add_argument("--img_size", default=(1088, 608))
    args = ap.parse_args()
    model = get_model(model_path).to(device)
    dataloader = datasets.LoadImages(args.input_video, args.img_size)
    img_path = os.path.join(target_path, 'img')
    feature_path = os.path.join(target_path, 'feature_data')
    features_list = []
    for path in [img_path, feature_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    for i, (path, img, img0) in enumerate(dataloader):
        input_tensor = torch.from_numpy(img).cuda().unsqueeze(0)
        outputs = model(input_tensor)
        if device == 'cuda':
            im_blob = torch.from_numpy(img).cuda().unsqueeze(1)
        else:
            im_blob = torch.from_numpy(img).unsqueeze(1)

        dets, features = decode(outputs[0], im_blob, img0)

        img = draw_picture(img0, dets)
        cv2.imwrite(os.path.join(img_path, "{:05d}.jpg".format(i)), img)
        # features_list.append(features)
        features_list.extend(features)
        # np.save(os.path.join(feature_path, '{:05d}.npy'.format(i)), features)
        # print(outputs[0]['id'].shape)
        # if i > 10:
        #     break
    np.save(os.path.join(feature_path, 'total.npy'), np.array(features_list))
    return

def main():
    # prune_model()
    model_path = '../models/yolov5m.pt'
    model_path = '../../../models/fairmot_lite.pth'

    # speedup_model(model_path)
    # speedup_yolo_model('yolov5s')
    # test_model()
    # test_load_model()

    # export_onnx(model_path)

    # model = get_yolo_model('yolov5m')
    # # print(f"{'#'*20}")
    # load_model(model, '../models/crowdhuman_yolov5m.pt')

    # get_model_layers("children")
    model_path = '/home/awifi/zzzj/projects/FairMOT/exp/mot/serhall_dla34_ft_data_modified/model_100.pth'
    # model_path = '/home/awifi/zzzj/projects/FairMOT/models/fairmot_dla34.pth'
    target_path = './model_outputs'
    get_model_outputs(model_path, target_path)

if __name__ == '__main__':
    # opt = opts().parse()
    main()
