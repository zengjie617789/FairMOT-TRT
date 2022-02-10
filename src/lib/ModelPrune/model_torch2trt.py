import tensorrt as trt
from torch2trt import torch2trt, TRTModule
from model_pruning import get_yolo_model
import torch
import os
from models.model import load_model
import sys
import time
from models.decode import mot_decode
import torch.nn.functional as F
import numpy as np
import cv2
from tracking_utils.visualization import get_color
import datasets.dataset.jde_yolov5 as datasets
from src.lib.models.networks.pose_dla_dcn import get_pose_net as get_pose_net_dla
from ModelPrune.model_pruning import decode, draw_picture

this_dir = os.path.dirname(__file__)
work_path = ''
save_path = os.path.join(this_dir, 'models')
if not os.path.exists(save_path):
        os.mkdir(save_path)

def model_trt(load_model_path, model_name='yolo'):
        device = 'cuda'
        workspace = 32
        if model_name == 'yolo':
            model = get_yolo_model('yolov5m', trt_compress=True)
        elif model_name == 'dla_34':
            heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}

            model = get_pose_net_dla(num_layers=34, heads=heads, trt_compress=True).to(device)
        else:
            raise Exception("not contain the model !!!")

        model = load_model(model, load_model_path)
        model.eval()

        # dummy_input = torch.randn((1, 3, 1088, 608)).to(device)
        dummy_input = torch.randn((1, 3, 608, 1088)).to(device)
        model_trt = torch2trt(
                model,
                [dummy_input],
                fp16_mode=True,
                log_level=trt.Logger.INFO,
                max_workspace_size=(1 << workspace),
                max_batch_size=8,
            )
        torch.save(model_trt.state_dict(), os.path.join(save_path, "{}_trt.pth".format(model_name)))

def load_model_trt(input_path, trt_file, device='cuda'):

        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(trt_file))
        dummy_input = torch.randn((1, 3, 1088, 608)).to(device)
        t1 = time.time()
        outputs = model_trt(dummy_input)
        dataloader = datasets.LoadImages(input_path)
        decode_outputs(dataloader, model_trt)
        t2 = time.time()
        print(f"time cost: {t2-t1}s")

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

def decode_outputs(dataloader, model, save_img=True, save_path='./outputs'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i, (path, img, img0) in enumerate(dataloader):
        im_blob = torch.from_numpy(img).cuda().unsqueeze(0)
        outputs = model(im_blob)
        output = {}
        output['hm'], output['wh'], output['id'], output['reg'] = outputs
        # print(f"model cost time :{time.time()-start}")
        dets, features = decode(output, im_blob, img0)
        # hm = output['hm'].sigmoid_()
        # wh = output['wh']
        # reg = output['reg']
        # id_feature = output['id']
        # id_feature = F.normalize(id_feature, dim=1)
        # dets, inds = mot_decode(hm, wh, reg=reg, ltrb=True, K=100)
        im = draw_picture(img0, dets)
        # im = plot_tracking(img0, dets, inds, frame_id=i)
        if save_img:
            cv2.imwrite(os.path.join(save_path, "{:05d}.jpg".format(i)), im)

def main():
    input_path = '/home/awifi/data/ServicesHallData/images/test/output02/img1'
    # load_model_trt(input_path, './models/dla_34_trt.pth')
    # load_model_path = '/home/awifi/zzzj/projects/FairMOT/exp/mot/serhall_yolov5m_ft_scale_data_mixup_reid_dim/model_100.pth'
    load_model_path = '/home/awifi/zzzj/projects/FairMOT/exp/mot/serhall_dla34_ft/model_30.pth'
    model_trt(load_model_path, model_name='dla_34')

if __name__ == '__main__':
    main()