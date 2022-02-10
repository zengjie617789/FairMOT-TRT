from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
# import datasets.dataset.jde as datasets
import datasets.dataset.jde_yolov5 as datasets
from track import eval_seq
import time

import utils.print_configs

from logger import Logger
# --arch yolo_m --load_model /home/awifi/zzzj/projects/FairMOT/exp/mot/serhall_yolov5m_ft_scale_data_mixup/model_last_best.pth --reid_dim 64  --conf_thres 0.4 --input-video /home/awifi/data/video/ServiceHallTestVideos/test02.mp4
# --arch yolo_s --load_model /home/awifi/zzzj/projects/FairMOT/models/fairmot_lite.pth --reid_dim 64  --conf_thres 0.4 --input-video /home/awifi/data/video/ServiceHallTestVideos/test02.mp4

#--arch yolo_m --load_model /home/awifi/zzzj/projects/FairMOT/exp/mot/serhall_yolov5m_ft_scale_data_reid_dim_256/model_80.pth --reid_dim 256  --conf_thres 0.5 --input-video /home/awifi/data/video/ServiceHallTestVideos/test02.mp4
#--arch dla_34 --load_model /home/awifi/zzzj/projects/FairMOT/exp/mot/serhall_dla34_ft/model_last_best.pth --reid_dim 128  --conf_thres 0.4 --input-video /home/awifi/data/video/ServiceHallTestVideos/test02.mp4
#--arch dla_34 --load_model /home/awifi/zzzj/projects/FairMOT/exp/mot/serhall_dla34_ft_data_modified/model_100.pth --reid_dim 128  --conf_thres 0.5 --input-video /home/awifi/data/video/ServiceHallTestVideos/test02.mp4
logger.setLevel(logging.INFO)

def get_time():
    from time import time, strftime, localtime
    return strftime("%Y-%m-%d-%H:%M:%S", localtime())

def demo(opt, trt_file=None, type='MMCV'):
    result_root = opt.output_root if opt.output_root != '' else '.' # ../demos
    video_format = ['mp4', 'ts', 'avi']

    if opt.input_video.split('.')[-1] in video_format:
        dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
        frame_rate = dataloader.frame_rate
        name = opt.input_video.split('/')[-1].split('.')[0]

    else:
        dataloader = datasets.LoadImages(opt.input_video, opt.img_size)
        frame_rate = 40
        name = os.path.split(os.path.split(opt.input_video)[0])[-1]

    result_root = result_root + '/' + name
    mkdir_if_missing(result_root)
    logger.info(f"result_root is {result_root}")
    logger.info('Starting tracking...')


    result_filename = os.path.join(result_root, f'{name}.txt')

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')

    t1 = time.time()
    frame_id, ave_time, calls = eval_seq(opt, dataloader, 'mot', result_filename, save_dir=frame_dir, show_image=False, frame_rate=frame_rate, use_cuda=opt.gpus!=[-1], trt_file=trt_file, type=type)
    print(f"frame_id:{frame_id}, ave_time:{ave_time}, calls: {calls}")
    print(f"average time cost:{(time.time() - t1)/int(calls)}")


    # logger_class = Logger(opt)
    print("start save video")

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, name+'-results.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    opt = opts().init()
    # trt_file = '/home/awifi/zzzj/Projects/FairMOT/models/onnx_engine/fairmot_dla34.trt'
    trt_file = '/home/zzzj/Projects/models/onnx_engine/fairmot_dla34_rename.engine' # 192.168.152.152
    # trt_file = '/home/zzzj/Projects/models/onnx_engine/fairmot_dla34_mmcv.trt'
    # trt_file = '/home/zzzj/Projects/trt-fairmot/TensorRT_ONNX_impl/fairmot.trt'
    # trt_file = '/home/zzzj/Projects/models/onnx_engine/yolo_lite_mmcv.trt'

    # demo(opt, trt_file=trt_file, type='MMCV')
    demo(opt, trt_file=trt_file, type='tensorrt')
    # demo(opt)
