from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker_TRT, JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

#--arch dla_34 --load_model /home/awifi/zzzj/projects/FairMOT/exp/mot/serhall_dla34_ft/model_30.pth --reid_dim 128  --conf_thres 0.5 --input-video /home/awifi/data/video/ServiceHallTestVideos/test01.mp4
#--arch yolo_m --load_model /home/awifi/zzzj/projects/FairMOT/exp/mot/serhall_yolov5m_ft_scale_data_mixup/model_30.pth --reid_dim 64  --conf_thres 0.5 --input-video /home/awifi/data/video/ServiceHallTestVideos/test01.mp4
def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


from collections import deque
pts = [deque(maxlen=30) for _ in range(9999)]
ids_state = dict.fromkeys(range(9999), False)
counter_memory = dict.fromkeys(range(10000), 0)

region = [{'x':693, 'y':831}, {'x':1122, 'y': 889}]   # for yuexian.mp4
# region = [{'x': 164, 'y': 412}, {'x': 1180, 'y': 412}]  # for MOT17-03.mp4
# region = [{'x':818, 'y':550}, {'x':1617, 'y': 550}]   # for MOT16-09.mp4
region_array = [(int(p['x'] * 100 / 100), int(p['y'] * 100 / 100)) for p in region]
task = 'COSSLINE'
# task = None
def tlwh_to_xyxy(bbox_tlwh, width, height):
    """
    TODO:
        Convert bbox from xtl_ytl_w_h to xc_yc_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    x, y, w, h = bbox_tlwh
    x1 = max(int(x), 0)
    x2 = min(int(x + w), width - 1)
    y1 = max(int(y), 0)
    y2 = min(int(y + h), height - 1)
    return x1, y1, x2, y2

def cross_line_detect_fast(img, boxes, line_arr, id, direction='up', angle=150):
    height, width, _ = img.shape
    boxes = tlwh_to_xyxy(boxes, width, height)
    global counter_memory
    (m0, n0), (m1, n1) = line_arr[0], line_arr[1]

    x1, y1, x2, y2 = boxes
    x_center, y_center = ((round((x1 + x2) / 2), round((y1 + y2) / 2)))
    dis = 50

    line_margin = np.array([[m0+dis*math.cos(math.radians(180+angle)), n0+dis*math.sin(math.radians(180+angle))],
                            [m1+dis*math.cos(math.radians(180+angle)), n1+dis*math.sin(math.radians(180+angle))],
                            [m1+dis*math.cos(math.radians(angle)), n1 + dis * math.sin(math.radians(angle))],
                            [m0+dis*math.cos(math.radians(angle)), n0 + dis * math.sin(math.radians(angle))]
                           ], dtype=np.int32)

    directions = {'up': (x_center, max(y1, y2)),
                  'down': (x_center, max(y1, y2)),
                  'left': (min(x1, x2), y_center),
                  'right': (max(x1, x2), y_center)}
    # center_ = directions[direction]
    center_ = (x_center, y_center)

    cv2.line(img, (m0, n0), (m1,n1), color=(0,0,255), thickness=2)
    cv2.polylines(img, [line_margin], isClosed=True, color=(0,0,255))
    inside = cv2.pointPolygonTest(line_margin, center_, False)
    if inside > 0 and counter_memory[id] != 1:
        pts[id].append(center_)
        ids_state[id] = True

    if inside <= 0 and ids_state[id]==True:
        ids_state[id] = False
        if len(pts[id])>1:
            if direction == "up":
                if center_[1] < np.mean([j for i,j in pts[id]]):  #image coordinate
                    counter_memory[id] = 1
                    alarm_num = 1
            elif direction == "down":
                if center_[1] > np.mean([j for i, j in pts[id]]):
                    counter_memory[id] = 1
                    alarm_num = 1
            else:
                pass

    text_scale = max(1, img.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(img.shape[1] / 500.))
    intbox = list(boxes)
    obj_id = int(id)
    id_text = '{}'.format(int(obj_id))
    _line_thickness = 1 if obj_id <= 0 else line_thickness
    # color = get_color(abs(obj_id))
    color = (0, 0, 255) if counter_memory[id] else (255, 0, 0)
    cv2.rectangle(img, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
    cv2.putText(img, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale,
                (0, 0, 255), thickness=text_thickness)
    cv2.circle(img, (int(x_center), int(y_center)), 2, color, thickness=2)

    return counter_memory[id], img




def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=False, frame_rate=30, use_cuda=True, trt_file=None, type='MMCV'):
    if save_dir:
        mkdir_if_missing(save_dir)

    if trt_file:
        tracker = JDETracker_TRT(opt, frame_rate=frame_rate, trt_file=trt_file, type=type)
    else:
        tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    cross_line_status = False

    for i, (path, img, img0) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # logger.info('Processing frame {} ({:.2f} fps {}) '.format(frame_id, 1. / max(1e-5, timer.average_time), timer.total_time))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        # online_im = img0
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            fea = t.features
            # vertical = tlwh[2] / tlwh[3] > 1.6
            # if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)

        if len(online_targets) == 0:
            print(f"{'#'*10}detect none!")
            online_im = img0

        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, online_scores,frame_id=frame_id, fps=1. / timer.average_time)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        if show_image:
            cv2.imshow('online_im', online_im)
            # cv2.waitKey(1)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'
    print(f"{'#'*20}reading {seqs} now")

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        print(osp.join(data_root, seq, 'img1'))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        # seqs_str = "MOT16-13"
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]
    print(f"data_root：{data_root}")
    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT17_test_public_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)
