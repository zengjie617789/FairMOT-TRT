from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts
import argparse


def main(data_root, det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    # result_root = os.path.join(data_root, '..', 'results', exp_name)
    # mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        result_filename = os.path.join(det_root, seq, '{}.txt'.format(seq))


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
    Evaluator.save_summary(summary, os.path.join(det_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    # opts().parser.add_argument('--save_name', default='MOT17_test_public_dla34')
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/home/awifi/data/datasets/MOT16/images/train/')
    ap.add_argument('--det_root', default='/home/awifi/zzzj/projects/FairMOT-original/FairMOT-original/demos/MOT16')
    ap.add_argument('--save_name', default='mot16')
    parser = ap.parse_args()

    # opt = opts().init()
    seqs_str = []
    for i in os.listdir(parser.det_root):
        if os.path.isdir(os.path.join(parser.det_root, i)):
            seqs_str.append(i)

    seqs = [seq.strip() for seq in seqs_str]

    main(data_root=parser.data_root,
         det_root=parser.det_root,
         seqs=seqs,
         exp_name=parser.save_name,
         show_image=False,
         save_images=False,
         save_videos=False)
