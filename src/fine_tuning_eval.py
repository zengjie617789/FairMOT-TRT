import os
import motmetrics as mm
import numpy as np
import torch
from tracking_utils.log import logger
from tracking_utils.evaluation import Evaluator
import subprocess

__all__ = ['get_idf1_result']

test_name=['output02','output04']
train_name = ['output01','output03','output05', 'output07', 'output09', 'output11']
names_dict = {'test': test_name, 'train': train_name}
def eval_in_finetuning(model_path, mode, model_mode, arch, reid_dim):
    model_name = os.path.split(model_path)[-1]
    load_model = os.path.join(model_path, 'model_for_eval.pth')
    # load_model = model
    output_path="/home/awifi/zzzj/projects/FairMOT/demos/ServicesHall/{}/serhall_nni/{}".format(model_mode, model_name)
    names = names_dict[mode]
    processes = []
    for i in names:
        python_bin = '/usr/local/miniconda3/envs/yolov5_crowdhuman/bin/python'
        cmd_string = f"{python_bin} demo.py  --load_model {load_model}\
                     --conf_thres 0.5 --reid_dim {reid_dim}   --arch {arch}\
                     --input-video /home/awifi/data/ServicesHallData/images/{mode}/{i}/img1\
                     --output-root {output_path}/{mode}\
                     --output-format text"

        print(cmd_string)
        # os.system(cmd_string[0])
        p = subprocess.Popen(cmd_string, shell=True)
        processes.append(p)
        # return_code = p.wait()
        # if return_code == 0:
        #     return
    for i in processes:
        i.wait()


def get_result(data_root, det_root=None, seqs=('MOT16-05',), exp_name='demo',):
    # result_root = os.path.join(data_root, '..', 'results', exp_name)
    # mkdir_if_missing(result_root)
    data_type = 'mot'
    # run tracking
    accs = []
    timer_avgs, timer_calls = [], []

    seqs_str = []
    for i in os.listdir(det_root):
        if os.path.isdir(os.path.join(det_root, i)):
            seqs_str.append(i)

    seqs = [seq.strip() for seq in seqs_str]

    for seq in seqs:
        result_filename = os.path.join(det_root, seq, '{}.txt'.format(seq))

        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    # logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    # print(strsummary)
    return summary['idf1']['OVERALL'], summary['mota']['OVERALL']
    # Evaluator.save_summary(summary, os.path.join(det_root, 'summary_{}.xlsx'.format(exp_name)))


def get_idf1_result(save_dir, mode, model_mode, arch, reid_dim):
    model_name = os.path.split(save_dir)[-1]
    eval_in_finetuning(save_dir, mode, model_mode, arch, reid_dim)
    data_root = f"/home/awifi/data/ServicesHallData/images/{mode}"
    det_root = f"/home/awifi/zzzj/projects/FairMOT/demos/ServicesHall/{model_mode}/serhall_nni/{model_name}/{mode}"
    idf1, mota = get_result(data_root, det_root)
    print(f"idf1: {idf1}")
    print(f"mota: {mota}")
    return mota

def main():
    model_name = 'serhall_yolos'
    save_dir = '/home/awifi/zzzj/projects/FairMOT/src/lib/../../exp/mot/serhall_yolos'
    save_dir = '/home/awifi/zzzj/projects/FairMOT/exp/mot/serhall_nni_ft/BadEB/'
    mode = 'test'
    model_mode = 'modified'
    # eval_in_finetuning('/home/awifi/zzzj/projects/FairMOT/exp/mot/serhall_yolos_fix', mode)
    arch = 'yolo_m'
    get_idf1_result(save_dir, mode, model_mode, arch, reid_dim=64)

    data_root = f"/home/awifi/data/ServicesHallData/images/{mode}"
    det_root = f"/home/awifi/zzzj/projects/FairMOT/demos/ServicesHall/{model_mode}/serhall_nni/{model_name}/{mode}"
    # get_result()

if __name__ == '__main__':
    main()
