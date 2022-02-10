import os.path as osp
import os
import numpy as np
import glob

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

'''
generate txt label files under labels_with_ids dir
root_path = "/home/awifi/data"
'''
def gen_labels(root_path, mode='train'):
    ROOT_PATH = os.path.expanduser(root_path)
    seq_root = os.path.join(ROOT_PATH, 'ServicesHallData/images', mode)
    label_root = os.path.join(ROOT_PATH, 'ServicesHallData/labels_with_ids', mode)
    if not os.path.exists(label_root):
        mkdirs(label_root)

    datas = []
    with open(os.path.join(seq_root, 'data_info.txt'), 'r') as f:
        content = f.readlines()
        for line in content:
            datas.append(line.strip())
        print(datas)

    seqs = [s for s in os.listdir(seq_root) if os.path.isdir(os.path.join(seq_root, s))]
    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        if seq in datas or seq.endswith('txt'):
            continue
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = osp.join(label_root, seq, 'img1')
        mkdirs(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, _ in gt:
            if (x or y or w or h) is None:
                print(x,y,w,h)
                continue
            # if mark == 0 or not label == 1:
            #     continue
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:05d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)

'''
generate custom train file under src/data dir
root_path = '/home/awifi/data/'
data_path = 'ServicesHallData/images/train/'
name = 'serhall.train'
'''
def gen_data_path_custom(root_path, data_path, name):
    real_path = os.path.join(root_path, data_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) if not s.endswith('txt') ]
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    datas = []
    with open(os.path.join(real_path, 'data_info.txt'), 'r') as f:
        content = f.readlines()
        for line in content:
            datas.append(line.strip())
        print(datas)

    with open(os.path.join(save_path, name), 'a+') as f:
        for seq_name in seq_names:
            if seq_name in datas:
                continue
            seq_path = os.path.join(real_path, seq_name, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            for i in range(0, len_all):
                image = images[i].split(root_path)[-1]
                #/home/awifi/data/ServicesHallData/train/output03/img1/00001.jpg
                print(image, file=f)
    f.close()

def main():
    # gen_labels('/home/awifi/data', mode='train')
    gen_labels('/home/awifi/data', mode='test')
    # gen_data_path_custom('/home/awifi/data/', 'ServicesHallData/images/train/', 'serhall.train')
    # gen_data_path_custom('/home/awifi/data/', 'ServicesHallData/images/test/', 'serhall.test')

if __name__ == '__main__':
    main()