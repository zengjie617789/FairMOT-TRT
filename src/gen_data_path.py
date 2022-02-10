import os
import glob
import _init_paths
from os.path import getsize
import shutil

def gen_caltech_path(root_path):
    label_path = 'Caltech/data/labels_with_ids'
    real_path = os.path.join(root_path, label_path)
    image_path = real_path.replace('labels_with_ids', 'images')
    images_exist = sorted(glob.glob(image_path + '/*.png'))
    with open('../src/data/caltech.all', 'w') as f:
        labels = sorted(glob.glob(real_path + '/*.txt'))
        for label in labels:
            image = label.replace('labels_with_ids', 'images').replace('.txt', '.png')
            if image in images_exist:
                print(image[22:], file=f)
    f.close()


def gen_data_path(root_path):
    mot_path = 'MOT17/images/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith('SDP')]
    with open('/home/yfzhang/PycharmProjects/fairmot/src/data/mot17.half', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half):
                image = images[i]
                print(image[22:], file=f)
    f.close()


def gen_data_path_mot17_val(root_path):
    mot_path = 'MOT17/images/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith('SDP')]
    with open('/home/yfzhang/PycharmProjects/fairmot2/src/data/mot17.val', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half, len_all):
                image = images[i]
                print(image[22:], file=f)
    f.close()


def gen_data_path_mot17_emb(root_path):
    mot_path = 'MOT17/images/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith('SDP')]
    with open('/home/yfzhang/PycharmProjects/fairmot2/src/data/mot17.emb', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half, len_all, 3):
                image = images[i]
                print(image[22:], file=f)
    f.close()

#for MOT format
def gen_data_path_custom(root_path, name, split_name='/home/awifi/data/'):
    # dir_path = os.path.join(root_path, data_path)
    seq_names = [s for s in sorted(os.listdir(root_path)) if os.path.isdir(os.path.join(root_path, s))]
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    with open(os.path.join(save_path, name), 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(root_path, seq_name, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            for i in range(0, len_all):
                image = images[i].split(split_name)[-1]
                #/home/awifi/data/ServicesHallData/train/output03/img1/00001.jpg
                print(image, file=f)
    f.close()

def remove_zero_label(root_path):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            if getsize(file_path) == 0:
                print(file_path)
                img_path = file_path.replace('txt', 'jpg').replace('labels_with_ids','images')
                print(img_path)
                os.remove(img_path)
                os.remove(file_path)

#for Img format
def gen_data_path_custom02(imgs_root, name, split_name='/home/awifi/data/datasets/'):
    # Imgs_path = os.path.join(root_path, 'images')
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    with open(os.path.join(save_path, name), 'w') as f:
        for imgs_dir in os.listdir(imgs_root):
            if imgs_dir.endswith('zip'):
                continue
            imgs_path = os.path.join(imgs_root, imgs_dir)
            images = sorted(glob.glob(imgs_path + '/*.jpg'))
            len_all = len(images)
            for i in range(0, len_all):
                image = images[i].split(split_name)[-1]
                #/home/awifi/data/ServicesHallData/train/output03/img1/00001.jpg
                print(image, file=f)
    f.close()

def change_gt_file(data_root):
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file == "gt.txt":
                print(os.path.join(root, file))
                file_path = os.path.join(root, file)
                with open(file_path, 'r+') as file:
                    lines = file.readlines()
                    newlines = []
                    for line in lines:
                        line = line.strip().split(',')
                        temp = [*line[:6], '1', '1', '1']
                        newline = ','.join(i for i in temp)
                        newlines.append(newline)

                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)

                with open(file_path, 'w+') as f:
                    for line in newlines:
                        f.write(line+'\n')



if __name__ == '__main__':
    # root = '/data/yfzhang/MOT/JDE'
    # gen_data_path_mot17_emb(root)

    # root_path = '/home/awifi/data/'
    # gen_data_path_custom(root_path, 'serhall.train')

    # generate image datasets path
    root_path = '/home/awifi/data/datasets/ServicesHallData/images'
    gen_data_path_custom02(root_path, 'serhall_img.train')

    #generate video datasets path
    # root_path = '/home/awifi/data/DanceTrack/images/train'
    # gen_data_path_custom(root_path, 'dance_track.train', split_name='/home/awifi/data/DanceTrack/')

    # change_gt_file("/home/awifi/data/ServicesHallData/images/train")
    # root_path = '/home/awifi/data/ServicesHallData/images/train'
    # gen_data_path_custom(root_path, 'serhall.train')

    # remove_zero_label(root_path)