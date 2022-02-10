import os
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import pickle
import re
import  os
import glob
import shutil
import  time

"""
在网络训练前先运行该函数获得数据的均值和标准差
"""
#/home/awifi/data/ServicesHallData/images/train/output01/img1/

# output01:
# <class 'type'>: normMean = [0.4725996  0.47627005 0.45940742]
# <class 'type'>: normstd = [0.23685737 0.23647648 0.22925246]
# output01+output03
# <class 'type'>: normMean = [0.46427318 0.46616882 0.4605752 ]
# <class 'type'>: normstd = [0.23573478 0.23837718 0.22661647]



class Dataloader():
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.dirs = ['train01', 'train02']

        self.means = [0, 0, 0]
        self.std = [0, 0, 0]

        self.transform = transforms.Compose([transforms.Resize((608, 1088)),
                                             transforms.ToTensor()  # 数据值从[0,255]范围转为[0,1]，相当于除以255操作
                                             ])

        # 因为这里使用的是ImageFolder，按文件夹给数据分类，一个文件夹为一类，label会自动标注好
        # self.dataset = {x: ImageFolder(os.path.join(dataroot, x), self.transform) for x in self.dirs}
        self.dataset = ImageFolder(dataroot, self.transform)
    def get_mean_std(self):
        """
        计算数据集的均值和标准差
        """
        num_imgs = len(self.dataset)
        for data in self.dataset:
            img = data[0]
            for i in range(3):
                # 计算每一个通道的均值和标准差
                self.means[i] += img[i, :, :].mean()
                self.std[i] += img[i, :, :].std()

        self.means = np.asarray(self.means) / num_imgs
        self.std = np.asarray(self.std) / num_imgs

        print("{}: normMean = {}".format(type, self.means))
        print("{}: normstd = {}".format(type, self.std))

        # # 将得到的均值和标准差写到文件中，之后就能够从中读取
        # with open(mean_std_path, 'wb') as f:
        #     pickle.dump(self.means, f)
        #     pickle.dump(self.stdevs, f)
        #     print('pickle done')



def dataset_accumulate(data_root):
    for i in glob.glob(os.path.join(data_root, 'train0[0-9]')):
        for img_name in os.listdir(i):
            new_img_name = i + '_' + img_name
            new_path = os.path.join(i, 'trains10', new_img_name)
            shutil.move(os.path.join(i, img_name), new_path)
            # print(os.path.join(i,img_name))

def rename_file_by_order(source_path, target_path=None, start_index=0):
    if target_path is None:
        source_name = os.path.split(source_path)[-1]
        target_name = source_name + '_renamed'
        target_path = source_path.replace(source_name, target_name)
        if not os.path.exists(target_path):
            os.mkdir(target_path)

    for i in os.listdir(source_path):
        if not i.endswith('jpg'):
            continue
        image_path = os.path.join(source_path, i)
        new_image_name = '{:05d}.jpg'.format(start_index)
        new_image_path = os.path.join(target_path, new_image_name)
        shutil.move(image_path, new_image_path)
        start_index += 1

def read_img_time(data_root):
    for imgPath in glob.glob(os.path.join(data_root, '*.jpg')):

        ImageDate = time.ctime(os.path.getmtime(imgPath))
        day = ImageDate.split(' ')[2]
        if day == '30':
            shutil.move(imgPath, imgPath.replace('temp', 'temp_to_remove'))
            print(imgPath)

if __name__ == '__main__':
    dataroot = '/home/awifi/data/ServicesHallData/images_shortcut'
    # data_root = '/home/awifi/data/datasets/ServicesHallData/images'

    # dataloader = Dataloader(data_root)
    # dataloader.get_mean_std()

    # data_root = '/home/awifi/data/datasets/ServicesHallData/images/'
    # dataset_accumulate(data_root)

    data_root = '/home/awifi/data/datasets/ServicesHallData/images/train03'
    # data_root = '/home/awifi/data/datasets/ServicesHallData/images_data'
    rename_file_by_order(data_root)

    # read_img_time('/home/awifi/data/datasets/ServicesHallData/images/temp/')