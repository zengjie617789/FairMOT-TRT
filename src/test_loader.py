import torch
from torchvision.transforms import transforms as T
from datasets.dataset_factory import get_dataset
import json
from opts import opts
import re
import io
from PIL import Image
import numpy as np
import base64
import time


def time_fun(func):
    def wrapper_fun(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print(f"cost time: {t2-t1}")
        return res
    return wrapper_fun

@time_fun
def b64_to_arr(b64):
    """
     base64转string
    :param b64:  base64
    :return:  string
    """
    base64_data = re.sub('^data:image/.+;base64,', '', b64)
    byte_data = base64.b64decode(base64_data)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data)
    return np.array(img)

@time_fun
def arr_to_b64(arr, format='jpeg'):
    # format jpeg bmp png gif ppm
    arr = Image.fromarray(arr.astype('uint8'))
    output_buffer = io.BytesIO()
    arr.save(output_buffer, format=format)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode()
    return base64_str


lr = 0.001
batch_size = 8

mean = [0.452, 0.459, 0.456]
std = [0.241, 0.245, 0.241]
transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])  # add the mean and std of ServicesHallDatasets
"""@nni.variable(nni.choice(1e-4, 5e-4, 1e-3), name=lr)"""
# """nni.variable(nni.choice(12, 8, 4), name=batch_size)"""

opt = opts().parse()
f = open(opt.data_cfg)
data_config = json.load(f)
TestDatasets = False if 'test' not in data_config.keys() else True
trainset_paths = data_config['train']
testset_paths = data_config['test'] if TestDatasets else None
dataset_root = data_config['root']
f.close()

Dataset = get_dataset(opt.dataset, opt.task)
dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
# test_dataset = Dataset(opt, dataset_root, testset_paths, (1088, 608), augment=True, transforms=transforms)
test_dataset = Dataset(opt, dataset_root, testset_paths, (1088, 1920), augment=True, transforms=transforms)


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    pin_memory=True,
    drop_last=True
)
for iter_id, batch in enumerate(test_loader):
    print(f"iter_id:{iter_id}, batch length : {len(batch)}")
    print(batch['input'][0].shape)
    data64 = arr_to_b64(batch['input'][0].data.numpy().transpose(1,2,0))

    b64_to_arr(data64)

