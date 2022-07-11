from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import utils, transforms
import torch
import os


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution, exp_setting):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        self.resolution = resolution
        self.transform = transform
        self.distance_dir = './distances'
        self.exp_setting = exp_setting
        id_path = os.path.join(self.distance_dir, f'index_{self.exp_setting}.txt')
        self.id_list = np.loadtxt(id_path)
        self.length = len(self.id_list)
        self.attr_val = {}
        attr_dict = {attr_name: f'{attr_name}_LARGE.txt' for attr_name in ['age', 'pose', 'glasses', 'hairlong', 'haircolor', 'smile', 'gender']}

        for attr in attr_dict.keys():
            self.attr_val[attr] = np.loadtxt(os.path.join(self.distance_dir, attr_dict[attr]))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_index = index
        index = int(self.id_list[index])
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        if self.exp_setting == 'glassessmileage':
            return img, index, self.attr_val['glasses'][index], self.attr_val['age'][index], self.attr_val['smile'][index]
        elif self.exp_setting == 'haircolorhairlonggender':
            return img, index, self.attr_val['haircolor'][index], self.attr_val['hairlong'][index], self.attr_val['gender'][index]
        elif self.exp_setting == 'glassesagepose':
            return img, index, self.attr_val['glasses'][index], self.attr_val['age'][index], self.attr_val['pose'][index]