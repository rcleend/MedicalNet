'''
Dataset for training
Written by Whalechen
'''

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage

class FibrosisDataset(Dataset):

    def __init__(self, root_dir, img_list, sets):
        with open(root_dir + img_list, 'r') as f:
            self.img_list = [line.strip() for line in f]

        print("Processing {} datas".format(len(self.img_list)))
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):

        # if self.phase == 'train':
        #     print(self.img_list[i])
        # elif self.phase == 'test':
        #     print('test')

        return self.img_list[i]