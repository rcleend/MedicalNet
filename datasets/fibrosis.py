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
import pandas as pd

class FibrosisDataset(Dataset):

    def __init__(self, root_dir, img_list, sets):
        self.data = pd.read_csv(root_dir + img_list)
        print("Processing {} rows".format(len(self.data)))

        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        df = self.data[i]

        # TODO: Get corresponding ct scan images and change them to tensor values
        # TODO: Get FVC value as y 
        # TODO: Get 


        return self.data[i][0]