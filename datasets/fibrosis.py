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
        # with open(root_dir + img_list, 'r') as f:
        #     self.entries = [line.strip() for line in f]
        self.entries = pd.read_csv(root_dir + img_list)

        print("Processing {} datas".format(len(self.entries) + 1))
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase


    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        # Create x (Images, Weeks, Percent, Smoking Status)
        # x_img
        x_wks = self.entries.iloc[i,1]
        x_pct = self.entries.iloc[i,3]
        x_smk = self.entries.iloc[i,6]

        x = [x_wks, x_pct, x_smk]

        # Create y (FVC, Age, Sex)
        y_fvc = self.entries.iloc[i,2]
        y_age = self.entries.iloc[i,4]
        y_sex = self.entries.iloc[i,5]

        y = [y_fvc, y_age, y_sex]


        # if self.phase == 'train':
        #     print(self.img_list[i])
        # elif self.phase == 'test':
        #     print('test')

        return (x,y)