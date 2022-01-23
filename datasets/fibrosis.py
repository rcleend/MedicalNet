'''
Dataset for training
Written by Whalechen
'''

import math
import os
import cv2
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import pandas as pd

class FibrosisDataset(Dataset):

    def __init__(self, root_dir, img_list, sets):
        self.entries = pd.read_csv(root_dir + img_list)

        print('Processing {} datas'.format(len(self.entries) + 1))
        self.img_dir = f'{root_dir}pre processed/images/'
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase


    def __len__(self):
        return len(self.entries)

    def __load_images__(self, images_path):
        images = []
        for filename in os.listdir(images_path):
            img = cv2.imread(f'{images_path}/{filename}')
            if img is not None:
                images.append(img)

        # Create np array
        images = np.array(images)
        # Remove rgb dimension
        images = images[:,:,:,0]
        return images

    def __getitem__(self, i):
        # Create x values (Weeks, Percent, Smoking Status, Images)
        x_wks = self.entries.iloc[i,1]
        x_pct = self.entries.iloc[i,3]
        x_smk = self.entries.iloc[i,6]
        # TODO get patien ct scan images and convert to 3d tensor
        # x_img = 
        print('loading images')
        x_img = self.__load_images__(self.img_dir + self.entries.iloc[i,0])
        print(x_img.shape)

        x = [x_wks, x_pct, x_smk]

        # Create y values (FVC, Age, Sex)
        y_fvc = self.entries.iloc[i,2]
        y_age = self.entries.iloc[i,4]
        y_sex = self.entries.iloc[i,5]

        y = [y_fvc, y_age, y_sex]

        # if self.phase == 'train':
        #     print(self.img_list[i])
        # elif self.phase == 'test':
        #     print('test')

        return x,y