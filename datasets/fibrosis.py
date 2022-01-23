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

        images = np.array(images)

        # Remove rgb dimension
        return images[:,:,:,0]

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __get_smoking_values(self, i):
        if self.entries.iloc[i,6] == 'Currently smokes':
            return 1, 0, 0 
        elif self.entries.iloc[i, 6] == 'Ex-smoker':
            return 0, 1, 0
        else:
            return 0, 0, 1



    def __getitem__(self, i):
        # Create x values (Weeks, Percent, Images)
        x_wks = self.entries.iloc[i,1]
        x_pct = self.entries.iloc[i,3]
        # TODO: evaluate if image actually contains relevant information and is not distorted
        x_img = self.__resize_data__(self.__load_images__(self.img_dir + self.entries.iloc[i,0]))

        x = [x_wks, x_pct, x_img]

        # Create y values (FVC, Age, Sex, Smoking)
        y_fvc = self.entries.iloc[i,2]
        y_age = self.entries.iloc[i,4]
        y_sex = self.entries.iloc[i,5]
        y_smk, y_ex_smk, y_non_smk = self.__get_smoking_values(i)

        y = [y_fvc, y_age, y_sex, y_smk, y_ex_smk, y_non_smk]

        return x,y