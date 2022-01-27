'''
Dataset for training
Written by Whalechen
'''

import math
import os
import cv2
import random
import torch

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import pandas as pd
from torchvision.io import ImageReadMode
from torchvision.io import read_image


class FibrosisDataset(Dataset):

    def __init__(self, root_dir, img_list, sets):
        self.entries = pd.read_csv(root_dir + img_list)
        # TODO: Replace with tensor also containing meta data

        print('Processing {} datas'.format(len(self.entries) + 1))
        self.img_dir = f'{root_dir}pre processed/images/'
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase


        self.patient = torch.empty((3,256,256,30))

    def __len__(self):
        return len(self.entries)

    def __load_images__(self, images_path):
        for i, filename in enumerate(os.listdir(images_path)):
            self.patient[:,:,:,i] = read_image(f'{images_path}/{filename}', mode=ImageReadMode.RGB)

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __ct2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
            
        return new_data


    def __get_smoking_values(self, i):
        if self.entries.iloc[i,6] == 'Currently smokes':
            return 1, 0, 0 
        elif self.entries.iloc[i, 6] == 'Ex-smoker':
            return 0, 1, 0
        else:
            return 0, 0, 1

    def __get_sex(self, i):
        if self.entries.iloc[i,5] == 'Male':
            return True
        else:
            return False


    def __getitem__(self, i):
        # Create x values (Weeks, Percent, Images)
        # x_wks = self.entries.iloc[i,1]
        # x_pct = self.entries.iloc[i,3]
        # TODO: evaluate if image actually contains relevant information and is not distorted
        # x_img = self.__ct2tensorarray__(
        #             self.__resize_data__(
        #                 self.__load_images__(self.img_dir + self.entries.iloc[i,0])
        #             )
        #         )

        # x = [x_wks, x_pct, x_img]

        self.__load_images__(self.img_dir + self.entries.iloc(i,0))

        # Create y values (FVC, Age, Sex, Smoking)
        y_fvc = self.entries.iloc[i,2]
        y_age = self.entries.iloc[i,4]
        y_is_male = self.__get_sex(i)

        # Get smoking values
        y_smk, y_ex_smk, y_non_smk = self.__get_smoking_values(i)

        y = torch.tensor(y_fvc, y_age, y_is_male, y_smk, y_ex_smk, y_non_smk)

        return self.patient,y