'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

from sre_constants import GROUPREF_EXISTS
from setting import parse_opts 
from datasets.fibrosis import FibrosisDataset 
from models.fibrosis import CustomLoss
from model import generate_model
import torch
from torch.utils.data import DataLoader
from utils.logger import log
import pandas as pd
import numpy as np
import torch.nn as nn

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

mse = nn.MSELoss()
bce = nn.BCELoss()

def test(data_loader, model, sets):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        update_accuracy(y_pred, y)

def update_accuracy(y_pred, y):
    # get RMSE for FVC
    fvc_rmse = rmse(y_pred, y)
    print('fvc rsme: ', fvc_rmse)
    print('fvc act: ', y)
    print('fvc pred: ', y_pred)

def rmse(pred, target):
    return torch.sqrt(mse(pred, target))

        
if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    
    # getting model
    model, _ = generate_model(sets) 
    checkpoint = torch.load(sets.resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    test_dataset = FibrosisDataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    test(data_loader, model,sets=sets) 
