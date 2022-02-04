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

def test(data_loader, model, accuracy, sets):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        update_accuracy(accuracy, y_pred, y)

def update_accuracy(accuracy, y_pred, y):
    # get RMSE for FVC
    fvc_rmse = rmse(y_pred[:,0], y[:,0])
    print('fvc act: ',y[:,0].value)
    print('fvc pred: ',y_pred[:,0].value)
    print('fvc RMSE: ',fvc_rmse.value)
    # accuracy['fvc'] += fvc_rmse
    # get RMSE for Age
    age_rmse = rmse(y_pred[:,1], y[:,1])
    print('age act: ',y[:,1].value)
    print('age pred: ',y_pred[:,1].value)
    print('age RMSE: ',age_rmse.value)
    # accuracy['age'] += age_rmse
    # get accuracy for sex
    sex_acc = bce(y_pred[:,2],y[:,2])
    print('sex: ',sex_acc.value)
    # accuracy['sex'].append(
    # get accuracy for smoking
    smok_acc = bce(y_pred[:,3:6],y[:,3:6])
    print('smk: ', smok_acc.value)
    # accuracy['smoking'].append()

def rmse(pred, target):
    return torch.sqrt(mse(torch.log(pred + 1), torch.log(target + 1)))

        
if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    
    # getting model
    model, _ = generate_model(sets) 
    checkpoint = torch.load(sets.resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    test_dataset = FibrosisDataset(sets.data_root, 'test.csv', sets)
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    accuracy = {}
    test(data_loader, model, accuracy, sets=sets) 
