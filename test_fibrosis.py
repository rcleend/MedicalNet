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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

mse = nn.MSELoss()

def test(data_loader, model, sets):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    all_fvc_pred = []
    all_fvc = []
    acc = {'fvc_sum': 0, 'age_sum': 0, 'sex_true': [], 'sex_pred': [], 'smk_true': [], 'smk_pred': []}
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        # Get sigmoid of y_pred and append to all_y_pred
        all_fvc_pred.append(torch.sigmoid(y_pred[:,2]).item())
        all_fvc.append(y[:,2].item())

        
        # update accuracy
        # update_acc(acc, y_pred, y, sets)
    print(all_fvc_pred)
    print(all_fvc)

    # fpr, tpr, _ = roc_curve(all_fvc, all_fvc_pred)
    # roc_auc = auc(fpr, tpr)
    # plot_roc(fpr,tpr, roc_auc)
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # print accuracy
    # log_acc(acc, sets, len(data_loader.dataset))

def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

def log_acc(acc, sets, n_data):
    if sets.multi_task == 'fvc':
        print('avg fvc loss: ', acc['fvc_sum'].item() / n_data)
    elif sets.multi_task == 'fvc_age':
        print('avg fvc loss: ', acc['fvc_sum'].item() / n_data)
        print('avg age loss: ', acc['age_sum'].item() / n_data)
    else:
        print('avg fvc loss: ', acc['fvc_sum'].item() / n_data)
        print('avg age loss: ', acc['age_sum'].item() / n_data)
        print('sex acc: ', accuracy_score(acc['sex_true'], acc['sex_pred']))
        print('smk acc: ', accuracy_score(acc['smk_true'], acc['smk_pred']))

def update_acc(acc, y_pred, y, sets):
    if sets.multi_task == 'fvc':
        acc['fvc_sum'] += rmse(y_pred, y)
    elif sets.multi_task == 'fvc_age':
        acc['fvc_sum'] += rmse(y_pred[:,0], y[:,0])
        acc['age_sum'] += rmse(y_pred[:,1], y[:,1])
    else:
        acc['fvc_sum'] += rmse(y_pred[:,0], y[:,0])
        acc['age_sum'] += rmse(y_pred[:,1], y[:,1])
        sex_true, sex_pred = sex_acc(y_pred[:,2], y[:,2])
        acc['sex_true'].append(sex_true)
        acc['sex_pred'].append(sex_pred)
        smk_true, smk_pred = smk_acc(y_pred[:,3:6], y[:,3:6])
        acc['smk_true'].append(smk_true)
        acc['smk_pred'].append(smk_pred)

def smk_acc(y_pred, y):
    true = torch.argmax(y, dim=1) 
    pred = torch.argmax(y_pred, dim=1) 
    return true.cpu().detach().numpy().tolist(), pred.cpu().detach().numpy().tolist()

def sex_acc(y_pred, y):
    return (y > 0.5).cpu().detach().numpy().tolist(), (y_pred > 0.5).cpu().detach().numpy().tolist()

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
