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

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


def test(data_loader, model, confidence_iter, sets):
    confidence_dict = {}
    eval_df = pd.DataFrame(columns = ['Patient_Week', 'FVC', 'Confidence'])

    # Create confidence probability distribution for all test data
    for i in range(confidence_iter):
        for j, (x, y, patient_id) in enumerate(data_loader):
            y_pred = model(y)

            # ID needs to exist out of patient id and week number
            patient_week = patient_id + x[1]
            fvc = y_pred.data[0]

            # Update confidence
            confidence_dict[patient_week].append(fvc)

            # On the last confidence iteration get confidence and add to eval_df
            if i == confidence_iter - 1:
                conf = np.std(confidence_dict[patient_week])
                eval_df.append([patient_week, fvc, conf])

    # TODO update name
    eval_df.to_csv('eval_test.csv')
        
if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    
    # getting model
    model, _ = generate_model(sets) 
    checkpoint = torch.load(sets.resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    test_dataset = FibrosisDataset(sets.data_root, 'test.csv', sets)
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    test(data_loader, model, confidence_iter=3, sets=sets) 
