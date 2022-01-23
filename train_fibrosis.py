'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

from setting import parse_opts 
from datasets.fibrosis import FibrosisDataset 
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import os

def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    # batches_per_epoch = len(data_loader)
    # log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    # loss_seg = nn.CrossEntropyLoss(ignore_index=-1)

    # print("Current setting is:")
    # print(sets)
    # print("\n\n")     
    # if not sets.no_cuda:
    #     loss_seg = loss_seg.cuda()
        
    # model.train()
    # train_time_sp = time.time()
    # for epoch in range(total_epochs):
        # log.info('Start epoch {}'.format(epoch))
        
        # scheduler.step()
        # log.info('lr = {}'.format(scheduler.get_lr()))
        
    for i, (x_batch, y_batch) in enumerate(data_loader):
        # TODO: Implement training code 
        print(x)
                            
    print('Finished training')            


if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    
    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 

    # optimizer
    params = [
            { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
            { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }
            ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # train from resume
    # if sets.resume_path:
    #     if os.path.isfile(sets.resume_path):
    #         print("=> loading checkpoint '{}'".format(sets.resume_path))
    #         checkpoint = torch.load(sets.resume_path)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #           .format(sets.resume_path, checkpoint['epoch']))

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True    

    # TODO: replace with FibrosisDataset()
    training_dataset = FibrosisDataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    # training
    train(data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 
