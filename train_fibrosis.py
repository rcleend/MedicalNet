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
from torchsummary import summary



def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))

    mse = nn.MSELoss()
    # multi_criterion = nn.MultiLabelSoftMarginLoss(weight=None, reduce=False)


    # Enable anomaly detection in backward pass
    torch.autograd.set_detect_anomaly(True)

    model.train()
    train_time_sp = time.time()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        
        log.info('lr = {}'.format(scheduler.get_lr()))

        for batch_id, (x_batch, y_batch) in enumerate(data_loader):
            y_batch = y_batch.to(device)
            batch_id_sp = epoch * batches_per_epoch
            optimizer.zero_grad()

            y_pred = model(x_batch)
            print('pred:', y_pred)
            print('acc:', y_batch)
            

            # Calculate loss using mean squared error
            loss = mse(y_pred.to(torch.float32), y_batch.to(torch.float32))
            # loss = multi_criterion(y_pred, y_batch)
            loss.backward()                
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, batch_id_sp, loss.item(), avg_batch_time))

            # Save model on specific intervals
            if batch_id_sp % save_interval == 0:
                save_model(save_folder, model, optimizer, epoch, batch_id)

        scheduler.step()
    
    print('Finished training')            


def save_model(save_folder, model, optimizer, epoch, batch_id):
    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
    model_save_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
                    
    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
    torch.save({
                'ecpoch': epoch,
                'batch_id': batch_id,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                model_save_path)

if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    
    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 

    model_stats = summary(model, (1,30,256,256))

    # optimizer
    params = [
            { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
            { 'params': parameters['new_parameters'], 'lr': sets.learning_rate }
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

    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True    

    training_dataset = FibrosisDataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    # training
    train(data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 
