'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

from setting import parse_opts 
from datasets.fibrosis import FibrosisDataset 
from models.fibrosis import CustomLoss
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
from torch.utils.tensorboard import SummaryWriter

mse = nn.MSELoss()
bce = nn.BCELoss()


def train(data_loader, test_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    writer = SummaryWriter()

    custom_loss = CustomLoss()
    # multi_criterion = nn.MultiLabelSoftMarginLoss(weight=None, reduce=False)


    # Enable anomaly detection in backward pass
    torch.autograd.set_detect_anomaly(True)

    train_time_sp = time.time()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    idx = 0
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        
        log.info('lr = {}'.format(scheduler.get_last_lr()))

        for batch_id, (x_batch, y_batch) in enumerate(data_loader):
            model.train()

            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)

            batch_id_sp = epoch * batches_per_epoch

            optimizer.zero_grad()

            y_pred = model(x_batch)

            # Calculate loss using mean squared error
            loss = custom_loss(y_pred.to(torch.float32), y_batch.to(torch.float32)) / sets.batch_size

            fvc_rmse, age_rmse = get_accuracy(y_pred, y_batch, sets)
            writer.add_scalar("Accuracy/train_fvc", fvc_rmse, idx)
            writer.add_scalar("Accuracy/train_age", age_rmse, idx)


            writer.add_scalar("Loss/train", loss, idx)

            loss.backward()                
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, idx, loss.item(), avg_batch_time))

            # Save model on specific intervals
            if idx % save_interval == 0:
                print('pred: ', y_pred)
                save_model(save_folder, model, optimizer, epoch, batch_id)

            model.eval()
            total_loss_test = 0
            for batch_id, (x_batch, y_batch) in enumerate(test_loader):
            
                y_batch = y_batch.to(device)
                x_batch = x_batch.to(device)

                y_pred = model(x_batch)

                total_loss_test += custom_loss(y_pred.to(torch.float32), y_batch.to(torch.float32))


            writer.add_scalar("Loss/test", total_loss_test / batches_per_epoch, idx)


            idx += 1

        

        
        scheduler.step()
    
    print('Finished training')            

def get_accuracy(y_pred, y, sets):
    # get RMSE for FVC
    fvc_rmse = rmse(y_pred, y)
    print('fvc rsme: ', fvc_rmse)
    print('fvc act: ', torch.mean(y))
    print('fvc pred: ', torch.mean(y_pred))
    # print('fvc RMSE: ',fvc_rmse / sets.batch_size)

    # get RMSE for Age
    # age_rmse = rmse(y_pred[:,1], y[:,1])
    # print('age rsme: ', age_rmse)
    # print('age act: ', torch.mean(y[:,1]))
    # print('age pred: ', torch.mean(y_pred[:,1]))
    # print('age RMSE: ',age_rmse / sets.batch_size)

    # get accuracy for sex
    # sex_acc = bce(y_pred[:,2],y[:,2])
    # print('sex: ',sex_acc)
    # accuracy['sex'].append(
    # get accuracy for smoking
    # smok_acc = bce(y_pred[:,3:6],y[:,3:6])
    # print('smk: ', smok_acc)
    # accuracy['smoking'].append()
    return (fvc_rmse, age_rmse)

def rmse(pred, target):
    return torch.sqrt(mse(pred, target))



def save_model(save_folder, model, optimizer, epoch, batch_id):
    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
    model_save_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
                    
    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
    torch.save({
                'epoch': epoch,
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

    # model_stats = summary(model, (1,30,256,256))

    optimizer = torch.optim.SGD(model.parameters(), lr=sets.learning_rate, momentum=0.9, weight_decay=1e-3)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(sets.resume_path, checkpoint['ecpoch']))

    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True    

    training_dataset = FibrosisDataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    test_dataset = FibrosisDataset(sets.data_root, 'test_interpolated.csv', sets)
    test_loader = DataLoader(test_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    # training
    train(data_loader, test_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 
