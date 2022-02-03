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

        for batch_id, (x_img_batch, x_wks_batch, y_batch) in enumerate(data_loader):
            model.train()

            y_batch = y_batch.to(device)
            x_img_batch = x_img_batch.to(device)
            x_wks_batch = x_wks_batch.to(device)
            batch_id_sp = epoch * batches_per_epoch

            optimizer.zero_grad()

            y_pred = model((x_img_batch, x_wks_batch))
            print(y_pred)

            # Calculate loss using mean squared error
            loss = custom_loss(y_pred.to(torch.float32), y_batch.to(torch.float32)) / sets.batch_size
            print(custom_loss(y_pred.to(torch.float32), y_batch.to(torch.float32)))

            writer.add_scalar("Loss/train", loss, idx)

            loss.backward()                
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, idx, loss.item(), avg_batch_time))

            # Save model on specific intervals
            if idx % save_interval == 0:
                save_model(save_folder, model, optimizer, epoch, batch_id)

            # model.eval()
            # total_loss_test = 0
            # for batch_id, (x_img_batch, x_wks_batch, y_batch) in enumerate(test_loader):
            
            #     y_batch = y_batch.to(device)
            #     x_img_batch = x_img_batch.to(device)
            #     x_wks_batch = x_wks_batch.to(device)

            #     y_pred = model((x_img_batch, x_wks_batch))

            #     # Calculate loss using mean squared error
            #     total_loss_test += custom_loss(y_pred.to(torch.float32), y_batch.to(torch.float32))


            # writer.add_scalar("Loss/test", total_loss_test / batches_per_epoch, idx)


            idx += 1

        
        scheduler.step()
    
    print('Finished training')            

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

    optimizer = torch.optim.Adam(model.parameters(), lr=sets.learning_rate, weight_decay=1e-3)   
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

    # test_dataset = FibrosisDataset(sets.data_root, 'test.csv', sets)
    # test_loader = DataLoader(test_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    test_loader = 0

    # training
    train(data_loader, test_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 
