import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from models.resnet import resnet10


class MedicalNet(nn.Module):

  def __init__(self, opt):
    super(MedicalNet, self).__init__()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    self.model = resnet10(
                            sample_input_D=opt.input_D, 
                            sample_input_H=opt.input_H, 
                            sample_input_W=opt.input_W, 
                            num_seg_classes=opt.n_seg_classes,
                            shortcut_type=opt.resnet_shortcut,
                            no_cuda=opt.no_cuda,
                        )

    self.model.conv_seg = nn.Sequential(
        nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
        nn.Flatten(start_dim=1),
        nn.Dropout(0.1)
    )
    net_dict = self.model.state_dict()
    pretrained_weights = torch.load(opt.pretrain_path, map_location=torch.device(device))
    pretrain_dict = {
        k.replace("module.", ""): v for k, v in pretrained_weights['state_dict'].items() if k.replace("module.", "") in net_dict.keys()
      }
    net_dict.update(pretrain_dict)
    self.model.load_state_dict(net_dict)
    self.fc = CustomDenseLayer(512)

  def forward(self, x):
    features = self.model(x)
    return self.fc(features)


class CustomDenseLayer(nn.Module):
  """
  This custom layer implements our custom network architecture. This dense layer is applied after the resnet model
  
  """
  def __init__(self, input=100):
    super().__init__()
    self.input = input
    self.relu = nn.ReLU(inplace=False)
    self.sigmoid = nn.Sigmoid()
    self.flatten = nn.Flatten()
    self.linear = nn.Linear(input,1)
    self.softmax = nn.Softmax()

  def forward(self, x):
    #   x = self.flatten(x)
    #   x = self.linear(x)
      #note: we clone the x tensors to prevent modification before computing the gradient
      x = self.relu(x) #FVC value
      x = self.linear(x)
    #   x[:,0] = self.relu(x[:,0].clone()) #FVC value
    #   x[:,1] = self.relu(x[:,1].clone()) ## Age
    #   x[:,2] = self.sigmoid(x[:,2].clone()) ##Male/female
    #   x[:,3:6] = self.softmax(x[:,3:6].clone())
      return x