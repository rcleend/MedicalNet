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
    img_features = self.model(x[0])
    return self.fc([img_features, x])

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        
    def fvc_loss(self, input, target):
       return torch.sqrt(self.mse(torch.log(input + 1), torch.log(target + 1)))

    def forward(self, input, target):
        return self.fvc_loss(input[:,0],target[:,0]) * 2 + self.mse(input[:,1],target[:,1]) + self.bce(input[:,2:6],target[:,2:6])

class CustomDenseLayer(nn.Module):
  """
  This custom layer implements our custom network architecture. This dense layer is applied after the resnet model
  
  """
  def __init__(self, input=100,n_hidden=10):
    super().__init__()
    self.n_hidden=n_hidden
    self.input = input
    self.relu = nn.ReLU(inplace=False)
    self.sigmoid = nn.Sigmoid()
    self.flatten = nn.Flatten()
    self.linear = nn.Linear(input,self.n_hidden)
    self.linear2 = nn.Linear(self.n_hidden,6)
    self.softmax = nn.Softmax()

  def forward(self, x):
      x = self.flatten(x)
      x = self.linear(x)
      x = self.linear2(x)
      #note: we clone the x tensors to prevent modification before computing the gradient
      x[:,2] = self.sigmoid(x[:,2].clone()) # Male/female
      x[:,3:6] = self.softmax(x[:,3:6].clone()) # Smoking Status
      return x