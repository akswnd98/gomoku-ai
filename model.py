# 1

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_model, save_model

class PolicyNet (nn.Module):
  def __init__ (self, hidden_layers_num=8):
    super(PolicyNet, self).__init__()

    self.input_layer = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(0.01)
    )
    self.hidden_layers = nn.Sequential(
      *(
        [
          nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
          )
        ] + [
          nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
          ) for _ in range(hidden_layers_num)
        ]
      )
    )
    self.output_layer = nn.Sequential(
      nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(1),
      nn.Flatten(start_dim=1),
      nn.Softmax(dim=1)
    )
  
  def forward (self, x):
    x = self.input_layer(x)
    x = self.hidden_layers(x)
    x = self.output_layer(x)
    return x;

class ValueNet (nn.Module):
  def __init__ (self, hidden_layers_num=8):
    super(ValueNet, self).__init__()
    self.input_layer = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(0.01)
    )
    self.hidden_layers = nn.Sequential(
      *(
        [
          nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
          )
        ] + [
          nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
          ) for _ in range(hidden_layers_num)
        ]
      )
    )
    self.output_layer = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(1),
      nn.Linear(128, 256, True),
      nn.LeakyReLU(0.01),
      nn.Linear(256, 1, True),
      nn.Tanh()
    )
  
  def forward (self, x):
    x = self.input_layer(x)
    x = self.hidden_layers(x)
    x = self.output_layer(x)
    return x;

if __name__ == '__main__':
  # policy_net = PolicyNet()
  # print(policy_net(torch.ones((2, 1, 9, 9), dtype=torch.float32)))
  # save_model(policy_net, 'policy_net.safetensors')

  # loaded_policy_net = PolicyNet()
  # load_model(loaded_policy_net, 'policy_net.safetensors')
  # print(loaded_policy_net(torch.ones((2, 1, 9, 9), dtype=torch.float32)))

  value_net = ValueNet()
  print(nn.BatchNorm2d(32)(nn.Conv2d(1, 32, 5, 1, 2, bias=False)(torch.ones((32, 1, 9, 9), dtype=torch.float32))).shape)
  print(value_net(torch.ones((2, 1, 9, 9), dtype=torch.float32)))
