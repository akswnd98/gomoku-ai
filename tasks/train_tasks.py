from tasks.task import *
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EnoughDataTriggerTask (TriggerTask):
  def __init__ (self, color, enough_size, task, context):
    super(EnoughDataTriggerTask, self).__init__(task, context)
    self.color = color
    self.enough_size = enough_size
  
  def check_condition (self):
    return len(self.context['datasets'][self.color]) >= self.enough_size
