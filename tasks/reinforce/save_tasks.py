from tasks.task import *
from safetensors.torch import save_model
import os

class SaveModelTask (Task):
  def __init__ (self, path, color, policy_net, context):
    super(SaveModelTask, self).__init__(context)
    self.path = path
    self.policy_net = policy_net
    self.color = color
    self.prefixes = ['black', 'white']

  def run (self):
    save_model(self.policy_net, os.path.join(self.path, 'ckpt', 'no{}_{}_policy_net.safetensors'.format(self.context['trained_nums'][self.color], self.prefixes[self.color])))
