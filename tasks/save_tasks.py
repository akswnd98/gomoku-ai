from tasks.task import *
import h5py
import numpy as np
from safetensors.torch import save_model
import os

class RegularSaveTriggerTask (TriggerTask):
  def __init__ (self, color, period, task, context):
    super(RegularSaveTriggerTask, self).__init__(task, context)
    self.color = color
    self.period = period
  
  def check_condition(self):
    return self.context['trained_nums'][self.color] >= self.context['prev_save_times'][self.color] + self.period

class SaveTask (Task):
  def __init__ (self, color, h, context):
    super(SaveTask, self).__init__(context)
    self.color = color
    self.h = h

class SaveTrainedNumTask (SaveTask):
  def run (self):
    self.h['trained_nums'][self.color] = self.context['trained_nums'][self.color]

class SaveLossTask (SaveTask):
  def __init__ (self, color, h, context):
    super(SaveLossTask, self).__init__(color, h, context)
    self.prefixes = ['black', 'white']

  def run (self):
    original_len = self.h['statistics']['losses'][self.prefixes[self.color]].shape[0]
    self.h['statistics']['losses'][self.prefixes[self.color]].resize((original_len + len(self.context['statistics']['losses'][self.color]), ))
    self.h['statistics']['losses'][self.prefixes[self.color]][original_len: ] = self.context['statistics']['losses'][self.color]
    self.context['statistics']['losses'][self.color] = []

class SaveModelTask (Task):
  def __init__ (self, path, color, policy_net, value_net, context):
    super(SaveModelTask, self).__init__(context)
    self.path = path
    self.policy_net = policy_net
    self.value_net = value_net
    self.color = color
    self.prefixes = ['black', 'white']

  def run (self):
    save_model(self.policy_net, os.path.join(self.path, 'ckpt', 'no{}_{}_policy_net.safetensors'.format(self.context['trained_nums'][self.color], self.prefixes[self.color])))
    save_model(self.value_net, os.path.join(self.path, 'ckpt', 'no{}_{}_value_net.safetensors'.format(self.context['trained_nums'][self.color], self.prefixes[self.color])))

class NoInterruptWhileSavingTask (Task):
  def run (self):
    print('saving... don\'t interrupt saving.')

class SavingOverTask (Task):
  def run (self):
    print('saving is over! it\' ok to interrupt.')

class UpdatePrevSaveTimeTask (Task):
  def __init__ (self, color, period, context):
    super(UpdatePrevSaveTimeTask, self).__init__(context)
    self.color = color
    self.period = period

  def run (self):
    self.context['prev_save_times'][self.color] += self.period

if __name__ == '__main__':
  h = h5py.File('hello.h5', 'w')
  h.create_dataset('dataset', dtype=np.float32, shape=(100, ), maxshape=(10000, ), chunks=True)
  h['dataset'].resize((3, ))
  h['dataset'][0: 3] = [1, 2, 3]
  h['dataset'].resize((6, ))
  h['dataset'][3: 6] = [2, 3, 4]
  print(h['dataset'][0: ])