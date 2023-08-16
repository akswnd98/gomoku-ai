import sys, getopt
import os
from help import *
import torch
from player import *
from model import *
from safetensors.torch import save_model
from safetensors.torch import load_model
from tasks.actor_critic.pipeline_task import *

if __name__ == '__main__':
  try:
    opts, args = getopt.getopt(sys.argv[1: ], 'hcd')
  except getopt.GetoptError as e:
    print(e)
    sys.exit(-1)

  for o, a in opts:
    if o == '-h':
      print_help()
      sys.exit(0)
    elif o == '-c':
      try:
        os.makedirs(args[0])
      except Exception as e:
        print(e)
        sys.exit(-1)
      
      policy_nets = [PolicyNet() for _ in range(2)]
      value_nets = [ValueNet() for _ in range(2)]
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      for net in value_nets + policy_nets:
        net.to(device)

      try:
        os.makedirs(os.path.join(args[0], 'ckpt'))
      except Exception as e:
        print(e)
        sys.exit(-1)

      model_prefixes = ['black', 'white']
      for c in range(2):
        save_model(policy_nets[c], os.path.join(args[0], 'ckpt', 'no0_{}_policy_net.safetensors'.format(model_prefixes[c])))
        save_model(value_nets[c], os.path.join(args[0], 'ckpt', 'no0_{}_value_net.safetensors'.format(model_prefixes[c])))

      hdf = h5py.File(os.path.join(args[0], 'progress.h5'), 'w')
      hdf.create_dataset('trained_nums', dtype=np.int64, shape=(2, ))
      hdf['trained_nums'][0: 2] = [0, 0]
      hdf.create_group('statistics')
      hdf_losses = hdf['statistics'].create_group('losses')
      hdf_losses.create_dataset('black', dtype=np.float32, shape=(0, ), maxshape=(10000000, ), chunks=True)
      hdf_losses.create_dataset('white', dtype=np.float32, shape=(0, ), maxshape=(10000000, ), chunks=True)
      context = {
        'trained_nums': [0, 0],
        'prev_save_times': [0, 0],
        'statistics': {
          'losses': [[], []]
        },
        'datasets': [[], []]
      }

    elif o == '-d':
      policy_nets = [PolicyNet() for _ in range(2)]
      value_nets = [ValueNet() for _ in range(2)]
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      for net in value_nets + policy_nets:
        net.to(device)

      model_prefixes = ['black', 'white']
      latest_model_num = [0, 0]
      for c in range(2):
        for file in list(filter(lambda x: x.split('_')[1] == model_prefixes[c], os.listdir(os.path.join(args[0], 'ckpt')))):
          latest_model_num[c] = max(latest_model_num[c], int(file.split('_')[0][2: ]))
      
      for c in range(2):
        load_model(policy_nets[c], os.path.join(args[0], 'ckpt', 'no{}_{}_policy_net.safetensors'.format(latest_model_num[c], model_prefixes[c])))
        load_model(value_nets[c], os.path.join(args[0], 'ckpt', 'no{}_{}_value_net.safetensors'.format(latest_model_num[c], model_prefixes[c])))

      hdf = h5py.File(os.path.join(args[0], 'progress.h5'), 'r+')
      context = {
        'trained_nums': hdf['trained_nums'][0: 2],
        'prev_save_times': hdf['trained_nums'][0: 2],
        'statistics':  {
          'losses': [list(hdf['statistics']['losses'][x]) for x in ['black', 'white']]
        },
        'datasets': [[], []]
      }
    
    players = [AiPlayer(device, policy_net, 0.2) for policy_net in policy_nets]
    pipeline_task = PipelineTask(args[0], device, players, 16384, policy_nets, value_nets, hdf, context)
    pipeline_task.run()
