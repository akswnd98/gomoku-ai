from model import *
from play import *
from player import *
import getopt
import sys
import os

if __name__ == '__main__':
  try:
    opts, args = getopt.getopt(sys.argv[1: ], '')
  except getopt.GetoptError as e:
    print(e)
    sys.exit(-1)

  policy_nets = [PolicyNet() for _ in range(2)]
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model_prefixes = ['black', 'white']

  latest_model_num = [0, 0]
  for c in range(2):
    for file in list(filter(lambda x: x.split('_')[1] == model_prefixes[c], os.listdir(os.path.join(args[0], 'ckpt')))):
      latest_model_num[c] = max(latest_model_num[c], int(file.split('_')[0][2: ]))

  for c in range(2):
    print('no{}_{}_policy_net.safetensors'.format(latest_model_num[c], model_prefixes[c]))
    load_model(policy_nets[c], os.path.join(args[0], 'ckpt', 'no{}_{}_policy_net.safetensors'.format(latest_model_num[c], model_prefixes[c])))
    policy_nets[c].to(device)

  datasets = SelfPlay([AiPlayer(device, policy_nets[i], 0) for i in range(2)], Game()).generate_dataset()

  for dataset in datasets:
    for l in dataset['next_board_state']:
      for x in l:
        if x == -1:
          print('e', end=' ')
        elif x == 0:
          print('b', end=' ')
        else:
          print('w', end=' ')
      print()
    print(dataset['player'])
    print(dataset['action'])
    print()
