from tasks.task import *
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TrainTask (Task):
  def __init__ (self, device, color, policy_nets, batch_size, gamma, context):
    super(TrainTask, self).__init__(context)
    self.device = device
    self.color = color
    self.policy_nets = policy_nets
    self.batch_size = batch_size
    self.gamma = gamma
    self.optimizer = optim.Adam(self.policy_nets[color].parameters(), lr=1e-4)

  def run (self):
    dataset = self.context['datasets'][self.color]
    random.shuffle(dataset)
    train_num = len(dataset) // self.batch_size * self.batch_size
    self.losses = []
    for i in range(0, train_num, self.batch_size):
      self.train(dataset[i: i + self.batch_size])

    self.context['datasets'][self.color] = dataset[len(dataset) - len(dataset) % self.batch_size: ]
    self.context['trained_nums'][self.color] += train_num
    self.context['statistics']['losses'][self.color].append(np.average(np.array(self.losses)))
  
  def train (self, dataset):
    rewards = torch.from_numpy(np.array([x['reward'] for x in dataset])).to(self.device)
    board_states = torch.unsqueeze(torch.from_numpy(np.array([x['board_state'] for x in dataset], dtype=np.float32)), 1).to(self.device)

    policies = self.policy_nets[self.color](board_states)
    actions = torch.from_numpy(np.array([x['action'][0] + x['action'][1] * 9 for x in dataset], dtype=np.int64)).to(self.device)
    one_hot_action = F.one_hot(actions, num_classes=81)
    action_prob = torch.sum(one_hot_action * policies)
    policy_loss = torch.log(action_prob + 1e-7) * rewards
    policy_loss = -torch.mean(policy_loss)

    loss = policy_loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    with torch.no_grad():
      self.losses.append(loss.cpu().numpy())
