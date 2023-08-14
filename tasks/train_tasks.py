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

class TrainTask (Task):
  def __init__ (self, device, color, policy_nets, value_nets, batch_size, gamma, context):
    super(TrainTask, self).__init__(context)
    self.device = device
    self.color = color
    self.policy_nets = policy_nets
    self.value_nets = value_nets
    self.batch_size = batch_size
    self.gamma = gamma
    self.optimizer = optim.Adam(list(self.policy_nets[color].parameters()) + list(self.value_nets[color].parameters()), lr=1e-4)

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
    with torch.no_grad():
      next_values = self.value_nets[self.color](torch.unsqueeze(torch.from_numpy(np.array([x['next_board_state'] for x in dataset], dtype=np.float32)).to(self.device), 1)).squeeze(1)
    rewards = torch.from_numpy(np.array([x['reward'] for x in dataset])).to(self.device)
    value_label = rewards + next_values * self.gamma
    board_states = torch.unsqueeze(torch.from_numpy(np.array([x['board_state'] for x in dataset], dtype=np.float32)), 1).to(self.device)
    values = self.value_nets[self.color](board_states).squeeze(1)
    value_loss = F.mse_loss(values, value_label, reduction='mean')
    
    with torch.no_grad():
      advantages = value_label - values
    policies = self.policy_nets[self.color](board_states)
    actions = torch.from_numpy(np.array([x['action'][0] + x['action'][1] * 9 for x in dataset], dtype=np.int64)).to(self.device)
    one_hot_action = F.one_hot(actions, num_classes=81)
    action_prob = torch.sum(one_hot_action * policies)
    policy_loss = torch.log(action_prob + 1e-7) * advantages
    policy_loss = -torch.mean(policy_loss)

    loss = value_loss + 0.2 * policy_loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    with torch.no_grad():
      self.losses.append(loss.cpu().numpy())
