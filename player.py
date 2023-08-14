# 1

import numpy as np
import torch
from functools import reduce

class Player:
  def step (self, board_state):
    pass

class AiPlayer (Player):
  def __init__ (self, device, policy_net, epsilon):
    self.device = device
    self.policy_net = policy_net
    self.epsilon = epsilon

  def step (self, board_state):
    with torch.no_grad():
      policy = self.policy_net(torch.from_numpy(board_state).view(1, 1, 9, 9).type(torch.float32).to(self.device))
      policy = torch.reshape(policy, (-1, )).cpu().numpy()
    possible = list(filter(
      lambda z: z[2] == -1,
      zip(
        np.arange(0, 9 * 9),
        policy,
        np.reshape(board_state, (9 * 9, ))
      )
    ))
    policy_idx = reduce(lambda a, b: a if a[1] > b[1] else b, possible, (0, -1, 0))[0]
    if np.random.choice([0, 1], 1, p=[1 - self.epsilon, self.epsilon]) == 0:
      idx = policy_idx
    else:
      a = [z[0] for z in possible]
      a.remove(policy_idx)
      idx, = np.random.choice(a, 1, p=np.ones((len(a), )) / len(a))

    return idx % 9, idx // 9
