# 1

import numpy as np
import torch

class Player:
  def step (self, board_state):
    pass

class AiPlayer (Player):
  def __init__ (self, device, policy_net):
    self.device = device
    self.policy_net = policy_net

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
      possible_probability = np.array([z[1] for z in possible])
      possible_probability = possible_probability / possible_probability.sum()
      idx, = np.random.choice([z[0] for z in possible], 1, p=possible_probability)
      return idx % 9, idx // 9
