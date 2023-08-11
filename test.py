import numpy as np
from play import *
from player import *
from model import *
import torch

# play = SelfPlay([AiPlayer(PolicyNet()), AiPlayer(PolicyNet())], Game())
# print(play.generate_dataset())

# player = AiPlayer(PolicyNet())
# board_state = np.random.randint(-1, 2, (9, 9), dtype=np.int8)
# print(board_state)
# print(
#   player.step(
#     board_state
#   )
# )

print(torch.sum(torch.Tensor([[1, 2, 3], [2, 3, 4]]), axis=1))
