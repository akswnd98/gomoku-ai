# 1

import numpy as np
from game import Game

class SelfPlay:
  def __init__ (self, players, game):
    self.players = players
    self.game = game
  
  def generate_dataset (self):
    turn = 0
    board_states = []
    next_board_states = []
    actions = []
    while not self.game.is_over:
      board_states.append(self.game.board_state.copy())
      action = self.players[turn].step(self.game.board_state)
      actions.append(action)
      self.game.step(*action, turn)
      next_board_states.append(self.game.board_state.copy())
      turn ^= 1
    return [
      {
        'board_state': board_state,
        'action': action,
        'next_board_state': next_board_state,
        'player': player,
        'reward': reward
      } for board_state, action, next_board_state, player, reward in zip(board_states, actions, next_board_states, [i % 2 for i in range(len(board_states))], [0] * (len(board_states) - 1) + [1])
    ]
