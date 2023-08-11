# 1

import numpy as np

class Game:
  def __init__ (self):
    self.board_state = np.ones((9, 9), dtype=np.int8) * -1
    self.is_over = False;
    self.winner = 0

  # c = color: black -> 0, white -> 1
  def step (self, x, y, c):
    self.board_state[y][x] = c
    self.update_over_state()

  def update_over_state (self):
    if self.check_draw():
      self.is_over = True
      self.winner = -1
      return

    dir = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]])
    for i in range(9):
      for j in range(9):
        for d in dir:
          l = list(
            filter(
              lambda x: x[0] >= 0 and x[0] < 9 and x[1] >= 0 and x[1] < 9 and self.board_state[x[1]][x[0]] != -1,
              [np.array([j, i]) + d * k for k in range(5)]
            )
          )
          if len(l) < 5:
            continue
          c_list = list(map(lambda x: self.board_state[x[1]][x[0]], l))
          if c_list.count(c_list[0]) >= 5:
            self.is_over = True
            self.winner = c_list[0]
            return

  def check_draw (self):
    for i in range(9):
      for j in range(9):
        if self.board_state[i][j] == -1:
          return False
    return True