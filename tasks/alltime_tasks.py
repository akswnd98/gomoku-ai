# 1

from tasks.task import *
from play import *

class PlayTask (Task):
  def __init__ (self, players, context):
    super(PlayTask, self).__init__(context)
    self.players = players
  
  def run (self):
    datasets = SelfPlay(self.players, Game()).generate_dataset()
    for c in range(2):
      self.context['datasets'][c] += list(filter(lambda x: x['player'] == c, datasets))
