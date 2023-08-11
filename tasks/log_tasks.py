# 1

from tasks.task import *

class TrainedTriggerTask (TriggerTask):
  def __init__ (self, color, task, context):
    super(TrainedTriggerTask, self).__init__(task, context)
    self.color = color
    self.last_trained_num = context['trained_nums'][color]
  
  def run (self):
    super(TrainedTriggerTask, self).run()
    self.last_trained_num = self.context['trained_nums'][self.color]

  def check_condition(self):
    return self.last_trained_num < self.context['trained_nums'][self.color]

class LogTrainedNumTask (Task):
  def __init__ (self, color, context):
    super(LogTrainedNumTask, self).__init__(context)
    self.color = color

  def run (self):
    print('{} trained num: {}'.format(['black', 'white'][self.color], self.context['trained_nums'][self.color]))

class LogTrainLossTask (Task):
  def __init__ (self, color, context):
    super(LogTrainLossTask, self).__init__(context)
    self.color = color

  def run (self):
    print('{} train loss: {}'.format(['black', 'white'][self.color], self.context['statistics']['losses'][self.color][-1]))

# class LogWinningRateTask (Task):
#   def run (self):
#     print('total games: {}'.format(self.context.statistics.total_games))
#     print('black wins: {}'.format(self.context.statistics.black_wins))
#     print('white wins: {}'.format(self.context.statistics.white_wins))
#     print(
#       'draw: {}'.format(
#         self.context.statistics.total_games
#         - self.context.statistics.black_wins
#         - self.context.statistics.white_wins
#       )
#     )
