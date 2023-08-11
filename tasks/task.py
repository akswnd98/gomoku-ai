# 1

from play import *
from game import *

class Task:
  def __init__ (self, context):
    self.context = context

  def run (self):
    pass

class SequentialTask (Task):
  def __init__ (self, tasks, *args, **kargs):
    super(SequentialTask, self).__init__(*args, **kargs)
    self.tasks = tasks

  def run (self):
    for task in self.tasks:
      task.run()

class InfiniteLoopTask (Task):
  def __init__ (self, initial_task, loop_task, *args, **kargs):
    super(InfiniteLoopTask, self).__init__(*args, **kargs)
    self.initial_task = initial_task
    self.loop_task = loop_task

  def run (self):
    self.initial_task.run()
    while True:
      self.loop_task.run()

class TriggerTask (Task):
  def __init__ (self, task, context):
    super(TriggerTask, self).__init__(context)
    self.task = task

  def run (self):
    if self.check_condition():
      self.task.run()

  def check_condition (self):
    pass
