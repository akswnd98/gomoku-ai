from tasks.task import *
from tasks.initial_tasks import *
from tasks.train_tasks import *
from tasks.alltime_tasks import *
from tasks.save_tasks import *
from tasks.log_tasks import *

class PipelineTask (InfiniteLoopTask):
  def __init__ (self, path, device, players, enough_size, policy_nets, value_nets, hdf, context):
    super(PipelineTask, self).__init__(
      SequentialTask([ # initial task
        WelcomePrinter(context)
      ], context),
      SequentialTask([ # loop task
        PlayTask(players, context),
        *[EnoughDataTriggerTask(c, enough_size, TrainTask(device, c, policy_nets, value_nets, 32, 1, context), context) for c in range(2)],
        *[TrainedTriggerTask(
          c,
          SequentialTask([
            LogTrainedNumTask(c, context),
            LogTrainLossTask(c, context),
          ], context),
          context
        ) for c in range(2)],
        *[RegularSaveTriggerTask(
          c,
          16384,
          SequentialTask([
            NoInterruptWhileSavingTask(context),
            SaveModelTask(path, c, policy_nets[c], value_nets[c], context),
            SaveTrainedNumTask(c, hdf, context),
            SaveLossTask(c, hdf, context),
            SavingOverTask(context),
            UpdatePrevSaveTimeTask(c, 16384, context)
          ], context),
          context
        ) for c in range(2)]
      ], context),
      context
    )
