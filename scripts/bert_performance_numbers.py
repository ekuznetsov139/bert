from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np
import sys


def get_event_acc(log_dir):
    event_acc = EventAccumulator(os.path.expanduser(log_dir))
    event_acc.Reload()
    return event_acc


def get_val(event_acc, scalar_name):
    w_times, step_nums, vals = zip(*event_acc.Scalars(scalar_name))
    return list(vals)


subfolders = [f.path for f in os.scandir("bert_perf_train") if f.is_dir()]

for x in subfolders:
    MI60_event_acc = get_event_acc(x)
    print(x + ": examples/sec avg:",
          np.mean(get_val(MI60_event_acc, "examples/sec")))
    print(x + ": global_step/sec avg:",
          np.mean(get_val(MI60_event_acc, "global_step/sec")))
    print(x + ": seconds per step avg:",
          1/np.mean(get_val(MI60_event_acc, "global_step/sec")))
