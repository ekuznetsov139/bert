from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np
import sys
import argparse


def get_event_acc(log_dir):
    event_acc = EventAccumulator(os.path.expanduser(log_dir))
    event_acc.Reload()
    return event_acc


def get_val(event_acc, scalar_name):
    w_times, step_nums, vals = zip(*event_acc.Scalars(scalar_name))
    return list(vals)


parser = argparse.ArgumentParser()
parser.add_argument('bert_train_dir', type=str, help='bert training directory')
args = parser.parse_args()


if args.bert_train_dir:
    try:
        event_acc = get_event_acc(args.bert_train_dir)
        print(args.bert_train_dir + ": examples/sec avg:",
              np.mean(get_val(event_acc, "examples/sec")))
        print(args.bert_train_dir + ": global_step/sec avg:",
              np.mean(get_val(event_acc, "global_step/sec")))
        print(args.bert_train_dir + ": seconds per step avg:",
              1/np.mean(get_val(event_acc, "global_step/sec")))
    except Exception as e:
        print(e)
        print("Make sure the input directory contains tfevents data")
