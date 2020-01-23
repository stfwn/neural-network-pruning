#! /bin/python
import os
import json
from collections import defaultdict
import numpy as np

def get_runs():
    runs = []
    for filename in os.listdir('.'):
        if '.log' in filename:
            with open(filename) as fp:
                run = json.loads(fp.read())
                if run in runs:
                    print(f'Duplicate run log! {filename} not counted.')
                runs.append(run)
    for run in runs:
        run['train_loss'] = np.array(run['train_loss'])
        run['train_acc'] = np.array(run['train_acc'])
        run['test_loss'] = np.array(run['test_loss'])
        run['test_acc'] = np.array(run['test_acc'])
    return runs

def filter_runs(runs, model=None, init=None):
    filtered = runs
    if model:
        filtered = [run for run in runs if run['args']['model'] == model]
    if init:
        filtered = [run for run in filtered if run['args']['initialization'] == init]
    return filtered

if __name__ == "__main__":
    main()
