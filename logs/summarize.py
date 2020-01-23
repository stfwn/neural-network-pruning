#! /bin/python
import os
import json
from collections import defaultdict
import helpers

def main():
    runs = helpers.get_runs()
    counts = defaultdict(lambda: defaultdict(int))
    for run in runs:
        for arg in run['args'].keys():
            if arg in ['load_last_pretrained', 'disable_cuda', 'forget_model']:
                continue
            val = run['args'][arg.lower()]
            counts[arg][val] += 1

    for key in counts.keys():
        print(f'{key.upper()}S')
        for arg in counts[key].keys():
            if key in ['initialization']:
                print(f'{arg}\t\t{counts[key][arg]}')
            else:
                print(f'{arg}\t\t\t{counts[key][arg]}')




if __name__ == "__main__":
    main()
