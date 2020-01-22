#! /bin/python
import os
import json
from collections import defaultdict

def main():
    runs = []
    for filename in os.listdir('.'):
        if '.log' in filename:
            with open(filename) as fp:
                run = json.loads(fp.read())
                if run in runs:
                    print(f'Duplicate run log! {filename} not counted.')
                runs.append(run)

    counts = defaultdict(lambda: defaultdict(int))
    for run in runs:
        for arg in run['args'].keys():
            if arg in ['load_last_pretrained', 'disable_cuda', 'forget_model']:
                continue
            val = run['args'][arg]
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
