import torch.nn as nn

def init_weights(model, method):
    if not method:
        print('Init method: default (?)')
        return
    elif method == 'kaiming-uniform':
        print('Init method: kaiming-uniform.')
        model.apply(kaiming_uniform)
    elif method == 'kaiming-normal':
        print('Init method: kaiming-normal.')
        model.apply(kaiming_normal)
    elif method == 'xavier-uniform':
        print('Init method: xavier-uniform.')
        model.apply(xavier_uniform)
    elif method == 'xavier-normal':
        print('Init method: xavier-normal.')
        model.apply(xavier_normal)
    else:
        raise ValueError(f'Init method {method} not implemented.')

def kaiming_uniform(m):
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight)

def kaiming_normal(m):
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)

def xavier_uniform(m):
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def xavier_normal(m):
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.xavier_normal_(m.weight)
