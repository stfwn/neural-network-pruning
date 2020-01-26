import torch.nn as nn

def init_weights(model, method):
    if not method:
        print('Init method: default (?)')
        return
    try:
        fn = methods[method]
        print(f'Init method: {method}')
        model.apply(fn)
    except:
        raise ValueError(f'Unkown init method {method}.')

""" The initial 4 schemes. """
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

""" Expanded selection on 2020-01-26. """
def uniform(m):
    """ Uniform in (0.0, 1.0) """
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.uniform_(m.weight)

def normal(m):
    """ Normal with mean 0. and std 1. """
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.normal_(m.weight)

def xavier_uniform_double(m):
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.xavier_uniform_(m.weight, gain=2)

def xavier_uniform_half(m):
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.xavier_uniform_(m.weight, gain=0.5)

def xavier_normal_double(m):
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.xavier_normal_(m.weight, gain=2)

def xavier_normal_half(m):
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.xavier_normal_(m.weight, gain=0.5)

methods = {
        'kaiming-uniform': kaiming_uniform,
        'kaiming-normal': kaiming_normal,
        'xavier-uniform': xavier_uniform,
        'xavier-normal': xavier_normal,
        'uniform': uniform,
        'normal': normal,
        'xavier-uniform-double': xavier_uniform_double,
        'xavier-uniform-half': xavier_uniform_half,
        'xavier-normal-double': xavier_normal_double,
        'xavier-normal-half': xavier_normal_half
        }
