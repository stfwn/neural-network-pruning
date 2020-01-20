import torch.nn as nn

def init_weights(model, method):
    if not method:
        print('Init method: default (?)')
        return
    methods = {
            'kaiming-uniform': kaiming_uniform,
            'kaiming-normal': kaiming_normal,
            'xavier-uniform': xavier_uniform,
            'xavier-normal': xavier_normal,
            }
    try:
        fn = methods[method]
        print(f'Init method: {method}')
        model.apply(fn)
    except:
        raise ValueError(f'Unkown init method {method}.')

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
