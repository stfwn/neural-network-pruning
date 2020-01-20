import torch.nn as nn

def init_weights(model, method):
    if method == 'kaiming-normal':
        model.apply(kaiming_normal)
    else:
        raise ValueError(f'Init method {method} not implemented.')

def kaiming_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
