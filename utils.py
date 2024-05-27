import torch
from torch.optim import SGD, Adam
from models.convnet import convnet3

def get_model(modelname:str, num_classes:int=10, channel:int=3, image_size:int=(32,32), **kwargs):
    if modelname.lower()=='convnet3':
        return convnet3(channel, num_classes, image_size)
    else:
        raise NotImplementedError
    
def get_optimizer(params, opt_name:str, **kwargs):
    if opt_name.lower()=='sgd':
        return SGD(params=params, **kwargs)
    elif opt_name.lower()=='adam':
        return Adam(params=params, **kwargs)
    else:
        raise NotImplementedError