import torch
from torch.optim import SGD, Adam
from models.convnet import convnet3
from models.vaes.cvae import ConditionalVAE
from dataset.cifar10 import CIFAR10

def get_model(modelname:str, num_classes:int=10, channel:int=3, image_size:int=(32,32), **kwargs):
    if modelname.lower()=='convnet3':
        return convnet3(channel=channel, num_classes=num_classes, image_size=image_size)
    elif modelname.lower()=='cvae':
        return ConditionalVAE(in_channels=channel,
                              num_classes=num_classes,
                              latent_dim=512,
                              img_size=image_size[0])
    else:
        raise NotImplementedError
    
def get_optimizer(params, opt_name:str, **kwargs):
    if opt_name.lower()=='sgd':
        return SGD(params=params, **kwargs)
    elif opt_name.lower()=='adam':
        return Adam(params=params, **kwargs)
    else:
        raise NotImplementedError
    
def get_dataset(dataset_name:str, data_path:str, *args, **kwargs):
    if dataset_name.lower() == 'cifar10':
        return CIFAR10(data_path=data_path, *args, **kwargs)
    else:
        raise NotImplementedError
    
