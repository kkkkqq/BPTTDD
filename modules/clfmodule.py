from torch.nn.modules import Module
from modules.basemodule import BaseModule
from augment.augment import DiffAug
import torch
from torch import Tensor
import torch.nn as nn


class ClassifierModule(BaseModule):

    def __init__(self, aug_args:dict=None):
        if aug_args is not None:
            self.augment = DiffAug(**aug_args)
        else:
            self.augment = DiffAug(strategy='')
        self.criterion = nn.CrossEntropyLoss()
        return None
    
    def forward_loss(self, backbone: Module, images:Tensor, targets:Tensor) -> tuple:
        device = next(backbone.parameters()).device
        if images.device != device:
            images = images.to(device)
        if targets.device != device:
            targets = targets.to(device)
        out = backbone(self.augment(images))
        loss = self.criterion(out, targets)
        return loss, out, targets
    
    def post_loss(self, loss:Tensor, out:Tensor, targets:Tensor):
        batchsize = out.shape[0]
        out_argmax = torch.argmax(out, dim=1).to(torch.long)
        if len(targets.shape)>1:
            targets_argmax = torch.argmax(out, dim=1).to(torch.long)
        acc = torch.sum(torch.eq(out_argmax, targets_argmax))
        return batchsize, {'loss':loss, 'acc':acc}



        