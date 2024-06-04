from torch.nn.modules import Module
from modules.basemodule import BaseModule
from augment.augment import DiffAug
import torch
from torch import Tensor
import torch.nn as nn
from models.vaes.cvae import ConditionalVAE
from typing import Tuple

class CVAEModule(BaseModule):

    def __init__(self, 
                 rescale_mean:Tuple[float,float,float],
                 rescale_std:Tuple[float,float,float],
                 aug_args:dict=None, 
                 kld_weight:float=1e-5, ):
        super().__init__()
        if aug_args is not None:
            self.augment = DiffAug(**aug_args)
        else:
            self.augment = DiffAug(strategy='')
        self.kld_weight = kld_weight
        with torch.no_grad():
            self.rescale_mean_tsr = torch.tensor(rescale_mean).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            self.rescale_std_tsr = torch.tensor(rescale_std).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        return None
    
    def forward_loss(self, backbone: Module, images:Tensor, targets:Tensor) -> tuple:
        if self._device is None:
            self._device = next(backbone.parameters()).device
        if images.device != self._device:
            images = images.to(self._device)
        if targets.device != self._device:
            targets = targets.to(self._device)
        images = self.augment(images)
        images = self.denormalize_images(images)
        recons, orig, mu, log_var = backbone(input=images, labels=targets)
        loss = ConditionalVAE.loss_function(recons, orig, mu, log_var, M_N=self.kld_weight)['loss']
        return loss, recons, orig, mu, log_var
    
    def post_loss(self, loss:Tensor, recons, mu, log_var):
        raise NotImplementedError

    def parse_batch(self, batch_out: tuple):
        return {'images':batch_out[0], 'targets':batch_out[1]}
    
    # def normalize_images(self, images:Tensor):
    #     '''scale images from (-1,1) to N(0,1))'''
    #     if self.rescale_mean_tsr.device != images.device:
    #         self.rescale_mean_tsr = self.rescale_mean_tsr.to(images.device)
    #         self.rescale_std_tsr = self.rescale_std_tsr.to(images.device)
    #     return images.sub(self.rescale_mean_tsr).div(self.rescale_std_tsr)

    def denormalize_images(self, images:Tensor):
        '''scale images from N(0,1) to (-1,1)'''
        if self.rescale_mean_tsr.device != images.device:
            self.rescale_mean_tsr = self.rescale_mean_tsr.to(images.device)
            self.rescale_std_tsr = self.rescale_std_tsr.to(images.device)
        return images.mul(self.rescale_std_tsr).add(self.rescale_mean_tsr).mul(2.).sub(1.)
    