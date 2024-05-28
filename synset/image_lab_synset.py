from synset.base_synset import BaseImageSynSet
import torch
import torch.nn.functional as F
from typing import Callable, Union, List, Tuple
from torch import Tensor
from torch.utils.data import DataLoader
from kornia.enhance import ZCAWhitening as ZCA
from augment.augment import DiffAug
import numpy as np
from utils import get_optimizer

class ImageLabSynSet(BaseImageSynSet):

    def __init__(self,
                 channel:int,
                 num_classes:int,
                 image_size:Tuple[int,int],
                 ipc:int,
                 zca:ZCA=None,
                 device='cpu',
                 train_images:bool=True,
                 images_opt_args:dict=None,
                 train_targets:bool=True,
                 targets_opt_args:dict=None,
                 init_type:str='noise_normal',
                 real_loader:DataLoader=None,
                 augment_args:dict = None):
        super().__init__(channel,
                         num_classes,
                         image_size,
                         ipc,
                         zca,
                         device)
        self.targets:Tensor = F.one_hot(self.labels, num_classes).to(torch.float)
        self.train_images = train_images
        self.train_targets = train_targets
        self.images_opt_args = images_opt_args
        self.targets_opt_args = targets_opt_args
        if train_images:
            self.trainables['images'] = self.images
            assert self.images_opt_args is not None
        if train_targets:
            self.trainables['targets'] = self.targets
            assert self.targets_opt_args is not None
        self.init_type = init_type
        init = self.init_type.lower().split('_')
        if 'noise' in init:
            if 'normal' in init:
                self.noise_init(True)
            else:
                self.noise_init(normalize=False)
        elif 'real' in init:
            if real_loader is None:
                raise AssertionError("chose real init, but real loader is None!")
            else:
                self.real_init(real_loader)
        else:
            raise ValueError("unrecognized initialization type {}".format(self.init_type))
        
        self.augment_args = augment_args
        if self.augment_args is None:
            self.augment = DiffAug(strategy='')
        else:
            self.augment = DiffAug(**self.augment_args)

        self.seed_shift = np.random.randint(10000)
        self.train()
        return None


    def __getitem__(self, idxes):
        return self.images[idxes], self.targets[idxes]
        
    def to(self, device):
        self.images = self.images.to(device)
        self.targets = self.targets.to(device)
        self.trainables['images'] = self.images
        self.trainables['targets'] = self.targets
        return None
    
    def batch(self, batch_idx:int, batch_size:int, class_idx:int=None, tracked:bool=True):
        if class_idx is None:
            sampler = self.sampler
            images = self.images
            targets = self.targets
            if batch_size<self.num_items:
                idxes = sampler.sample_idxes(batch_idx, batch_size)
                imgs = images[idxes]
                tgts = targets[idxes]
            else:
                imgs = images
                tgts = targets
        else:
            sampler = self.class_samplers[class_idx]
            start_idx = class_idx * self.ipc
            end_idx = start_idx + self.ipc
            images = self.images[start_idx:end_idx]
            targets = self.targets[start_idx:end_idx]
            if batch_size<self.ipc:
                idxes = sampler.sample_idxes(batch_idx, batch_size)
                imgs = images[idxes]
                tgts = targets[idxes]
            else:
                imgs = images
                tgts = targets
        
        if tracked:
            seed = self.seed_shift + batch_idx * batch_size
        else:
            seed = -1
        
        imgs = self.augment(imgs, seed=seed)
        return imgs, tgts
    
    def shuffle(self, shuffle_classes: bool = False):
        super().shuffle(shuffle_classes)
        self.seed_shift = np.random.randint(10000)
    
    def make_optimizers(self):
        opt_dct = dict()
        if self.train_images:
            opt_dct['images'] = get_optimizer([self.images], **self.images_opt_args)
        if self.train_targets:
            opt_dct['targets'] = get_optimizer([self.targets], **self.targets_opt_args)
        return opt_dct