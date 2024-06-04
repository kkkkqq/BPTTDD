from synset.base_synset import BaseImageSynSet
import torch
import torch.nn.functional as F
from typing import Callable, Union, List, Tuple
from torch import Tensor
from torch.utils.data import DataLoader
from kornia.enhance import ZCAWhitening as ZCA
from augment.augment import DiffAug
import numpy as np
from utils import get_optimizer, get_model

class ImageLabelerSynSet(BaseImageSynSet):

    def __init__(self,
                 channel:int,
                 num_classes:int,
                 image_size:Tuple[int,int],
                 ipc:int,
                 labeler_args:dict,
                 labeler_path:str=None,
                 zca:ZCA=None,
                 device='cuda',
                 train_images:bool=True,
                 images_opt_args:dict=None,
                 train_labeler:bool=True,
                 labeler_opt_args:dict=None,
                 init_type:str='noise_normal',
                 real_loader:DataLoader=None,
                 augment_args:dict = None,
                 classwise:bool=False):
        super().__init__(channel,
                         num_classes,
                         image_size,
                         ipc,
                         zca,
                         device,
                         classwise)
        self.labeler_args = labeler_args
        self.labeler_path = labeler_path
        if 'channel' not in labeler_args:
            labeler_args.update({'channel': channel,
                                 'num_classes': num_classes,
                                 'image_size': image_size})
        self.labeler = get_model(**labeler_args).to(self.device)
        if self.labeler_path is not None:
            self.labeler.load_state_dict(torch.load(self.labeler_path))
        self.train_images = train_images
        self.train_labeler = train_labeler
        self.images_opt_args = images_opt_args
        self.labeler_opt_args = labeler_opt_args
        if train_images:
            self.trainables['images'] = self.get_images_lst()
            assert self.images_opt_args is not None
        if train_labeler:
            self.trainables['labeler'] = list(self.labeler.parameters())
            assert self.labeler_opt_args is not None
        self.init_type = init_type
        init = self.init_type.lower().split('_')
        self.real_loader = real_loader
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

        self._labels = None
        return None


    def __getitem__(self, idxes):
        if not self.classwise:
            images = self.images[idxes]
        else:
            raise NotImplementedError
        return images, self.labeler
        
    def to(self, device):
        super().to(device)
        self.trainables['images'] = self.get_images_lst()
        self.labeler.to(self.device)
        self.trainables['labeler'] = list(self.labeler.parameters())
        return None
    
    def batch(self, batch_idx:int, batch_size:int, class_idx:int=None, tracked:bool=True, soft_targets:bool=True):
        if not soft_targets:
            if self._labels is None:
                self._labels = self.labels.to(self.device)
            else:
                if self._labels.device != self.device:
                    self._labels = self.labels.to(self.device)

        if class_idx is None:
            if self.classwise:
                if batch_size >= self.num_items:
                    imgs = torch.cat(self.images_lst, dim=0)
                    if not soft_targets:
                        tgts = self._labels
                else:
                    raise NotImplementedError("currently only support full batch for classwise synsets")
            else:
                sampler = self.sampler
                images = self.images
                if batch_size<self.num_items:
                    idxes = sampler.sample_idxes(batch_idx, batch_size)
                    imgs = images[idxes]
                    if not soft_targets:
                        tgts = self._labels[idxes]
                else:
                    imgs = images
                    if not soft_targets:
                        tgts = self._labels
        else:
            sampler = self.class_samplers[class_idx]
            start_idx = class_idx * self.ipc
            end_idx = start_idx + self.ipc 
            if self.classwise:
                images = self.images_lst[class_idx]
            else: 
                images = self.images[start_idx:end_idx]
            if batch_size<self.ipc:
                idxes = sampler.sample_idxes(batch_idx, batch_size)
                imgs = images[idxes]
                if not soft_targets:
                    tgts = self._labels[start_idx:end_idx][idxes]
            else:
                imgs = images
                if not soft_targets:
                    tgts = self._labels[start_idx:end_idx]
        
        if tracked:
            seed = self.seed_shift + batch_idx * batch_size
        else:
            seed = -1
        
        imgs = self.augment(imgs, seed=seed)
        if not soft_targets:
            if len(tgts.shape)==1:
                tgts = torch.nn.functional.one_hot(tgts, num_classes = self.num_classes).to(torch.float)
        else:
            tgts = torch.nn.functional.softmax(self.labeler(imgs),1)
        return imgs, tgts
    
    def shuffle(self, shuffle_classes: bool = False):
        super().shuffle(shuffle_classes)
        self.seed_shift = np.random.randint(10000)
    
    def make_optimizers(self):
        opt_dct = dict()
        if self.train_images:
            opt_dct['images'] = get_optimizer(self.get_images_lst(), **self.images_opt_args)
        if self.train_labeler:
            opt_dct['labeler'] = get_optimizer(self.labeler.parameters(), **self.labeler_opt_args)
        return opt_dct

    def train(self):
        super().train()
        self.labeler.train()
        return None
    
    def eval(self):
        super().eval()
        self.labeler.eval()
        return None