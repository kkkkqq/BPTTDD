import torch
from typing import Callable, Union, List, Tuple
from torch import Tensor
from torch.utils.data import DataLoader
from kornia.enhance import ZCAWhitening as ZCA
from torchvision.utils import make_grid
import numpy as np

class IdxSampler():

    def __init__(self, num_items, variety:int=100) -> None:
        self.num_items = num_items
        self.variety = variety
        self.idxes = [np.random.permutation(num_items) for _ in range(variety)]
        return None
    
    def shuffle(self):
        self.idxes = [np.random.permutation(self.num_items) for _ in range(self.variety)]
    
    def sample_idxes(self, batch_idx:int, batch_size:int):
        batch_per_line = self.num_items//batch_size
        line_idx = (batch_idx // batch_per_line)//self.variety
        start_idx = (batch_idx%batch_per_line)*batch_size
        end_idx = start_idx + batch_size
        idxes = self.idxes[line_idx][start_idx:end_idx]
        return idxes



class BaseSynSet():

    def __init__(self, num_items:int, device='cpu'):
        """
        Baseclass for synset.
        """
        self.trainables:dict = dict()
        self.num_items:int = num_items
        self.device = device
        self.sampler = IdxSampler(self.num_items)

    def __getitem__(self, idxes):
        raise NotImplementedError

    def __len__(self):
        return self.num_items
    
    def to(self, device):
        raise NotImplementedError
    
    def batch(self, batch_idx:int, batch_size:int, class_idx:int=None, tracked:bool=True):
        raise NotImplementedError

    def shuffle(self):
        self.sampler.shuffle()
        return None
    
    def eval(self):
        for trainable in self.trainables:
            if isinstance(trainable, Tensor):
                trainable.requires_grad_(False)
            elif isinstance(trainable, list) or isinstance(trainable, tuple):
                for itm in trainable:
                    itm.requires_grad_(False)
            elif isinstance(trainable, dict):
                for val in trainable.values():
                    val.requires_grad_(False)
            else:
                raise TypeError("values of synset.trainables must be Tensor|List[Tensor]|Dict[str,Tensor], but got ", trainable)
        return None
    
    def train(self):
        for trainable in self.trainables:
            if isinstance(trainable, Tensor):
                trainable.requires_grad_(True)
            elif isinstance(trainable, list) or isinstance(trainable, tuple):
                for itm in trainable:
                    itm.requires_grad_(True)
            elif isinstance(trainable, dict):
                for val in trainable.values():
                    val.requires_grad_(True)
            else:
                raise TypeError("values of synset.trainables must be Tensor|List[Tensor]|Dict[str,Tensor], but got ", trainable)
        return None
    
    def make_optimizers(self):
        """
        returns a dict of optimizers attached to self.trainables
        """
        raise NotImplementedError
    
class BaseImageSynSet(BaseSynSet):

    def __init__(self,
                 channel:int,
                 num_classes:int,
                 image_size:Tuple[int,int],
                 ipc:int,
                 zca:ZCA=None,
                 device='cuda',
                 classwise:bool=False):
        self.num_classes = num_classes
        self.ipc = ipc
        self.num_items = ipc * num_classes
        super().__init__(self.num_items, device)
        self.channel = channel
        self.image_size = image_size
        self.zca = zca
        self.classwise = classwise
        if not self.classwise:
            self.images:Tensor = torch.zeros((self.num_items, self.channel, *self.image_size)).to(self.device)
            
        else:
            self.images_lst:List[Tensor] = [torch.zeros((self.ipc, self.channel, *self.image_size)).to(self.device) for _ in range(self.num_classes)]
        self.labels:Tensor = torch.repeat_interleave(torch.arange(self.num_classes), self.ipc, dim=0)#labels are not targets
        self.class_samplers = [IdxSampler(ipc) for _ in range(self.num_classes)]
        return None
    
    def shuffle(self, shuffle_classes:bool=False):
        super().shuffle()
        if shuffle_classes:
            for smpl in self.class_samplers:
                smpl.shuffle()
        return None

    @staticmethod
    def zca_inverse_images(imgs:Tensor, zca_trans:ZCA):
        with torch.no_grad():
            imgs = zca_trans.inverse_transform(imgs)
        return imgs
    
    def zca_inverse(self, images:Tensor):
        if self.zca is not None:
            return self.zca_inverse_images(images.detach().clone(), self.zca)
        else:
            return images.detach().clone()
    
    @staticmethod
    def upsample_images(imgs:Tensor, repeats:int):
        with torch.no_grad():
            imgs = torch.repeat_interleave(imgs, repeats, 2)
            imgs = torch.repeat_interleave(imgs, repeats, 3)
        return imgs
    
    def upsample(self, images:Tensor):
        if self.image_size[0]<128:
            if self.image_size[0]<64:
                return self.upsample_images(images.detach().clone(), 4)
            else:
                return self.upsample_images(images.detach().clone(), 2)
        else:
            return images.detach().clone()

    @staticmethod
    def make_grid_images(imgs:Tensor, 
                         nrow:int=8, 
                         padding:int=2,
                         normalize:bool=False, 
                         value_range:Tuple[int,int]=None,
                         scale_each:bool=False,
                         pad_value:float=0,
                         **kwargs):
        with torch.no_grad():
            grids=make_grid(imgs,
                            nrow,
                            padding,
                            normalize,
                            value_range,
                            scale_each,
                            pad_value,
                            **kwargs)
        return grids
    
    def make_grid(self, images:Tensor):
        if self.num_classes>10:
            for i in range(10):
                if self.num_classes%(10-i)==0:
                    nrow = 10-i
                    break
        else:
            nrow = self.num_classes
        return self.make_grid_images(images.detach().clone(), nrow)
    
    @staticmethod
    def clip_images(imgs:Tensor,
                    clip_val:float):
        with torch.no_grad():
            std = torch.std(imgs)
            mean = torch.mean(imgs)
            imgs = torch.clip(imgs.detach().clone(), min = mean-clip_val*str, max = mean+clip_val*std)
        return imgs
    

    def clip(self, images:Tensor):
        return self.clip_images(images.detach().clone(), 2.5)
    
    def image_for_display(self, display_ipc:int, clip:bool=False)->Tensor:
        imgs = []
        if not self.classwise:
            for cls in range(self.num_classes):
                start_idx = cls*self.ipc
                end_idx = start_idx + display_ipc
                chunk = self.images[start_idx:end_idx].detach().clone()
                if len(chunk.shape)<4:
                    chunk.unsqueeze_(0)
                imgs.append(chunk)
        else:
            imgs = [self.images_lst[cat][:display_ipc] for cat in range(self.num_classes)]
        imgs = torch.cat(imgs, dim=0)
        imgs = self.zca_inverse(imgs)
        if clip:
            imgs = self.clip(imgs)
        imgs = self.upsample(imgs)
        imgs = self.make_grid(imgs)
        return imgs.detach().cpu()

    @staticmethod
    def noise_init_images(images:Tensor, normalize:bool=True):
        with torch.no_grad():
            images[:] = torch.randn_like(images)
            if normalize:
                images_norm = torch.norm(images, dim=(1,2,3), keepdim=True)
                images.div_(images_norm)
        return None
    
    def noise_init(self, normalize:bool=True):
        if not self.classwise:
            self.noise_init_images(self.images, normalize)
        else:
            for cat_imgs in self.images_lst:
                self.noise_init_images(cat_imgs, normalize)
        return None
    
    @staticmethod
    def real_init_images(images:Union[Tensor, List[Tensor]], 
                         labels:Tensor, 
                         ipc:int, 
                         num_classes:int, 
                         real_loader:DataLoader, 
                         classwise:bool=False):
        with torch.no_grad():
            if len(labels.shape)>1:
                labels = torch.argmax(labels, dim=1)
            cls_imgs_lst = [[] for _ in range(num_classes)]
            cls_num_lst = [0 for _ in range(num_classes)]
            for batch in real_loader:
                imgs = batch[0]
                labs = batch[1]
                if len(labs.shape)>1:
                    labs = torch.argmax(labs, dim=1)
                for cat in range(num_classes):
                    idxes = labs==cat
                    num_data = int(torch.sum(idxes))
                    cls_imgs = imgs[idxes]
                    if len(cls_imgs.shape)<4:
                        cls_imgs.unsqueeze_(0)
                    cls_imgs_lst[cat].append(cls_imgs)
                    cls_num_lst[cat] += num_data
                if all([num>=ipc for num in cls_num_lst]):
                    break
            cls_imgs_lst = [torch.cat(eles, dim=0) for eles in cls_imgs_lst]
            cls_imgs_lst = [ele[:ipc] for ele in cls_imgs_lst]
            if not classwise:
                for cat in range(num_classes):
                    images[labels==cat] = cls_imgs_lst[cat]
            else:
                for cat in range(num_classes):
                    images[cat][:] = cls_imgs_lst[cat]
        return None

    def real_init(self, real_loader:DataLoader):
        if not self.classwise:
            self.real_init_images(self.images, self.labels, self.ipc, self.num_classes, real_loader)
        else:
            self.real_init_images(self.images_lst, self.labels, self.ipc, self.num_classes, real_loader, True)
        return None
    
    def get_images_lst(self):
        if self.classwise:
            return self.images_lst
        else:
            return [self.images]
        
    def to(self, device):
        if self.classwise:
            self.images_lst = [img.to(device) for img in self.images_lst]
        else:
            self.images = self.images.to(device)
        return None
    
    




            






