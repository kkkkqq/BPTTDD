import torch
from torch.utils.data import Dataset
from typing import Tuple, List
import kornia as K

class ImageDataSet():

    def __init__(self):
        self.dataset_name:str = None
        self.data_path:str = None
        self.channel:int=None
        self.num_classes:int=None
        self.image_size:Tuple[int,int] = None
        self.mean:list = None
        self.std:list = None
        self.zca:bool = False
        self.zca_trans:K.enhance.ZCAWhitening = None
        self.dst_train:Dataset = None
        self.dst_test:Dataset = None
