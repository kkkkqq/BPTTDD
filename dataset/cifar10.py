import torch
from torch.utils.data import Dataset, TensorDataset
import kornia as K
from dataset.baseset import ImageDataSet
from torchvision import datasets, transforms
import tqdm

class CIFAR10(ImageDataSet):

    def __init__(self, data_path:str, zca:bool=False):
        super().__init__()
        self.dataset_name = 'cifar10'
        self.data_path = data_path
        self.channel = 3
        self.num_classes = 10
        self.image_size = (32,32)
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.zca = zca
        if self.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])
        self.dst_train = datasets.CIFAR10(data_path, True, transform, None, False)
        self.dst_test = datasets.CIFAR10(data_path, False, transform, None, False)
        if self.zca:
            dst_train = self.dst_train
            dst_test = self.dst_test
            images = []
            labels = []
            print("Train ZCA")
            for i in tqdm.tqdm(range(len(dst_train))):
                im, lab = dst_train[i]
                images.append(im)
                labels.append(lab)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            images = torch.stack(images, dim=0).to(device)
            labels = torch.tensor(labels, dtype=torch.long, device="cpu")
            zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
            zca.fit(images)
            zca_images = zca(images).to("cpu")
            self.dst_train = TensorDataset(zca_images, labels)

            images = []
            labels = []
            print("Test ZCA")
            for i in tqdm.tqdm(range(len(dst_test))):
                im, lab = dst_test[i]
                images.append(im)
                labels.append(lab)
            images = torch.stack(images, dim=0).to(device)
            labels = torch.tensor(labels, dtype=torch.long, device="cpu")

            zca_images = zca(images).to("cpu")
            self.dst_test = TensorDataset(zca_images, labels)

            self.zca_trans = zca
        return None

        
