from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import pdb
import h5py
import numpy as np
import torch
import scipy

class ARAD1K(Dataset):
    def __init__(self, data_path="./data/ARAD1K", data_type = 'mat', transform = None, device = 'cuda'):
        self.data_path = sorted(glob(os.path.join(data_path, f'*.{data_type}')))
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        try:
            with h5py.File(self.data_path[idx], 'r') as f:
                scene = np.array(f['cube'][5:])
            scene = torch.from_numpy(scene).float()
        except Exception as e:
            print(f"[open failed] {self.data_path[idx]}: {e}")

        if self.transform:
            scene = self.transform(scene)

        return {'scene': scene.to(self.device)}
    
class ICVL(Dataset):
    def __init__(self, data_path='./data/ICVL', data_type = 'mat', transform = None, device = 'cuda'):
        self.data_path = sorted(glob(os.path.join(data_path, f'*.{data_type}')))
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        try:
            with h5py.File(self.data_path[idx], 'r') as f:
                scene = np.array(f['rad'][5:])
            scene /= np.max(scene)
            scene = torch.from_numpy(scene).float()
        except Exception as e:
            print(f"[open failed] {self.data_path[idx]}: {e}")

        if self.transform:
            scene = self.transform(scene)

        return {'scene': scene.to(self.device)}
    
class CAVE(Dataset):
    def __init__(self, data_path = './data/CAVE', data_type = 'png', transform = None, device = 'cuda'):
        self.data_path = sorted(glob(os.path.join(data_path, '*')))
        self.transform = transform
        self.device = device
        self.data_type = data_type

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        root_folder = self.data_path[idx]
        img_folder = os.path.join(root_folder, os.listdir(root_folder)[0])
        img_folder = sorted(glob(os.path.join(img_folder, f'*.{self.data_type}')))
        img_folder = img_folder[5:]
        scene = []
        for img_path in img_folder:
            img = np.array(Image.open(img_path))
            scene.append(img)
        scene = np.array(scene)
        scene = scene/np.max(scene)
        scene = torch.from_numpy(scene).float()

        if self.transform:
            scene = self.transform(scene)

        return {'scene': scene.to(self.device)}
    

   
class Harvard(Dataset):
    def __init__(self, data_path='./data/Harvard', data_type = 'mat', transform = None, device = 'cuda'):
        self.data_path = sorted(glob(os.path.join(data_path, f'*.{data_type}')))
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        try:
            scene = scipy.io.loadmat(self.data_path[idx])
            scene = scene['ref'][:,:,5:]
            scene /= np.max(scene)
            scene = torch.from_numpy(scene).float()
            scene = scene.permute((2,0,1))
        except Exception as e:
            print(f"[open failed] {self.data_path[idx]}: {e}")

        if self.transform:
            scene = self.transform(scene)

        return {'scene': scene.to(self.device)}


class KAUST(Dataset):
    def __init__(self, data_path="./data/KAUST", data_type = 'h5', transform = None, device = 'cuda'):
        self.data_path = sorted(glob(os.path.join(data_path, f'*.{data_type}')))
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        try:
            with h5py.File(self.data_path[idx], 'r') as f:
                scene = np.array(f['img\\'])[5:-3]
            scene /= np.max(scene)
            scene = torch.from_numpy(scene).float()
        except Exception as e:
            print(f"[open failed] {self.data_path[idx]}: {e}")

        if self.transform:
            scene = self.transform(scene)

        return {'scene': scene.to(self.device), 'path': self.data_path[idx]}