from hzhu_gen import *

import torch, pickle, copy
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import os

class DataHandle(Dataset):
    
    def __init__(self, path):
        self.path = path
        self.init()
        
    def init(self):
        self.file_list = ls_file(path=self.path)
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, i):
        local_path = self.path+'/'+self.file_list[i]        
        return torch.load(local_path)
    
    def plot(self, i):
        data = self[i]
        plt.figure(figsize=(8,10))
        for i, key in enumerate(data):
            plt.subplot(2,3,1+i)
            shape = data[key].shape
            if len(shape)==3:
                plt.imshow(data[key][0,:,:])
            elif len(shape)==2:
                plt.imshow(data[key])
            plt.title('%s\n%s\n%.3f\n%.3f'%(data[key].dtype, data[key].shape, data[key].min(), data[key].max()))
    
class DataMaster:
    
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        name_list = ['Train','Test','Valid']
        num_workers = torch.get_num_threads()-1 if torch.get_num_threads()<=9 else 8
        self.handle = {item:DataHandle(self.path+'/'+item) for item in name_list}
        self.dataLoader = {item:DataLoader(
            self.handle[item], batch_size=self.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, prefetch_factor=4) for item in name_list}
        
    def __call__(self, key):
        return self.dataLoader[key]
    