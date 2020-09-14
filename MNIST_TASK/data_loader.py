import torch.utils.data as data
from os.path import exists, join, split
from os import listdir
from os.path import join
from PIL import Image
import numpy as np
from torchvision import transforms
import torch


class DatasetFromFolder(data.Dataset):

    def __init__(self, data_feature, data_target,transform=None,phase='train'):
        self.data_feature = data_feature
        self.data_target = data_target
        #self.mean_X_train = mean_X_train
        #self.std_X_train = std_X_train
        #self.mean_y_train = mean_y_train
        #self.std_y_train = std_y_train
        self.transform =transform
        #self.transformed_target = self.transforms_target()
        #self.transformed_target_test = self.transforms_target_test()
        self.phase=phase

    def __len__(self):
        return len(self.data_feature)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)

        #data_feature = torch.from_numpy(self.data_feature[index]).float()
        #data_target =  torch.from_numpy(self.data_target[index]).float()
        data_feature = (self.data_feature[index])
        data_target = (self.data_target[index])

        if self.transform:
            sample = self.transform(data_feature)

        return sample,data_target


        return data_feature,data_target

