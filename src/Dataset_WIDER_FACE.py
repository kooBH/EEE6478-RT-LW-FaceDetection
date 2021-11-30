import torch
import torchvision.transforms as transforms
import PIL
import os,glob
import numpy as np


class Dataset_WIDER_FACE(torch.utils.data.Dataset):
    def __init__(self, hp, is_train=True):
        self.hp = hp
        self.root = hp.data.root

        if is_train :
            self.list_data= [x for x in glob.glob(os.path.join(self.root,'WIDER_train','**','*.jpg'),recursive=True)]
        else :
            self.list_data= [x for x in glob.glob(os.path.join(self.root,'WIDER_test','**','*.jpg'),recursive=True)]
        self.convert = transforms.Compose(transforms.Resize(),transforms.ToTensor())

    def __getitem__(self, index):
        path_data = self.list_data[index]
        img = PIL.Image.open(path_data)

        pt = self.convert(img)

        # Process data_item if necessary.

        data = data_item

        return data

    def __len__(self):
        return len(self.data_list)


