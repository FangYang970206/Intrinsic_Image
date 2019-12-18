import os, glob, random
import torch
import torch.utils.data as Data
import scipy.misc
import numpy as np


class Shapenet_Dataset(Data.Dataset):
    def __init__(self, directory, size_per_dataset, mode):
        self.size_per_dataset = size_per_dataset
        self.mode = mode
        self.datasets = directory
        
        if mode == 'test':
            self.selections = ['input', 'albedo', 'shading', 'shape']
        print(self.selections)
        print(self.datasets)
        print(os.listdir(os.path.join(self.datasets, self.selections[0])))
        self.alldata_files = glob.glob(os.path.join(os.path.join(self.datasets, self.selections[0]), '*.png'))
        print(len(self.alldata_files))
        self.image_names = []
        for i in range(len(self.alldata_files)):
            self.image_names.append(self.alldata_files[i].split('\\')[-1])
        random.shuffle(self.image_names)
        print(len(self.image_names))

    def __read_image(self, img_name, sel):
        path = os.path.join(os.path.join(self.datasets, sel), img_name)
        if sel == 'shading':
            from skimage import io
            img = io.imread(path, as_grey=True)
            img = img[np.newaxis, :, :]
        else:
            img = scipy.misc.imread(path, mode='RGB')
            img = img.transpose(2,0,1) / 255.
        return img

    def __getitem(self, idx):
        outputs = []
        for sel in self.selections:
            img_name = self.image_names[idx]
            out = self.__read_image(img_name, sel)
            outputs.append(out)
        return outputs

    def __getitem__(self, idx):
        outputs = self.__getitem(idx)
        return outputs

    def __len__(self):
        return self.size_per_dataset
