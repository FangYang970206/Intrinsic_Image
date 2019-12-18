import os, glob, random
import cv2
import torch
import torch.utils.data as Data
import scipy.misc
import numpy as np


class MIT_Dataset(Data.Dataset):
    def __init__(self, directory, size_per_dataset, mode, type_names):
        self.directory = directory
        self.size_per_dataset = size_per_dataset
        self.mode = mode
        self.type_names = type_names
        if mode == 'test':
            self.selections = ['diffuse', 'reflectance', 'shading', 'mask']
        if mode == 'train':
            self.inp = []
            self.refl = []
            self.shading = []
            self.mask = []
            self.selections = ['diffuse', 'light01', 'light02', 'light03', 
                               'light04', 'light05', 'light06', 'light07', 
                               'light08', 'light09', 'light10']
            for type_name in type_names:
                for sel in self.selections:
                    self.inp.append(type_name + '_' + sel + '.png')
                    self.refl.append(type_name + '_' + 'reflectance.png')
                    self.mask.append(type_name + '_' + 'mask.png')
                    if sel == 'diffuse':
                        self.shading.append(type_name + '_' + 'shading.png')
                    else:
                        self.shading.append(type_name + '_shad_' + sel + '.png')

    def __read_image(self, img_name, sel):
        path = os.path.join(self.directory, img_name)
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256))
        if sel == 'shading':
            img = img.transpose(2,0,1)
            img = img[0, :, :]
            img = img[np.newaxis, :, :]/255.
        else:
            img = img.transpose(2,0,1) / 255.
        return img

    def __getitem(self, idx):
        outputs = []
        if self.mode == 'test':
            for sel in self.selections:
                img_name = self.type_names[idx]
                img_name = img_name + '_' + sel + '.png'
                out = self.__read_image(img_name, sel)
                outputs.append(out)
        if self.mode == 'train':
            outputs.append(self.__read_image(self.inp[idx], 'inp'))
            outputs.append(self.__read_image(self.refl[idx], 'refl'))
            outputs.append(self.__read_image(self.shading[idx], 'shading'))
            outputs.append(self.__read_image(self.mask[idx], 'mask'))
        return outputs

    def __getitem__(self, idx):
        outputs = self.__getitem(idx)
        return outputs

    def __len__(self):
        return self.size_per_dataset
