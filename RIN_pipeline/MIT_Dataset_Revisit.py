import os
import random

import torch
import torch.utils.data as Data
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


class MIT_Dataset_Revisit(Data.Dataset):
    def __init__(self, FileName, mode='train', sel=['reflectance', 'shading', 'mask'], transform = None, refl_multi_size=None, shad_multi_size=None, image_size=None, fullsize=False,):
        self.FileName = FileName
        self.toTensor = ToTensor()
        self.mode = mode
        self.sel = sel
        self.names = []
        self.refl_multi_size = refl_multi_size
        self.shad_multi_size = shad_multi_size
        self.image_size = image_size
        self.fullsize = fullsize
        with open(FileName) as f:
            for line in f.readlines():
                self.names.append(line)

    def __getitem__(self, idx):
        inp_path = self.names[idx].strip()
        albedo_path = self.names[idx].replace('input', self.sel[0]).strip()
        shading_path = self.names[idx].replace('input', self.sel[1]).strip()
        mask_path = self.names[idx].replace('input', self.sel[2]).strip()

        if self.fullsize:
            input_image = Image.open(inp_path).convert('RGB')
            albedo_image = Image.open(albedo_path).convert('RGB')
            shading_image = Image.open(shading_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')
            input_image = self._pad(input_image)
            albedo_image = self._pad(albedo_image)
            shading_image = self._pad(shading_image)
            mask = self._pad(mask)
        else:
            if self.mode == 'train':
                input_image = Image.open(inp_path).resize((256, 256)).convert('RGB')
                albedo_image = Image.open(albedo_path).resize((256, 256)).convert('RGB')
                shading_image = Image.open(shading_path).resize((256, 256)).convert('RGB')
                mask = Image.open(mask_path).resize((256, 256)).convert('RGB')
            else:
                input_image = Image.open(inp_path).convert('RGB')
                albedo_image = Image.open(albedo_path).convert('RGB')
                shading_image = Image.open(shading_path).convert('RGB')
                mask = Image.open(mask_path).convert('RGB')

        if self.mode == 'train':
            if random.random() < 0.5:
                input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
                albedo_image = albedo_image.transpose(Image.FLIP_LEFT_RIGHT)
                shading_image = shading_image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        if self.refl_multi_size:
            h, w = albedo_image.size
            albedo_frame1 = albedo_image.resize((h // 4, w // 4))
            albedo_frame2 = albedo_image.resize((h // 2, w // 2))
        if self.shad_multi_size:
            h, w = shading_image.size
            shading_frame1 = shading_image.resize((h // 4, w // 4))
            shading_frame2 = shading_image.resize((h // 2, w // 2))
        if self.refl_multi_size and self.shad_multi_size:
            return self.toTensor(input_image), self.toTensor(albedo_image), self.toTensor(shading_image), \
                   self.toTensor(mask), [self.toTensor(albedo_frame1), self.toTensor(albedo_frame2)], \
                                        [self.toTensor(shading_frame1), self.toTensor(shading_frame2)]
        if self.refl_multi_size:
            return self.toTensor(input_image), self.toTensor(albedo_image), self.toTensor(shading_image), \
                   self.toTensor(mask), [self.toTensor(albedo_frame1), self.toTensor(albedo_frame2)]
        if self.shad_multi_size:
            return self.toTensor(input_image), self.toTensor(albedo_image), self.toTensor(shading_image), \
                   self.toTensor(mask), [self.toTensor(shading_frame1), self.toTensor(shading_frame2)]
        else:
            return self.toTensor(input_image), self.toTensor(albedo_image), self.toTensor(shading_image), self.toTensor(mask)
    
    def __len__(self):
        return len(self.names)

    def _pad(self, img):
        arr = np.array(img)
        h, w, ch = arr.shape
        h_pad = 16 - h % 16
        w_pad = 16 - w % 16
        if h_pad != 0:
            arr = np.concatenate((arr, np.zeros((h_pad,w,ch)).astype('uint8')), axis=0)
        if w_pad != 0:
            arr = np.concatenate((arr, np.zeros((h+h_pad,w_pad,ch)).astype('uint8')), axis=1)
        img = Image.fromarray(arr.astype('uint8')).convert('RGB')
        return img
        
        