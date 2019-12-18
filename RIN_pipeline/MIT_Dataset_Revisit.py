import os
import random

import torch
import torch.utils.data as Data
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


class MIT_Dataset_Revisit(Data.Dataset):
    def __init__(self, FileName, mode='train', sel=['reflectance', 'shading', 'mask'], transform = None):
        self.FileName = FileName
        self.toTensor = ToTensor()
        self.mode = mode
        self.sel = sel
        self.names = []
        with open(FileName) as f:
            for line in f.readlines():
                self.names.append(line)

    def __getitem__(self, idx):
        inp_path = self.names[idx].strip()
        albedo_path = self.names[idx].replace('input', self.sel[0]).strip()
        shading_path = self.names[idx].replace('input', self.sel[1]).strip()
        mask_path = self.names[idx].replace('input', self.sel[2]).strip()

        input_image = Image.open(inp_path).resize((256, 256)).convert('RGB')
        albedo_image = Image.open(albedo_path).resize((256, 256)).convert('RGB')
        shading_image = Image.open(shading_path).resize((256, 256)).convert('RGB')
        mask = Image.open(mask_path).resize((256, 256)).convert('RGB')

        # if self.mode == 'train':
        #     if random.random() < 0.5:
        #         input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
        #         albedo_image = albedo_image.transpose(Image.FLIP_LEFT_RIGHT)
        #         shading_image = shading_image.transpose(Image.FLIP_LEFT_RIGHT)
        #         mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        #     if random.random() < 0.5:
        #         input_image = input_image.transpose(Image.FLIP_TOP_BOTTOM)
        #         albedo_image = albedo_image.transpose(Image.FLIP_TOP_BOTTOM)
        #         shading_image = shading_image.transpose(Image.FLIP_TOP_BOTTOM)
        #         mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return self.toTensor(input_image), self.toTensor(albedo_image), self.toTensor(shading_image), self.toTensor(mask)
    
    def __len__(self):
        return len(self.names)
        
        