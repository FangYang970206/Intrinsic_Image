import os
import random

import torch
import torch.utils.data as Data
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


class MPI_Dataset_Revisit(Data.Dataset):
    def __init__(self, FileName, mode='train', sel=['albedo', 'shading', 'mask'], transform = None, refl_multi_size=None, shad_multi_size=None, image_size=None):
        self.FileName = FileName
        self.toTensor = ToTensor()
        self.sel = sel
        self.transform = transform
        self.names = []
        self.refl_multi_size = refl_multi_size
        self.shad_multi_size = shad_multi_size
        self.image_size = image_size
        with open(FileName) as f:
            for line in f.readlines():
                self.names.append(line)

    def __getitem__(self, idx):
        inp_path = self.names[idx].strip()
        if "input" in inp_path:
            albedo_path = self.names[idx].replace('input', self.sel[0]).strip()
            shading_path = self.names[idx].replace('input', self.sel[1]).strip()
            mask_path = self.names[idx].replace('input', self.sel[2]).strip()
        if "clean" in inp_path:
            albedo_path = self.names[idx].replace('clean', self.sel[0]).strip()
            shading_path = self.names[idx].replace('clean', self.sel[1]).strip()
            mask_path = self.names[idx].replace('clean', self.sel[2]).strip()

        input_image = Image.open(inp_path).convert('RGB')
        albedo_image = Image.open(albedo_path).convert('RGB')
        shading_image = Image.open(shading_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        if self.refl_multi_size:
            albedo_frame1 = albedo_image.resize((self.image_size // 4, self.image_size // 4))
            albedo_frame2 = albedo_image.resize((self.image_size // 2, self.image_size // 2))
        if self.shad_multi_size:
            shading_frame1 = shading_image.resize((self.image_size // 4, self.image_size // 4))
            shading_frame2 = shading_image.resize((self.image_size // 2, self.image_size // 2))
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
            if self.transform is not None:
                # print("transform...")
                return self.transform(input_image, albedo_image, shading_image, mask)
            else:
                return self.toTensor(input_image), self.toTensor(albedo_image), self.toTensor(shading_image), self.toTensor(mask)
    
    def __len__(self):
        return len(self.names)


if __name__ == "__main__":
    directory = 'F:\\sintel'
    device = 'cpu'
    from MPI_transform import MPI_Test_Agumentation
    test_transform = MPI_Test_Agumentation()
    dataset = MPI_Dataset(directory, mode='test', transform=test_transform)
    dataload = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for loader in dataload:
        labeled = [t.to(device) for t in loader]
        print(labeled[0].size, labeled[1].size())
        
        