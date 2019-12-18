import os
import random

import torch
import torch.utils.data as Data
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


sintel_scenes = dict(
    train=['alley_1', 'bamboo_1', 'bandage_1', 'cave_2', 'market_2', 'market_6', 'shaman_2', 'sleeping_1', 'temple_2'],
    test=['alley_2', 'bamboo_2', 'bandage_2', 'cave_4', 'market_5', 'mountain_1', 'shaman_3', 'sleeping_2', 'temple_3'])


class MPI_Dataset(Data.Dataset):
    def __init__(self, directory, mode='train', sel=['RS', 'albedo', 'shading', 'albedo_defect_mask'], transform = None):
        self.directory = directory
        self.mode = mode
        self.sel = sel
        self.sintel_type = sintel_scenes[mode]
        self.names = []
        self.transform = transform
        for scene_name in self.sintel_type:
            input_directory = os.path.join(self.directory, sel[1])
            input_names = os.listdir(os.path.join(input_directory, scene_name))
            for i in range(len(input_names)):
                self.names.append(os.path.join(scene_name, input_names[i]))

    def __getitem__(self, idx):
        inp_path = os.path.join(self.directory, self.sel[0])
        albedo_path = os.path.join(self.directory, self.sel[1])
        shading_path = os.path.join(self.directory, self.sel[2])
        albedo_mask_path = os.path.join(self.directory, self.sel[3])
        input_image = Image.open(os.path.join(inp_path, self.names[idx])).convert('RGB')
        albedo_image = Image.open(os.path.join(albedo_path, self.names[idx])).convert('RGB')
        shading_image = Image.open(os.path.join(shading_path, self.names[idx])).convert('RGB')
        albedo_mask = Image.open(os.path.join(albedo_mask_path, self.names[idx])).convert('RGB')
        return self.transform(input_image, albedo_image, shading_image, albedo_mask)
    
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
        
        