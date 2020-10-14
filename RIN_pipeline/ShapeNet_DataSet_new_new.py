import os, glob, random
import torch
import torch.utils.data as Data
import scipy.misc
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


class ShapeNet_Dateset_new_new(Data.Dataset):
    def __init__(self, directory, size_per_dataset, mode, image_size, remove_names, refl_multi_size=None, shad_multi_size=None):
        self.size_per_dataset = size_per_dataset
        self.mode = mode
        self.datasets = os.path.join(directory, self.mode)
        self.image_size = image_size
        self.selections = ['albedo', 'shading', 'input', 'mask']
        self.alldata_files = glob.glob(os.path.join(os.path.join(self.datasets, 'albedo'), '*.png'))
        self.image_names = []
        self.refl_multi_size = refl_multi_size
        self.shad_multi_size = shad_multi_size
        for i in range(len(self.alldata_files)):
            if self.alldata_files[i].split('\\')[-1] not in remove_names:
                self.image_names.append(self.alldata_files[i].split('\\')[-1])
        random.shuffle(self.image_names)
        self.image_names = self.image_names[:self.size_per_dataset]
        self.toTensor = ToTensor()

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        albedo_path = os.path.join(self.datasets, 'albedo', img_name)
        shading_path = os.path.join(self.datasets, 'shading', img_name)
        mask_path = os.path.join(self.datasets, 'mask', img_name)
        
        albedo_image = Image.open(albedo_path).resize((self.image_size, self.image_size)).convert('RGB')
        shading_image = Image.open(shading_path).resize((self.image_size, self.image_size)).convert('RGB')
        mask = Image.open(mask_path).resize((self.image_size, self.image_size)).convert('RGB')

        input_image = self.toTensor(albedo_image) * self.toTensor(shading_image)

        if self.refl_multi_size:
            albedo_frame1 = albedo_image.resize((self.image_size // 4, self.image_size // 4))
            albedo_frame2 = albedo_image.resize((self.image_size // 2, self.image_size // 2))
        if self.shad_multi_size:
            shading_frame1 = shading_image.resize((self.image_size // 4, self.image_size // 4))
            shading_frame2 = shading_image.resize((self.image_size // 2, self.image_size // 2))
        if self.refl_multi_size and self.shad_multi_size:
            return input_image, self.toTensor(albedo_image), self.toTensor(shading_image), \
                   self.toTensor(mask), [self.toTensor(albedo_frame1), self.toTensor(albedo_frame2)], \
                                        [self.toTensor(shading_frame1), self.toTensor(shading_frame2)]
        if self.refl_multi_size:
            return input_image, self.toTensor(albedo_image), self.toTensor(shading_image), \
                   self.toTensor(mask), [self.toTensor(albedo_frame1), self.toTensor(albedo_frame2)]
        if self.shad_multi_size:
            return input_image, self.toTensor(albedo_image), self.toTensor(shading_image), \
                   self.toTensor(mask), [self.toTensor(shading_frame1), self.toTensor(shading_frame2)]
        else:
            return input_image, self.toTensor(albedo_image), self.toTensor(shading_image), self.toTensor(mask)

    def __len__(self):
        return self.size_per_dataset


if __name__ == "__main__":
    import time
    directory = 'F:\\ShapeNet'
    size_per_dataset = 200
    mode = 'val'
    remove_names = os.listdir('F:\\ShapeNet\\remove')
    dset = ShapeNet_Dateset_new_new(directory, size_per_dataset, mode,image_size=256, remove_names=remove_names)
    dataload = torch.utils.data.DataLoader(dset, batch_size=16, num_workers=4)
    print('done init')
    time_0 = time.time()
    for i, inp in enumerate(dataload):
        print(len(inp))
        print(i, inp[0].size(), inp[1].size())
        print(torch.sum(inp[1]*inp[2] - inp[0]))
    print('total time: ', time.time() - time_0)