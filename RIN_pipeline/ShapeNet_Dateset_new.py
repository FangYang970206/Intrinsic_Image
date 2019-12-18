import os, glob, random
import torch
import torch.utils.data as Data
import scipy.misc
import numpy as np


class ShapeNet_Dateset_new(Data.Dataset):
    def __init__(self, directory, size_per_dataset, mode, img_size, remove_names):
        self.size_per_dataset = size_per_dataset
        self.mode = mode
        self.datasets = os.path.join(directory, self.mode)
        self.img_size = img_size
        if mode == 'train':
            self.selections = ['albedo', 'shading', 'input', 'mask']
        elif mode == 'val':
            self.selections = ['albedo', 'shading', 'input', 'mask']
        elif mode == 'test':
            self.selections = ['albedo', 'shading', 'input', 'mask']
        self.alldata_files = glob.glob(os.path.join(os.path.join(self.datasets, self.selections[0]), '*.png'))
        self.image_names = []
        for i in range(len(self.alldata_files)):
            if self.alldata_files[i].split('\\')[-1] not in remove_names:
                self.image_names.append(self.alldata_files[i].split('\\')[-1])
        random.shuffle(self.image_names)
        print("imageset length: %d" % len(self.image_names))

    def __read_image(self, img_name, sel):
        path = os.path.join(os.path.join(self.datasets, sel), img_name)
        img = scipy.misc.imread(path, mode='RGB')
        # img = scipy.misc.imresize(img, self.img_size) #resize image!
        if sel == 'sha':
            img = img[np.newaxis, :, :]/255.
        else:
            img = img.transpose(2,0,1) / 255.
        return img

    def __getitem(self, idx):
        outputs = []
        if self.mode == "train":
            for sel in self.selections:
                if sel == 'input':
                    out = outputs[0]*outputs[1]
                    outputs.append(out)
                else:
                    img_name = self.image_names[idx]
                    out = self.__read_image(img_name, sel)
                    outputs.append(out)
        elif self.mode == 'val':
            for sel in self.selections:
                img_name = self.image_names[idx]
                out = self.__read_image(img_name, sel)
                outputs.append(out)
        else:
            for sel in self.selections:
                if sel == 'input':
                    out = outputs[0]*outputs[1]
                    outputs.append(out)
                else:
                    img_name = self.image_names[idx]
                    out = self.__read_image(img_name, sel)
                    outputs.append(out)
        return outputs

    def __getitem__(self, idx):
        outputs = self.__getitem(idx)
        return outputs[2], outputs[0], outputs[1], outputs[3]

    def __len__(self):
        return self.size_per_dataset


if __name__ == "__main__":
    import time
    directory = 'F:\\BOLD'
    size_per_dataset = 200
    mode = 'val'
    dset = ShapeNet_Dateset_new(directory, size_per_dataset, mode)
    print(dset.selections)
    dataload = torch.utils.data.DataLoader(dset, batch_size=16, num_workers=4)
    print('done init')
    time_0 = time.time()
    for i, inp in enumerate(dataload):
        print(len(inp))
        print(i, inp[0].size(), inp[1].size())
    print('total time: ', time.time() - time_0)