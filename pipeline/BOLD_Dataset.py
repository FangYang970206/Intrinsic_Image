import os, glob, random
import torch
import torch.utils.data as Data
import scipy.misc
import numpy as np


class BOLD_Dataset(Data.Dataset):
    '''
    use train data to supervision training,
    use val data to unsupervision training,
    use test data to test.
    '''
    def __init__(self, directory, size_per_dataset, mode):
        self.size_per_dataset = size_per_dataset
        self.mode = mode.split('_')[0]
        self.datasets = os.path.join(directory, self.mode)
        if mode == 'train_refl' or mode == 'test_refl':
            self.selections = ['orig', 'refl']
        elif mode == 'train_sha' or mode == 'test_sha':
            self.selections = ['orig', 'sha']
        elif mode == 'val' or mode == 'test':
            self.selections = ['orig', 'refl', 'sha']
        self.alldata_files = glob.glob(os.path.join(os.path.join(self.datasets, self.selections[0]), '*.png'))
        # print(len(self.alldata_files))
        self.image_names = []
        for i in range(len(self.alldata_files)):
            self.image_names.append(self.alldata_files[i].split('\\')[-1])
        random.shuffle(self.image_names)
        # print(len(self.image_names))

    def __read_image(self, img_name, sel):
        path = os.path.join(os.path.join(self.datasets, sel), img_name)
        img = scipy.misc.imread(path)
        if sel == 'sha':
            img = img[np.newaxis, :, :]/255.
        else:
            img = img.transpose(2,0,1) / 255.
        return img

    def __getitem(self, idx):
        outputs = []
        if self.mode == "train":
            for sel in self.selections:
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
                img_name = self.image_names[idx]
                out = self.__read_image(img_name, sel)
                outputs.append(out)
        return outputs

    def __getitem__(self, idx):
        outputs = self.__getitem(idx)
        return outputs

    def __len__(self):
        return self.size_per_dataset


if __name__ == "__main__":
    import time
    directory = 'F:\\BOLD'
    size_per_dataset = 200
    mode = 'val'
    dset = BOLD_Dataset(directory, size_per_dataset, mode)
    print(dset.selections)
    dataload = torch.utils.data.DataLoader(dset, batch_size=16, num_workers=4)
    print('done init')
    time_0 = time.time()
    for i, inp in enumerate(dataload):
        print(len(inp))
        print(i, inp[0].size(), inp[1].size())
    print('total time: ', time.time() - time_0)