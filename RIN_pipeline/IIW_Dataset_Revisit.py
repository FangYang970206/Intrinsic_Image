import os
import random

import torch
import torch.utils.data as Data
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


class IIW_Dataset_Revisit(Data.Dataset):
    '''
    out_mode : tensor or txt
    '''
    def __init__(self, FileName, transform = None, out_mode='tensor'):
        self.FileName = FileName
        self.toTensor = ToTensor()
        self.transform = transform
        self.out_mode = out_mode
        self.names = []
        with open(FileName) as f:
            for line in f.readlines():
                self.names.append(line)

    def __getitem__(self, idx):
        inp_path = self.names[idx].strip()
        input_image = Image.open(inp_path).convert('RGB')
        label_txt = inp_path.replace('png', 'txt')
        
        if self.out_mode == 'tensor':
            label = []
            with open(label_txt) as f:
                for line in f.readlines():
                    strs = line.split(',')
                    for i in range(len(strs)):
                        strs[i] = float(strs[i])
                    label.append(strs)
            return self.toTensor(input_image), torch.from_numpy(np.array(label))
        else:
            return self.toTensor(input_image), label_txt
    
    def __len__(self):
        return len(self.names)


if __name__ == "__main__":
    directory = 'F:\\revisit_IID\\iiw-dataset\\iiw_Learning_Lightness_train.txt'
    dataset = IIW_Dataset_Revisit(directory)
    dataload = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for loader in dataload:
        labeled = [t for t in loader]#if isinstance(t, str) else t.to('cuda') 
        # print(labeled)
        img, label = labeled
        print(img.size(), label[0])
        
        