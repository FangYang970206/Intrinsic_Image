import os
import torch
import scipy.misc as misc
import torch.nn as nn


def main():
    target_path = 'F:\\BOLD\\resize_test'
    pred_path = 'F:\\BOLD\\test_result'
    selections = ['orig', 'refl', 'sha']

    for sel in selections:
        img_names = os.listdir(os.path.join(target_path, sel))[:1]
        loss = 0.0
        for ind, img_name in enumerate(img_names):
            target_img = misc.imread(os.path.join(os.path.join(target_path, sel), img_name))
            pred_img = misc.imread(os.path.join(os.path.join(pred_path, sel), img_name))
            if sel == 'sha':
                loss += ((pred_img/255. - target_img/255.)**2).sum()
            else:
                loss += ((pred_img - target_img)**2).sum()
        loss = loss/(ind+1)
        print('{}_loss:{}\n'.format(sel, loss))
        with open('mse.txt', 'r+') as f:
            f.write('{}_loss:{}\n'.format(sel, loss)) 


if __name__ == '__main__':
    main()