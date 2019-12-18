import os
import argparse

import cv2
import torch
import torch.optim as optim
from torch.backends import cudnn
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt

import RIN
import RIN_pipeline


def main():
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',          type=str,   default='F:\\DB_BOLD\\test\\orig',
    help='base folder of datasets')
    parser.add_argument('--save_path',          type=str,   default='F:\\DB_BOLD\\test_result_RIN_attention_CosBF_VGG0.1')
    parser.add_argument('--checkpoint',         type=bool,  default=True)
    parser.add_argument('--img_resize_shape',   type=str,   default=(256, 256))
    parser.add_argument('--state_dict',         type=str,   default='logs_new\\RIN_DB_BOLD_CosBF_VGG0.1\\composer_state_99.t7')
    args = parser.parse_args()

    # pylint: disable=E1101
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        os.mkdir(os.path.join(args.save_path, 'orig'))
        os.mkdir(os.path.join(args.save_path, 'refl'))
        os.mkdir(os.path.join(args.save_path, 'shad'))
    
    shader = RIN.Shader()
    # shader.load_state_dict(torch.load('logs/shader/shader_state_59.t7'))
    decomposer = RIN.Decomposer()
    # reflection.load_state_dict(torch.load('reflection_state.t7'))
    composer = RIN.Composer(decomposer, shader).to(device)

    if args.checkpoint:
        composer.load_state_dict(torch.load(args.state_dict))
        print('load checkpoint success!')
    
    img_names = os.listdir(args.data_path)

    # img_names = img_names[:1]

    composer.eval()
    with torch.no_grad():
        for img_name in img_names:
            print(img_name)
            img = misc.imread(os.path.join(args.data_path, img_name))
            # img = misc.imresize(img, args.img_resize_shape)

            img = img.transpose(2,0,1) / 255.
            img = img[np.newaxis, :, :, :]
            img = torch.from_numpy(img).float().to(device)
            recon_pred, refl_pred, shad_pred, _ = composer.forward(img)

            recon_pred = np.clip(recon_pred.squeeze().detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            refl_pred = np.clip(refl_pred.squeeze().detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            shad_pred = np.clip(shad_pred.repeat(1, 3, 1, 1).squeeze().detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            shad_pred = cv2.cvtColor(shad_pred, cv2.COLOR_BGR2GRAY)

            misc.imsave(os.path.join(args.save_path+'\\orig', img_name), recon_pred)
            misc.imsave(os.path.join(args.save_path+'\\refl', img_name), refl_pred)
            misc.imsave(os.path.join(args.save_path+'\\shad', img_name), shad_pred)

            # plt.imshow(shad_pred, cmap='Greys')
            # plt.show()


if __name__ == "__main__":
    main()
