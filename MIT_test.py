import os
import random
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.backends import cudnn
import numpy as np
import scipy.misc

# import RIN
import RIN
import RIN_pipeline

from utils import *

def main():
    random.seed(9999)
    torch.manual_seed(9999)
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',          type=str,   default='MIT_logs\\RIID_origin_RIN_updateLR0.0005_4_bf_cosLoss_VGG0.1_400epochs_bs22\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--state_dict',         type=str,   default='composer_state_200.t7')
    args = parser.parse_args()

    check_folder(args.save_path)

    # pylint: disable=E1101
    device = torch.device("cuda: 1" if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101
    shader = RIN.Shader()
    reflection = RIN.Decomposer()
    composer = RIN.Composer(reflection, shader).to(device)

    MIT_test_txt = 'MIT_TXT\\MIT_BarronSplit_test.txt'
    MIT_test_fullsize_txt = 'MIT_TXT\\MIT_BarronSplit_fullsize_test.txt'

    composer.load_state_dict(torch.load(os.path.join(args.save_path, args.state_dict)))
    print('load checkpoint success!')

    test_set = RIN_pipeline.MIT_Dataset_Revisit(MIT_test_txt, mode='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)
    
    check_folder(os.path.join(args.save_path, "refl_target"))
    check_folder(os.path.join(args.save_path, "shad_target"))
    check_folder(os.path.join(args.save_path, "refl_output"))
    check_folder(os.path.join(args.save_path, "shad_output"))
    check_folder(os.path.join(args.save_path, "shape_output"))
    check_folder(os.path.join(args.save_path, "mask"))

    fullsize_names = []
    with open(MIT_test_fullsize_txt, 'r') as f:
        for line in f.readlines():
            fullsize_names.append(line)

    composer.eval()
    with torch.no_grad():
        for ind, tensors in enumerate(test_loader):
            print(ind)
            inp = [t.to(device) for t in tensors]
            input_g, _, _, mask_g = inp
            _, albedo_fake, shading_fake, shape_fake = composer.forward(input_g)
            # albedo_fake  = albedo_fake*mask_g
            # shading_fake = shading_fake*mask_g
            shape_fake = shape_fake* mask_g

            # lab_refl_targ = albedo_g.squeeze().cpu().numpy().transpose(1,2,0)
            # lab_sha_targ = shading_g.squeeze().cpu().numpy().transpose(1,2,0)
            # mask = mask_g.squeeze().cpu().numpy().transpose(1,2,0)
            refl_pred = albedo_fake.squeeze().cpu().numpy().transpose(1,2,0)
            sha_pred = shading_fake.squeeze().cpu().numpy().transpose(1,2,0)
            shape_pred = shape_fake.squeeze().cpu().numpy().transpose(1,2,0)

            # lab_refl_targ = np.clip(lab_refl_targ, 0, 1)
            # lab_sha_targ = np.clip(lab_sha_targ, 0, 1)
            refl_pred = np.clip(refl_pred, 0, 1)
            sha_pred = np.clip(sha_pred, 0, 1)
            shape_pred = np.clip(shape_pred, 0, 1)
            # mask = np.clip(mask, 0, 1)

            scipy.misc.imsave(os.path.join(args.save_path, "refl_output", "{}.png".format(ind)), refl_pred)
            scipy.misc.imsave(os.path.join(args.save_path, "shad_output", "{}.png".format(ind)), sha_pred)
            scipy.misc.imsave(os.path.join(args.save_path, "shape_output", "{}.png".format(ind)), shape_pred)

            # inp_path = fullsize_names[ind].strip()
            albedo_path = fullsize_names[ind].replace('input', 'reflectance').strip()
            shading_path = fullsize_names[ind].replace('input', 'shading').strip()
            mask_path = fullsize_names[ind].replace('input', 'mask').strip()

            refl_label = scipy.misc.imread(albedo_path)
            shad_label = scipy.misc.imread(shading_path)
            mask_label = scipy.misc.imread(mask_path)

            refl_output = scipy.misc.imread(os.path.join(args.save_path, "refl_output", "{}.png".format(ind)))
            shad_output = scipy.misc.imread(os.path.join(args.save_path, "shad_output", "{}.png".format(ind)))
            shape_output = scipy.misc.imread(os.path.join(args.save_path, "shape_output", "{}.png".format(ind)))

            refl_output = scipy.misc.imresize(refl_output, refl_label.shape)
            shad_output = scipy.misc.imresize(shad_output, shad_label.shape)
            shape_output = scipy.misc.imresize(shape_output, shad_label.shape)

            scipy.misc.imsave(os.path.join(args.save_path, "refl_target", "{}.png".format(ind)), refl_label)
            scipy.misc.imsave(os.path.join(args.save_path, "shad_target", "{}.png".format(ind)), shad_label)
            scipy.misc.imsave(os.path.join(args.save_path, "mask", "{}.png".format(ind)), mask_label)
            scipy.misc.imsave(os.path.join(args.save_path, "refl_output", "{}.png".format(ind)), refl_output)
            scipy.misc.imsave(os.path.join(args.save_path, "shad_output", "{}.png".format(ind)), shad_output)
            scipy.misc.imsave(os.path.join(args.save_path, "shape_output", "{}.png".format(ind)), shape_output)
            

if __name__ == "__main__":
    main()
