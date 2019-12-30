import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.backends import cudnn
import torchvision.transforms as transforms

import RIN_new as RIN
import RIN_pipeline
import numpy as np
import scipy.misc

from utils import *

def main():
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',              type=str,   default='SceneSplit')
    parser.add_argument('--save_path',          type=str,   default='MPI_logs_new\\GAN_RIID_updateLR3_epoch160_CosbfVGG_SceneSplit_refl-se-skip_shad-se-low_multi_new_shadSqueeze_grad\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--refl_checkpoint',    type=str,   default='refl_checkpoint')
    parser.add_argument('--shad_checkpoint',    type=str,   default='shad_checkpoint')
    parser.add_argument('--state_dict_refl',         type=str,   default='composer_reflectance_state_81.t7')
    parser.add_argument('--state_dict_shad',         type=str,   default='composer_shading_state_81.t7')
    parser.add_argument('--refl_skip_se',       type=StrToBool,  default=False)
    parser.add_argument('--shad_skip_se',       type=StrToBool,  default=False)
    parser.add_argument('--refl_low_se',        type=StrToBool,  default=False)
    parser.add_argument('--shad_low_se',        type=StrToBool,  default=False)
    parser.add_argument('--refl_multi_size',    type=StrToBool,  default=False)
    parser.add_argument('--shad_multi_size',    type=StrToBool,  default=False)
    parser.add_argument('--refl_detach_flag',   type=StrToBool,  default=False)
    parser.add_argument('--shad_detach_flag',   type=StrToBool,  default=False)
    parser.add_argument('--shad_squeeze_flag',  type=StrToBool,  default=False)
    parser.add_argument('--refl_reduction',     type=StrToInt,   default=8)
    parser.add_argument('--shad_reduction',     type=StrToInt,   default=8)
    parser.add_argument('--cuda',               type=str,       default='cuda')
    parser.add_argument('--fullsize',           type=StrToBool, default=False)
    args = parser.parse_args()

    device = torch.device(args.cuda)
    reflectance = RIN.SEDecomposerSingle(multi_size=args.refl_multi_size, low_se=args.refl_low_se, skip_se=args.refl_skip_se, detach=args.refl_detach_flag, reduction=args.refl_reduction).to(device)
    shading = RIN.SEDecomposerSingle(multi_size=args.shad_multi_size, low_se=args.shad_low_se, skip_se=args.shad_skip_se, se_squeeze=args.shad_squeeze_flag, reduction=args.shad_reduction, detach=args.shad_detach_flag).to(device)
    reflectance.load_state_dict(torch.load(os.path.join(args.save_path, args.refl_checkpoint, args.state_dict_refl)))
    shading.load_state_dict(torch.load(os.path.join(args.save_path, args.shad_checkpoint, args.state_dict_shad)))
    print('load checkpoint success!')
    composer = RIN.SEComposer(reflectance, shading, args.refl_multi_size, args.shad_multi_size).to(device)

    if args.fullsize:
        print('test fullsize....')
        MPI_Image_Split_test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_imageSplit-fullsize-ChenSplit-test.txt'
        MPI_Scene_Split_test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_sceneSplit-fullsize-NoDefect-test.txt'
        h, w = 436,1024
        pad_h,pad_w = clc_pad(h,w,16)
        print(pad_h, pad_w)
        tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
        tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
    else:
        print('test size256....')
        MIT_test_txt = 'MIT_TXT\\MIT_BarronSplit_test.txt'

    if args.fullsize:
        test_txt = MPI_Image_Split_test_txt
    else:
        test_txt = MIT_test_txt

    test_set = RIN_pipeline.MIT_Dataset_Revisit(test_txt, mode='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)

    if args.fullsize:
        check_folder(os.path.join(args.save_path, "refl_target_fullsize"))
        check_folder(os.path.join(args.save_path, "refl_output_fullsize"))
        check_folder(os.path.join(args.save_path, "shad_target_fullsize"))
        check_folder(os.path.join(args.save_path, "shad_output_fullsize"))
        check_folder(os.path.join(args.save_path, "mask"))
    else:
        check_folder(os.path.join(args.save_path, "refl_target"))
        check_folder(os.path.join(args.save_path, "shad_target"))
        check_folder(os.path.join(args.save_path, "refl_output"))
        check_folder(os.path.join(args.save_path, "shad_output"))
        check_folder(os.path.join(args.save_path, "mask"))

    ToPIL = transforms.ToPILImage()

    composer.eval()
    with torch.no_grad():
        for ind, tensors in enumerate(test_loader):
            print(ind)
            inp = [t.to(device) for t in tensors]
            input_g, albedo_g, shading_g, mask_g = inp
            if args.fullsize:
                input_g = tmp_pad(input_g)
            if args.refl_multi_size and args.shad_multi_size:
                albedo_fake, shading_fake, _, _ = composer.forward(input_g)
            elif args.refl_multi_size or args.shad_multi_size:
                albedo_fake, shading_fake, _ = composer.forward(input_g)
            else:
                albedo_fake, shading_fake = composer.forward(input_g)
            if args.fullsize:
                albedo_fake, shading_fake = tmp_inversepad(albedo_fake), tmp_inversepad(shading_fake)

            albedo_fake  = albedo_fake * mask_g
            shading_fake = shading_fake * mask_g

            albedo_fake = albedo_fake.cpu().clamp(0, 1)
            shading_fake = shading_fake.cpu().clamp(0, 1)
            albedo_g = albedo_g.cpu().clamp(0, 1)
            shading_g = shading_g.cpu().clamp(0, 1)

            lab_refl_targ = ToPIL(albedo_g.squeeze())
            lab_sha_targ = ToPIL(shading_g.squeeze())
            refl_pred = ToPIL(albedo_fake.squeeze())
            sha_pred = ToPIL(shading_fake.squeeze())
            mask_g = ToPIL(mask_g.cpu().squeeze())

            lab_refl_targ.save(os.path.join(args.save_path, "refl_target_fullsize" if args.fullsize else "refl_target", "{}.png".format(ind)))
            lab_sha_targ.save(os.path.join(args.save_path, "shad_target_fullsize" if args.fullsize else "shad_target", "{}.png".format(ind)))
            refl_pred.save(os.path.join(args.save_path, "refl_output_fullsize" if args.fullsize else "refl_output", "{}.png".format(ind)))
            sha_pred.save(os.path.join(args.save_path, "shad_output_fullsize" if args.fullsize else "shad_output", "{}.png".format(ind)))
            mask_g.save(os.path.join(args.save_path, "mask", "{}.png".format(ind)))

if __name__ == "__main__":
    main()
