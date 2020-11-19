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
    parser.add_argument('--save_path',          type=str,   default='logs_vqvae\\MIT_base_256x256_noRetinex_withBf_leakyrelu_BNUP_Sigmiod_inception_bs4_finetune_woMultiPredict\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--refl_checkpoint',    type=str,   default='refl_checkpoint')
    parser.add_argument('--shad_checkpoint',    type=str,   default='shad_checkpoint')
    parser.add_argument('--state_dict_refl',    type=str,   default='composer_reflectance_state.t7')
    parser.add_argument('--state_dict_shad',    type=str,   default='composer_shading_state.t7')
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
    parser.add_argument('--cuda',               type=str,        default='cuda')
    parser.add_argument('--fullsize',           type=StrToBool,  default=True)
    parser.add_argument('--shad_out_conv',      type=StrToInt,   default=3)
    parser.add_argument('--dataset',            type=str,        default='mit')
    parser.add_argument('--shapenet_g',         type=StrToBool,  default=False)
    parser.add_argument('--vq_flag',            type=StrToBool,  default=False)
    parser.add_argument('--use_tanh',           type=StrToBool,  default=False)
    parser.add_argument('--use_inception',      type=StrToBool,  default=True)
    parser.add_argument('--use_skip',           type=StrToBool,  default=True)
    parser.add_argument('--use_multiPredict',   type=StrToBool,  default=False)
    parser.add_argument('--vae',                type=StrToBool,  default=True)
    args = parser.parse_args()

    device = torch.device(args.cuda)
    if args.vae:
        reflectance = RIN.VQVAE(vq_flag=args.vq_flag, use_tanh=args.use_tanh, use_inception=args.use_inception, use_skip=args.use_skip, use_multiPredict=args.use_multiPredict).to(device)
        shading = RIN.VQVAE(vq_flag=args.vq_flag, use_tanh=args.use_tanh, use_inception=args.use_inception, use_skip=args.use_skip, use_multiPredict=args.use_multiPredict).to(device)
    else:
        reflectance = RIN.SEDecomposerSingle(multi_size=args.refl_multi_size, low_se=args.refl_low_se, skip_se=args.refl_skip_se, detach=args.refl_detach_flag, reduction=args.refl_reduction).to(device)
        shading = RIN.SEDecomposerSingle(multi_size=args.shad_multi_size, low_se=args.shad_low_se, skip_se=args.shad_skip_se, se_squeeze=args.shad_squeeze_flag, reduction=args.shad_reduction, detach=args.shad_detach_flag, last_conv_ch=args.shad_out_conv).to(device)
    reflectance.load_state_dict(torch.load(os.path.join(args.save_path, args.refl_checkpoint, args.state_dict_refl)))
    shading.load_state_dict(torch.load(os.path.join(args.save_path, args.shad_checkpoint, args.state_dict_shad)))
    print('load checkpoint success!')
    composer = RIN.SEComposer(reflectance, shading, args.refl_multi_size, args.shad_multi_size).to(device)
    
    if args.dataset == 'mit':
        if args.fullsize:
            print('test fullsize....')
            test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MIT_TXT\\MIT_BarronSplit_fullsize_test.txt'
        else:
            print('test size256....')
            test_txt = 'MIT_TXT\\MIT_BarronSplit_test.txt'
        test_set = RIN_pipeline.MIT_Dataset_Revisit(test_txt, mode='test')
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)
    else:
        remove_names = os.listdir('F:\\ShapeNet\\remove')
        if args.shapenet_g:
            test_set = RIN_pipeline.ShapeNet_Dateset_new_new('F:\\ShapeNet', size_per_dataset=9000, mode='test', image_size=256, remove_names=remove_names, shapenet_g=args.shapenet_g)
        else:
            test_set = RIN_pipeline.ShapeNet_Dateset_new_new('F:\\ShapeNet', size_per_dataset=9000, mode='test', image_size=256, remove_names=remove_names)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)
    
    if args.shapenet_g:
        check_folder(os.path.join(args.save_path, "refl_target_G"))
        check_folder(os.path.join(args.save_path, "shad_target_G"))
        check_folder(os.path.join(args.save_path, "refl_output_G"))
        check_folder(os.path.join(args.save_path, "shad_output_G"))
        check_folder(os.path.join(args.save_path, "mask_G"))
    else:
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
                h,w = input_g.size()[2], input_g.size()[3]
                pad_h,pad_w = clc_pad(h,w,16)
                print(pad_h, pad_w)
                tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
                tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
                input_g = tmp_pad(input_g)
            if args.refl_multi_size and args.shad_multi_size:
                albedo_fake, shading_fake, _, _ = composer.forward(input_g)
            elif args.refl_multi_size or args.shad_multi_size:
                albedo_fake, shading_fake, _ = composer.forward(input_g)
            else:
                albedo_fake, shading_fake = composer.forward(input_g)
            if args.fullsize:
                albedo_fake, shading_fake = tmp_inversepad(albedo_fake), tmp_inversepad(shading_fake)

            if args.use_tanh:
                albedo_fake = (albedo_fake + 1) / 2
                shading_fake = (shading_fake + 1) / 2
                albedo_g = (albedo_g + 1) / 2
                shading_g = (shading_g + 1) / 2

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

            if args.shapenet_g:
                lab_refl_targ.save(os.path.join(args.save_path, "refl_target_G", "{}.png".format(ind)))
                lab_sha_targ.save(os.path.join(args.save_path, "shad_target_G", "{}.png".format(ind)))
                refl_pred.save(os.path.join(args.save_path, "refl_output_G", "{}.png".format(ind)))
                sha_pred.save(os.path.join(args.save_path, "shad_output_G", "{}.png".format(ind)))
                mask_g.save(os.path.join(args.save_path, "mask_G", "{}.png".format(ind)))
            else:
                lab_refl_targ.save(os.path.join(args.save_path, "refl_target_fullsize" if args.fullsize else "refl_target", "{}.png".format(ind)))
                lab_sha_targ.save(os.path.join(args.save_path, "shad_target_fullsize" if args.fullsize else "shad_target", "{}.png".format(ind)))
                refl_pred.save(os.path.join(args.save_path, "refl_output_fullsize" if args.fullsize else "refl_output", "{}.png".format(ind)))
                sha_pred.save(os.path.join(args.save_path, "shad_output_fullsize" if args.fullsize else "shad_output", "{}.png".format(ind)))
                mask_g.save(os.path.join(args.save_path, "mask", "{}.png".format(ind)))

if __name__ == "__main__":
    main()
