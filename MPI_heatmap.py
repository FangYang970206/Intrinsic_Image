import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.backends import cudnn
import torchvision.transforms as transforms
from PIL import Image
import RIN_new as RIN
import RIN_pipeline
import numpy as np
import scipy.misc
import cv2
from utils import *

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def cam(x, size = (1024, 436)):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, size)
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def main():
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',              type=str,   default='SceneSplit')
    parser.add_argument('--save_path',          type=str,   default='MPI_logs_new\\GAN_RIID_updateLR3_epoch160_CosbfVGG_SceneSplit_refl-se-skip_shad-se-low_multi_new_shadSqueeze_grad\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--refl_checkpoint',    type=str,        default='refl_checkpoint')
    parser.add_argument('--shad_checkpoint',    type=str,        default='shad_checkpoint')
    parser.add_argument('--state_dict_refl',    type=str,        default='composer_reflectance_state_81.t7')
    parser.add_argument('--state_dict_shad',    type=str,        default='composer_shading_state_81.t7')
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
    parser.add_argument('--heatmap',            type=StrToBool,  default=False)
    args = parser.parse_args()

    device = torch.device(args.cuda)
    model = RIN.SEDecomposerSingle(multi_size=args.refl_multi_size, low_se=args.refl_low_se, skip_se=args.refl_skip_se, detach=args.refl_detach_flag, reduction=args.refl_reduction, heatmap=args.heatmap).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_path, args.refl_checkpoint, args.state_dict_refl)))
    print('load checkpoint success!')

    if args.fullsize:
        print('test fullsize....')
        MPI_Image_Split_test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_imageSplit-fullsize-ChenSplit-test.txt'
        MPI_Scene_Split_test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_sceneSplit-fullsize-NoDefect-test.txt'
        h, w = 436,1024
        pad_h,pad_w = clc_pad(h,w,16)
        print(pad_h, pad_w)
        tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
        tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
        tmp_inversepad_heatmap = nn.ReflectionPad2d((0,0,0,-3))
    else:
        print('test size256....')
        MPI_Image_Split_test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_imageSplit-256-test.txt'
        MPI_Scene_Split_test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_sceneSplit-256-test.txt'

    if args.split == 'ImageSplit':
        test_txt = MPI_Image_Split_test_txt
        print('Image split mode')
    else:
        test_txt = MPI_Scene_Split_test_txt
        print('Scene split mode')

    test_set = RIN_pipeline.MPI_Dataset_Revisit(test_txt)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)

    if args.fullsize:
        check_folder(os.path.join(args.save_path, "refl_heapmapin"))
        check_folder(os.path.join(args.save_path, "refl_heapmapout"))

    ToPIL = transforms.ToPILImage()

    model.eval()
    with torch.no_grad():
        for ind, tensors in enumerate(test_loader):
            print(ind)
            inp = [t.to(device) for t in tensors]
            input_g, albedo_g, shading_g, mask_g = inp
            if args.fullsize:
                input_g = tmp_pad(input_g)
            if args.refl_multi_size:
                albedo_fake, _, heapmap = model.forward(input_g)
            else:
                albedo_fake = model.forward(input_g)
            if args.fullsize:
                input_g = tmp_inversepad(input_g)
                heapmap[0] = tmp_inversepad_heatmap(heapmap[0])
                heapmap[1] = tmp_inversepad_heatmap(heapmap[1])

            # albedo_fake  = albedo_fake*mask_g

            # lab_refl_targ = albedo_g.squeeze().cpu().numpy().transpose(1,2,0)
            # lab_sha_targ = shading_g.squeeze().cpu().numpy().transpose(1,2,0)
            # refl_pred = albedo_fake.squeeze().cpu().numpy().transpose(1,2,0)
            # sha_pred = shading_fake.squeeze().cpu().numpy().transpose(1,2,0)
            print(heapmap[0].squeeze().size())
            heapmap[0] = torch.sum(heapmap[0], dim=1, keepdim=True)
            heapmap[1] = torch.sum(heapmap[1], dim=1, keepdim=True)
            heapmapin = tensor2numpy(heapmap[0][0])
            heapmapout = tensor2numpy(heapmap[1][0])
            #heapmapout = torch.sum(heapmap[1].squeeze(), dim=0, keepdim=True).cpu().clamp(0,1).numpy().transpose(1,2,0)
            print(heapmapin.shape)
            heapmapin = cam(heapmapin)
            print(heapmapin.shape)
            heapmapout = cam(heapmapout)
            # heapmapin = heapmapin.transpose(2,0,1)
            # heapmapout = heapmapout.transpose(2,0,1)
            # input_g = input_g.squeeze().cpu().clamp(0, 1).numpy()
            # print(heapmapin.shape)
            # print(input_g.shape)
            # heapmapin = np.concatenate((heapmapin, input_g), 1).astype(np.float32)
            # heapmapout = np.concatenate((heapmapout, input_g), 1).astype(np.float32)
            # print(heapmapin.shape)
            # heapmapin = torch.from_numpy(heapmapin)
            # heapmapout = torch.from_numpy(heapmapout)
            # lab_refl_targ = ToPIL(input_g.squeeze())
            # refl_pred = ToPIL(albedo_fake.squeeze())

            # heapmapin = torch.cat([heapmapin, torch.zeros(2, h // 4, w // 4)])
            # heapmapout = torch.cat([heapmapout, torch.zeros(2, h // 4, w // 4)])

            # print(heapmapin.size)
            # print(heapmapout.size)
            cv2.imwrite(os.path.join(args.save_path, "refl_heapmapin", '{}.png'.format(ind)), heapmapin * 255.0)
            cv2.imwrite(os.path.join(args.save_path, "refl_heapmapout", '{}.png'.format(ind)), heapmapout * 255.0)
            # heapmapinimage = ToPIL(heapmapin)
            # heapmapoutimage = ToPIL(heapmapout)

            # print(heapmapinimage.size)
            # print(heapmapinimage.size)
            # print(lab_refl_targ.size)

            # heapmapinimage = Image.blend(lab_refl_targ, heapmapinimage, 0.3)
            # heapmapoutimage = Image.blend(lab_refl_targ, heapmapoutimage, 0.3)

            # heapmapinimage.save(os.path.join(args.save_path, "refl_heapmapin", "{}.png".format(ind)))
            # heapmapoutimage.save(os.path.join(args.save_path, "refl_heapmapout", "{}.png".format(ind)))

            # lab_refl_targ = np.clip(lab_refl_targ, 0, 1)
            # lab_sha_targ = np.clip(lab_sha_targ, 0, 1)
            # refl_pred = np.clip(refl_pred, 0, 1)
            # sha_pred = np.clip(sha_pred, 0, 1)

            # scipy.misc.imsave(os.path.join(args.save_path, "refl_target_fullsize" if args.fullsize else "refl_target", "{}.png".format(ind)), lab_refl_targ)
            # scipy.misc.imsave(os.path.join(args.save_path, "refl_output_fullsize" if args.fullsize else "refl_output", "{}.png".format(ind)), refl_pred)
            # scipy.misc.imsave(os.path.join(args.save_path, "shad_target_fullsize" if args.fullsize else "shad_target", "{}.png".format(ind)), lab_sha_targ)
            # scipy.misc.imsave(os.path.join(args.save_path, "shad_output_fullsize" if args.fullsize else "shad_output", "{}.png".format(ind)), sha_pred)


if __name__ == "__main__":
    main()
