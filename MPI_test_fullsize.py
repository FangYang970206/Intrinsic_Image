import os
import random
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.backends import cudnn

# import RIN
import RIN
import RIN_pipeline
import numpy as np
import scipy.misc

from utils import *

def main():
    random.seed(9999)
    torch.manual_seed(9999)
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',              type=str,   default='SceneSplit')
    parser.add_argument('--mode',               type=str,   default='test')
    parser.add_argument('--save_path',          type=str,   default='MPI_logs\\RIID_origin_RIN_updateLR_CosBF_VGG0.1_shading_SceneSplit\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--state_dict',         type=str,   default='composer_state_179.t7')
    args = parser.parse_args()

    # pylint: disable=E1101
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101
    shader = RIN.Shader(output_ch=3)
    reflection = RIN.Decomposer()
    composer = RIN.Composer(reflection, shader).to(device)

    MPI_Image_Split_test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_imageSplit-fullsize-ChenSplit-test.txt'
    MPI_Scene_Split_test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_sceneSplit-fullsize-NoDefect-test.txt'

    if args.split == 'ImageSplit':
        test_txt = MPI_Image_Split_test_txt
        print('Image split mode')
    else:
        test_txt = MPI_Scene_Split_test_txt
        print('Scene split mode')


    composer.load_state_dict(torch.load(os.path.join(args.save_path, args.state_dict)))
    print('load checkpoint success!')

    test_set = RIN_pipeline.MPI_Dataset_Revisit(test_txt)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)
    
    check_folder(os.path.join(args.save_path, "refl_target_fullsize"))
    check_folder(os.path.join(args.save_path, "shad_target_fullsize"))
    check_folder(os.path.join(args.save_path, "refl_output_fullsize"))
    check_folder(os.path.join(args.save_path, "shad_output_fullsize"))
    check_folder(os.path.join(args.save_path, "shape_output_fullsize"))
    check_folder(os.path.join(args.save_path, "mask_fullsize"))

    composer.eval()
    with torch.no_grad():
        for ind, tensors in enumerate(test_loader):
            print(ind)
            inp = [t.to(device) for t in tensors]
            input_g, albedo_g, shading_g, mask_g = inp
            lab_refl_pred = np.zeros((input_g.size()[2], input_g.size()[3], input_g.size()[1]))
            lab_sha_pred = np.zeros_like(lab_refl_pred)
            lab_shape_pred = np.zeros_like(lab_refl_pred)
            for ind2 in range(8):
                if ind2 % 2 == 0:
                    input_g_tmp =  input_g[:, :, :256, (ind2 // 2) * 256 : (ind2 // 2 + 1) * 256]
                    mask_g_tmp = mask_g[:, :, :256, (ind2 // 2) * 256 : (ind2 // 2 + 1) * 256]
                    _, albedo_fake, shading_fake, shape_fake = composer.forward(input_g_tmp)
                    albedo_fake  = albedo_fake*mask_g_tmp
                    lab_refl_pred[:256, (ind2 // 2) * 256 : (ind2 // 2 + 1) * 256, :] += albedo_fake.squeeze().cpu().numpy().transpose(1,2,0)
                    lab_sha_pred[:256, (ind2 // 2) * 256 : (ind2 // 2 + 1) * 256, :] += shading_fake.squeeze().cpu().numpy().transpose(1,2,0)
                    lab_shape_pred[:256, (ind2 // 2) * 256 : (ind2 // 2 + 1) * 256, :] += shape_fake.squeeze().cpu().numpy().transpose(1,2,0)
                else:
                    input_g_tmp = input_g[:, :, 180:, (ind2 // 2) * 256 : (ind2 // 2 + 1) * 256]
                    mask_g_tmp = mask_g[:, :, 180:, (ind2 // 2) * 256 : (ind2 // 2 + 1) * 256]
                    _, albedo_fake, shading_fake, shape_fake = composer.forward(input_g_tmp)
                    albedo_fake  = albedo_fake*mask_g_tmp
                    lab_refl_pred[180:, (ind2 // 2) * 256 : (ind2 // 2 + 1) * 256, :] += albedo_fake.squeeze().cpu().numpy().transpose(1,2,0)
                    lab_sha_pred[180:, (ind2 // 2) * 256 : (ind2 // 2 + 1) * 256, :] += shading_fake.squeeze().cpu().numpy().transpose(1,2,0)
                    lab_shape_pred[180:, (ind2 // 2) * 256 : (ind2 // 2 + 1) * 256, :] += shape_fake.squeeze().cpu().numpy().transpose(1,2,0)
            
            lab_refl_pred[180:256, :, :] /= 2
            lab_sha_pred[180:256, :, :] /= 2
            lab_shape_pred[180:256, :, :] /= 2

            lab_refl_targ = albedo_g.squeeze().cpu().numpy().transpose(1,2,0)
            lab_sha_targ = shading_g.squeeze().cpu().numpy().transpose(1,2,0)
            mask = mask_g.squeeze().cpu().numpy().transpose(1,2,0)
            # refl_pred = albedo_fake.squeeze().cpu().numpy().transpose(1,2,0)
            # sha_pred = shading_fake.squeeze().cpu().numpy().transpose(1,2,0)
            # shape_pred = shape_fake.squeeze().cpu().numpy().transpose(1,2,0)
            refl_pred = lab_refl_pred
            sha_pred = lab_sha_pred
            shape_pred = lab_shape_pred

            lab_refl_targ = np.clip(lab_refl_targ, 0, 1)
            lab_sha_targ = np.clip(lab_sha_targ, 0, 1)
            refl_pred = np.clip(refl_pred, 0, 1)
            sha_pred = np.clip(sha_pred, 0, 1)
            shape_pred = np.clip(shape_pred, 0, 1)
            mask = np.clip(mask, 0, 1)

            scipy.misc.imsave(os.path.join(args.save_path, "refl_target_fullsize", "{}.png".format(ind)), lab_refl_targ)
            scipy.misc.imsave(os.path.join(args.save_path, "shad_target_fullsize", "{}.png".format(ind)), lab_sha_targ)
            scipy.misc.imsave(os.path.join(args.save_path, "mask_fullsize", "{}.png".format(ind)), mask)
            scipy.misc.imsave(os.path.join(args.save_path, "refl_output_fullsize", "{}.png".format(ind)), refl_pred)
            scipy.misc.imsave(os.path.join(args.save_path, "shad_output_fullsize", "{}.png".format(ind)), sha_pred)
            scipy.misc.imsave(os.path.join(args.save_path, "shape_output_fullsize", "{}.png".format(ind)), shape_pred)


if __name__ == "__main__":
    main()
