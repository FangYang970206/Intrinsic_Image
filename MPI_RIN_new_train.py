import os
import random
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.backends import cudnn
import numpy as np

# import RIN
import RIN_new as RIN 
import RIN_pipeline

from utils import *

def main():
    random.seed(520)
    torch.manual_seed(520)
    torch.cuda.manual_seed(520)
    np.random.seed(520)
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',          type=str,   default='F:\\sintel',
    help='base folder of datasets')
    parser.add_argument('--split',              type=str,   default='SceneSplit')
    parser.add_argument('--mode',               type=str,   default='two')
    parser.add_argument('--save_path',          type=str,   default='MPI_logs_new\\RIID_new_RIN_updateLR1_epoch240_CosBF_VGG0.1_shading_SceneSplit_selayer1_reflmultiSize\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--lr',                 type=float, default=0.0005,
    help='learning rate')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--save_model',         type=bool,  default=True,
    help='whether to save model or not')
    parser.add_argument('--num_epochs',         type=int,   default=120)
    parser.add_argument('--batch_size',         type=int,   default=20)
    parser.add_argument('--checkpoint',         type=bool,  default=False)
    parser.add_argument('--state_dict',         type=str,   default='composer_state.t7')
    parser.add_argument('--cuda',               type=str,   default='cuda')
    parser.add_argument('--choose',             type=str,   default='refl')
    parser.add_argument('--refl_skip_se',       type=StrToBool,  default=False)
    parser.add_argument('--shad_skip_se',       type=StrToBool,  default=False)
    parser.add_argument('--refl_low_se',        type=StrToBool,  default=False)
    parser.add_argument('--shad_low_se',        type=StrToBool,  default=False)
    parser.add_argument('--refl_multi_size',    type=StrToBool,  default=False)
    parser.add_argument('--shad_multi_size',    type=StrToBool,  default=False)
    parser.add_argument('--refl_vgg_flag',      type=StrToBool,  default=False)
    parser.add_argument('--shad_vgg_flag',      type=StrToBool,  default=False)
    parser.add_argument('--refl_bf_flag',       type=StrToBool,  default=False)
    parser.add_argument('--shad_bf_flag',       type=StrToBool,  default=False)
    parser.add_argument('--refl_cos_flag',      type=StrToBool,  default=False)
    parser.add_argument('--shad_cos_flag',      type=StrToBool,  default=False)
    parser.add_argument('--image_size',         type=StrToInt, default=256)
    args = parser.parse_args()

    check_folder(args.save_path)
    # pylint: disable=E1101
    device = torch.device(args.cuda)
    # pylint: disable=E1101
    reflectance = RIN.SEDecomposerSingle(multi_size=args.refl_multi_size, low_se=args.refl_low_se, skip_se=args.refl_skip_se).to(device)
    shading = RIN.SEDecomposerSingle(multi_size=args.shad_multi_size, low_se=args.shad_low_se, skip_se=args.shad_skip_se).to(device)
    composer = RIN.SEComposer(reflectance, shading, args.refl_multi_size, args.shad_multi_size).to(device)

    MPI_Image_Split_train_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_imageSplit-256-train.txt'
    MPI_Image_Split_test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_imageSplit-256-test.txt'
    MPI_Scene_Split_train_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_sceneSplit-256-train.txt'
    MPI_Scene_Split_test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_sceneSplit-256-test.txt'

    if args.split == 'ImageSplit':
        train_txt = MPI_Image_Split_train_txt
        test_txt = MPI_Image_Split_test_txt
        print('Image split mode')
    else:
        train_txt = MPI_Scene_Split_train_txt
        test_txt = MPI_Scene_Split_test_txt
        print('Scene split mode')

    cur_epoch = 0
    if args.checkpoint:
        composer.load_state_dict(torch.load(os.path.join(args.save_path, args.state_dict)))
        print('load checkpoint success!')

    train_set = RIN_pipeline.MPI_Dataset_Revisit(train_txt, refl_multi_size=args.refl_multi_size, shad_multi_size=args.shad_multi_size, image_size=args.image_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)

    test_set = RIN_pipeline.MPI_Dataset_Revisit(test_txt)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=False)
    
    writer = SummaryWriter(log_dir=args.save_path)

    trainer = RIN_pipeline.OctaveTrainer(composer, train_loader, device, writer, args)

    best_albedo_loss = 9999
    best_shading_loss = 9999

    for epoch in range(cur_epoch, args.num_epochs):
        print('<Main> Epoch {}'.format(epoch))
        
        trainer.train()

        if (epoch + 1) % 40 == 0:
            args.lr = args.lr * 0.75
            trainer.update_lr(args.lr)
            
        albedo_test_loss, shading_test_loss = RIN_pipeline.MPI_test_unet(composer, test_loader, device, refl_multi_size=args.refl_multi_size, shad_multi_size=args.shad_multi_size)
        average_loss = (albedo_test_loss + shading_test_loss) / 2
        writer.add_scalar('A_mse', albedo_test_loss, epoch)
        writer.add_scalar('S_mse', shading_test_loss, epoch)
        writer.add_scalar('aver_mse', average_loss, epoch)

        with open(os.path.join(args.save_path, 'loss_every_epoch.txt'), 'a+') as f:
            f.write('epoch{} --- average_loss: {}, albedo_loss:{}, shading_loss:{}\n'.format(epoch, average_loss, albedo_test_loss, shading_test_loss))

        if albedo_test_loss < best_albedo_loss:
            best_albedo_loss = albedo_test_loss
            if args.save_model:
                state = composer.reflectance.state_dict()
                torch.save(state, os.path.join(args.save_path, 'composer_reflectance_state.t7'))
            with open(os.path.join(args.save_path, 'reflectance_loss.txt'), 'a+') as f:
                f.write('epoch{} --- albedo_loss:{}\n'.format(epoch, albedo_test_loss))
        
        if shading_test_loss < best_shading_loss:
            best_shading_loss = shading_test_loss
            if args.save_model:
                state = composer.shading.state_dict()
                torch.save(state, os.path.join(args.save_path, 'composer_shading_state.t7'))
            with open(os.path.join(args.save_path, 'shading_loss.txt'), 'a+') as f:
                f.write('epoch{} --- shading_loss:{}\n'.format(epoch, shading_test_loss))


if __name__ == "__main__":
    main()
