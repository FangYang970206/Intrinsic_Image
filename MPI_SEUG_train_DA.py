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
    random.seed(9999)
    torch.manual_seed(9999)
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',          type=str,   default='F:\\MPI-sintel-PyrResNet\\Sintel\\images\\',
    help='base folder of datasets')
    parser.add_argument('--split',              type=str,   default='SceneSplit')
    parser.add_argument('--mode',               type=str,   default='train')
    parser.add_argument('--save_path',          type=str,   default='MPI_logs\\RIID_new_RIN_updateLR1_epoch240_CosBF_VGG0.1_shading_SceneSplit_GAN_selayer1_ReflMultiSize_DA\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--lr',                 type=float, default=0.0005,
    help='learning rate')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--save_model',         type=bool,  default=True,
    help='whether to save model or not')
    parser.add_argument('--num_epochs',         type=int,   default=1000)
    parser.add_argument('--batch_size',         type=int,   default=20)
    parser.add_argument('--checkpoint',         type=bool,  default=False)
    parser.add_argument('--state_dict',         type=str,   default='composer_state.t7')
    parser.add_argument('--skip_se',            type=StrToBool, default=False)
    parser.add_argument('--cuda',               type=str,   default='cuda:1')
    parser.add_argument('--dilation',           type=StrToBool,   default=False)
    parser.add_argument('--se_improved',        type=StrToBool,  default=False)
    parser.add_argument('--weight_decay',       type=float, default=0.0001)
    parser.add_argument('--refl_multi_size',    type=bool,  default=False)
    parser.add_argument('--shad_multi_size',    type=bool,  default=False)
    parser.add_argument('--data_augmentation',  type=bool,  default=True)
    args = parser.parse_args()

    check_folder(args.save_path)
    # pylint: disable=E1101
    device = torch.device(args.cuda)
    # pylint: disable=E1101
    # shader = RIN.Shader(output_ch=3)
    print(args.skip_se)
    Generator_R = RIN.SESingleGenerator(multi_size=args.refl_multi_size).to(device)
    Generator_S = RIN.SESingleGenerator(multi_size=args.shad_multi_size).to(device)
    composer = RIN.SEComposerGenerater(Generator_R, Generator_S, args.refl_multi_size, args.shad_multi_size).to(device)
    Discriminator_R = RIN.SEUG_Discriminator().to(device)
    Discriminator_S = RIN.SEUG_Discriminator().to(device)
    # composer = RIN.Composer(reflection, shader).to(device)

    
    MPI_Image_Split_train_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_imageSplit-256-train.txt'
    MPI_Image_Split_test_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_imageSplit-256-test.txt'
    if args.data_augmentation:
        MPI_Scene_Split_train_txt = 'D:\\fangyang\\intrinsic_by_fangyang\\MPI_TXT\\MPI_main_sceneSplit-fullsize-NoDefect-train.txt'
    else:
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
        # cur_epoch = int(args.state_dict.split('_')[-1].split('.')[0]) + 1
    if args.data_augmentation:
        train_transform = RIN_pipeline.MPI_Train_Agumentation()
    train_set = RIN_pipeline.MPI_Dataset_Revisit(train_txt, transform=train_transform if args.data_augmentation else None, refl_multi_size=args.refl_multi_size, shad_multi_size=args.shad_multi_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)

    test_set = RIN_pipeline.MPI_Dataset_Revisit(test_txt)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=False)
    
    writer = SummaryWriter(log_dir=args.save_path)

    trainer = RIN_pipeline.SEUGTrainer(composer, Discriminator_R, Discriminator_S, train_loader, args.lr, device, writer, weight_decay=args.weight_decay, refl_multi_size=args.refl_multi_size, shad_multi_size=args.shad_multi_size)

    best_albedo_loss = 9999
    best_shading_loss = 9999

    for epoch in range(cur_epoch, args.num_epochs):
        print('<Main> Epoch {}'.format(epoch))
        
        step = trainer.train()

        if (epoch + 1) % 100 == 0:
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
