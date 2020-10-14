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
    parser.add_argument('--mode',               type=str,   default='train')
    parser.add_argument('--data_path',          type=str,   default='F:\\ShapeNet',
    help='base folder of datasets')
    parser.add_argument('--save_path',          type=str,   default='logs_shapenet\\RIID_new_RIN_updateLR1_epoch160_CosBF_VGG0.1_shading_SceneSplit_GAN_selayer1_ReflMultiSize_320x320\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--refl_checkpoint',    type=str,   default='refl_checkpoint')
    parser.add_argument('--shad_checkpoint',    type=str,   default='shad_checkpoint')
    parser.add_argument('--lr',                 type=float, default=0.001,
    help='learning rate')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--save_model',         type=bool,  default=True,
    help='whether to save model or not')
    parser.add_argument('--num_epochs',         type=int,   default=40)
    parser.add_argument('--batch_size',         type=int,   default=20)
    parser.add_argument('--checkpoint',         type=StrToBool,  default=False)
    parser.add_argument('--state_dict_refl',    type=str,   default='composer_reflectance_state.t7')
    parser.add_argument('--state_dict_shad',    type=str,    default='composer_shading_state.t7')
    parser.add_argument('--remove_names',       type=str,   default='F:\\ShapeNet\\remove')
    parser.add_argument('--cur_epoch',          type=StrToInt,   default=0)
    parser.add_argument('--skip_se',            type=StrToBool,  default=False)
    parser.add_argument('--cuda',               type=str,        default='cuda:1')
    parser.add_argument('--dilation',           type=StrToBool,  default=False)
    parser.add_argument('--se_improved',        type=StrToBool,  default=False)
    parser.add_argument('--weight_decay',       type=float,      default=0.0001)
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
    parser.add_argument('--refl_grad_flag',     type=StrToBool,  default=False)
    parser.add_argument('--shad_grad_flag',     type=StrToBool,  default=False)
    parser.add_argument('--refl_detach_flag',   type=StrToBool,  default=False)
    parser.add_argument('--shad_detach_flag',   type=StrToBool,  default=False)
    parser.add_argument('--refl_D_weight_flag', type=StrToBool,  default=False)
    parser.add_argument('--shad_D_weight_flag', type=StrToBool,  default=False)
    parser.add_argument('--shad_squeeze_flag',  type=StrToBool,  default=False)
    parser.add_argument('--refl_reduction',     type=StrToInt,   default=8)
    parser.add_argument('--shad_reduction',     type=StrToInt,   default=8)
    parser.add_argument('--refl_bn',            type=StrToBool,  default=True)
    parser.add_argument('--shad_bn',            type=StrToBool,  default=True)
    parser.add_argument('--refl_act',           type=str,        default='relu')
    parser.add_argument('--shad_act',           type=str,        default='relu')
    # parser.add_argument('--refl_gan',           type=StrToBool,  default=False)
    # parser.add_argument('--shad_gan',           type=StrToBool,  default=False)
    parser.add_argument('--data_augmentation',  type=StrToBool,  default=False)
    parser.add_argument('--fullsize',           type=StrToBool,  default=False)
    parser.add_argument('--fullsize_test',      type=StrToBool,  default=False)
    parser.add_argument('--image_size',         type=StrToInt,   default=256)
    parser.add_argument('--ttur',               type=StrToBool,  default=False)
    args = parser.parse_args()

    check_folder(args.save_path)
    check_folder(os.path.join(args.save_path, args.refl_checkpoint))
    check_folder(os.path.join(args.save_path, args.shad_checkpoint))
    # pylint: disable=E1101
    device = torch.device(args.cuda)
    # pylint: disable=E1101
    reflectance = RIN.SEDecomposerSingle(multi_size=args.refl_multi_size, low_se=args.refl_low_se, skip_se=args.refl_skip_se, detach=args.refl_detach_flag, reduction=args.refl_reduction, bn=args.refl_bn, act=args.refl_act).to(device)
    shading = RIN.SEDecomposerSingle(multi_size=args.shad_multi_size, low_se=args.shad_low_se, skip_se=args.shad_skip_se, se_squeeze=args.shad_squeeze_flag, reduction=args.shad_reduction, detach=args.shad_detach_flag, bn=args.shad_bn, act=args.shad_act).to(device)
    cur_epoch = 0
    if args.checkpoint:
        reflectance.load_state_dict(torch.load(os.path.join(args.save_path, args.refl_checkpoint, args.state_dict_refl)))
        shading.load_state_dict(torch.load(os.path.join(args.save_path, args.shad_checkpoint, args.state_dict_shad)))
        cur_epoch = args.cur_epoch
        print('load checkpoint success!')
    composer = RIN.SEComposer(reflectance, shading, args.refl_multi_size, args.shad_multi_size).to(device)
    
    if not args.ttur:
        Discriminator_R = RIN.SEUG_Discriminator().to(device)
        Discriminator_S = RIN.SEUG_Discriminator().to(device)
    else:
        Discriminator_R = RIN.SEUG_Discriminator_new().to(device)
        Discriminator_S = RIN.SEUG_Discriminator_new().to(device)
    
    remove_names = os.listdir(args.remove_names)
    train_set = RIN_pipeline.ShapeNet_Dateset_new_new(args.data_path, size_per_dataset=90000, mode='train', image_size=args.image_size, remove_names=remove_names,refl_multi_size=args.refl_multi_size, shad_multi_size=args.shad_multi_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)
    test_set = RIN_pipeline.ShapeNet_Dateset_new_new(args.data_path, size_per_dataset=1000, mode='test', image_size=args.image_size, remove_names=remove_names)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=False)

    if args.mode == 'test':
        print('test mode .....')
        albedo_test_loss, shading_test_loss = RIN_pipeline.MPI_test_unet(composer, test_loader, device, args)
        print('albedo_test_loss: ', albedo_test_loss)
        print('shading_test_loss: ', shading_test_loss)
        return

    writer = SummaryWriter(log_dir=args.save_path)

    if not args.ttur:
        trainer = RIN_pipeline.SEUGTrainer(composer, Discriminator_R, Discriminator_S, train_loader, device, writer, args)
    else:
        trainer = RIN_pipeline.SEUGTrainerNew(composer, Discriminator_R, Discriminator_S, train_loader, device, writer, args)

    best_albedo_loss = 9999
    best_shading_loss = 9999

    for epoch in range(cur_epoch, args.num_epochs):
        print('<Main> Epoch {}'.format(epoch))
        
        trainer.train()

        if (epoch + 1) % 10 == 0:
            args.lr = args.lr * 0.75
            trainer.update_lr(args.lr)
        
        # if (epoch + 1) % 10 == 0:
        albedo_test_loss, shading_test_loss = RIN_pipeline.MPI_test_unet(composer, test_loader, device, args)
        average_loss = (albedo_test_loss + shading_test_loss) / 2
        writer.add_scalar('A_mse', albedo_test_loss, epoch)
        writer.add_scalar('S_mse', shading_test_loss, epoch)
        writer.add_scalar('aver_mse', average_loss, epoch)
        with open(os.path.join(args.save_path, 'loss_every_epoch.txt'), 'a+') as f:
            f.write('epoch{} --- average_loss: {}, albedo_loss:{}, shading_loss:{}\n'.format(epoch, average_loss, albedo_test_loss, shading_test_loss))
        if args.save_model:
            state = composer.reflectance.state_dict()
            torch.save(state, os.path.join(args.save_path, args.refl_checkpoint, 'composer_reflectance_state_{}.t7'.format(epoch)))
            state = composer.shading.state_dict()
            torch.save(state, os.path.join(args.save_path, args.shad_checkpoint, 'composer_shading_state_{}.t7'.format(epoch)))
        if albedo_test_loss < best_albedo_loss:
            best_albedo_loss = albedo_test_loss
            # if args.save_model:
            #     state = composer.reflectance.state_dict()
            #     torch.save(state, os.path.join(args.save_path, args.refl_checkpoint, 'composer_reflectance_state_{}.t7'.format(epoch)))
            with open(os.path.join(args.save_path, 'reflectance_loss.txt'), 'a+') as f:
                f.write('epoch{} --- albedo_loss:{}\n'.format(epoch, albedo_test_loss))
        if shading_test_loss < best_shading_loss:
            best_shading_loss = shading_test_loss
            # if args.save_model:
            #     state = composer.shading.state_dict()
            #     torch.save(state, os.path.join(args.save_path, args.shad_checkpoint, 'composer_shading_state_{}.t7'.format(epoch)))
            with open(os.path.join(args.save_path, 'shading_loss.txt'), 'a+') as f:
                f.write('epoch{} --- shading_loss:{}\n'.format(epoch, shading_test_loss))

if __name__ == "__main__":
    main()
