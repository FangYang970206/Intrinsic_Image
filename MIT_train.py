import os
import random
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.backends import cudnn
import logging  # 引入logging模块
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import warnings
warnings.filterwarnings("ignore")
import numpy as np

# import RIN
import RIN_new as RIN 
import RIN_pipeline

from utils import *


def main():
    random.seed(6666)
    torch.manual_seed(6666)
    torch.cuda.manual_seed(6666)
    np.random.seed(6666)
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',               type=str,   default='train')
    parser.add_argument('--save_path',          type=str,   default='MIT_logs\\RIID_origin_RIN_updateLR0.0005_4_bf_cosLoss_VGG0.1_400epochs_bs22\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--refl_checkpoint',    type=str,   default='refl_checkpoint')
    parser.add_argument('--shad_checkpoint',    type=str,   default='shad_checkpoint')
    parser.add_argument('--lr',                 type=float, default=0.0005,
    help='learning rate')
    parser.add_argument('--save_model',         type=bool,  default=True,
    help='whether to save model or not')
    parser.add_argument('--num_epochs',         type=int,   default=100)
    parser.add_argument('--batch_size',         type=StrToInt,   default=16)
    parser.add_argument('--checkpoint',         type=StrToBool,  default=False)
    parser.add_argument('--state_dict_refl',    type=str,   default='composer_reflectance_state.t7')
    parser.add_argument('--state_dict_shad',    type=str,    default='composer_shading_state.t7')
    parser.add_argument('--cur_epoch',          type=StrToInt,   default=0)
    parser.add_argument('--skip_se',            type=StrToBool,  default=False)
    parser.add_argument('--cuda',               type=str,        default='cuda')
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
    # parser.add_argument('--fullsize_test',      type=StrToBool,  default=False)
    parser.add_argument('--image_size',         type=StrToInt,   default=256)
    parser.add_argument('--shad_out_conv',      type=StrToInt,   default=3)
    parser.add_argument('--finetune',           type=StrToBool,  default=False)
    parser.add_argument('--vae',                type=StrToBool,  default=False)
    args = parser.parse_args()

    check_folder(args.save_path)
    check_folder(os.path.join(args.save_path, args.refl_checkpoint))
    check_folder(os.path.join(args.save_path, args.shad_checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101
    reflectance = RIN.SEDecomposerSingle(multi_size=args.refl_multi_size, low_se=args.refl_low_se, skip_se=args.refl_skip_se, detach=args.refl_detach_flag, reduction=args.refl_reduction, bn=args.refl_bn, act=args.refl_act).to(device)
    shading = RIN.SEDecomposerSingle(multi_size=args.shad_multi_size, low_se=args.shad_low_se, skip_se=args.shad_skip_se, se_squeeze=args.shad_squeeze_flag, reduction=args.shad_reduction, detach=args.shad_detach_flag, bn=args.shad_bn, act=args.shad_act, last_conv_ch=args.shad_out_conv).to(device)
    cur_epoch = 0
    if args.checkpoint:
        reflectance.load_state_dict(torch.load(os.path.join(args.save_path, args.refl_checkpoint, args.state_dict_refl)))
        shading.load_state_dict(torch.load(os.path.join(args.save_path, args.shad_checkpoint, args.state_dict_shad)))
        cur_epoch = args.cur_epoch
        print('load checkpoint success!')
    composer = RIN.SEComposer(reflectance, shading, args.refl_multi_size, args.shad_multi_size).to(device)
    # shader = RIN.Shader()
    # reflection = RIN.Decomposer()
    # composer = RIN.Composer(reflection, shader).to(device)
    Discriminator_R = RIN.SEUG_Discriminator().to(device)
    # if args.shad_gan:
    Discriminator_S = RIN.SEUG_Discriminator().to(device)

    MIT_train_fullsize_txt = 'MIT_TXT\\MIT_BarronSplit_fullsize_train.txt'
    MIT_test_fullsize_txt = 'MIT_TXT\\MIT_BarronSplit_fullsize_test.txt'
    MIT_train_txt = 'MIT_TXT\\MIT_BarronSplit_train.txt'
    MIT_test_txt = 'MIT_TXT\\MIT_BarronSplit_test.txt'
    if args.fullsize and not args.finetune:
        train_set = RIN_pipeline.MIT_Dataset_Revisit(MIT_train_fullsize_txt, mode='train', refl_multi_size=args.refl_multi_size, shad_multi_size=args.shad_multi_size, image_size=args.image_size, fullsize=args.fullsize)
    else:
        train_set = RIN_pipeline.MIT_Dataset_Revisit(MIT_train_txt, mode='train', refl_multi_size=args.refl_multi_size, shad_multi_size=args.shad_multi_size, image_size=args.image_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)

    test_set = RIN_pipeline.MIT_Dataset_Revisit(MIT_test_fullsize_txt, mode='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)
    
    # test_loader_2 = torch.utils.data.DataLoader(test_set, batch_size=10, num_workers=args.loader_threads, shuffle=False)

    writer = SummaryWriter(log_dir=args.save_path)
    best_albedo_loss = 9999
    best_shading_loss = 9999
    best_avg_lmse = 9999
    flag = True
    # trainer = RIN_pipeline.MIT_TrainerOrigin(composer, train_loader, args.lr, device, writer)
    trainer = RIN_pipeline.SEUGTrainer(composer, Discriminator_R, Discriminator_S, train_loader, device, writer, args)
    logging.info('start training....')
    for epoch in range(cur_epoch, args.num_epochs):
        print('<Main> Epoch {}'.format(epoch))

        trainer.train()
        
        if epoch >= 80 and args.finetune:
            if flag and args.finetune:
                flag = False
                train_set = RIN_pipeline.MIT_Dataset_Revisit(MIT_train_fullsize_txt, mode='train', refl_multi_size=args.refl_multi_size, shad_multi_size=args.shad_multi_size, image_size=args.image_size, fullsize=args.fullsize)
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, num_workers=args.loader_threads, shuffle=True)
                trainer = RIN_pipeline.SEUGTrainer(composer, Discriminator_R, Discriminator_S, train_loader, device, writer, args)
        

        albedo_test_loss, shading_test_loss = RIN_pipeline.MIT_test_unet(composer, test_loader, device, args)
        # albedo_test_loss, shading_test_loss = 0, 0
        # with torch.no_grad():
        #     composer.eval()
        #     criterion = torch.nn.MSELoss(size_average=True).to(device)
        #     for _, labeled in enumerate(test_loader):
        #         labeled = [t.to(device) for t in labeled]
        #         input_g, albedo_g, shading_g, mask_g = labeled
        #         lab_inp_pred, lab_refl_pred, lab_shad_pred, _ = composer.forward(input_g)
        #         lab_inp_pred = lab_inp_pred * mask_g
        #         lab_refl_pred = lab_refl_pred * mask_g
        #         lab_shad_pred = lab_shad_pred * mask_g
        #         refl_loss = criterion(lab_refl_pred, albedo_g)
        #         shad_loss = criterion(lab_shad_pred, shading_g)
        #         # recon_loss = criterion(lab_inp_pred, input_g)
        #         albedo_test_loss = refl_loss.item()
        #         shading_test_loss = shad_loss.item()
        #         writer.add_scalar('test_refl_loss', refl_loss.item(), epoch)
        #         writer.add_scalar('test_shad_loss', shad_loss.item(), epoch)
                # writer.add_scalar('test_recon_loss', recon_loss.item(), epoch)
                # cur_aver_loss = (refl_loss.item() + shad_loss.item()) / 2
                # writer.add_scalar('cur_aver_loss', cur_aver_loss, epoch)
                # if cur_aver_loss < best_loss:
                #     best_loss = cur_aver_loss
        
        if (epoch + 1) % 40 == 0:
            args.lr *= 0.75
            # logging.info('epoch{} learning rate : {}'.format(epoch, args.lr))
            trainer.update_lr(args.lr)

        average_loss = (albedo_test_loss + shading_test_loss) / 2
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
        # if ave_lmse < best_avg_lmse:
        #     best_avg_lmse = ave_lmse
        #     # if args.save_model:
        #     #     state = composer.shading.state_dict()
        #     #     torch.save(state, os.path.join(args.save_path, args.shad_checkpoint, 'composer_shading_state_{}.t7'.format(epoch)))
        #     with open(os.path.join(args.save_path, 'LMSE.txt'), 'a+') as f:
        #         f.write('epoch{}---AVG_LMSE:{}\n'.format(epoch, best_avg_lmse))
        # with open(os.path.join(args.save_path, 'loss_every_epoch.txt'), 'a+') as f:
        #     f.write('epoch{} --- average_loss: {}, albedo_loss:{}, shading_loss:{}\n'.format(epoch, best_loss, cur_refl_loss, cur_shad_loss))

        # if best_loss < best_loss_before:
        #     best_loss_before = best_loss
        #     if args.save_model:
        #         state = composer.state_dict()
        #         torch.save(state, os.path.join(args.save_path, 'composer_state_{}.t7'.format(epoch)))
        #         logging.info('save model --- composer_state_{}.t7'.format(epoch))
        #     with open(os.path.join(args.save_path, 'loss.txt'), 'a+') as f:
        #         f.write('epoch{} --- average_loss: {}, albedo_loss:{}, shading_loss:{}\n'.format(epoch, best_loss, cur_refl_loss, cur_shad_loss))

        #     RIN_pipeline.visualize_MPI(composer, test_loader, device, os.path.join(args.save_path, 'image_{}.png'.format(epoch)))

if __name__ == "__main__":
    main()
