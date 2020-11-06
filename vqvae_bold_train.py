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
    parser.add_argument('--data_path',          type=str,   default='E:\\BOLD',
    help='base folder of datasets')
    parser.add_argument('--mode',               type=str,   default='train')
    parser.add_argument('--save_path',          type=str,   default='logs_vqvae\\BOLD_base_256x256\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--refl_checkpoint',    type=str,   default='refl_checkpoint')
    parser.add_argument('--shad_checkpoint',    type=str,   default='shad_checkpoint')
    parser.add_argument('--lr',                 type=float, default=0.0005,
    help='learning rate')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--save_model',         type=bool,  default=True,
    help='whether to save model or not')
    parser.add_argument('--num_epochs',         type=int,   default=60)
    parser.add_argument('--batch_size',         type=int,   default=4)
    parser.add_argument('--checkpoint',         type=StrToBool,  default=False)
    parser.add_argument('--cur_epoch',          type=StrToInt,   default=0)
    parser.add_argument('--cuda',               type=str,        default='cuda')
    parser.add_argument('--weight_decay',       type=float,      default=0.0001)
    parser.add_argument('--refl_multi_size',    type=StrToBool,  default=False)
    parser.add_argument('--shad_multi_size',    type=StrToBool,  default=False)
    parser.add_argument('--refl_vgg_flag',      type=StrToBool,  default=True)
    parser.add_argument('--shad_vgg_flag',      type=StrToBool,  default=True)
    parser.add_argument('--refl_bf_flag',       type=StrToBool,  default=True)
    parser.add_argument('--shad_bf_flag',       type=StrToBool,  default=True)
    parser.add_argument('--refl_cos_flag',      type=StrToBool,  default=False)
    parser.add_argument('--shad_cos_flag',      type=StrToBool,  default=False)
    parser.add_argument('--refl_grad_flag',     type=StrToBool,  default=False)
    parser.add_argument('--shad_grad_flag',     type=StrToBool,  default=False)
    parser.add_argument('--vae',                type=StrToBool,  default=False)
    parser.add_argument('--fullsize_test',      type=StrToBool,  default=False)
    parser.add_argument('--vq_flag',            type=StrToBool,  default=False)
    parser.add_argument('--img_resize_shape',   type=str,        default=(256, 256))
    parser.add_argument('--use_tanh',           type=StrToBool,  default=False)
    parser.add_argument('--use_inception',      type=StrToBool,  default=False)
    parser.add_argument('--init_weights',       type=StrToBool,  default=False)
    parser.add_argument('--adam_flag',          type=StrToBool,  default=False)
    args = parser.parse_args()

    check_folder(args.save_path)
    check_folder(os.path.join(args.save_path, args.refl_checkpoint))
    check_folder(os.path.join(args.save_path, args.shad_checkpoint))
    # pylint: disable=E1101
    device = torch.device(args.cuda)
    # pylint: disable=E1101
    reflectance = RIN.VQVAE(vq_flag=args.vq_flag, init_weights=args.init_weights, use_tanh=args.use_tanh, use_inception=args.use_inception).to(device)
    shading = RIN.VQVAE(vq_flag=args.vq_flag, init_weights=args.init_weights, use_tanh=args.use_tanh, use_inception=args.use_inception).to(device)
    cur_epoch = 0
    if args.checkpoint:
        reflectance.load_state_dict(torch.load(os.path.join(args.save_path, args.refl_checkpoint, args.state_dict_refl)))
        shading.load_state_dict(torch.load(os.path.join(args.save_path, args.shad_checkpoint, args.state_dict_shad)))
        cur_epoch = args.cur_epoch
        print('load checkpoint success!')
    composer = RIN.SEComposer(reflectance, shading, args.refl_multi_size, args.shad_multi_size).to(device)
    
    # train_txt = "BOLD_TXT\\train_list.txt"
    # test_txt = "BOLD_TXT\\test_list.txt"

    supervision_train_set = RIN_pipeline.BOLD_Dataset(args.data_path, size_per_dataset=40000, mode='train', img_size=args.img_resize_shape)
    train_loader = torch.utils.data.DataLoader(supervision_train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)
    test_set = RIN_pipeline.BOLD_Dataset(args.data_path, size_per_dataset=None, mode='val', img_size=args.img_resize_shape)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=False)

    if args.mode == 'test':
        print('test mode .....')
        albedo_test_loss, shading_test_loss = RIN_pipeline.MPI_test_unet(composer, test_loader, device, args)
        print('albedo_test_loss: ', albedo_test_loss)
        print('shading_test_loss: ', shading_test_loss)
        return

    writer = SummaryWriter(log_dir=args.save_path)

    trainer = RIN_pipeline.BOLDVQVAETrainer(composer, train_loader, device, writer, args)

    best_albedo_loss = 9999
    best_shading_loss = 9999

    for epoch in range(cur_epoch, args.num_epochs):
        print('<Main> Epoch {}'.format(epoch))
        
        trainer.train()

        if (epoch + 1) % 20 == 0:
            args.lr = args.lr * 0.75
            trainer.update_lr(args.lr)
        
        if (epoch + 1) % 5 == 0:
            albedo_test_loss, shading_test_loss = RIN_pipeline.MPI_test_unet(composer, test_loader, device, args)
            average_loss = (albedo_test_loss + shading_test_loss) / 2
            writer.add_scalar('A_mse', albedo_test_loss, epoch)
            writer.add_scalar('S_mse', shading_test_loss, epoch)
            writer.add_scalar('aver_mse', average_loss, epoch)
            with open(os.path.join(args.save_path, 'loss_every_epoch.txt'), 'a+') as f:
                f.write('epoch{} --- average_loss: {}, albedo_loss:{}, shading_loss:{}\n'.format(epoch, average_loss, albedo_test_loss, shading_test_loss))
            if albedo_test_loss < best_albedo_loss:
                state = composer.reflectance.state_dict()
                torch.save(state, os.path.join(args.save_path, args.refl_checkpoint, 'composer_reflectance_state_{}.t7'.format(epoch)))
                best_albedo_loss = albedo_test_loss
                with open(os.path.join(args.save_path, 'reflectance_loss.txt'), 'a+') as f:
                    f.write('epoch{} --- albedo_loss:{}\n'.format(epoch, albedo_test_loss))
            if shading_test_loss < best_shading_loss:
                best_shading_loss = shading_test_loss
                state = composer.shading.state_dict()
                torch.save(state, os.path.join(args.save_path, args.shad_checkpoint, 'composer_shading_state_{}.t7'.format(epoch)))
                with open(os.path.join(args.save_path, 'shading_loss.txt'), 'a+') as f:
                    f.write('epoch{} --- shading_loss:{}\n'.format(epoch, shading_test_loss))

if __name__ == "__main__":
    main()
