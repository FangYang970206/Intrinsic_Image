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
import RIN
import RIN_pipeline

from utils import *


def main():
    random.seed(9999)
    torch.manual_seed(9999)
    torch.cuda.manual_seed(9999)
    np.random.seed(9999)
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',               type=str,   default='train')
    parser.add_argument('--save_path',          type=str,   default='MIT_logs\\RIID_origin_RIN_updateLR0.0005_4_bf_cosLoss_VGG0.1_400epochs_bs22\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--lr',                 type=float, default=0.0005,
    help='learning rate')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--save_model',         type=bool,  default=True,
    help='whether to save model or not')
    parser.add_argument('--num_epochs',         type=int,   default=400)
    parser.add_argument('--batch_size',         type=int,   default=22)
    parser.add_argument('--checkpoint',         type=bool,  default=False)
    parser.add_argument('--state_dict',         type=str,   default='composer_state.t7')
    args = parser.parse_args()

    check_folder(args.save_path)
    device = torch.device("cuda: 1" if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101
    shader = RIN.Shader()
    reflection = RIN.Decomposer()
    composer = RIN.Composer(reflection, shader).to(device)

    MIT_train_txt = 'MIT_TXT\\MIT_BarronSplit_train.txt'
    MIT_test_txt = 'MIT_TXT\\MIT_BarronSplit_test.txt'

    cur_epoch = 0
    if args.checkpoint:
        composer.load_state_dict(torch.load(os.path.join(args.save_path, args.state_dict)))
        logging.info('load checkpoint success --- ' + os.path.join(args.save_path, args.state_dict))
        cur_epoch = int(args.state_dict.split('_')[-1].split('.')[0]) + 1

    train_set = RIN_pipeline.MIT_Dataset_Revisit(MIT_train_txt, mode='train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)

    test_set = RIN_pipeline.MIT_Dataset_Revisit(MIT_test_txt, mode='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)
    
    test_loader_2 = torch.utils.data.DataLoader(test_set, batch_size=10, num_workers=args.loader_threads, shuffle=False)

    writer = SummaryWriter(log_dir=args.save_path)
    cur_aver_loss = 0
    best_loss = 9999
    best_loss_before = 9999
    trainer = RIN_pipeline.MIT_TrainerOrigin(composer, train_loader, args.lr, device, writer)
    logging.info('start training....')
    for epoch in range(cur_epoch, args.num_epochs):
        print('<Main> Epoch {}'.format(epoch))
        
        step = trainer.train()

        cur_refl_loss, cur_shad_loss = 0, 0
        with torch.no_grad():
            composer.eval()
            criterion = torch.nn.MSELoss(size_average=True).to(device)
            for _, labeled in enumerate(test_loader_2):
                labeled = [t.to(device) for t in labeled]
                input_g, albedo_g, shading_g, mask_g = labeled
                lab_inp_pred, lab_refl_pred, lab_shad_pred, _ = composer.forward(input_g)
                lab_inp_pred = lab_inp_pred * mask_g
                lab_refl_pred = lab_refl_pred * mask_g
                lab_shad_pred = lab_shad_pred * mask_g
                refl_loss = criterion(lab_refl_pred, albedo_g)
                shad_loss = criterion(lab_shad_pred, shading_g)
                recon_loss = criterion(lab_inp_pred, input_g)
                cur_refl_loss = refl_loss.item()
                cur_shad_loss = shad_loss.item()
                writer.add_scalar('test_refl_loss', refl_loss.item(), epoch)
                writer.add_scalar('test_shad_loss', shad_loss.item(), epoch)
                writer.add_scalar('test_recon_loss', recon_loss.item(), epoch)
                cur_aver_loss = (refl_loss.item() + shad_loss.item()) / 2
                writer.add_scalar('cur_aver_loss', cur_aver_loss, epoch)
                if cur_aver_loss < best_loss:
                    best_loss = cur_aver_loss
        
        if (epoch + 1) % 100 == 0:
            args.lr *= 0.75
            logging.info('epoch{} learning rate : {}'.format(epoch, args.lr))
            trainer.update_lr(args.lr)
        
        with open(os.path.join(args.save_path, 'loss_every_epoch.txt'), 'a+') as f:
            f.write('epoch{} --- average_loss: {}, albedo_loss:{}, shading_loss:{}\n'.format(epoch, best_loss, cur_refl_loss, cur_shad_loss))

        if best_loss < best_loss_before:
            best_loss_before = best_loss
            if args.save_model:
                state = composer.state_dict()
                torch.save(state, os.path.join(args.save_path, 'composer_state_{}.t7'.format(epoch)))
                logging.info('save model --- composer_state_{}.t7'.format(epoch))
            with open(os.path.join(args.save_path, 'loss.txt'), 'a+') as f:
                f.write('epoch{} --- average_loss: {}, albedo_loss:{}, shading_loss:{}\n'.format(epoch, best_loss, cur_refl_loss, cur_shad_loss))

            RIN_pipeline.visualize_MPI(composer, test_loader, device, os.path.join(args.save_path, 'image_{}.png'.format(epoch)))
    logging.info('end training....')

if __name__ == "__main__":
    main()
