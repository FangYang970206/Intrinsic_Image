import os
import random
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.backends import cudnn

# import RIN
import RIN_Unet as RIN 
import RIN_pipeline

from utils import *

def main():
    random.seed(9999)
    torch.manual_seed(9999)
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',          type=str,   default='F:\\sintel',
    help='base folder of datasets')
    parser.add_argument('--split',              type=str,   default='SceneSplit')
    parser.add_argument('--mode',               type=str,   default='train')
    parser.add_argument('--save_path',          type=str,   default='MPI_logs\\RIID_Unet_updateLR1_epoch240_CosBF_VGG0.1_shading_SceneSplit\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--lr',                 type=float, default=0.0005,
    help='learning rate')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--save_model',         type=bool,  default=True,
    help='whether to save model or not')
    parser.add_argument('--num_epochs',         type=int,   default=240)
    parser.add_argument('--batch_size',         type=int,   default=16)
    parser.add_argument('--checkpoint',         type=bool,  default=False)
    parser.add_argument('--state_dict',         type=str,   default='composer_state.t7')
    parser.add_argument('--device',             type=str,   default='cuda')
    args = parser.parse_args()

    check_folder(args.save_path)
    # pylint: disable=E1101
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101
    # shader = RIN.Shader(output_ch=3)
    composer = RIN.Decomposer().to(device)
    # composer = RIN.Composer(reflection, shader).to(device)

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
        cur_epoch = int(args.state_dict.split('_')[-1].split('.')[0]) + 1

    train_set = RIN_pipeline.MPI_Dataset_Revisit(train_txt)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)

    test_set = RIN_pipeline.MPI_Dataset_Revisit(test_txt)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)
    
    writer = SummaryWriter(log_dir=args.save_path)

    trainer = RIN_pipeline.Unet_TrainerOrigin(composer, train_loader, args.lr, device, writer)

    best_average_loss = 9999

    for epoch in range(cur_epoch, args.num_epochs):
        print('<Main> Epoch {}'.format(epoch))
        
        step = trainer.train()

        if (epoch + 1) % 120 == 0:
            args.lr = args.lr * 0.75
            trainer.update_lr(args.lr)
            
        albedo_test_loss, shading_test_loss = RIN_pipeline.MPI_test_unet(composer, test_loader, device)
        average_loss = (albedo_test_loss + shading_test_loss) / 2
        writer.add_scalar('A_mse', albedo_test_loss, epoch)
        writer.add_scalar('S_mse', shading_test_loss, epoch)
        writer.add_scalar('aver_mse', average_loss, epoch)

        with open(os.path.join(args.save_path, 'loss_every_epoch.txt'), 'a+') as f:
            f.write('epoch{} --- average_loss: {}, albedo_loss:{}, shading_loss:{}\n'.format(epoch, average_loss, albedo_test_loss, shading_test_loss))

        if average_loss < best_average_loss:
            best_average_loss = average_loss
            if args.save_model:
                state = composer.state_dict()
                torch.save(state, os.path.join(args.save_path, 'composer_state.t7'))
            with open(os.path.join(args.save_path, 'loss.txt'), 'a+') as f:
                f.write('epoch{} --- average_loss: {}, albedo_loss:{}, shading_loss:{}\n'.format(epoch, average_loss, albedo_test_loss, shading_test_loss))


if __name__ == "__main__":
    main()
