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
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    # cudnn.benchmark = True
    # cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',          type=str,   default='IIW_logs\\RIID_new_RIN_updateLR1_epoch240\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--lr',                 type=float, default=0.0005,
    help='learning rate')
    parser.add_argument('--loader_threads',     type=float, default=8,
    help='number of parallel data-loading threads')
    parser.add_argument('--save_model',         type=bool,  default=True,
    help='whether to save model or not')
    parser.add_argument('--num_epochs',         type=int,   default=40)
    parser.add_argument('--batch_size',         type=int,   default=1)
    parser.add_argument('--checkpoint',         type=bool,  default=False)
    parser.add_argument('--state_dict',         type=str,   default='composer_state.t7')
    parser.add_argument('--cuda',               type=str,   default='cuda')
    parser.add_argument('--image_size',         type=StrToInt, default=256)
    args = parser.parse_args()

    check_folder(args.save_path)
    # pylint: disable=E1101
    device = torch.device(args.cuda)
    # pylint: disable=E1101
    composer = RIN.SEDecomposerSingle().to(device)

    IIW_train_txt = 'F:\\revisit_IID\\iiw-dataset\\iiw_Learning_Lightness_train.txt'
    IIW_test_txt = 'F:\\revisit_IID\\iiw-dataset\\iiw_Learning_Lightness_test.txt'

    train_set = RIN_pipeline.IIW_Dataset_Revisit(IIW_train_txt)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)

    test_set = RIN_pipeline.IIW_Dataset_Revisit(IIW_test_txt, out_mode='txt')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=False)
    
    writer = SummaryWriter(log_dir=args.save_path)

    trainer = RIN_pipeline.IIWTrainer(composer, train_loader, device, writer, args)

    best_score = 9999

    for epoch in range(args.num_epochs):
        print('<Main> Epoch {}'.format(epoch))
        
        trainer.train()

        if (epoch + 1) % 40 == 0:
            args.lr = args.lr * 0.75
            trainer.update_lr(args.lr)
            
        score = RIN_pipeline.IIW_test_unet(composer, test_loader, device)
        writer.add_scalar('score', score, epoch)

        with open(os.path.join(args.save_path, 'score.txt'), 'a+') as f:
            f.write('epoch{} --- score: {}\n'.format(epoch, score))

        if score < best_score:
            best_score = score
            if args.save_model:
                state = composer.state_dict()
                torch.save(state, os.path.join(args.save_path, 'composer_state.t7'))
            with open(os.path.join(args.save_path, 'score_best.txt'), 'a+') as f:
                f.write('epoch{} --- score:{}\n'.format(epoch, score))


if __name__ == "__main__":
    main()
