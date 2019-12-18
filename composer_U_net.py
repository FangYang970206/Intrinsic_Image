import os
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch.backends import cudnn

import U_Net
import pipeline

def main():
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',          type=str,   default='F:\\BOLD\\',
    help='base folder of datasets')
    parser.add_argument('--mode',               type=list,  default=['val', 'test'])
    parser.add_argument('--save_path',          type=str,   default='logs\\composer\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--lr',                 type=float, default=0.001,
    help='learning rate')
    parser.add_argument('--loader_threads',     type=float, default=16,
    help='number of parallel data-loading threads')
    parser.add_argument('--save_model',         type=bool,  default=True,
    help='whether to save model or not')
    parser.add_argument('--train_set_size',     type=int,   default=10170,
    help='number of images in an epoch')
    parser.add_argument('--num_epochs',         type=int,   default=60)
    parser.add_argument('--batch_size',         type=int,   default=8)
    parser.add_argument('--checkpoint',         type=bool,  default=False)
    parser.add_argument('--state_dict',         type=str,   default='composer_state.t7')
    args = parser.parse_args()

    # pylint: disable=E1101
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101

    shader = U_Net.Shader()
    shader.load_state_dict(torch.load('logs/shader/shader_state_59.t7'))
    reflection = U_Net.Reflection()
    reflection.load_state_dict(torch.load('reflection_state.t7'))
    composer = U_Net.Composer(reflection, shader).to(device)

    if args.checkpoint:
        composer.load_state_dict(torch.load(args.state_dict))
        print('load checkpoint success!')

    train_set = pipeline.BOLD_Dataset(args.data_path, size_per_dataset=args.train_set_size, mode=args.mode[0])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)
    
    val_set = pipeline.BOLD_Dataset(args.data_path, size_per_dataset=20, mode=args.mode[1])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)
    
    writer = SummaryWriter(log_dir=args.save_path)
    optimizer = optim.Adam(composer.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40])

    dummy_input = torch.rand(3, 3, 512, 512).to(device)
    writer.add_graph(composer, dummy_input)
    step = 0
    for epoch in range(args.num_epochs):
        print('<Main> Epoch {}'.format(epoch))
        trainer = pipeline.ComposerTrainer(composer, train_loader, device, optimizer, writer, step)
        step = trainer.train()
        if args.save_model:
                state = composer.state_dict()
                torch.save(state, os.path.join(args.save_path, 'composer_state_{}.t7'.format(epoch)))
        
        loss = pipeline.visualize_composer(composer, val_loader, device, os.path.join(args.save_path, '{}.png'.format(epoch)))
        writer.add_scalar('test_recon_loss', loss[0], step)
        writer.add_scalar('test_refl_loss', loss[1], step)
        writer.add_scalar('test_sha_loss', loss[2], step)
        scheduler.step()


if __name__ == "__main__":
    main()
