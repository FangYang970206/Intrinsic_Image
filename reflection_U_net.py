import os
import argparse
import U_Net
import pipeline
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',          type=str,   default='F:\\BOLD\\',
    help='base folder of datasets')
    parser.add_argument('--mode',               type=list,  default=['train_refl', 'test_refl'])
    parser.add_argument('--save_path',          type=str,   default='logs\\reflection\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--lr',                 type=float, default=0.005,
    help='learning rate')
    parser.add_argument('--loader_threads',     type=float, default=16,
    help='number of parallel data-loading threads')
    parser.add_argument('--save_model',         type=bool,  default=True,
    help='whether to save model or not')
    parser.add_argument('--train_set_size',     type=int,   default=68800,
    help='number of images in an epoch')
    parser.add_argument('--num_epochs',         type=int,   default=150)
    parser.add_argument('--batch_size',         type=int,   default=16)
    parser.add_argument('--checkpoint',         type=bool,  default=False)
    parser.add_argument('--state_dict',         type=str,   default='reflection_state.t7')
    args = parser.parse_args()

    # pylint: disable=E1101
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101

    model = U_Net.Reflection().to(device)
    
    if args.checkpoint:
        model.load_state_dict(torch.load(args.state_dict))
        print('load checkpoint success!')

    train_set = pipeline.BOLD_Dataset(args.data_path, size_per_dataset=args.train_set_size, mode=args.mode[0])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)
    
    val_set = pipeline.BOLD_Dataset(args.data_path, size_per_dataset=20, mode=args.mode[1])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=20, num_workers=args.loader_threads, shuffle=False)
    
    writer = SummaryWriter(log_dir=args.save_path)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[80, 120])

    dummy_input = torch.rand(3, 3, 512, 512).to(device)
    writer.add_graph(model, dummy_input)
    step = 0
    for epoch in range(args.num_epochs):
        print('<Main> Epoch {}'.format(epoch))
        trainer = pipeline.ReflectionTrainer(model, train_loader, device, optimizer, writer, step)
        step = trainer.train()
        if args.save_model:
                state = model.state_dict()
                torch.save(state, 'reflection_state.t7')
        
        test_loss = pipeline.visualize_reflection(model, val_loader, device, os.path.join(args.save_path, '{}.png'.format(epoch)))
        writer.add_scalar('test_refl_loss', test_loss, step)
        scheduler.step()

if __name__ == "__main__":
    main()
