import os
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch.backends import cudnn
import scipy
import numpy as np

from utils import *

# import RIN
import RIN
import RIN_pipeline

def main():
    cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',          type=str,   default='F:\\BOLD_dataset',
    help='base folder of datasets')
    parser.add_argument('--mode',               type=str,  default='train')
    parser.add_argument('--save_path',          type=str,   default='logs\\lihao\\',
    help='save path of model, visualizations, and tensorboard')
    parser.add_argument('--optimizer',          type=str,   default='adam')
    parser.add_argument('--lr',                 type=float, default=0.0001,
    help='learning rate')
    parser.add_argument('--loader_threads',     type=float, default=4,
    help='number of parallel data-loading threads')
    parser.add_argument('--save_model',         type=bool,  default=True,
    help='whether to save model or not')
    parser.add_argument('--supervision_set_size',     type=int,   default=33900)
    # parser.add_argument('--unsupervision_set_size',     type=int,   default=10170)
    parser.add_argument('--num_epochs',         type=int,   default=80)
    parser.add_argument('--batch_size',         type=int,   default=16)
    parser.add_argument('--checkpoint',         type=bool,  default=False)
    parser.add_argument('--img_resize_shape',   type=str,   default=(256, 256))
    parser.add_argument('--state_dict',         type=str,   default='composer_state.t7')
    parser.add_argument('--dataset',            type=str,   default='BOLD')
    parser.add_argument('--remove_names',       type=str,   default='F:\\ShapeNet\\remove')
    args = parser.parse_args()

    # pylint: disable=E1101
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101

    shader = RIN.Shader()
    # shader.load_state_dict(torch.load('logs/shader/shader_state_59.t7'))
    decomposer = RIN.Decomposer()
    # reflection.load_state_dict(torch.load('reflection_state.t7'))
    composer = RIN.Composer(decomposer, shader).to(device)
    # RIN.init_weights(composer, init_type='kaiming')

    cur_epoch = 0
    if args.checkpoint:
        # cur_epoch = int(args.state_dict.split('.')[0].split('_')[-1])
        composer.load_state_dict(torch.load(os.path.join(args.save_path, args.state_dict)))
        print('load checkpoint success!')
    
    if args.mode == 'train':
        if args.dataset == "ShapeNet":
            remove_names = os.listdir(args.remove_names)
            supervision_train_set = RIN_pipeline.ShapeNet_Dateset_new(args.data_path, size_per_dataset=args.supervision_set_size, mode='train', img_size=args.img_resize_shape, remove_names=remove_names)
            sv_train_loader = torch.utils.data.DataLoader(supervision_train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)
            test_set = RIN_pipeline.ShapeNet_Dateset_new(args.data_path, size_per_dataset=20, mode='test', img_size=args.img_resize_shape, remove_names=remove_names)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)
        else:
            train_txt = 'lihao/train_list.txt'
            test_txt = 'lihao/test_list.txt'
            supervision_train_set = RIN_pipeline.BOLD_Dataset(args.data_path, size_per_dataset=args.supervision_set_size, mode='train', img_size=args.img_resize_shape, file_name=train_txt)
            sv_train_loader = torch.utils.data.DataLoader(supervision_train_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=True)
            test_set = RIN_pipeline.BOLD_Dataset(args.data_path, size_per_dataset=20, mode='test', img_size=args.img_resize_shape, file_name=test_txt)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)

        writer = SummaryWriter(log_dir=args.save_path)

        step = 0
        trainer = RIN_pipeline.ShapeNetSupervisionTrainer(composer, sv_train_loader, args.lr, device, writer, step, optim_choose=args.optimizer)
        for epoch in range(cur_epoch+1, args.num_epochs):
            print('<Main> Epoch {}'.format(epoch))
            if epoch % 10 == 0:
                trainer.update_lr(args.lr * 0.75)
            if epoch < 100:
                step = trainer.train()
            # else:
            #     trainer = RIN_pipeline.UnsupervisionTrainer(composer, unsv_train_loader, args.lr, device, writer, new_step)
            #     new_step = trainer.train()

            if args.save_model:
                state = composer.state_dict()
                torch.save(state, os.path.join(args.save_path, 'composer_state.t7'))

            # step += new_step
            # loss = RIN_pipeline.visualize_composer(composer, test_loader, device, os.path.join(args.save_path, '{}.png'.format(epoch)))
            # writer.add_scalar('test_recon_loss', loss[0], epoch)
            # writer.add_scalar('test_refl_loss', loss[1], epoch)
            # writer.add_scalar('test_sha_loss', loss[2], epoch)
    else:
        check_folder(os.path.join(args.save_path, "refl_target"))
        check_folder(os.path.join(args.save_path, "shad_target"))
        check_folder(os.path.join(args.save_path, "refl_output"))
        check_folder(os.path.join(args.save_path, "shad_output"))
        check_folder(os.path.join(args.save_path, "shape_output"))
        if args.dataset == "ShapeNet":
            # check_folder(os.path.join(args.save_path, "mask"))
            remove_names = os.listdir(args.remove_names)
            test_set = RIN_pipeline.ShapeNet_Dateset_new(args.data_path, size_per_dataset=9488, mode='test', img_size=args.img_resize_shape, remove_names=remove_names)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)
        else:
            test_txt = 'lihao/test_list.txt'
            test_set = RIN_pipeline.BOLD_Dataset(args.data_path, size_per_dataset=18984, mode='test', img_size=args.img_resize_shape, file_name=test_txt)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=args.loader_threads, shuffle=False)
        
        composer.load_state_dict(torch.load(os.path.join(args.save_path, args.state_dict)))
        composer.eval()
        with torch.no_grad():
            for ind, tensors in enumerate(test_loader):
            
                inp = [t.float().to(device) for t in tensors]
                try:
                    lab_inp, lab_refl_targ, lab_sha_targ, mask = inp
                except ValueError:
                    lab_inp, lab_refl_targ, lab_sha_targ = inp
                else:
                    print('input dim should be 3 or 4')
                lab_inp, lab_refl_targ, lab_sha_targ = inp
                recon_pred, refl_pred,  sha_pred, shape_pred = composer.forward(lab_inp)

                lab_refl_targ = lab_refl_targ.squeeze().cpu().numpy().transpose(1,2,0)
                lab_sha_targ = lab_sha_targ.squeeze().cpu().numpy().transpose(1,2,0)
                # mask = mask.squeeze().cpu().numpy().transpose(1,2,0)
                refl_pred = refl_pred.squeeze().cpu().numpy().transpose(1,2,0)
                sha_pred = sha_pred.squeeze().cpu().numpy().transpose(1,2,0)
                shape_pred = shape_pred.squeeze().cpu().numpy().transpose(1,2,0)
                lab_refl_targ = np.clip(lab_refl_targ, 0, 1)
                lab_sha_targ = np.clip(lab_sha_targ, 0, 1)
                refl_pred = np.clip(refl_pred, 0, 1)
                sha_pred = np.clip(sha_pred, 0, 1)
                shape_pred = np.clip(shape_pred, 0, 1)
                # mask = np.clip(mask, 0, 1)
                scipy.misc.imsave(os.path.join(args.save_path, "refl_target", "{}.png".format(ind)), lab_refl_targ)
                scipy.misc.imsave(os.path.join(args.save_path, "shad_target", "{}.png".format(ind)), lab_sha_targ)
                # scipy.misc.imsave(os.path.join(args.save_path, "mask", "{}.png".format(ind)), mask)
                scipy.misc.imsave(os.path.join(args.save_path, "refl_output", "{}.png".format(ind)), refl_pred)
                scipy.misc.imsave(os.path.join(args.save_path, "shad_output", "{}.png".format(ind)), sha_pred)
                scipy.misc.imsave(os.path.join(args.save_path, "shape_output", "{}.png".format(ind)), shape_pred)

            

if __name__ == "__main__":
    main()
