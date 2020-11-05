import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_ssim
import adabound
import itertools
import random
from tqdm import tqdm
from torch.autograd import Variable
from .COS_Loss import COS_Loss
from .Bilateral_Loss import clc_Loss_albedo, clc_Loss_shading
from .VGG_Encoder import VGG_Encoder
from .ranger import Ranger
from .gradient_loss import GradientLoss


class BOLDVQVAETrainer:
    def __init__(self, Generator, loader, device, writer, args):
        self.G = Generator
        self.loader = loader
        self.device = device
        self.writer = writer
        self.mse = nn.MSELoss(size_average=True).to(device)
        self.vgg = VGG_Encoder(device=device)
        self.cos = COS_Loss().to(device)
        self.grad = GradientLoss(self.device)
        self.args = args
        self.lr = args.lr
        self.refl_multi_size = args.refl_multi_size
        self.shad_multi_size = args.shad_multi_size
        self.refl_vgg_flag = args.refl_vgg_flag
        self.shad_vgg_flag = args.shad_vgg_flag
        self.refl_bf_flag = args.refl_bf_flag
        self.shad_bf_flag = args.shad_bf_flag
        self.refl_cos_flag = args.refl_cos_flag
        self.shad_cos_flag = args.shad_cos_flag
        self.refl_grad_flag = args.refl_grad_flag
        self.shad_grad_flag = args.shad_grad_flag
        self.weight_decay = args.weight_decay
        self.step = 0
        self.vq_flag = args.vq_flag
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=self.lr)
        
        if self.refl_multi_size:
            print('refl_multi_size mode on')
        if self.shad_multi_size:
            print('shad_multi_size mode on')
        if self.refl_vgg_flag:
            print('refl_vgg_flag mode on')
        if self.shad_vgg_flag:
            print('shad_vgg_flag mode on')
        if self.refl_bf_flag:
            print('refl_bf_flag mode on')
        if self.shad_bf_flag:
            print('shad_bf_flag mode on')
        if self.refl_cos_flag:
            print('refl_cos_flag mode on')
        if self.shad_cos_flag:
            print('shad_cos_flag mode on')
        if self.refl_grad_flag:
            print('refl_grad_flag mode on')
        if self.shad_grad_flag:
            print('shad_grad_flag mode on')

    def _epoch(self):
        self.G.train()

        progress = tqdm( total=len(self.loader.dataset))

        for _, labeled in enumerate(self.loader):
            for i in range(len(labeled)):
                labeled[i] = labeled[i].to(self.device)
            input_g, albedo_g, shading_g = labeled

            self.optimizer_G.zero_grad()
            
            if self.vq_flag:
                refl_list, shad_list = self.G.forward(input_g)
                lab_refl_pred, refl_diff = refl_list
                lab_shad_pred, shad_diff = shad_list
            else:
                lab_refl_pred, lab_shad_pred = self.G.forward(input_g)

            refl_mse_loss = self.mse(lab_refl_pred, albedo_g)
            
            # if self.refl_cos_flag:
            #     refl_cos_loss = self.cos(lab_refl_pred, albedo_g)
            
            # if self.refl_multi_size:
            #     refl_multi_size_loss = 0.6 * self.mse(lab_refl_pred_list[0], refl_frame_list[0]) + \
            #                            0.8 * self.mse(lab_refl_pred_list[1], refl_frame_list[1])
            # if self.refl_bf_flag:
            #     refl_bf_loss = clc_Loss_albedo(lab_refl_pred, albedo_g, self.device)

            if self.refl_vgg_flag:
                refl_content_loss = self.vgg(lab_refl_pred, albedo_g)
            if self.vq_flag:
                refl_diff_loss = refl_diff.mean()
            # if self.refl_grad_flag:
            #     refl_grad_loss = self.grad(lab_refl_pred, albedo_g)

            refl_loss = refl_mse_loss + \
                        (refl_content_loss*0.1 if self.refl_vgg_flag else 0) + \
                        (refl_diff_loss if self.vq_flag else 0)
            #             (refl_multi_size_loss if self.refl_multi_size else 0) + \
            #             (refl_bf_loss if self.refl_bf_flag else 0) + \
            #             (refl_grad_loss if self.refl_grad_flag else 0)
            #             (refl_cos_loss if self.refl_cos_flag else 0) + \

            refl_loss.backward()

            shad_mse_loss = self.mse(lab_shad_pred, shading_g)
            
            # if self.shad_multi_size:
            #     shad_multi_size_loss = 0.6 * self.mse(lab_shad_pred_list[0], shad_frame_list[0]) + \
            #                            0.8 * self.mse(lab_shad_pred_list[1], shad_frame_list[1])
            
            # if self.shad_cos_flag:
            #     shad_cos_loss = self.cos(lab_shad_pred, shading_g)

            # if self.shad_bf_flag:
            #     shad_bf_loss = clc_Loss_shading(lab_shad_pred, shading_g, self.device)

            if self.shad_vgg_flag:
                shad_content_loss = self.vgg(lab_shad_pred, shading_g)
            if self.vq_flag:
                shad_diff_loss = shad_diff.mean()
            # if self.shad_grad_flag:
            #     shad_grad_loss = self.grad(lab_shad_pred, shading_g)
            shad_loss = shad_mse_loss + \
                        (shad_content_loss*0.1 if self.shad_vgg_flag else 0) + \
                        (shad_diff_loss if self.vq_flag else 0)
            # shad_loss = shad_mse_loss + shad_D_loss + \
            #             (shad_cos_loss if self.shad_cos_flag else 0) + \
            #             (shad_content_loss*0.1 if self.shad_vgg_flag else 0) + \
            #             (shad_multi_size_loss if self.shad_multi_size else 0) + \
            #             (shad_bf_loss if self.shad_bf_flag else 0) + \
            #             (shad_grad_loss if self.shad_grad_flag else 0)
            shad_loss.backward()
            
            # loss.backward()
            self.optimizer_G.step()

            self.writer.add_scalar('refl_mse_loss', refl_mse_loss.item(), self.step)
            # self.writer.add_scalar('refl_D_loss', refl_D_loss.item(), self.step)
            if self.refl_vgg_flag:
                self.writer.add_scalar('refl_content_loss', refl_content_loss.item(), self.step)
            if self.vq_flag:
                self.writer.add_scalar('refl_diff_loss', refl_diff_loss.item(), self.step)
            self.writer.add_scalar('shad_mse_loss', shad_mse_loss.item(), self.step)
            if self.shad_vgg_flag:
                self.writer.add_scalar('shad_content_loss', shad_content_loss.item(), self.step)
            if self.vq_flag:
                self.writer.add_scalar('shad_diff_loss', shad_diff_loss.item(), self.step)
            # self.writer.add_scalar('shad_D_loss', shad_D_loss.item(), self.step)
            # if self.shad_grad_flag:
            #     self.writer.add_scalar('shad_grad_loss', shad_grad_loss.item(), self.step)

            self.step += 1

            progress.update(self.loader.batch_size)
            # progress.set_description( '%.5f | %.5f| %.5f| %.5f |' % (refl_mse_loss.item(), shad_mse_loss.item(), refl_diff_loss.item(), shad_diff_loss.item()) )

    def train(self):
        self._epoch()

    def update_lr(self, lr):
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
    
