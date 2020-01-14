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


class SEUGTrainerNew:
    def __init__(self, Generator, Discriminator_R, Discriminator_S, loader, device, writer, args):
        self.G = Generator
        self.D_R = Discriminator_R
        self.D_S = Discriminator_S
        self.loader = loader
        self.device = device
        self.writer = writer
        self.mse = nn.MSELoss(size_average=True).to(device)
        self.vgg = VGG_Encoder(device=device)
        self.cos = COS_Loss().to(device)
        self.grad = GradientLoss(self.device)
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
        self.refl_D_weight_flag = args.refl_D_weight_flag
        self.shad_D_weight_flag = args.shad_D_weight_flag
        self.weight_decay = args.weight_decay
        self.step = 0
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.optimizer_D = optim.Adam(itertools.chain(self.D_R.parameters(), self.D_S.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        
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
        if self.refl_D_weight_flag:
            print('refl_D_weight_flag mode on')
            self.refl_D_weight = [4, 1, 1, 4]
        else:
            self.refl_D_weight = [1, 1, 1, 1]
        if self.shad_D_weight_flag:
            print('shad_D_weight_flag mode on')
            self.shad_D_weight = [4, 1, 1, 4]
        else:
            self.shad_D_weight = [1, 1, 1, 1]


    def _epoch(self):
        self.G.train()
        self.D_R.train()
        self.D_S.train()

        progress = tqdm( total=len(self.loader.dataset))

        for _, labeled in enumerate(self.loader):
            for i in range(len(labeled)):
                if isinstance(labeled[i], list):
                    if len(labeled[i]) != 2:
                        raise ValueError('list must be 2')
                    for j in range(len(labeled[i])):
                        labeled[i][j] = labeled[i][j].to(self.device)
                else:
                    labeled[i] = labeled[i].to(self.device)

            # labeled = [t.to(self.device) for t in labeled]
            if self.refl_multi_size and self.shad_multi_size:
                input_g, albedo_g, shading_g, _, refl_frame_list, shad_frame_list = labeled
            elif self.refl_multi_size:
                input_g, albedo_g, shading_g, _, refl_frame_list = labeled
            elif self.shad_multi_size:
                input_g, albedo_g, shading_g, _, shad_frame_list = labeled
            else:
                input_g, albedo_g, shading_g, _ = labeled

            for i in range(5):
                if self.refl_multi_size and self.shad_multi_size:
                    lab_refl_pred, lab_shad_pred, lab_refl_pred_list, lab_shad_pred_list = self.G.forward(input_g)
                elif self.refl_multi_size:
                    lab_refl_pred, lab_shad_pred, lab_refl_pred_list = self.G.forward(input_g)
                elif self.shad_multi_size:
                    lab_refl_pred, lab_shad_pred, lab_shad_pred_list = self.G.forward(input_g)
                else:
                    lab_refl_pred, lab_shad_pred = self.G.forward(input_g)

                if i == 0:
                    new_albedo_g = albedo_g
                    new_shading_g = shading_g
                    new_lab_refl_pred = lab_refl_pred
                    new_lab_shad_pred = lab_shad_pred
                else:
                    x_start = random.randint(0, 255 - 32)
                    y_start = random.randint(0, 255 - 32)
                    # new_input_g = input_g[:, :, x_start:x_start+32, y_start:y_start+32]
                    new_albedo_g = albedo_g[:, :, x_start:x_start+32, y_start:y_start+32]
                    new_shading_g = shading_g[:, :, x_start:x_start+32, y_start:y_start+32]
                    new_lab_refl_pred = lab_refl_pred[:, :, x_start:x_start+32, y_start:y_start+32]
                    new_lab_shad_pred = lab_shad_pred[:, :, x_start:x_start+32, y_start:y_start+32]

                # print(new_albedo_g.size())
                real_logit_R = self.D_R(new_albedo_g)
                real_logit_S = self.D_S(new_shading_g)

                fake_logit_R = self.D_R(new_lab_refl_pred)
                fake_logit_S = self.D_S(new_lab_shad_pred)

                D_R_loss = - torch.mean(real_logit_R) + torch.mean(fake_logit_R)
                D_S_loss = - torch.mean(real_logit_S) + torch.mean(fake_logit_S)

                D_loss = D_R_loss + D_S_loss
            
                self.optimizer_D.zero_grad()
                D_loss.backward()
                self.optimizer_D.step()

                # wgan-gp loss
                alpha_R = torch.rand(new_albedo_g.size(0), 1, 1, 1).cuda().expand_as(new_albedo_g)
                interpolated_R = Variable(alpha_R * new_albedo_g.data + (1 - alpha_R) * new_lab_refl_pred.data, requires_grad=True)
                out_R = self.D_R(interpolated_R)
                grad_R = torch.autograd.grad(outputs=out_R,
                                           inputs=interpolated_R,
                                           grad_outputs=torch.ones(out_R.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]
                grad_R = grad_R.view(grad_R.size(0), -1)
                grad_l2norm_R = torch.sqrt(torch.sum(grad_R ** 2, dim=1))
                R_D_loss_gp = torch.mean((grad_l2norm_R - 1) ** 2)
                
                alpha_S = torch.rand(new_shading_g.size(0), 1, 1, 1).cuda().expand_as(new_shading_g)
                interpolated_S = Variable(alpha_S * new_shading_g.data + (1 - alpha_S) * new_lab_shad_pred.data, requires_grad=True)
                out_S = self.D_S(interpolated_S)
                grad_S = torch.autograd.grad(outputs=out_S,
                                           inputs=interpolated_S,
                                           grad_outputs=torch.ones(out_S.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]
                grad_S = grad_S.view(grad_S.size(0), -1)
                grad_l2norm_S = torch.sqrt(torch.sum(grad_S ** 2, dim=1))
                S_D_loss_gp = torch.mean((grad_l2norm_S - 1) ** 2)

                D_loss_gp = 10 * (R_D_loss_gp + S_D_loss_gp)
                self.optimizer_D.zero_grad()
                D_loss_gp.backward()
                self.optimizer_D.step()

            self.optimizer_G.zero_grad()
            
            if self.refl_multi_size and self.shad_multi_size:
                lab_refl_pred, lab_shad_pred, lab_refl_pred_list, lab_shad_pred_list = self.G.forward(input_g)
            elif self.refl_multi_size:
                lab_refl_pred, lab_shad_pred, lab_refl_pred_list = self.G.forward(input_g)
            elif self.shad_multi_size:
                lab_refl_pred, lab_shad_pred, lab_shad_pred_list = self.G.forward(input_g)
            else:
                lab_refl_pred, lab_shad_pred = self.G.forward(input_g)

            fake_logit_R = self.D_R(lab_refl_pred)
            fake_logit_S = self.D_S(lab_shad_pred)

            refl_mse_loss = self.mse(lab_refl_pred, albedo_g)
            
            refl_D_loss = - fake_logit_R.mean()
            # refl_D_loss = self.refl_D_weight[0] * self.mse(fake_logit_R[0], torch.ones_like(fake_logit_R[0]).to(self.device)) + \
            #               self.refl_D_weight[1] * self.mse(fake_logit_R[1], torch.ones_like(fake_logit_R[1]).to(self.device)) + \
            #               self.refl_D_weight[2] * self.mse(fake_logit_R[2], torch.ones_like(fake_logit_R[2]).to(self.device)) + \
            #               self.refl_D_weight[3] * self.mse(fake_logit_R[3], torch.ones_like(fake_logit_R[3]).to(self.device))
            if self.refl_cos_flag:
                refl_cos_loss = self.cos(lab_refl_pred, albedo_g)
            
            if self.refl_multi_size:
                refl_multi_size_loss = 0.6 * self.mse(lab_refl_pred_list[0], refl_frame_list[0]) + \
                                       0.8 * self.mse(lab_refl_pred_list[1], refl_frame_list[1])
            if self.refl_bf_flag:
                refl_bf_loss = clc_Loss_albedo(lab_refl_pred, albedo_g, self.device)

            if self.refl_vgg_flag:
                refl_content_loss = self.vgg(lab_refl_pred, albedo_g)

            if self.refl_grad_flag:
                refl_grad_loss = self.grad(lab_refl_pred, albedo_g)

            refl_loss = refl_mse_loss + refl_D_loss + \
                        (refl_cos_loss if self.refl_cos_flag else 0) + \
                        (refl_content_loss*0.1 if self.refl_vgg_flag else 0) + \
                        (refl_multi_size_loss if self.refl_multi_size else 0) + \
                        (refl_bf_loss if self.refl_bf_flag else 0) + \
                        (refl_grad_loss if self.refl_grad_flag else 0)

            refl_loss.backward()

            shad_mse_loss = self.mse(lab_shad_pred, shading_g)

            shad_D_loss = - fake_logit_S.mean()
            # shad_D_loss = self.shad_D_weight[0] * self.mse(fake_logit_S[0], torch.ones_like(fake_logit_S[0]).to(self.device)) + \
            #               self.shad_D_weight[1] * self.mse(fake_logit_S[1], torch.ones_like(fake_logit_S[1]).to(self.device)) + \
            #               self.shad_D_weight[2] * self.mse(fake_logit_S[2], torch.ones_like(fake_logit_S[2]).to(self.device)) + \
            #               self.shad_D_weight[3] * self.mse(fake_logit_S[3], torch.ones_like(fake_logit_S[3]).to(self.device))
            
            if self.shad_multi_size:
                shad_multi_size_loss = 0.6 * self.mse(lab_shad_pred_list[0], shad_frame_list[0]) + \
                                       0.8 * self.mse(lab_shad_pred_list[1], shad_frame_list[1])
            
            if self.shad_cos_flag:
                shad_cos_loss = self.cos(lab_shad_pred, shading_g)

            if self.shad_bf_flag:
                shad_bf_loss = clc_Loss_shading(lab_shad_pred, shading_g, self.device)

            if self.shad_vgg_flag:
                shad_content_loss = self.vgg(lab_shad_pred, shading_g)

            if self.shad_grad_flag:
                shad_grad_loss = self.grad(lab_shad_pred, shading_g)
            
            shad_loss = shad_mse_loss + shad_D_loss + \
                        (shad_cos_loss if self.shad_cos_flag else 0) + \
                        (shad_content_loss*0.1 if self.shad_vgg_flag else 0) + \
                        (shad_multi_size_loss if self.shad_multi_size else 0) + \
                        (shad_bf_loss if self.shad_bf_flag else 0) + \
                        (shad_grad_loss if self.shad_grad_flag else 0)

            shad_loss.backward()
            self.optimizer_G.step()

            self.writer.add_scalar('refl_mse_loss', refl_mse_loss.item(), self.step)
            self.writer.add_scalar('refl_D_loss', refl_D_loss.item(), self.step)
            if self.refl_grad_flag:
                self.writer.add_scalar('refl_grad_loss', refl_grad_loss.item(), self.step)
            
            self.writer.add_scalar('shad_mse_loss', shad_mse_loss.item(), self.step)
            self.writer.add_scalar('shad_D_loss', shad_D_loss.item(), self.step)
            if self.shad_grad_flag:
                self.writer.add_scalar('shad_grad_loss', shad_grad_loss.item(), self.step)

            self.step += 1

            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f | %.5f| %.5f| %.5f |' % (refl_mse_loss.item(), shad_mse_loss.item(), refl_D_loss.item(), shad_D_loss.item()) )

    def train(self):
        self._epoch()

    def update_lr(self, lr):
        for optim in [self.optimizer_G, self.optimizer_D]:
            for param_group in optim.param_groups:
                param_group['lr'] = lr
    