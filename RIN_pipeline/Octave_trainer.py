import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_ssim
import adabound
from tqdm import tqdm

from .COS_Loss import COS_Loss
from .Bilateral_Loss import clc_Loss_albedo, clc_Loss_shading
from .VGG_Encoder import VGG_Encoder
from .ranger import Ranger


class OctaveTrainer:
    def __init__(self, model, loader, device, writer, args):
        self.model = model
        self.loader = loader
        self.device = device
        self.writer = writer
        self.step = 0
        self.lr = args.lr
        self.mode = args.mode
        self.choose = args.choose
        self.refl_multi_size = args.refl_multi_size
        self.shad_multi_size = args.shad_multi_size
        self.refl_vgg_flag = args.refl_vgg_flag
        self.shad_vgg_flag = args.shad_vgg_flag
        self.refl_bf_flag = args.refl_bf_flag
        self.shad_bf_flag = args.shad_bf_flag
        self.mse = nn.MSELoss(size_average=True).to(device)
        self.cos = COS_Loss().to(device)
        self.vgg = VGG_Encoder(device=device)
        self.optimizer = Ranger(self.model.parameters(), lr=self.lr)
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

    def _epoch(self):
        self.model.train()

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

            if self.refl_multi_size and self.shad_multi_size:
                input_g, albedo_g, shading_g, _, refl_frame_list, shad_frame_list = labeled
            elif self.refl_multi_size:
                input_g, albedo_g, shading_g, _, refl_frame_list = labeled
            elif self.shad_multi_size:
                input_g, albedo_g, shading_g, _, shad_frame_list = labeled
            else:
                input_g, albedo_g, shading_g, _ = labeled
            
            if self.mode == 'two':
                if self.refl_multi_size and self.shad_multi_size:
                    lab_refl_pred, lab_shad_pred, lab_refl_pred_list, lab_shad_pred_list = self.model.forward(input_g)
                elif self.refl_multi_size:
                    lab_refl_pred, lab_shad_pred, lab_refl_pred_list = self.model.forward(input_g)
                elif self.shad_multi_size:
                    lab_refl_pred, lab_shad_pred, lab_shad_pred_list = self.model.forward(input_g)
                else:
                    lab_refl_pred, lab_shad_pred = self.model.forward(input_g)
            else:
                if self.choose == 'refl':
                    lab_refl_pred = self.model.forward(input_g)
                else:
                    lab_shad_pred = self.model.forward(input_g)

            self.optimizer.zero_grad()
            if self.mode == 'two':
                refl_mse_loss = self.mse(lab_refl_pred, albedo_g)
                refl_cos_loss = self.cos(lab_refl_pred, albedo_g)
                if self.refl_multi_size:
                    refl_multi_size_loss = 0.6 * self.mse(lab_refl_pred_list[0], refl_frame_list[0]) + \
                                           0.8 * self.mse(lab_refl_pred_list[1], refl_frame_list[1])
                if self.refl_bf_flag:
                    refl_bf_loss = clc_Loss_albedo(lab_refl_pred, albedo_g, self.device)
                if self.refl_vgg_flag:
                    refl_content_loss = self.vgg(lab_refl_pred, albedo_g)

                refl_loss = refl_mse_loss + refl_cos_loss + (refl_content_loss*0.1 if self.refl_vgg_flag else 0) + \
                                                            (refl_multi_size_loss if self.refl_multi_size else 0) + \
                                                            (refl_bf_loss if self.refl_bf_flag else 0)
                
                shad_mse_loss = self.mse(lab_shad_pred, shading_g)
                shad_cos_loss = self.cos(lab_shad_pred, shading_g)
                if self.shad_multi_size:
                    shad_multi_size_loss = 0.6 * self.mse(lab_shad_pred_list[0], shad_frame_list[0]) + \
                                           0.8 * self.mse(lab_shad_pred_list[1], shad_frame_list[1])
                if self.shad_bf_flag:
                    shad_bf_loss = clc_Loss_shading(lab_shad_pred, shading_g, self.device)
                if self.shad_vgg_flag:
                    shad_content_loss = self.vgg(lab_shad_pred, shading_g)
                shad_loss = shad_mse_loss + shad_cos_loss + (shad_content_loss*0.1 if self.shad_vgg_flag else 0) + \
                                                            (shad_multi_size_loss if self.shad_multi_size else 0) + \
                                                            (shad_bf_loss if self.shad_bf_flag else 0)
            else:
                if self.choose == 'refl':
                    refl_mse_loss = self.mse(lab_refl_pred, albedo_g)
                    refl_bf_loss = clc_Loss_albedo(lab_refl_pred, albedo_g)
                    refl_cos_loss = self.cos(lab_refl_pred, albedo_g)
                    refl_content_loss = self.vgg(lab_refl_pred, albedo_g)
                    refl_loss = refl_bf_loss + refl_cos_loss + refl_content_loss*0.1 + refl_mse_loss
                else:
                    shad_bf_loss = clc_Loss_shading(lab_shad_pred, shading_g)
                    shad_mse_loss = self.mse(lab_shad_pred, shading_g)
                    shad_cos_loss = self.cos(lab_shad_pred, shading_g)
                    shad_content_loss = self.vgg(lab_shad_pred, shading_g)
                    shad_loss = shad_bf_loss + shad_content_loss*0.1 + shad_mse_loss
            if self.mode == 'two':
                loss = refl_loss + shad_loss
            else:
                if self.choose == 'refl':
                    loss = refl_loss
                else:
                    loss = shad_loss
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('refl_loss', refl_loss.item(), self.step)
            self.writer.add_scalar('shad_loss', shad_loss.item(), self.step)
            self.writer.add_scalar('refl_mse_loss', refl_mse_loss.item(), self.step)
            self.writer.add_scalar('shad_mse_loss', shad_mse_loss.item(), self.step)
            self.writer.add_scalar('refl_cos_loss', refl_cos_loss.item(), self.step)
            self.writer.add_scalar('shad_cos_loss', shad_cos_loss.item(), self.step)
            self.step += 1

            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f | %.5f|' % (refl_mse_loss.item(), shad_mse_loss.item()))

    def train(self):
        self._epoch()
    
    def update_lr(self, lr):
        for optim in [self.optimizer]:
            for param_group in optim.param_groups:
                param_group['lr'] = lr
    
    def clamp_model_weights(self, Vmin=-0.1, Vmax=0.1):
        for _, x in enumerate(self.model):
            for p in x.parameters():
                p.data.clamp_(Vmin, Vmax)
        return self
    
