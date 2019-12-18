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



class SupervisionTrainer:
    def __init__(self, model, loader, lr, device, writer, step, optim_choose='adam'):
        self.model = model
        self.loader = loader
        self.criterion1 = nn.MSELoss(size_average=True).to(device)
        # self.criterion2 = pytorch_ssim.SSIM()
        self.criterion3 = COS_Loss().to(device)
        self.criterion4 = VGG_Encoder()
        self.device = device
        self.writer = writer
        self.step = step
        parameters1 = []
        parameters2 = []
        parameters1.append( {'params': self.model.decomposer.encoder2.parameters(), 'lr': lr} )
        parameters1.append( {'params': self.model.decomposer.decoder_normals.parameters(), 'lr': lr} )
        parameters1.append( {'params': self.model.decomposer.decoder_lights.parameters(), 'lr': lr} )
        parameters1.append( {'params': self.model.shader.parameters(), 'lr': lr} )
        parameters2.append( {'params': self.model.decomposer.encoder1.parameters(), 'lr': lr} )
        parameters2.append( {'params': self.model.decomposer.decoder_reflectance.parameters(), 'lr': lr} )
        # parameters1.append( {'params': self.model.reflection.parameters(), 'lr': lr} )
        # parameters2.append( {'params': self.model.shader.parameters(), 'lr': lr} )
        if optim_choose == 'adam':
            self.optimizer1 = optim.Adam(parameters1, lr=lr)
            self.optimizer2 = optim.Adam(parameters2, lr=lr)
        elif optim_choose == 'adabound':
            self.optimizer1 = adabound.AdaBound(parameters1, lr=lr, final_lr=0.1)
            self.optimizer2 = adabound.AdaBound(parameters2, lr=lr, final_lr=0.1)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def __epoch(self):
        self.model.train()

        progress = tqdm( total=len(self.loader.dataset))

        for ind, labeled in enumerate(self.loader):
            labeled = [t.float().to(self.device) for t in labeled]

            lab_inp, lab_refl_targ, lab_shad_targ = labeled
            
            _, lab_refl_pred, lab_shad_pred, shape_pred = self.model.forward(lab_inp)
            # _, lab_refl_pred, lab_shad_pred = self.model.forward(lab_inp)

            self.optimizer2.zero_grad()   
            # refl_loss_SSIM = -self.criterion2(lab_refl_targ, lab_refl_pred)
            refl_mse_loss = self.criterion1(lab_refl_pred, lab_refl_targ)
            refl_bf_loss = clc_Loss_albedo(lab_refl_pred, lab_refl_targ)
            refl_cos_loss = self.criterion3(lab_refl_pred, lab_refl_targ)
            refl_content_loss = self.criterion4(lab_refl_pred, lab_refl_targ)
            # refl_loss = refl_mse_loss + refl_cos_loss
            refl_loss = refl_bf_loss + refl_cos_loss + refl_content_loss*0.1
            # refl_loss = refl_bf_loss + refl_cos_loss
            # refl_loss = refl_content_loss + refl_cos_loss
            refl_loss.backward()
            self.optimizer2.step()

            self.optimizer1.zero_grad()
            shad_bf_loss = clc_Loss_shading(lab_shad_pred, lab_shad_targ)
            shad_mse_loss = self.criterion1(lab_shad_pred, lab_shad_targ)
            shad_cos_loss = self.criterion3(lab_shad_pred, lab_shad_targ)
            shape_cos_loss = self.criterion3(shape_pred, lab_shad_targ)
            # print(lab_shad_pred.size())
            # print(lab_shad_targ.size())
            shad_content_loss = self.criterion4(lab_shad_pred, lab_shad_targ)
            # shad_loss = shad_mse_loss + shape_cos_loss
            shad_loss = shad_bf_loss + shad_content_loss*0.1 + shape_cos_loss
            # shad_loss = shad_content_loss + shape_cos_loss
            shad_loss.backward()
            self.optimizer1.step()

            self.writer.add_scalar('supervision_refl_bf_loss', refl_bf_loss.item(), self.step)
            self.writer.add_scalar('supervision_refl_content_loss', refl_content_loss.item(), self.step)
            self.writer.add_scalar('supervision_refl_cos_loss', refl_cos_loss.item(), self.step)
            self.writer.add_scalar('supervision_shad_bf_loss', shad_bf_loss.item(), self.step)
            self.writer.add_scalar('supervision_shad_content_loss', shad_content_loss.item(), self.step)
            self.writer.add_scalar('supervision_shad_mse_loss', shad_mse_loss.item(), self.step)
            self.writer.add_scalar('supervision_shad_cos_loss', shad_cos_loss.item(), self.step)
            self.writer.add_scalar('supervision_refl_mse_loss', refl_mse_loss.item(), self.step)
            self.step += 1

            # self.optimizer.zero_grad()
            # refl_loss = self.criterion1(lab_refl_pred, lab_refl_targ)
            # shad_loss = self.criterion1(lab_shad_pred, lab_shad_targ)
            # un_loss = self.criterion1(un_recon, un_inp)
            # un_loss.backward()
            # self.optimizer.step()

            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f | %.5f| %.5f| %.5f | %.5f| %.5f' % (refl_mse_loss.item(), shad_mse_loss.item(), refl_bf_loss.item(), shad_bf_loss.item(), refl_content_loss.item(), shad_content_loss.item()) )
 
        return self.step

    def train(self):
        errors = self.__epoch()
        return errors



