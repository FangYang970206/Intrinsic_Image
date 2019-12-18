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



class MPI_TrainerOriginRemoveShape:
    def __init__(self, model, loader, lr, device, writer):
        self.model = model
        self.loader = loader
        self.criterion1 = nn.MSELoss(size_average=True).to(device)
        self.criterion3 = COS_Loss().to(device)
        self.criterion4 = VGG_Encoder(device=device)
        self.device = device
        self.writer = writer
        self.step = 0
        parameters1 = []
        parameters2 = []
        parameters1.append( {'params': self.model.decomposer.encoder2.parameters(), 'lr': lr} )
        # parameters1.append( {'params': self.model.decomposer.decoder_normals.parameters(), 'lr': lr} )
        parameters1.append( {'params': self.model.decomposer.decoder_lights.parameters(), 'lr': lr} )
        parameters1.append( {'params': self.model.shader.parameters(), 'lr': lr} )
        parameters2.append( {'params': self.model.decomposer.encoder1.parameters(), 'lr': lr} )
        parameters2.append( {'params': self.model.decomposer.decoder_reflectance.parameters(), 'lr': lr} )
        self.optimizer_R = optim.Adam(parameters2, lr=lr)
        self.optimizer_S = optim.Adam(parameters1, lr=lr)

    def _epoch(self):
        self.model.train()

        progress = tqdm( total=len(self.loader.dataset))

        for _, labeled in enumerate(self.loader):
            labeled = [t.to(self.device) for t in labeled]

            input_g, albedo_g, shading_g, _ = labeled
            
            # input_fake, albedo_fake, shading_fake = self.model.forward(input_g)
            _, lab_refl_pred, lab_shad_pred = self.model.forward(input_g)

            # albedo_fake  = albedo_fake*mask_g

            # self.model.reflection.clamp_model_weights()
            # self.model.shader.clamp_model_weights()
            self.optimizer_R.zero_grad()
            refl_mse_loss = self.criterion1(lab_refl_pred, albedo_g)
            # refl_recon_loss = self.criterion1(lab_refl_pred*shading_g.repeat(1,3,1,1), input_g)
            refl_bf_loss = clc_Loss_albedo(lab_refl_pred, albedo_g)
            refl_cos_loss = self.criterion3(lab_refl_pred, albedo_g)
            refl_content_loss = self.criterion4(lab_refl_pred, albedo_g)
            refl_loss = refl_bf_loss + refl_cos_loss + refl_content_loss*0.1 + refl_mse_loss
            # refl_loss = refl_bf_loss + refl_cos_loss + refl_content_loss*0.1 + refl_recon_loss
            refl_loss.backward()
            self.optimizer_R.step()

            self.optimizer_S.zero_grad()
            # shad_recon_loss = self.criterion1(lab_shad_pred.repeat(1,3,1,1) * albedo_g, input_g)
            shad_bf_loss = clc_Loss_shading(lab_shad_pred, shading_g)
            shad_mse_loss = self.criterion1(lab_shad_pred, shading_g)
            shad_cos_loss = self.criterion3(lab_shad_pred, shading_g)
            # shape_cos_loss = self.criterion3(shape_pred, shading_g)
            shad_content_loss = self.criterion4(lab_shad_pred, shading_g)
            # shad_loss = shad_bf_loss + shad_content_loss*0.1 + shape_cos_loss + shad_recon_loss
            shad_loss = shad_bf_loss + shad_content_loss*0.1 + shad_mse_loss
            shad_loss.backward()
            self.optimizer_S.step()

            self.writer.add_scalar('supervision_refl_bf_loss', refl_bf_loss.item(), self.step)
            self.writer.add_scalar('supervision_refl_content_loss', refl_content_loss.item(), self.step)
            self.writer.add_scalar('supervision_refl_cos_loss', refl_cos_loss.item(), self.step)
            self.writer.add_scalar('supervision_shad_bf_loss', shad_bf_loss.item(), self.step)
            self.writer.add_scalar('supervision_shad_content_loss', shad_content_loss.item(), self.step)
            self.writer.add_scalar('supervision_shad_mse_loss', shad_mse_loss.item(), self.step)
            self.writer.add_scalar('supervision_shad_cos_loss', shad_cos_loss.item(), self.step)
            self.writer.add_scalar('supervision_refl_mse_loss', refl_mse_loss.item(), self.step)
            self.step += 1

            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f | %.5f| %.5f| %.5f | %.5f| %.5f' % (refl_mse_loss.item(), shad_mse_loss.item(), refl_bf_loss.item(), shad_bf_loss.item(), refl_content_loss.item(), shad_content_loss.item()) )
        return self.step

    def train(self):
        step = self._epoch()
        return step
    
    def update_lr(self, lr):
        for optim in [self.optimizer_R, self.optimizer_S]:
            for param_group in optim.param_groups:
                param_group['lr'] = lr
    
    def clamp_model_weights(self, Vmin=-0.1, Vmax=0.1):
        for _, x in enumerate(self.model):
            for p in x.parameters():
                p.data.clamp_(Vmin, Vmax)
        return self
    
