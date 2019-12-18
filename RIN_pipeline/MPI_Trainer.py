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



class MPI_Trainer:
    def __init__(self, model, loader, lr, device, writer, step):
        self.model = model
        self.loader = loader
        self.criterion1 = nn.MSELoss(size_average=True).to(device)
        # self.criterion3 = COS_Loss().to(device)
        self.criterion4 = VGG_Encoder()
        self.device = device
        self.writer = writer
        self.step = step
        # self.optimizer1 = optim.Adam(self.model.reflection.parameters(), lr=lr)
        # self.optimizer2 = optim.Adam(self.model.shader.parameters(), lr=lr)
    def __epoch(self):
        self.model.train()

        progress = tqdm( total=len(self.loader.dataset))

        for _, labeled in enumerate(self.loader):
            labeled = [t.to(self.device) for t in labeled]

            input_g, albedo_g, shading_g, mask_g = labeled
            
            input_fake, albedo_fake, shading_fake = self.model.forward(input_g)

            # albedo_fake  = albedo_fake*mask_g

            self.model.reflection.clamp_model_weights()
            self.model.shader.clamp_model_weights()

            self.model.reflection.optimizer_zerograd()
            self.model.shader.optimizer_zerograd()
            albedo_bf_loss = clc_Loss_albedo(albedo_fake, albedo_g)
            albedo_content_loss = self.criterion4(albedo_fake, albedo_g)
            albedo_loss = albedo_bf_loss + albedo_content_loss*0.5

            shad_bf_loss = clc_Loss_shading(shading_fake, shading_g)
            shad_content_loss = self.criterion4(shading_fake, shading_g)
            shad_loss = shad_bf_loss + shad_content_loss*0.5

            recon_loss = self.criterion1(input_fake, input_g)

            loss = albedo_loss + shad_loss + recon_loss
            loss.backward()
            self.model.reflection.optimizer_step()
            self.model.shader.optimizer_step()

            self.writer.add_scalar('albedo_content_loss', albedo_content_loss.item(), self.step)
            self.writer.add_scalar('albedo_bf_loss', albedo_bf_loss.item(), self.step)
            self.writer.add_scalar('shad_bf_loss', shad_bf_loss.item(), self.step)
            self.writer.add_scalar('shad_content_loss', shad_content_loss.item(), self.step)
            self.writer.add_scalar('recon_loss', recon_loss.item(), self.step)
            self.writer.add_scalar('shad_loss', shad_loss.item(), self.step)
            self.writer.add_scalar('albedo_loss', albedo_loss.item(), self.step)
            self.writer.add_scalar('loss', loss.item(), self.step)
            self.step += 1

            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f | %.5f| %.5f' % (shad_loss.item(), albedo_loss.item(), loss.item()))
 
        return self.step

    def train(self):
        errors = self.__epoch()
        return errors
    
    def update_lr(self, lr):
        for optim in self.optimizer:
            for param_group in optim.param_group:
                param_group['lr'] = lr
    
    def clamp_model_weights(self, Vmin=-0.1, Vmax=0.1):
        for _, x in enumerate(self.model):
            for p in x.parameters():
                p.data.clamp_(Vmin, Vmax)
        return self
    
