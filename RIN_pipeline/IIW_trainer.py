import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_ssim
import adabound
from tqdm import tqdm

# from .WHDRHingeLossPara import WHDRHingeLossPara, WHDRHingeLossParaModule
from .WHDRHingeLossParaPro import WHDRHingeLossParaProModule
from .eval_meterics import clc_pad
from .ranger import Ranger


class IIWTrainer:
    def __init__(self, model, loader, device, writer, args):
        self.model = model
        self.loader = loader
        self.device = device
        self.writer = writer
        self.step = 0
        self.lr = args.lr
        self.whdr_loss = WHDRHingeLossParaProModule()
        self.optimizer = Ranger(self.model.parameters(), lr=self.lr)

    def _epoch(self):
        self.model.train()

        progress = tqdm( total=len(self.loader.dataset))

        for _, labeled in enumerate(self.loader):
            input_g, label_tensor = labeled
            _, _, h, w = input_g.size()
            pad_h,pad_w = clc_pad(h,w,16)
            input_g = input_g.to(self.device)
            tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
            tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
            input_g = tmp_pad(input_g)
            label_tensor = label_tensor.float().to(self.device)
            self.optimizer.zero_grad()
            print(self.model.encoder1[-1].weight.data)
            pred = self.model(input_g)
            pred = tmp_inversepad(pred)
            loss = self.whdr_loss.forward(pred, label_tensor)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('loss', loss.item(), self.step)
            self.step += 1

            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f |' % loss.item())

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
    
