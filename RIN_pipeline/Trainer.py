import sys, math, numpy as np, pdb
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.optim as optim
from torch.autograd import Variable
import pipeline

class Trainer:
    def __init__(self, model, loader, lr,epoch_size,device):
        self.model = model
        self.loader = loader
        self.criterion = nn.MSELoss(size_average=True).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epoch_size = epoch_size
        self.device = device

    def __epoch(self):
        self.model.train()
        losses = pipeline.AverageMeter(1)
        progress = tqdm( total=len(self.loader.dataset) )
        tqdm.monitor_interval = 0
        for ind, tensors in enumerate(self.loader):

            inp = [ t.float().to(self.device) for t in tensors[:-1] ]
            targ = tensors[-1].float().to(self.device)

            self.optimizer.zero_grad()
            out = self.model.forward(*inp)
            loss = self.criterion(out, targ)
            loss.backward()
            self.optimizer.step()

            losses.update( [loss.item()] )
            progress.update(self.loader.batch_size)
            progress.set_description( str(loss.item()) )
            if losses.count * self.loader.batch_size > self.epoch_size:
                break
        return losses.avgs

    def train(self):
        # t = trange(iters)
        # for i in t:
        err = self.__epoch()
            # t.set_description( str(err) )
        #return self.model
        return err





