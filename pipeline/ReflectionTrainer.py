from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

 

class ReflectionTrainer:
    def __init__(self, model, loader, device, optimizer, writer, step):
        self.model = model
        self.loader = loader
        self.criterion = nn.MSELoss(size_average=True).to(device)
        self.device = device
        self.optimizer = optimizer
        self.writer = writer
        self.step = step

    def __epoch(self):
        self.model.train()

        progress = tqdm(total=len(self.loader.dataset))

        for _, batch in enumerate(self.loader):
            batch_data = [t.float().to(self.device) for t in batch]

            lab_inp, lab_refl_targ = batch_data
            
            lab_refl_pred = self.model.forward(lab_inp)
            
            self.optimizer.zero_grad()
            refl_loss = self.criterion(lab_refl_pred, lab_refl_targ)
            refl_loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('train_refl_loss', refl_loss.item(), self.step)
            self.step += 1
            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f' % (refl_loss.item()))

        return self.step

    def train(self):
        return self.__epoch()
        




